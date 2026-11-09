// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Krait: GPU-Accelerated DNA Sequence Aligner for short reads.
// Copyright (C) 2026 Valentyn Zhukovetskyi <zukovetski@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://gnu.org/licenses/>.

use crossbeam::channel::Sender;
use flate2::read::MultiGzDecoder;
use std::io::Read;
use std::time::Instant;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    num::NonZero,
    ops::Range,
    path::Path,
};

use crate::telemetry::StageMetrics;

pub struct SingleOffsets {
    pub name: Range<usize>,
    pub seq: Range<usize>,
    pub qual: Range<usize>,
}

pub struct FastqChunk {
    pub data: Vec<u8>,
    pub offsets: Vec<SingleOffsets>,
}

/// Dedicated single-file FASTQ worker with bulk decompression and SIMD line parsing.
pub fn run_single_reader(
    path: String,
    batch_size: usize,
    chunk_tx: Sender<FastqChunk>,
) -> std::thread::JoinHandle<StageMetrics> {
    std::thread::Builder::new()
        .name(format!(
            "krait-reader-{}",
            Path::new(&path).file_name().unwrap().to_string_lossy()
        ))
        .spawn(move || {
            let mut metrics = StageMetrics::default();
            let mut t_busy = Instant::now();
            let mut reader = open_flexible_reader(&path, 2);

            let mut data_size = batch_size * 220;
            let mut offsets_size = batch_size;
            let mut data_buffer = Vec::with_capacity(data_size);
            let mut offsets = Vec::with_capacity(offsets_size);
            let mut scratchpad = vec![0u8; data_size.div_ceil(10)];
            let mut parse_cursor = 0;
            let mut eof = false;

            loop {
                // 1. Bulk read decompressed bytes directly into data_buffer
                while offsets.len() < batch_size && !eof {
                    let prev_len = data_buffer.len();

                    match reader.read(&mut scratchpad) {
                        Ok(0) => {
                            eof = true;
                        }
                        Ok(n) => {
                            data_buffer.extend_from_slice(&scratchpad[..n]);
                            // Parse records out of the newly available bytes
                            parse_records_in_memory(
                                &data_buffer,
                                &mut parse_cursor,
                                &mut offsets,
                                batch_size,
                            );
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                            data_buffer.truncate(prev_len);
                        }
                        Err(e) => panic!("Decompression read error in {path}: {e}"),
                    }
                }

                if offsets.is_empty() && eof {
                    break; // Normal EOF reached
                }

                // 2. We have accumulated a full batch of records (or reached final EOF)
                let records_in_batch = offsets.len().min(batch_size);

                // End of the last full record in this batch
                let cutoff_byte = if records_in_batch > 0 {
                    offsets[records_in_batch - 1].qual.end + 1 // +1 skips the trailing '\n'
                } else {
                    0
                };

                data_size = data_size.max(cutoff_byte);
                offsets_size = offsets_size.max(offsets.len());

                // 3. Extract the leftover unparsed tail for the next chunk
                let mut leftover = Vec::with_capacity(data_size);
                leftover.extend_from_slice(&data_buffer[cutoff_byte..]);
                data_buffer.truncate(cutoff_byte);

                metrics.items += offsets.len() as u64;
                // 4. Send the perfectly sized, contiguous chunk
                let chunk = FastqChunk {
                    data: std::mem::replace(&mut data_buffer, leftover),
                    offsets: std::mem::replace(&mut offsets, Vec::with_capacity(offsets_size)),
                };

                metrics.busy += t_busy.elapsed();
                t_busy = Instant::now();

                if chunk_tx.send(chunk).is_err() {
                    break;
                }

                parse_cursor = 0;

                // Re-parse the leftover in the new buffer coordinates if any full lines exist
                if !data_buffer.is_empty() {
                    parse_records_in_memory(
                        &data_buffer,
                        &mut parse_cursor,
                        &mut offsets,
                        batch_size,
                    );
                }
            }
            metrics
        })
        .unwrap()
}

pub fn open_flexible_reader(path: &str, worker_count: usize) -> Box<dyn BufRead + Send> {
    let file = File::open(path).unwrap_or_else(|_| panic!("Failed to open file: {path}"));

    let is_gz = Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz") || ext.eq_ignore_ascii_case("bgz"));

    if !is_gz {
        // Uncompressed raw FASTQ: direct buffered I/O
        return Box::new(BufReader::with_capacity(128 * 1024, file));
    }

    // 1. Try multi-threaded BGZF reader (pure Rust via zlib-rs + rayon)
    let buf_file = BufReader::with_capacity(128 * 1024, file);
    let mut bgzf_reader = noodles_bgzf::io::MultithreadedReader::with_worker_count(
        NonZero::new(worker_count).unwrap(),
        buf_file,
    );

    // Probe the first block: if it's a valid BGZF, return it!
    if bgzf_reader.fill_buf().is_ok() {
        return Box::new(bgzf_reader);
    }

    // 2. Fallback: standard legacy monolithic GZIP (flate2 + zlib-rs)
    // Reopen the file since the probe might have partially advanced the stream
    let raw_file = File::open(path).expect("File must exist on fallback");
    let buf_file = BufReader::with_capacity(128 * 1024, raw_file);
    let gz_decoder = MultiGzDecoder::new(buf_file);

    Box::new(BufReader::with_capacity(128 * 1024, gz_decoder))
}

/// Parses FASTQ records directly from memory using fast memchr newline scanning.
fn parse_records_in_memory(
    data: &[u8],
    cursor: &mut usize,
    offsets: &mut Vec<SingleOffsets>,
    max_records: usize,
) {
    while offsets.len() < max_records {
        let start = *cursor;
        let rem = &data[start..];

        // Line 1: Header / Name
        let Some(nl1) = memchr::memchr(b'\n', rem) else {
            break;
        };
        let name_raw = &rem[..nl1];
        let name_clean = clean_fastq_name(name_raw);
        let name_start = start + usize::from(name_raw.starts_with(b"@"));
        let name_range = name_start..name_start + name_clean.len();

        // Line 2: Sequence
        let rem2 = &rem[nl1 + 1..];
        let Some(nl2) = memchr::memchr(b'\n', rem2) else {
            break;
        };
        let seq_len = trim_trailing_cr(&rem2[..nl2]).len();
        let seq_start = start + nl1 + 1;
        let seq_range = seq_start..seq_start + seq_len;

        // Line 3: '+' Separator (skip line)
        let rem3 = &rem2[nl2 + 1..];
        let Some(nl3) = memchr::memchr(b'\n', rem3) else {
            break;
        };

        // Line 4: Quality
        let rem4 = &rem3[nl3 + 1..];
        let Some(nl4) = memchr::memchr(b'\n', rem4) else {
            break;
        };
        let qual_len = trim_trailing_cr(&rem4[..nl4]).len();
        let qual_start = start + nl1 + 1 + nl2 + 1 + nl3 + 1;
        let qual_range = qual_start..qual_start + qual_len;

        // Advance cursor past the 4th newline
        *cursor = qual_start + nl4 + 1;

        offsets.push(SingleOffsets {
            name: name_range,
            seq: seq_range,
            qual: qual_range,
        });
    }
}

fn trim_trailing_cr(mut s: &[u8]) -> &[u8] {
    while !s.is_empty() && (s[s.len() - 1] == b'\n' || s[s.len() - 1] == b'\r') {
        s = &s[..s.len() - 1];
    }
    s
}

fn clean_fastq_name(mut name: &[u8]) -> &[u8] {
    name = trim_trailing_cr(name);
    if name.starts_with(b"@") {
        name = &name[1..];
    }
    if let Some(pos) = memchr::memchr2(b' ', b'\t', name) {
        name = &name[..pos];
    }
    if name.ends_with(b"/1") || name.ends_with(b"/2") {
        name = &name[..name.len() - 2];
    }
    name
}
