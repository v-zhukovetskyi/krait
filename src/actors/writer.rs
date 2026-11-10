use crossbeam::channel::Receiver;
use indicatif::ProgressBar;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use crate::telemetry::StageMetrics;
use crate::{
    actors::producer::{PairedMeta, ReferenceGenome, SingleMeta, WriteBatch},
    constants::{
        CIGAR_CAPACITY, CIGAR_SENTINEL, MIN_MAPPED_SCORE, SCORES_PER_READ, WRITER_BUFFER_CAPACITY,
        sam_flags,
    },
};
pub struct ScoreBatch {
    pub scores_and_coords: Vec<i32>,
    pub cigar_bin: Vec<u32>,
}
const CHUNK_RECORDS: usize = 512;
const APPROX_BYTES_PER_RECORD: usize = 200;

/// Spawns the dedicated background thread to format and stream SAM records to disk.
pub fn run_writer_worker(
    output_path: String,
    genome: Arc<ReferenceGenome>,
    meta_rx: Receiver<WriteBatch>,
    score_rx: Receiver<ScoreBatch>,
    progress_bar: ProgressBar,
) -> std::thread::JoinHandle<StageMetrics> {
    std::thread::Builder::new()
        .name("krait-writer".to_string())
        .spawn(move || {
            let mut metrics = StageMetrics::default();

            // Select output destination: stdout or file
            let target: Box<dyn Write + Send> = if output_path == "-" {
                Box::new(std::io::stdout())
            } else {
                let file = File::create(&output_path)
                    .unwrap_or_else(|e| panic!("Failed to create output file '{output_path}': {e}"));
                Box::new(file)
            };
            let mut writer = BufWriter::with_capacity(WRITER_BUFFER_CAPACITY, target);

            write_sam_header(&mut writer, &genome).expect("Failed to write SAM header");

            let mut total_processed: u64 = 0;
            let mut cur_meta: Option<BatchCursor<WriteBatch>> = None;
            let mut cur_score: Option<BatchCursor<ScoreBatch>> = None;

            loop {
                let t_idle = Instant::now();
                // Refill exhausted cursors from crossbeam channels
                if cur_meta.is_none() {
                    cur_meta = meta_rx.recv().ok().map(BatchCursor::new);
                }
                if cur_score.is_none() {
                    cur_score = score_rx.recv().ok().map(BatchCursor::new);
                }

                metrics.idle += t_idle.elapsed();
                let t_busy = Instant::now();

                // Terminate when either stream is closed and drained
                let (Some(meta), Some(score)) = (&mut cur_meta, &mut cur_score) else {
                    break;
                };

                // Enforce even task count for paired reads to keep pairs together
                let available = meta.remaining().min(score.remaining());
                let tasks_to_process = if meta.is_paired() {
                    available & !1
                } else {
                    available
                };

                if tasks_to_process == 0 {
                    if meta.remaining() <= 1 { cur_meta = None; }
                    if score.remaining() <= 1 { cur_score = None; }
                    continue;
                }

                let meta_range = meta.offset..meta.offset + tasks_to_process;
                let score_range = score.offset..score.offset + tasks_to_process;

                meta.advance(tasks_to_process);
                score.advance(tasks_to_process);

                // Render SAM records in parallel chunks into contiguous byte buffers
                let sam_chunks = render_sam_chunks(
                    &meta.batch,
                    meta_range,
                    &score.batch,
                    score_range,
                    &genome,
                );

                // Bulk write contiguous byte blocks directly to disk
                for chunk in sam_chunks {
                    let _ = writer.write_all(&chunk);
                }

                total_processed += tasks_to_process as u64;
                progress_bar.inc(tasks_to_process as u64);

                // Evict cursors that have been completely written
                if cur_meta.as_ref().is_some_and(<BatchCursor<WriteBatch>>::is_exhausted) {
                    cur_meta = None;
                }
                if cur_score.as_ref().is_some_and(<BatchCursor<ScoreBatch>>::is_exhausted) {
                    cur_score = None;
                }

                metrics.busy += t_busy.elapsed();
                metrics.items += tasks_to_process as u64;
            }

            let _ = writer.flush();
            if progress_bar.is_hidden() {
                eprintln!("📝 [Writer] File written successfully. Total reads processed: {total_processed}");
            }

            metrics
        })
        .unwrap()
}

/// Writes standard SAM headers (@HD and @SQ contig definitions).
fn write_sam_header<W: Write>(writer: &mut W, genome: &ReferenceGenome) -> std::io::Result<()> {
    writer.write_all(b"@HD\tVN:1.6\tSO:unsorted\n")?;
    for contig in &genome.contigs {
        writeln!(writer, "@SQ\tSN:{}\tLN:{}", contig.name, contig.len)?;
    }
    Ok(())
}

/// Tracks reading progress within a partially processed batch.
struct BatchCursor<T> {
    batch: T,
    offset: usize,
}

impl<T> BatchCursor<T> {
    const fn new(batch: T) -> Self {
        Self { batch, offset: 0 }
    }

    const fn advance(&mut self, count: usize) {
        self.offset += count;
    }
}

impl BatchCursor<WriteBatch> {
    const fn total(&self) -> usize {
        match &self.batch {
            WriteBatch::Single(b) => b.meta.len(),
            WriteBatch::Paired(b) => b.meta.len(),
        }
    }

    const fn remaining(&self) -> usize {
        self.total() - self.offset
    }

    const fn is_paired(&self) -> bool {
        matches!(&self.batch, WriteBatch::Paired(_))
    }

    const fn is_exhausted(&self) -> bool {
        self.offset >= self.total()
    }
}

impl BatchCursor<ScoreBatch> {
    const fn total(&self) -> usize {
        self.batch.scores_and_coords.len() / SCORES_PER_READ
    }

    const fn remaining(&self) -> usize {
        self.total() - self.offset
    }

    const fn is_exhausted(&self) -> bool {
        self.offset >= self.total()
    }
}

/// Dispatches parallel rendering of SAM chunks according to batch sequencing mode.
fn render_sam_chunks(
    meta_batch: &WriteBatch,
    meta_range: Range<usize>,
    score_batch: &ScoreBatch,
    score_range: Range<usize>,
    genome: &ReferenceGenome,
) -> Vec<Vec<u8>> {
    let score_slice = &score_batch.scores_and_coords
        [score_range.start * SCORES_PER_READ..score_range.end * SCORES_PER_READ];
    let cigar_slice = &score_batch.cigar_bin
        [score_range.start * CIGAR_CAPACITY..score_range.end * CIGAR_CAPACITY];

    match meta_batch {
        WriteBatch::Single(b) => {
            let metas = &b.meta[meta_range];
            render_single_chunks(metas, &b.raw_data, score_slice, cigar_slice, genome)
        }
        WriteBatch::Paired(b) => {
            let metas = &b.meta[meta_range];
            render_paired_chunks(
                metas,
                &b.raw_data1,
                &b.raw_data2,
                score_slice,
                cigar_slice,
                genome,
            )
        }
    }
}

/// Formats Single-End records into chunked byte buffers in parallel.
fn render_single_chunks(
    metas: &[SingleMeta],
    raw_data: &[u8],
    score_slice: &[i32],
    cigar_slice: &[u32],
    genome: &ReferenceGenome,
) -> Vec<Vec<u8>> {
    metas
        .par_chunks(CHUNK_RECORDS)
        .enumerate()
        .map(|(chunk_idx, sub_metas)| {
            let offset = chunk_idx * CHUNK_RECORDS;
            let mut buffer = Vec::with_capacity(sub_metas.len() * APPROX_BYTES_PER_RECORD);

            for (i, meta) in sub_metas.iter().enumerate() {
                let task_idx = offset + i;
                let s_slice =
                    &score_slice[task_idx * SCORES_PER_READ..(task_idx + 1) * SCORES_PER_READ];
                let c_slice =
                    &cigar_slice[task_idx * CIGAR_CAPACITY..(task_idx + 1) * CIGAR_CAPACITY];

                write_single_record(&mut buffer, meta, raw_data, s_slice, c_slice, genome);
            }

            buffer
        })
        .collect()
}

/// Formats Paired-End records into chunked byte buffers in parallel.
fn render_paired_chunks(
    metas: &[PairedMeta],
    raw_data1: &[u8],
    raw_data2: &[u8],
    score_slice: &[i32],
    cigar_slice: &[u32],
    genome: &ReferenceGenome,
) -> Vec<Vec<u8>> {
    metas
        .par_chunks(CHUNK_RECORDS * 2)
        .enumerate()
        .map(|(chunk_idx, sub_metas)| {
            let pair_offset = chunk_idx * CHUNK_RECORDS;
            let mut buffer = Vec::with_capacity(sub_metas.len() * APPROX_BYTES_PER_RECORD);

            let (chunks, _) = sub_metas.as_chunks::<2>();

            for (i, pair) in chunks.iter().enumerate() {
                let pair_idx = pair_offset + i;
                let task_base = pair_idx * 2;
                let s_slice =
                    &score_slice[task_base * SCORES_PER_READ..(task_base + 2) * SCORES_PER_READ];
                let c_slice =
                    &cigar_slice[task_base * CIGAR_CAPACITY..(task_base + 2) * CIGAR_CAPACITY];

                write_paired_record(
                    &mut buffer,
                    pair,
                    raw_data1,
                    raw_data2,
                    s_slice,
                    c_slice,
                    genome,
                );
            }

            buffer
        })
        .collect()
}

/// Serializes a single alignment record directly into the destination byte buffer.
fn write_single_record(
    buffer: &mut Vec<u8>,
    meta: &SingleMeta,
    raw_data: &[u8],
    score_slice: &[i32],
    cigar_slice: &[u32],
    genome: &ReferenceGenome,
) {
    let score = score_slice[0];
    let start_j: usize = score_slice[1]
        .try_into()
        .expect("Score offset cannot be negative");

    let name = &raw_data[meta.name.clone()];

    // Fast path: unmapped read
    if score < MIN_MAPPED_SCORE {
        buffer.extend_from_slice(name);
        buffer.extend_from_slice(b"\t4\t*\t0\t0\t*\t*\t0\t0\t");
        write_seq(buffer, raw_data, meta.seq.clone(), meta.is_rev);
        buffer.push(b'\t');
        write_qual(buffer, raw_data, meta.qual.clone(), meta.is_rev);
        buffer.push(b'\n');
        return;
    }

    let flag = if meta.is_rev { sam_flags::REVERSE } else { 0 };
    let global_pos = meta.pos as usize + start_j;

    if let Some((chrom, local_pos)) = genome.resolve_coords(global_pos) {
        let seq_len = meta.seq.len();
        buffer.extend_from_slice(name);
        let _ = write!(buffer, "\t{flag}\t{chrom}\t{local_pos}\t60\t");
        write_cigar(buffer, cigar_slice, seq_len);
        buffer.extend_from_slice(b"\t*\t0\t0\t");
        write_seq(buffer, raw_data, meta.seq.clone(), meta.is_rev);
        buffer.push(b'\t');
        write_qual(buffer, raw_data, meta.qual.clone(), meta.is_rev);
        let _ = writeln!(buffer, "\tAS:i:{score}");
    } else {
        buffer.extend_from_slice(name);
        buffer.extend_from_slice(b"\t4\t*\t0\t0\t*\t*\t0\t0\t");
        write_seq(buffer, raw_data, meta.seq.clone(), meta.is_rev);
        buffer.push(b'\t');
        write_qual(buffer, raw_data, meta.qual.clone(), meta.is_rev);
        buffer.push(b'\n');
    }
}

/// Serializes a paired-end alignment record pair directly into the destination byte buffer.
#[allow(clippy::too_many_lines)]
fn write_paired_record(
    buffer: &mut Vec<u8>,
    pair: &[PairedMeta],
    raw_data1: &[u8],
    raw_data2: &[u8],
    score_slice: &[i32],
    cigar_slice: &[u32],
    genome: &ReferenceGenome,
) {
    let s1 = score_slice[0];
    let start_j1: usize = score_slice[1]
        .try_into()
        .expect("Score offset cannot be negative");

    let s2 = score_slice[SCORES_PER_READ];
    let start_j2: usize = score_slice[SCORES_PER_READ + 1]
        .try_into()
        .expect("Score offset cannot be negative");

    let r1_mapped = s1 >= MIN_MAPPED_SCORE;
    let r2_mapped = s2 >= MIN_MAPPED_SCORE;

    let name = &raw_data1[pair[0].name.clone()];

    // Fast path: both mates unmapped
    if !r1_mapped && !r2_mapped {
        let f1 =
            sam_flags::PAIRED | sam_flags::UNMAPPED | sam_flags::MATE_UNMAPPED | sam_flags::READ1;
        let f2 =
            sam_flags::PAIRED | sam_flags::UNMAPPED | sam_flags::MATE_UNMAPPED | sam_flags::READ2;

        buffer.extend_from_slice(name);
        let _ = write!(buffer, "\t{f1}\t*\t0\t0\t*\t*\t0\t0\t");
        write_seq(buffer, raw_data1, pair[0].seq.clone(), pair[0].is_rev);
        buffer.push(b'\t');
        write_qual(buffer, raw_data1, pair[0].qual.clone(), pair[0].is_rev);
        buffer.push(b'\n');

        buffer.extend_from_slice(name);
        let _ = write!(buffer, "\t{f2}\t*\t0\t0\t*\t*\t0\t0\t");
        write_seq(buffer, raw_data2, pair[1].seq.clone(), pair[1].is_rev);
        buffer.push(b'\t');
        write_qual(buffer, raw_data2, pair[1].qual.clone(), pair[1].is_rev);
        buffer.push(b'\n');
        return;
    }

    let g_pos1 = pair[0].pos as usize + start_j1;
    let g_pos2 = pair[1].pos as usize + start_j2;

    let (chrom1, p1) = if r1_mapped {
        genome.resolve_coords(g_pos1).unwrap_or(("*", 0))
    } else {
        ("*", 0)
    };
    let (chrom2, p2) = if r2_mapped {
        genome.resolve_coords(g_pos2).unwrap_or(("*", 0))
    } else {
        ("*", 0)
    };

    let same_chrom = r1_mapped && r2_mapped && (chrom1 == chrom2);
    let is_proper = same_chrom && pair[0].is_proper;
    let proper_bit = if is_proper { sam_flags::PROPER_PAIR } else { 0 };

    let mut flag1 = sam_flags::PAIRED | proper_bit | sam_flags::READ1;
    if !r1_mapped {
        flag1 |= sam_flags::UNMAPPED;
    }
    if !r2_mapped {
        flag1 |= sam_flags::MATE_UNMAPPED;
    }
    if r1_mapped && pair[0].is_rev {
        flag1 |= sam_flags::REVERSE;
    }
    if r2_mapped && pair[1].is_rev {
        flag1 |= sam_flags::MATE_REVERSE;
    }

    let mut flag2 = sam_flags::PAIRED | proper_bit | sam_flags::READ2;
    if !r2_mapped {
        flag2 |= sam_flags::UNMAPPED;
    }
    if !r1_mapped {
        flag2 |= sam_flags::MATE_UNMAPPED;
    }
    if r2_mapped && pair[1].is_rev {
        flag2 |= sam_flags::REVERSE;
    }
    if r1_mapped && pair[0].is_rev {
        flag2 |= sam_flags::MATE_REVERSE;
    }

    let rnext1 = if same_chrom { "=" } else { chrom2 };
    let rnext2 = if same_chrom { "=" } else { chrom1 };

    let tlen1 = if is_proper {
        if p1 <= p2 {
            i32::try_from(pair[0].tlen).unwrap()
        } else {
            -i32::try_from(pair[0].tlen).unwrap()
        }
    } else {
        0
    };
    let tlen2 = if is_proper { -tlen1 } else { 0 };

    let cigar_r1 = &cigar_slice[..CIGAR_CAPACITY];
    let cigar_r2 = &cigar_slice[CIGAR_CAPACITY..];

    // --- Mate 1 ---
    buffer.extend_from_slice(name);
    let _ = write!(buffer, "\t{flag1}\t{chrom1}\t{p1}\t60\t");
    if r1_mapped {
        write_cigar(buffer, cigar_r1, pair[0].seq.len());
    } else {
        buffer.push(b'*');
    }
    let _ = write!(buffer, "\t{rnext1}\t{p2}\t{tlen1}\t");
    write_seq(buffer, raw_data1, pair[0].seq.clone(), pair[0].is_rev);
    buffer.push(b'\t');
    write_qual(buffer, raw_data1, pair[0].qual.clone(), pair[0].is_rev);
    let _ = writeln!(buffer, "\tAS:i:{s1}");

    // --- Mate 2 ---
    buffer.extend_from_slice(name);
    let _ = write!(buffer, "\t{flag2}\t{chrom2}\t{p2}\t60\t");
    if r2_mapped {
        write_cigar(buffer, cigar_r2, pair[1].seq.len());
    } else {
        buffer.push(b'*');
    }
    let _ = write!(buffer, "\t{rnext2}\t{p1}\t{tlen2}\t");
    write_seq(buffer, raw_data2, pair[1].seq.clone(), pair[1].is_rev);
    buffer.push(b'\t');
    write_qual(buffer, raw_data2, pair[1].qual.clone(), pair[1].is_rev);
    let _ = writeln!(buffer, "\tAS:i:{s2}");
}

/// Appends sequence bases directly to the buffer, applying reverse-complement on the fly without intermediate allocation.
fn write_seq(buffer: &mut Vec<u8>, raw_bytes: &[u8], range: Range<usize>, is_rev: bool) {
    let slice = &raw_bytes[range];
    if is_rev {
        buffer.extend(slice.iter().rev().map(|&b| match b.to_ascii_uppercase() {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            _ => b'N',
        }));
    } else {
        buffer.extend_from_slice(slice);
    }
}

/// Appends quality scores directly to the buffer, reversing them in place if the read is on the reverse strand.
fn write_qual(buffer: &mut Vec<u8>, raw_bytes: &[u8], range: Range<usize>, is_rev: bool) {
    let slice = &raw_bytes[range];
    if is_rev {
        buffer.extend(slice.iter().rev().copied());
    } else {
        buffer.extend_from_slice(slice);
    }
}

/// Decodes packed CIGAR words and writes operations directly into the buffer without allocating a `String`.
fn write_cigar(buffer: &mut Vec<u8>, cigar_slice: &[u32], raw_len: usize) {
    let mut cur_op = 'X';
    let mut cur_count = 0usize;
    let mut query_consumed = 0usize;
    let mut has_written = false;

    for &val in cigar_slice.iter().rev() {
        if val == CIGAR_SENTINEL {
            continue;
        }

        let count = (val >> 2) as usize;
        let op = match val & 3 {
            0 => 'M',
            1 => 'I',
            2 => 'D',
            _ => 'S',
        };

        if matches!(op, 'M' | 'I' | 'S') {
            query_consumed += count;
        }

        if op == cur_op {
            cur_count += count;
        } else {
            if cur_op != 'X' {
                let _ = write!(buffer, "{cur_count}{cur_op}");
                has_written = true;
            }
            cur_op = op;
            cur_count = count;
        }
    }

    if raw_len > query_consumed {
        let extra = raw_len - query_consumed;
        if cur_op == 'S' {
            cur_count += extra;
        } else {
            if cur_op != 'X' {
                let _ = write!(buffer, "{cur_count}{cur_op}");
                has_written = true;
            }
            cur_op = 'S';
            cur_count = extra;
        }
    }

    if cur_op != 'X' {
        let _ = write!(buffer, "{cur_count}{cur_op}");
        has_written = true;
    }

    if !has_written {
        buffer.push(b'*');
    }
}
