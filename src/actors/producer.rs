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

use fxhash::{FxBuildHasher, FxHashMap};
use rayon::prelude::*;

use std::time::Instant;
use std::{borrow::Cow, collections::HashMap, sync::Arc};

use crate::actors::reader::{FastqChunk, SingleOffsets};
use crate::telemetry::StageMetrics;
use crate::{
    cli::Args,
    constants::{
        FLANK, MAX_CANDIDATE_PAIRS, MAX_INSERT, MIN_INSERT, READ_LEN, REF_LEN, RESCUE_OFFSET,
        RESCUE_TLEN,
    },
};
pub struct SingleMeta {
    pub name: std::ops::Range<usize>,
    pub pos: u32,
    pub seq: std::ops::Range<usize>,
    pub qual: std::ops::Range<usize>,
    pub is_rev: bool,
}

pub struct PairedMeta {
    pub name: std::ops::Range<usize>,
    pub pos: u32,
    pub seq: std::ops::Range<usize>,
    pub qual: std::ops::Range<usize>,
    pub is_rev: bool,
    pub is_proper: bool,
    pub tlen: i32,
}

pub struct SingleWriteBatch {
    pub raw_data: Vec<u8>,
    pub meta: Vec<SingleMeta>,
}

pub struct PairedWriteBatch {
    pub raw_data1: Vec<u8>,
    pub raw_data2: Vec<u8>,
    pub meta: Vec<PairedMeta>,
}

pub enum WriteBatch {
    Single(SingleWriteBatch),
    Paired(PairedWriteBatch),
}

pub struct ComputeChunk {
    pub task_arena: Vec<u32>,
    pub task_count: usize,
}

pub struct Contig {
    pub name: String,
    pub len: usize,
    pub global_start: usize,
}

pub struct ReferenceGenome {
    pub seq: Vec<u8>,
    pub contigs: Vec<Contig>,
}

impl ReferenceGenome {
    /// Maps a global 0-based coordinate to its contig name and 1-based local position.
    pub fn resolve_coords(&self, global_pos: usize) -> Option<(&str, usize)> {
        // Binary search for the contig satisfying global_start <= global_pos
        let idx = self
            .contigs
            .partition_point(|c| c.global_start <= global_pos)
            .checked_sub(1)?;

        let contig = &self.contigs[idx];
        let local_pos = global_pos - contig.global_start;

        if local_pos < contig.len {
            Some((&contig.name, local_pos + 1)) // SAM specification uses 1-based indexing
        } else {
            None // Falls into the inter-contig 'N' padding gap
        }
    }
}

pub enum ReaderInput {
    Single(crossbeam::channel::Receiver<FastqChunk>),
    Paired(
        crossbeam::channel::Receiver<FastqChunk>,
        crossbeam::channel::Receiver<FastqChunk>,
    ),
}

/// Spawns the producer thread responsible for k-mer indexing and chunk processing.
pub fn run_producer_worker(
    args: Args,
    genome: Arc<ReferenceGenome>,
    input: ReaderInput,
    producer_tx: crossbeam::channel::Sender<ComputeChunk>,
    meta_tx: crossbeam::channel::Sender<WriteBatch>,
) -> std::thread::JoinHandle<StageMetrics> {
    std::thread::Builder::new()
        .name("krait-producer".to_string())
        .spawn(move || {
            let mut metrics = StageMetrics::default();
            let index = build_kmer_index(&genome, args.kmer_size);
            let mut base_read_id = 0u64;

            let mut t_idle = Instant::now();
            match input {
                ReaderInput::Single(rx) => {
                    while let Ok(chunk) = rx.recv() {
                        metrics.idle += t_idle.elapsed();
                        let t_busy = Instant::now();
                        let num_read = chunk.data.len();

                        let (wb, cb) =
                            process_single_chunk(chunk, &genome, &index, &args, &mut base_read_id);
                        if producer_tx.send(cb).is_err() || meta_tx.send(wb).is_err() {
                            break;
                        }
                        if args.limit > 0 && base_read_id >= args.limit {
                            return metrics;
                        }
                        metrics.busy += t_busy.elapsed();
                        metrics.items += num_read as u64;
                        t_idle = Instant::now();
                    }
                }
                ReaderInput::Paired(r1_rx, r2_rx) => {
                    while let (Ok(chunk1), Ok(chunk2)) = (r1_rx.recv(), r2_rx.recv()) {
                        metrics.idle += t_idle.elapsed();
                        let t_busy = Instant::now();
                        let num_pairs = chunk1.offsets.len().min(chunk2.offsets.len());

                        let (wb, cb) = process_paired_chunks(
                            chunk1,
                            chunk2,
                            &genome,
                            &index,
                            &args,
                            &mut base_read_id,
                        );
                        if producer_tx.send(cb).is_err() || meta_tx.send(wb).is_err() {
                            break;
                        }
                        if args.limit > 0 && base_read_id >= args.limit {
                            return metrics;
                        }

                        metrics.busy += t_busy.elapsed();
                        metrics.items += (num_pairs * 2) as u64;
                        t_idle = Instant::now();
                    }
                }
            }
            metrics
        })
        .unwrap()
}

fn build_kmer_index(
    genome: &Arc<ReferenceGenome>,
    kmer_size: usize,
) -> HashMap<u32, u32, std::hash::BuildHasherDefault<fxhash::FxHasher>> {
    eprintln!("[Producer] Building k-mer index (k={kmer_size})...");
    let mut index: FxHashMap<u32, u32> = FxHashMap::default();

    if genome.seq.len() >= kmer_size {
        for i in 0..=(genome.seq.len() - kmer_size) {
            if let Some(kmer) = pack_kmer(&genome.seq[i..i + kmer_size], kmer_size) {
                index
                    .entry(kmer)
                    .and_modify(|pos| *pos = u32::MAX) // Invalidate non-unique k-mers
                    .or_insert_with(|| u32::try_from(i).unwrap());
            }
        }
    }

    index.retain(|_, &mut pos| pos != u32::MAX);
    eprintln!("[Producer] Index built. Unique anchors: {}", index.len());
    index
}

/// Processes a Single-End FASTQ chunk, extracting candidate alignment tasks.
fn process_single_chunk(
    chunk: FastqChunk,
    genome: &Arc<ReferenceGenome>,
    index: &HashMap<u32, u32, FxBuildHasher>,
    args: &Args,
    base_read_id: &mut u64,
) -> (WriteBatch, ComputeChunk) {
    let genome_len = genome.seq.len();
    let num_seq = chunk.data.len();

    // 1. Parallel candidate discovery and task generation via Rayon
    let results: Vec<(SingleMeta, [u32; 24])> = chunk
        .offsets
        .par_iter()
        .flat_map_iter(|offset| {
            let mut seq = Cow::Borrowed(&chunk.data[offset.seq.clone()]);
            let mut seeds = find_seeds(&seq, genome_len, index, args.kmer_size);
            let is_rev = seeds.is_empty();

            if seeds.is_empty() {
                seq = Cow::Owned(reverse_complement(&chunk.data[offset.seq.clone()]));
                seeds = find_seeds(&seq, genome_len, index, args.kmer_size);
            }

            let (packed_read, mask_n_read) = pack_read_stack(&seq);
            let n_cands = u32::try_from(seeds.len).unwrap();

            (0..seeds.len).into_iter().map(move |x| {
                let ref_snippet = &genome.seq[seeds.data[x]..seeds.data[x] + REF_LEN];
                let (packed_ref, mask_n_ref) = pack_ref_stack(ref_snippet);
                let group_info = (n_cands << 2) | (u32::try_from(x).unwrap() & 3);

                let task_words = assemble_task_words(
                    packed_ref,
                    packed_read,
                    mask_n_ref,
                    mask_n_read,
                    group_info,
                );

                let meta = SingleMeta {
                    name: offset.name.clone(),
                    pos: u32::try_from(seeds.data[x]).unwrap(),
                    seq: offset.seq.clone(),
                    qual: offset.qual.clone(),
                    is_rev,
                };

                (meta, task_words)
            })
        })
        .collect();

    // 2. Flatten tasks into continuous arena and collect metadata
    let mut task_arena = Vec::with_capacity(results.len() * 24);
    let mut meta_list = Vec::with_capacity(results.len());

    for (meta, task_words) in results {
        task_arena.extend_from_slice(&task_words);
        meta_list.push(meta);
    }

    let compute_chunk = ComputeChunk {
        task_arena,
        task_count: meta_list.len(),
    };

    let write_batch = WriteBatch::Single(SingleWriteBatch {
        raw_data: chunk.data,
        meta: meta_list,
    });

    *base_read_id += num_seq as u64;

    (write_batch, compute_chunk)
}

/// Stack-allocated container for tripartite seed hits (at most 3 elements).
/// Zero heap allocations!
#[derive(Default, Clone, Copy)]
pub struct SeedHits {
    pub data: [usize; 3],
    pub len: usize,
}

impl SeedHits {
    pub const fn push(&mut self, val: usize) {
        if self.len < 3 {
            self.data[self.len] = val;
            self.len += 1;
        }
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.data[..self.len]
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn sort_unstable(&mut self) {
        self.data[..self.len].sort_unstable();
    }
}

/// Finds candidate genomic alignment coordinates using tripartite anchor seeds.
pub fn find_seeds(
    seq: &[u8],
    genome_len: usize,
    index: &FxHashMap<u32, u32>,
    kmer_size: usize,
) -> SeedHits {
    let read_len = seq.len();
    if read_len < kmer_size {
        return SeedHits::default();
    }

    let seed_offsets = [0, (read_len - kmer_size) / 2, read_len - kmer_size];
    let mut hits = SeedHits::default();

    for &offset in &seed_offsets {
        let Some(kmer) = pack_kmer(&seq[offset..offset + kmer_size], kmer_size) else {
            continue;
        };

        let Some(&hit_pos) = index.get(&kmer) else {
            continue;
        };

        let ref_start = (hit_pos as usize)
            .saturating_sub(offset)
            .saturating_sub(FLANK);

        let ref_window_len = read_len + 2 * FLANK;
        if ref_start + ref_window_len <= genome_len {
            // Check duplicates in stack slice:
            let is_duplicate = hits
                .as_slice()
                .iter()
                .any(|&pos| pos.abs_diff(ref_start) < 50);
            if !is_duplicate {
                hits.push(ref_start);
            }
        }
    }

    hits
}

/// 2-bit packs a k-mer into a u32 integer. Returns `None` if the k-mer contains 'N' or lowercase (soft-masked) bases.
fn pack_kmer(seq: &[u8], k: usize) -> Option<u32> {
    if seq.len() < k {
        return None;
    }
    let mut kmer = 0u32;
    for &b in &seq[0..k] {
        let val = match b {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => return None, // Exclude soft-masked lowercase letters and ambiguous bases
        };
        kmer = (kmer << 2) | val;
    }
    Some(kmer)
}

/// Generates the reverse complement of a nucleotide sequence.
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&b| match b.to_ascii_uppercase() {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            _ => b'N',
        })
        .collect()
}

/// 2-bit packs query read bases into seven u32 words and four u32 'N' mask words.
fn pack_read_stack(seq: &[u8]) -> ([u32; 7], [u32; 4]) {
    let mut packed = [0u32; 7];
    let mut n_mask = [0u32; 4];

    for (i, &b) in seq.iter().enumerate() {
        let (val, is_n) = match b.to_ascii_uppercase() {
            b'A' => (0, false),
            b'C' => (1, false),
            b'G' => (2, false),
            b'T' => (3, false),
            _ => (0, true),
        };
        packed[i / 16] |= val << ((i % 16) * 2);
        if is_n {
            n_mask[i / 32] |= 1 << (i % 32);
        }
    }
    (packed, n_mask)
}

/// 2-bit packs reference bases into eight u32 words and four u32 'N' mask words.
fn pack_ref_stack(seq: &[u8]) -> ([u32; 8], [u32; 4]) {
    let mut packed = [0u32; 8];
    let mut n_mask = [0u32; 4];

    for (i, &b) in seq.iter().enumerate() {
        let (val, is_n) = match b.to_ascii_uppercase() {
            b'A' => (0, false),
            b'C' => (1, false),
            b'G' => (2, false),
            b'T' => (3, false),
            _ => (0, true),
        };
        packed[i / 16] |= val << ((i % 16) * 2);
        if is_n {
            n_mask[i / 32] |= 1 << (i % 32);
        }
    }
    (packed, n_mask)
}

/// Assembles the 24-word GPU task payload.
fn assemble_task_words(
    packed_ref: [u32; 8],
    packed_read: [u32; 7],
    mask_n_ref: [u32; 4],
    mask_n_read: [u32; 4],
    group_info: u32,
) -> [u32; 24] {
    let mut words = [0u32; 24];
    words[0..8].copy_from_slice(&packed_ref);
    words[8..15].copy_from_slice(&packed_read);
    words[15..19].copy_from_slice(&mask_n_ref);
    words[19..23].copy_from_slice(&mask_n_read);
    words[23] = group_info;
    words
}

const PRODUCER_CHUNK_SIZE: usize = 512;

/// Processes a batch of paired-end reads, performing seed discovery and GPU task generation.
pub fn process_paired_chunks(
    chunk1: FastqChunk,
    chunk2: FastqChunk,
    genome: &Arc<ReferenceGenome>,
    index: &HashMap<u32, u32, FxBuildHasher>,
    args: &Args,
    base_read_id: &mut u64,
) -> (WriteBatch, ComputeChunk) {
    let genome_len = genome.seq.len();
    let num_pairs = chunk1.offsets.len().min(chunk2.offsets.len());
    let c1 = &chunk1;
    let c2 = &chunk2;

    // 1. Partition the pair indices into coarse-grained ranges (e.g. 512 pairs per chunk).
    // This creates only ~32-64 tasks for Rayon, avoiding all work-stealing overhead.
    let chunk_ranges: Vec<(usize, usize)> = (0..num_pairs)
        .step_by(PRODUCER_CHUNK_SIZE)
        .map(|start| (start, (start + PRODUCER_CHUNK_SIZE).min(num_pairs)))
        .collect();

    // 2. Parallel processing: each thread fills a single preallocated buffer for all 512 pairs.
    // Zero heap allocations per individual read pair!
    let chunk_results: Vec<(Vec<u32>, Vec<PairedMeta>)> = chunk_ranges
        .into_par_iter()
        .map(|(start_idx, end_idx)| {
            let count = end_idx - start_idx;
            // Preallocate assuming ~2 tasks per pair (typical Illumina concordance)
            let mut local_arena = Vec::with_capacity(count * 2 * 24);
            let mut local_meta = Vec::with_capacity(count * 2);

            for i in start_idx..end_idx {
                let o1 = &chunk1.offsets[i];
                let o2 = &chunk2.offsets[i];

                let (candidates, ctx) = PairAlignmentContext::find(
                    o1,
                    o2,
                    &c1.data,
                    &c2.data,
                    genome_len,
                    index,
                    args.kmer_size,
                );

                for (cand_idx, cand) in candidates.into_iter().enumerate() {
                    if cand.pos1 + REF_LEN <= genome_len && cand.pos2 + REF_LEN <= genome_len {
                        // Assembles tasks on the stack directly as [mate1, mate2]
                        let [mate1, mate2] =
                            assemble_pair_tasks(o1, o2, &cand, &ctx, genome, cand_idx);

                        // Push directly into the contiguous arena: zero intermediate Vec!
                        local_arena.extend_from_slice(&mate1.1);
                        local_meta.push(mate1.0);

                        local_arena.extend_from_slice(&mate2.1);
                        local_meta.push(mate2.0);
                    }
                }
            }

            (local_arena, local_meta)
        })
        .collect(); // Simple flat collection of 32 items. Zero LinkedList / SpinLatch!

    // 3. Fast bulk concatenation of chunk buffers
    let total_tasks: usize = chunk_results.iter().map(|(_, m)| m.len()).sum();
    let mut task_arena = Vec::with_capacity(total_tasks * 24);
    let mut meta_list = Vec::with_capacity(total_tasks);

    for (mut arena, mut metas) in chunk_results {
        task_arena.append(&mut arena);
        meta_list.append(&mut metas);
    }

    let compute_chunk = ComputeChunk {
        task_arena,
        task_count: meta_list.len(),
    };

    let write_batch = WriteBatch::Paired(PairedWriteBatch {
        raw_data1: chunk1.data,
        raw_data2: chunk2.data,
        meta: meta_list,
    });

    *base_read_id += num_pairs as u64;

    (write_batch, compute_chunk)
}

/// Evaluation context containing precomputed sequence slices and reverse complements.
pub struct PairAlignmentContext<'a> {
    pub r1_raw: &'a [u8],
    pub r2_raw: &'a [u8],
    pub r2_rev: Vec<u8>,
    pub r1_rev: Option<Vec<u8>>,
}

impl<'a> PairAlignmentContext<'a> {
    /// Discovers candidate pair alignments and returns them alongside the sequence context.
    pub fn find(
        offset1: &SingleOffsets,
        offset2: &SingleOffsets,
        chunk_bytes1: &'a [u8],
        chunk_bytes2: &'a [u8],
        genome_len: usize,
        index: &HashMap<u32, u32, FxBuildHasher>,
        kmer_size: usize,
    ) -> (Vec<CandidatePair>, Self) {
        let r1_raw = &chunk_bytes1[offset1.seq.clone()];
        let r2_raw = &chunk_bytes2[offset2.seq.clone()];
        let r2_rev = reverse_complement(r2_raw);

        // 1. Try standard FR orientation: R1 on (+), R2 on (-)
        let mut r1_fwd = find_seeds(r1_raw, genome_len, index, kmer_size);
        let mut r2_rev_seeds = find_seeds(&r2_rev, genome_len, index, kmer_size);

        let mut candidates = Vec::with_capacity(MAX_CANDIDATE_PAIRS);

        if !r1_fwd.is_empty() && !r2_rev_seeds.is_empty() {
            r1_fwd.sort_unstable();
            r2_rev_seeds.sort_unstable();

            candidates.extend(
                find_pairs_two_pointers(r1_fwd.as_slice(), r2_rev_seeds.as_slice(), true)
                    .into_iter()
                    .map(|(p1, p2, tlen)| CandidatePair {
                        pos1: p1,
                        is_rev1: false,
                        pos2: p2,
                        is_rev2: true,
                        tlen,
                    }),
            );
        }

        // 2. Try inverted RF orientation: R1 on (-), R2 on (+) if no candidates found
        let mut r1_rev = None;
        let mut r1_rev_seeds = SeedHits::default();
        let mut r2_fwd_seeds = SeedHits::default();

        if candidates.is_empty() {
            let rev1 = reverse_complement(r1_raw);
            r1_rev_seeds = find_seeds(&rev1, genome_len, index, kmer_size);
            r2_fwd_seeds = find_seeds(r2_raw, genome_len, index, kmer_size);

            if !r1_rev_seeds.is_empty() && !r2_fwd_seeds.is_empty() {
                r1_rev_seeds.sort_unstable();
                r2_fwd_seeds.sort_unstable();

                candidates.extend(
                    find_pairs_two_pointers(
                        r1_rev_seeds.as_slice(),
                        r2_fwd_seeds.as_slice(),
                        false,
                    )
                    .into_iter()
                    .map(|(p1, p2, tlen)| CandidatePair {
                        pos1: p1,
                        is_rev1: true,
                        pos2: p2,
                        is_rev2: false,
                        tlen,
                    }),
                );
            }

            r1_rev = Some(rev1);
        }

        // 3. Fallback: single-end rescue heuristic if paired discovery yielded no results
        if candidates.is_empty() {
            candidates = rescue_candidate(
                r1_fwd.as_slice(),
                r1_rev_seeds.as_slice(),
                r2_fwd_seeds.as_slice(),
                r2_rev_seeds.as_slice(),
                genome_len,
            );
        }

        let ctx = Self {
            r1_raw,
            r2_raw,
            r2_rev,
            r1_rev,
        };

        (candidates, ctx)
    }
}

/// Identifies concordant pairs within insert size limits using a two-pointer scan.
fn find_pairs_two_pointers(
    p1_list: &[usize],
    p2_list: &[usize],
    r1_is_fwd: bool,
) -> Vec<(usize, usize, i32)> {
    let mut i = 0;
    let mut j = 0;
    let mut found_pairs = Vec::with_capacity(MAX_CANDIDATE_PAIRS);

    while i < p1_list.len() && j < p2_list.len() {
        let p1 = p1_list[i];
        let p2 = p2_list[j];

        let (left, right) = if r1_is_fwd { (p1, p2) } else { (p2, p1) };

        if right < left {
            if r1_is_fwd {
                j += 1;
            } else {
                i += 1;
            }
        } else {
            let tlen = (right + READ_LEN) - left;

            if tlen > MAX_INSERT {
                if r1_is_fwd {
                    i += 1;
                } else {
                    j += 1;
                }
            } else if tlen < MIN_INSERT {
                if r1_is_fwd {
                    j += 1;
                } else {
                    i += 1;
                }
            } else {
                // Concordant pair found
                found_pairs.push((p1, p2, i32::try_from(tlen).unwrap()));

                if found_pairs.len() >= MAX_CANDIDATE_PAIRS {
                    break;
                }

                // Advance pointers to search for alternative candidate alignments
                if r1_is_fwd {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }
    }

    found_pairs
}

/// Genomic candidate locations and orientations for a read pair.
#[derive(Debug, Clone, Copy)]
pub struct CandidatePair {
    pub pos1: usize,
    pub is_rev1: bool,
    pub pos2: usize,
    pub is_rev2: bool,
    pub tlen: i32,
}

/// Fallback heuristic: infers missing mate coordinate when only one end anchored.
fn rescue_candidate(
    r1_fwd: &[usize],
    r1_rev: &[usize],
    r2_fwd: &[usize],
    r2_rev: &[usize],
    genome_len: usize,
) -> Vec<CandidatePair> {
    let mut rescued = Vec::with_capacity(1);

    if let Some(&p1) = r1_fwd.first() {
        let p2 = p1 + RESCUE_OFFSET;
        if p2 + REF_LEN <= genome_len {
            rescued.push(CandidatePair {
                pos1: p1,
                is_rev1: false,
                pos2: p2,
                is_rev2: true,
                tlen: RESCUE_TLEN,
            });
        }
    } else if let Some(&p1) = r1_rev.first() {
        let p2 = p1.saturating_sub(RESCUE_OFFSET);
        rescued.push(CandidatePair {
            pos1: p1,
            is_rev1: true,
            pos2: p2,
            is_rev2: false,
            tlen: RESCUE_TLEN,
        });
    } else if let Some(&p2) = r2_fwd.first() {
        let p1 = p2 + RESCUE_OFFSET;
        if p1 + REF_LEN <= genome_len {
            rescued.push(CandidatePair {
                pos1: p1,
                is_rev1: true,
                pos2: p2,
                is_rev2: false,
                tlen: RESCUE_TLEN,
            });
        }
    } else if let Some(&p2) = r2_rev.first() {
        let p1 = p2.saturating_sub(RESCUE_OFFSET);
        rescued.push(CandidatePair {
            pos1: p1,
            is_rev1: false,
            pos2: p2,
            is_rev2: true,
            tlen: RESCUE_TLEN,
        });
    }

    rescued
}

/// Assembles paired tasks for both mates on the stack without heap allocation.
fn assemble_pair_tasks(
    offset1: &SingleOffsets,
    offset2: &SingleOffsets,
    cand: &CandidatePair,
    ctx: &PairAlignmentContext,
    genome: &ReferenceGenome,
    candidate_idx: usize,
) -> [(PairedMeta, [u32; 24]); 2] {
    // 1. Resolve read sequence for R1 (using cached reverse complement if available)
    let r1_bytes: Cow<[u8]> = if cand.is_rev1 {
        ctx.r1_rev
            .as_deref()
            .map_or_else(|| Cow::Owned(reverse_complement(ctx.r1_raw)), Cow::Borrowed)
    } else {
        Cow::Borrowed(ctx.r1_raw)
    };

    let mate1 = encode_mate_task(
        offset1.name.clone(),
        offset1.seq.clone(),
        offset1.qual.clone(),
        &r1_bytes,
        cand.pos1,
        cand.is_rev1,
        cand.tlen,
        candidate_idx,
        genome,
    );

    // 2. Resolve read sequence for R2
    let r2_bytes = if cand.is_rev2 {
        &ctx.r2_rev
    } else {
        ctx.r2_raw
    };

    let mate2 = encode_mate_task(
        offset2.name.clone(),
        offset2.seq.clone(),
        offset2.qual.clone(),
        r2_bytes,
        cand.pos2,
        cand.is_rev2,
        cand.tlen,
        candidate_idx,
        genome,
    );

    [mate1, mate2]
}

/// Formats and packs one mate of an aligned pair into GPU task words and SAM metadata.
#[allow(clippy::too_many_arguments)]
fn encode_mate_task(
    name: std::ops::Range<usize>,
    seq_range: std::ops::Range<usize>,
    qual_range: std::ops::Range<usize>,
    read_seq: &[u8],
    ref_pos: usize,
    is_rev: bool,
    tlen: i32,
    candidate_idx: usize,
    genome: &ReferenceGenome,
) -> (PairedMeta, [u32; 24]) {
    let ref_snippet = &genome.seq[ref_pos..ref_pos + REF_LEN];
    let (packed_ref, mask_n_ref) = pack_ref_stack(ref_snippet);
    let (packed_read, mask_n_read) = pack_read_stack(read_seq);

    // Group info bits: (batch_cand_count << 2) | cand_sub_index
    let group_info = (1 << 2) | (u32::try_from(candidate_idx).unwrap() & 3);
    let task_words =
        assemble_task_words(packed_ref, packed_read, mask_n_ref, mask_n_read, group_info);

    let meta = PairedMeta {
        name,
        pos: u32::try_from(ref_pos).unwrap(),
        seq: seq_range,
        qual: qual_range,
        is_rev,
        is_proper: true,
        tlen,
    };

    (meta, task_words)
}
