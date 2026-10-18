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

use std::sync::Arc;

use bio::io::fasta;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use tokio::sync::mpsc;

use crate::{
    actors::{
        gpu_actor::GpuBatch,
        reader::{ChunkType, ReadChunk, open_flexible_reader},
        writer::{CandidateMeta, MappedPair, MetaBatch, ReadMeta},
    },
    cli::Args,
    constants::{N_MASK_WORDS, PADDING_LEN, READ_LEN, READ_WORDS, REF_LEN, REF_WORDS},
};

struct ReadPair {
    name: Vec<u8>,
    r1_seq: Vec<u8>,
    r1_qual: Vec<u8>,
    r2_seq: Vec<u8>,
    r2_qual: Vec<u8>,
}

// Эта функция берет ОДИН рид и пытается найти ему место в геноме

#[derive(Clone)]
pub struct PairedMeta {
    pub name: Vec<u8>,
    pub seq1: Vec<u8>,
    pub qual1: Vec<u8>,
    pub seq2: Vec<u8>,
    pub qual2: Vec<u8>,
    pub pos1: u32,
    pub pos2: u32,
    pub is_rev1: bool,
    pub is_rev2: bool,
    pub is_proper: bool,
    pub tlen: i32,
}
pub struct ReferenceGenome {
    pub seq: Vec<u8>,
    pub contigs: Vec<Contig>,
}

impl ReferenceGenome {
    // Быстрый перевод глобальной координаты в (Имя_хромосомы, 1-based_позиция)
    pub fn resolve_coords(&self, global_pos: usize) -> Option<(&str, usize)> {
        // Бинарный поиск: находим контиг, у которого global_start <= global_pos
        let idx = self
            .contigs
            .partition_point(|c| c.global_start <= global_pos)
            .checked_sub(1)?;

        let contig = &self.contigs[idx];
        let local_pos = global_pos - contig.global_start;

        if local_pos < contig.len {
            Some((&contig.name, local_pos + 1)) // SAM использует 1-based индекс!
        } else {
            None // Попали в защитную зону из 'N' между хромосомами
        }
    }
}

fn map_read_candidates(
    seq: &[u8],
    qual: &[u8],
    name: &[u8],
    is_revcomp: bool,
    genome: &ReferenceGenome,
    index: &FxHashMap<u32, u32>,
    kmer_size: usize,
) -> Vec<MappedPair> {
    let seed_offsets = [0, 42, READ_LEN.saturating_sub(kmer_size)];
    let mut candidates = Vec::with_capacity(3);
    let mut seen_positions = Vec::with_capacity(3);

    for &offset in &seed_offsets {
        let Some(kmer) = pack_kmer(&seq[offset..offset + kmer_size], kmer_size) else {
            continue;
        };

        let Some(&hit_pos) = index.get(&kmer) else {
            continue;
        };

        let ref_start = hit_pos
            .saturating_sub(u32::try_from(offset).unwrap())
            .saturating_sub(14) as usize;

        // Фильтр дубликатов: если этот сид указывает на тот же самый локус (+/- 50 bp), игнорируем
        let is_duplicate = seen_positions
            .iter()
            .any(|&pos: &usize| pos.abs_diff(ref_start) < 50);
        if is_duplicate {
            continue;
        }

        if ref_start + REF_LEN <= genome.seq.len() {
            seen_positions.push(ref_start);

            let ref_snippet = &genome.seq[ref_start..ref_start + REF_LEN];
            let (packed_ref, mask_n_ref) = pack_ref_stack(ref_snippet);
            let (packed_read, mask_n_read) = pack_read_stack(seq);

            candidates.push(MappedPair {
                meta: ReadMeta {
                    name: name.to_vec(),
                    pos: u32::try_from(ref_start).unwrap(),
                    seq: seq.to_vec(),
                    qual: qual.to_vec(),
                    is_revcomp,
                },
                packed_ref,
                packed_read,
                mask_n_ref,
                mask_n_read,
            });
        }
    }

    candidates
}

fn reverse_complement(seq: &[u8]) -> Vec<u8> {
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

// Упаковка рида (100 bp)
#[inline]
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

// Упаковка референса (128 bp)
#[inline]
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

// Возвращает (2-bit packed, N-mask)
fn pack_dna_with_mask(seq: &[u8], expected_words: usize) -> (Vec<u32>, Vec<u32>) {
    let mut packed = vec![0u32; expected_words];
    let n_mask_words = expected_words.div_ceil(2); // 16 баз на u32 в 2-битном vs 32 базы на u32 в маске
    let mut n_mask = vec![0u32; n_mask_words.max(4)];

    for (i, &b) in seq.iter().enumerate() {
        let (val, is_n) = match b.to_ascii_uppercase() {
            b'A' => (0, false),
            b'C' => (1, false),
            b'G' => (2, false),
            b'T' => (3, false),
            _ => (0, true), // 'N' или мусор: пишем 0, но ставим бит в маске N!
        };

        packed[i / 16] |= val << ((i % 16) * 2);
        if is_n {
            n_mask[i / 32] |= 1 << (i % 32);
        }
    }
    (packed, n_mask)
}

fn pack_kmer(seq: &[u8], k: usize) -> Option<u32> {
    if seq.len() < k {
        return None;
    }
    let mut kmer = 0u32;
    for &b in &seq[0..k] {
        // Soft-masking: если буква строчная (повтор) или 'N', мы возвращаем None!
        let val = match b {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => return None, // Строчные a,c,g,t тоже попадут сюда и отфильтруются!
        };
        kmer = (kmer << 2) | val;
    }
    Some(kmer)
}

pub fn load_reference(path: &str) -> ReferenceGenome {
    let fasta_reader = fasta::Reader::new(open_flexible_reader(path));
    let mut seq = Vec::new();
    let mut contigs = Vec::new();

    for record in fasta_reader.records() {
        let rec = record.unwrap();
        let name = rec.id().to_string();
        let raw_seq = rec.seq();
        let len = raw_seq.len();

        let global_start = seq.len();
        seq.extend_from_slice(raw_seq);

        contigs.push(Contig {
            name,
            len,
            global_start,
        });

        // Вставляем подушку из N, чтобы разнести хромосомы
        seq.extend(std::iter::repeat_n(b'N', PADDING_LEN));
    }

    ReferenceGenome { seq, contigs }
}

const MIN_INSERT: usize = 100;
const MAX_INSERT: usize = 800;

// Возвращает (pos_left, pos_right, tlen, r1_is_left), если пара корректна
fn check_proper_pair(
    pos1: usize,
    is_rev1: bool,
    pos2: usize,
    is_rev2: bool,
) -> Option<(usize, usize, i32, bool)> {
    // 1. Цепи обязаны быть противоположными
    if is_rev1 == is_rev2 {
        return None;
    }

    // В Illumina FR: рид на прямой цепи ВСЕГДА идет левее рида на обратной цепи!
    let (left_pos, right_pos, r1_is_left) = if !is_rev1 && is_rev2 {
        (pos1, pos2, true)
    } else if is_rev1 && !is_rev2 {
        (pos2, pos1, false)
    } else {
        return None;
    };

    if right_pos < left_pos {
        return None;
    }

    let tlen = (right_pos + READ_LEN) - left_pos;

    if (MIN_INSERT..=MAX_INSERT).contains(&tlen) {
        Some((left_pos, right_pos, tlen as i32, r1_is_left))
    } else {
        None
    }
}

#[derive(Clone)]
pub struct Contig {
    pub name: String,
    pub len: usize,
    pub global_start: usize,
}
use rayon::prelude::*; // Обязательно подключаем трейты Rayon!

pub fn producer_task(
    args: Args,
    genome: Arc<ReferenceGenome>,
    mut chunk_rx: mpsc::Receiver<ChunkType>,
    gpu_tx: mpsc::Sender<GpuBatch>,
    meta_tx: mpsc::Sender<MetaBatch>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        println!(
            "🧠 [Producer] Строим k-mer индекс (k={})...",
            args.kmer_size
        );
        let mut index: FxHashMap<u32, u32> = FxHashMap::default();

        if genome.seq.len() >= args.kmer_size {
            for i in 0..=(genome.seq.len() - args.kmer_size) {
                if let Some(kmer) = pack_kmer(&genome.seq[i..i + args.kmer_size], args.kmer_size) {
                    index
                        .entry(kmer)
                        .and_modify(|pos| *pos = u32::MAX)
                        .or_insert_with(|| u32::try_from(i).unwrap());
                }
            }
        }

        index.retain(|_, &mut pos| pos != u32::MAX);
        println!(
            "✅ [Producer] Индекс построен. Уникальных якорей: {}",
            index.len()
        );

        let mut batch_id = 0;
        let mut base_read_id = 0u64;

        while let Some(chunk) = chunk_rx.blocking_recv() {
            let chunk_len = chunk.reads.len();

            // =================================================================
            // 1. ПАРАЛЛЕЛЬНАЯ ЧАСТЬ (RAYON): ВСЕ ЯДРА CPU ЗАГРУЖЕНЫ НА 100%
            // =================================================================
            // Каждый поток независимо считает сиды, N-маски и обратную цепь.
            // Никаких мьютексов и общих векторов!
            let processed_chunk: Vec<(u64, Vec<MappedPair>)> = chunk
                .reads
                .into_par_iter()
                .enumerate()
                .map(|(local_idx, mut raw_read)| {
                    let read_id = base_read_id + local_idx as u64;

                    // Прямая цепь
                    let mut candidates = map_read_candidates(
                        &raw_read.seq,
                        &raw_read.qual,
                        &raw_read.name,
                        false,
                        &genome,
                        &index,
                        args.kmer_size,
                    );

                    // Если пусто — обратная цепь
                    if candidates.is_empty() {
                        let rev_seq = reverse_complement(&raw_read.seq);
                        raw_read.qual.reverse();
                        candidates = map_read_candidates(
                            &rev_seq,
                            &raw_read.qual,
                            &raw_read.name,
                            true,
                            &genome,
                            &index,
                            args.kmer_size,
                        );
                    }

                    (read_id, candidates)
                })
                .collect(); // Rayon параллельно собрал результаты

            base_read_id += chunk_len as u64;

            // =================================================================
            // 2. БЫСТРАЯ ПОСЛЕДОВАТЕЛЬНАЯ СБОРКА В SoA (занимает ~1-2 мс)
            // =================================================================
            let mut soa_packed_a = vec![0u32; args.batch_size * REF_WORDS];
            let mut soa_packed_b = vec![0u32; args.batch_size * READ_WORDS];
            let mut soa_mask_n_a = vec![0u32; args.batch_size * N_MASK_WORDS];
            let mut soa_mask_n_b = vec![0u32; args.batch_size * N_MASK_WORDS];
            let mut soa_group_info = vec![0u32; args.batch_size];
            let mut temp_candidates = Vec::with_capacity(args.batch_size);
            let mut count: usize = 0;

            for (read_id, candidates) in processed_chunk {
                // Если добавление кандидатов превысит размер батча GPU — останавливаемся
                if count + candidates.len() > args.batch_size {
                    break;
                }

                let n_cands = candidates.len() as u32;
                let start_idx = count as u32;
                // Упаковываем: [30 бит: start_idx | 2 бита: n_cands]
                let info = (start_idx << 2) | (n_cands & 3);

                soa_group_info[count] = info;

                for cand in candidates {
                    // Пишем НАПРЯМУЮ в SoA-массивы, минуя 4 промежуточных вектора!
                    for w in 0..REF_WORDS {
                        soa_packed_a[w * args.batch_size + count] = cand.packed_ref[w];
                    }
                    for w in 0..READ_WORDS {
                        soa_packed_b[w * args.batch_size + count] = cand.packed_read[w];
                    }
                    for w in 0..N_MASK_WORDS {
                        soa_mask_n_a[w * args.batch_size + count] = cand.mask_n_ref[w];
                    }
                    for w in 0..N_MASK_WORDS {
                        soa_mask_n_b[w * args.batch_size + count] = cand.mask_n_read[w];
                    }

                    temp_candidates.push(CandidateMeta {
                        read_id,
                        meta: cand.meta,
                    });

                    count += 1;
                }
            }

            // 3. ОТПРАВКА НА GPU И В WRITER
            if count > 0 {
                if gpu_tx
                    .blocking_send(GpuBatch {
                        id: batch_id,
                        packed_a: soa_packed_a,
                        packed_b: soa_packed_b,
                        mask_n_a: soa_mask_n_a,
                        mask_n_b: soa_mask_n_b,
                        group_info: soa_group_info,
                        valid_reads: count,
                    })
                    .is_err()
                {
                    break;
                }

                if meta_tx
                    .blocking_send(MetaBatch {
                        candidates: temp_candidates,
                    })
                    .is_err()
                {
                    break;
                }

                batch_id += 1;
                if args.limit > 0 && batch_id >= args.limit {
                    return;
                }
            }
        }
    })
}
