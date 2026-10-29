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

use std::{borrow::Cow, collections::HashMap, sync::Arc};

use bio::io::fasta;
use fxhash::{FxBuildHasher, FxHashMap};
use rayon::prelude::*;
use tokio::sync::mpsc;

use crate::{
    actors::reader::{ChunkBuffer, PairChunkBuffer, SingleChunkBuffer, open_flexible_reader},
    cli::Args,
    constants::{PADDING_LEN, READ_LEN, REF_LEN},
};

// Эта функция берет ОДИН рид и пытается найти ему место в геноме
#[derive(Clone)]
pub struct SingleMeta {
    pub id: u64,
    pub name: std::ops::Range<usize>,
    pub pos: u32,
    pub seq: std::ops::Range<usize>,
    pub qual: std::ops::Range<usize>,
    pub is_rev: bool,
}

#[derive(Clone)]
pub struct PairedMeta {
    pub id: u64,
    pub name: std::ops::Range<usize>,
    pub pos: u32,
    pub seq: std::ops::Range<usize>,
    pub qual: std::ops::Range<usize>,
    pub is_rev: bool,
    pub is_proper: bool,
    pub tlen: i32,
}
pub struct SingleWriteBatch {
    raw_data: Vec<u8>,
    meta: Vec<SingleMeta>,
    id_range: std::ops::Range<u64>,
}

pub struct PairedWriteBatch {
    raw_data: Vec<u8>,
    meta: Vec<PairedMeta>,
    id_range: std::ops::Range<u64>,
}
pub enum WriteBatch {
    Single(SingleWriteBatch),
    Paired(PairedWriteBatch),
}

pub struct ComputeChunk {
    pub task_arena: Vec<u32>, // Сплошной массив, где каждые 24 числа = 1 задача
    pub task_count: usize,    // Сколько всего задач (arena.len() / 24)
    pub id_range: std::ops::Range<u64>, // Сплошной массив, где каждые 24 числа = 1 задача   // Сколько всего задач (arena.len() / 24)
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

pub fn producer_task(
    args: Args,
    genome: Arc<ReferenceGenome>,
    mut chunk_rx: mpsc::Receiver<ChunkBuffer>,
    gpu_tx: mpsc::Sender<ComputeChunk>,
    meta_tx: mpsc::Sender<WriteBatch>,
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

        let mut base_read_id = 0u64;

        while let Some(chunk) = chunk_rx.blocking_recv() {
            let (write_batch, compute_chunk) = match chunk {
                ChunkBuffer::Single(chunk) => {
                    process_single_chunk(chunk, &genome, &index, &args, &mut base_read_id)
                }
                ChunkBuffer::Paired(chunk) => {
                    process_paired_chunk(chunk, &genome, &index, &args, &mut base_read_id)
                }
            };

            if gpu_tx.blocking_send(compute_chunk).is_err() {
                break;
            }
            if meta_tx.blocking_send(write_batch).is_err() {
                break;
            }

            if args.limit > 0 && base_read_id >= args.limit {
                return;
            }
        }
    })
}

const FLANK: usize = 14; // Запас слева для центрирования в полосе W=32

/// Ищет до 3 уникальных предполагаемых точек посадки (ref_start) для рида в геноме
pub fn find_seeds(
    seq: &[u8],
    genome_len: usize,
    index: &FxHashMap<u32, u32>,
    kmer_size: usize,
) -> Vec<usize> {
    let read_len = seq.len();
    if read_len < kmer_size {
        return Vec::new();
    }

    // 1. Формула геометрии сидов: Начало, Центр, Конец
    let seed_offsets = [0, (read_len - kmer_size) / 2, read_len - kmer_size];

    let mut hits = Vec::with_capacity(3);

    for &offset in &seed_offsets {
        // Упаковываем k-мер (если встретится 'N' или мусор — pack_kmer вернет None)
        let Some(kmer) = pack_kmer(&seq[offset..offset + kmer_size], kmer_size) else {
            continue;
        };

        // Заглядываем в хэш-таблицу
        let Some(&hit_pos) = index.get(&kmer) else {
            continue;
        };

        // Проецируем координату сида на начало рида с отступом FLANK:
        // ref_start = hit_pos - offset - FLANK
        let ref_start = (hit_pos as usize)
            .saturating_sub(offset)
            .saturating_sub(FLANK);

        // Проверяем, что окно не вылезет за конец генома
        // (Окно референса = read_len + 2 * FLANK)
        let ref_window_len = read_len + 2 * FLANK;
        if ref_start + ref_window_len <= genome_len {
            // Фильтр дубликатов: если этот сид попал в то же самое место (+/- 50 bp),
            // что и предыдущий сид этого же рида — не дублируем
            let is_duplicate = hits.iter().any(|&pos: &usize| pos.abs_diff(ref_start) < 50);
            if !is_duplicate {
                hits.push(ref_start);
            }
        }
    }

    hits
}

#[inline]
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

fn process_single_chunk(
    chunk: SingleChunkBuffer,
    genome: &Arc<ReferenceGenome>,
    index: &HashMap<u32, u32, FxBuildHasher>,
    args: &Args,
    base_read_id: &mut u64,
) -> (WriteBatch, ComputeChunk) {
    let genome_len = genome.seq.len();
    let num_seq = chunk.single.len();
    let start_id = *base_read_id; // Защита от ругани компилятора

    // 1. ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА (ЧИСТЫЙ RAYON БЕЗ АЛЛОКАЦИЙ ВНУТРИ)
    let results: Vec<(SingleMeta, [u32; 24])> = chunk
        .single
        .par_iter()
        .enumerate()
        .flat_map_iter(|(local_idx, single)| {
            let read_id = start_id + local_idx as u64;
            let mut seq = Cow::Borrowed(&chunk.data[single.seq.clone()]);
            let mut seeds = find_seeds(&seq, genome_len, index, args.kmer_size);
            let mut is_rev = false;

            if seeds.is_empty() {
                seq = Cow::Owned(reverse_complement(&chunk.data[single.seq.clone()]));
                seeds = find_seeds(&seq, genome_len, index, args.kmer_size);
                is_rev = true;
            };

            let (packed_read, mask_n_read) = pack_read_stack(&seq);
            let is_rev = is_rev;
            let n_cands = seeds.len() as u32;

            // Ленивый итератор выдает заготовки задач (слово 23 пока оставляем 0)
            seeds.into_iter().enumerate().map(move |(x, seed)| {
                let ref_snippet = &genome.seq[seed..seed + REF_LEN];
                let (packed_ref, mask_n_ref) = pack_ref_stack(ref_snippet);
                let group_info = (n_cands << 2) | (x as u32 & 3);

                let task_words = assemble_task_words(
                    packed_ref,
                    packed_read,
                    mask_n_ref,
                    mask_n_read,
                    group_info,
                );

                let meta = SingleMeta {
                    id: read_id,
                    name: single.name.clone(),
                    pos: seed as u32,
                    seq: single.seq.clone(),
                    qual: single.qual.clone(),
                    is_rev,
                };

                (meta, task_words)
            })
        })
        .collect();

    // 2. БЫСТРАЯ СБОРКА АРЕНЫ И РАСЧЕТ ЧЕСТНЫХ GPU-ИНДЕКСОВ GROUP_INFO (~0.2 мс)
    let mut task_arena = Vec::with_capacity(results.len() * 24);
    let mut meta_list = Vec::with_capacity(results.len());

    for (meta, task_words) in results {
        task_arena.extend_from_slice(&task_words);
        meta_list.push(meta);
    }

    let compute_chunk = ComputeChunk {
        task_arena,
        task_count: meta_list.len(),
        id_range: start_id..start_id + num_seq as u64,
    };

    // Забираем chunk.data целиком с собой для Райтера!
    let write_batch = WriteBatch::Single(SingleWriteBatch {
        raw_data: chunk.data,
        meta: meta_list,
        id_range: start_id..start_id + num_seq as u64,
    });

    *base_read_id += num_seq as u64;

    (write_batch, compute_chunk)
}

const MAX_CANDIDATE_PAIRS: usize = 3;

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
                // НАШЛИ ВАЛИДНУЮ ПАРУ!
                found_pairs.push((p1, p2, tlen as i32));

                // Если уже нашли 3 альтернативных места в геноме — хватит, уходим
                if found_pairs.len() >= MAX_CANDIDATE_PAIRS {
                    break;
                }

                // ВМЕСТО ВЫХОДА ДВИГАЕМСЯ ДАЛЬШЕ: ищем, нет ли дупликаций дальше по геному!
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

fn process_paired_chunk(
    chunk: PairChunkBuffer,
    genome: &Arc<ReferenceGenome>,
    index: &HashMap<u32, u32, FxBuildHasher>,
    args: &Args,
    base_read_id: &mut u64,
) -> (WriteBatch, ComputeChunk) {
    let genome_len = genome.seq.len();
    let num_pairs = chunk.pairs.len();
    let start_id = *base_read_id; // Копируем для потокобезопасного захвата в Rayon
    let chunk_bytes: &[u8] = &chunk.data;

    let results: Vec<(PairedMeta, [u32; 24])> = chunk
        .pairs
        .par_iter()
        .enumerate()
        .flat_map_iter(|(local_idx, pair)| {
            let read_id = start_id + local_idx as u64;
            let mut best_pairs: Vec<(usize, bool, usize, bool, i32)> = Vec::with_capacity(3);

            // --- ШАГ 1: FR-сценарий №1 (R1 на '+', R2 на '-') ---
            let rev2 = reverse_complement(&chunk_bytes[pair.seq2.clone()]);
            let mut r1_fwd = find_seeds(
                &chunk_bytes[pair.seq1.clone()],
                genome_len,
                index,
                args.kmer_size,
            );
            let mut r2_rev = find_seeds(&rev2, genome_len, index, args.kmer_size);

            if !r1_fwd.is_empty() && !r2_rev.is_empty() {
                r1_fwd.sort_unstable();
                r2_rev.sort_unstable();
                best_pairs = find_pairs_two_pointers(&r1_fwd, &r2_rev, true)
                    .into_iter()
                    .map(|(p1, p2, tlen)| (p1, false, p2, true, tlen))
                    .collect();
            }

            // --- ШАГ 2: Ленивый FR-сценарий №2 (R1 на '-', R2 на '+') ---
            let mut r1_rev = Vec::new();
            let mut r2_fwd = Vec::new();
            let mut lazy_rev1 = None;

            if best_pairs.is_empty() {
                let rev1 = reverse_complement(&chunk_bytes[pair.seq1.clone()]);
                r1_rev = find_seeds(&rev1, genome_len, index, args.kmer_size);
                r2_fwd = find_seeds(
                    &chunk_bytes[pair.seq2.clone()],
                    genome_len,
                    index,
                    args.kmer_size,
                );

                if !r1_rev.is_empty() && !r2_fwd.is_empty() {
                    r1_rev.sort_unstable();
                    r2_fwd.sort_unstable();
                    best_pairs = find_pairs_two_pointers(&r1_rev, &r2_fwd, false)
                        .into_iter()
                        .map(|(p1, p2, tlen)| (p1, true, p2, false, tlen))
                        .collect();
                }
                lazy_rev1 = Some(rev1);
            }

            // --- ШАГ 3: БЕЗОПАСНЫЙ RESCUE (Без лишних аллокаций) ---
            if best_pairs.is_empty() {
                if let Some(&p1) = r1_fwd.first() {
                    let p2 = p1 + 350;
                    if p2 + REF_LEN <= genome_len {
                        best_pairs.push((p1, false, p2, true, 450));
                    }
                } else if let Some(&p1) = r1_rev.first() {
                    let p2 = p1.saturating_sub(350);
                    best_pairs.push((p1, true, p2, false, 450));
                } else if let Some(&p2) = r2_fwd.first() {
                    let p1 = p2 + 350;
                    if p1 + REF_LEN <= genome_len {
                        best_pairs.push((p1, true, p2, false, 450));
                    }
                } else if let Some(&p2) = r2_rev.first() {
                    let p1 = p2.saturating_sub(350);
                    best_pairs.push((p1, false, p2, true, 450));
                }
            }

            // --- ШАГ 4: СБОРКА КАНДИДАТОВ ---

            best_pairs
                .into_iter()
                // Фильтруем валидные границы генома прямо на лету
                .filter(|&(p1, _, p2, _, _)| {
                    p1 + REF_LEN <= genome_len && p2 + REF_LEN <= genome_len
                })
                .enumerate()
                .flat_map(move |(x, (p1, is_rev1, p2, is_rev2, tlen))| {
                    // --- ОБРАБОТКА R1 ---
                    let ref_snippet_1 = &genome.seq[p1..p1 + REF_LEN];
                    let (packed_ref_1, mask_n_ref_1) = pack_ref_stack(ref_snippet_1);

                    let (packed_read_1, mask_n_read_1) = if is_rev1 {
                        let r1_bytes =
                            lazy_rev1.as_deref().map(Cow::Borrowed).unwrap_or_else(|| {
                                Cow::Owned(reverse_complement(&chunk_bytes[pair.seq1.clone()]))
                            });
                        pack_read_stack(&r1_bytes)
                    } else {
                        pack_read_stack(&chunk_bytes[pair.seq1.clone()])
                    };

                    // Твой group_info (биты можно настроить под нужды GPU)
                    let group_info_1 = (1 << 2) | (x as u32 & 3);
                    let task_words_1 = assemble_task_words(
                        packed_ref_1,
                        packed_read_1,
                        mask_n_ref_1,
                        mask_n_read_1,
                        group_info_1,
                    );

                    let meta1 = PairedMeta {
                        id: read_id,
                        name: pair.name.clone(), // Копирование Range (дешево)
                        pos: p1 as u32,
                        seq: pair.seq1.clone(),
                        qual: pair.qual1.clone(),
                        is_rev: is_rev1,
                        is_proper: true,
                        tlen,
                    };

                    // --- ОБРАБОТКА R2 ---
                    let ref_snippet_2 = &genome.seq[p2..p2 + REF_LEN];
                    let (packed_ref_2, mask_n_ref_2) = pack_ref_stack(ref_snippet_2);

                    let (packed_read_2, mask_n_read_2) = if is_rev2 {
                        pack_read_stack(&rev2)
                    } else {
                        pack_read_stack(&chunk_bytes[pair.seq2.clone()])
                    };

                    let group_info_2 = (1 << 2) | (x as u32 & 3);
                    let task_words_2 = assemble_task_words(
                        packed_ref_2,
                        packed_read_2,
                        mask_n_ref_2,
                        mask_n_read_2,
                        group_info_2,
                    );

                    let meta2 = PairedMeta {
                        id: read_id,
                        name: pair.name.clone(),
                        pos: p2 as u32,
                        seq: pair.seq2.clone(),
                        qual: pair.qual2.clone(),
                        is_rev: is_rev2,
                        is_proper: true,
                        tlen,
                    };

                    // Магия стека: возвращаем массив из двух кортежей.
                    // Rust автоматически превратит его в итератор без аллокаций в куче!
                    [(meta1, task_words_1), (meta2, task_words_2)]
                })
        })
        .collect();

    let mut task_arena = Vec::with_capacity(results.len() * 24);
    let mut meta_list = Vec::with_capacity(results.len());

    for (meta, task_words) in results {
        task_arena.extend_from_slice(&task_words);
        meta_list.push(meta);
    }

    let compute_chunk = ComputeChunk {
        task_arena,
        task_count: meta_list.len(),
        id_range: start_id..start_id + num_pairs as u64,
    };

    // Забираем chunk.data целиком с собой для Райтера!
    let write_batch = WriteBatch::Paired(PairedWriteBatch {
        raw_data: chunk.data,
        meta: meta_list,
        id_range: start_id..start_id + num_pairs as u64,
    });

    *base_read_id += num_pairs as u64;

    (write_batch, compute_chunk)
}
