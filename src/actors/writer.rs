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

use crate::{
    actors::producer::ReferenceGenome,
    constants::{MATCH_SCORE, READ_LEN},
};
use indicatif::ProgressBar;
use rayon::prelude::*;
use std::fmt::Write;
use tokio::{io::AsyncWriteExt, sync::mpsc};

pub struct ScoreBatch {
    pub scores_and_coords: Vec<i32>, // Только 2 числа на рид: [score, align_start_j]
    pub cigar_bin: Vec<u32>,         // 16 бинарных чисел CIGAR
}
pub struct MetaBatch {
    pub candidates: Vec<CandidateMeta>,
}

#[derive(Clone)]
pub struct ReadMeta {
    pub name: Vec<u8>,
    pub pos: u32,
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
    pub is_revcomp: bool,
}

pub struct CandidateMeta {
    pub read_id: u64,
    pub meta: ReadMeta,
}

pub struct MappedPair {
    pub meta: ReadMeta,
    pub packed_ref: [u32; 8],
    pub packed_read: [u32; 7],
    pub mask_n_ref: [u32; 4],
    pub mask_n_read: [u32; 4],
}

// 1. Компактный и изолированный декодер CIGAR
fn decode_cigar(cigar_slice: &[u32]) -> String {
    let mut final_cigar = String::with_capacity(32);
    let mut cur_op = 'X';
    let mut cur_count = 0;

    for &val in cigar_slice.iter().rev() {
        if val != 0xFFFF_FFFF {
            let count = val >> 2;
            let op = match val & 3 {
                0 => 'M',
                1 => 'I',
                2 => 'D',
                _ => 'S',
            };

            if op == cur_op {
                cur_count += count; // Склеиваем одинаковые соседние операции
            } else {
                if cur_op != 'X' {
                    let _ = write!(final_cigar, "{cur_count}{cur_op}");
                }
                cur_op = op;
                cur_count = count;
            }
        }
    }

    if cur_op != 'X' {
        let _ = write!(final_cigar, "{cur_count}{cur_op}");
    }

    if final_cigar.is_empty() {
        "*".to_string()
    } else {
        final_cigar
    }
}

// 2. Расчет MAPQ по модели BWA
fn calculate_mapq(s1: i32, s2: i32, max_score: i32) -> u8 {
    if s1 <= 0 || s1 == s2 {
        return 0;
    }
    if s2 <= 0 {
        return 60;
    }
    let delta = (s1 - s2) as f32;
    let s1_f = s1 as f32;
    let max_f = max_score as f32;

    let mapq = 60.0 * (delta / s1_f) * (s1_f / max_f);
    (mapq.round() as u8).min(60)
}

pub async fn writer_task(
    output_path: String,
    genome: Arc<ReferenceGenome>,
    mut meta_rx: mpsc::Receiver<MetaBatch>,
    mut score_rx: mpsc::Receiver<ScoreBatch>,
    pb: ProgressBar,
) {
    let file = tokio::fs::File::create(&output_path).await.unwrap();
    let mut writer = tokio::io::BufWriter::new(file);

    writer
        .write_all(b"@HD\tVN:1.6\tSO:unsorted\n")
        .await
        .unwrap();

    // Заголовки для всех контигов/хромосом
    for contig in &genome.contigs {
        writer
            .write_all(format!("@SQ\tSN:{}\tLN:{}\n", contig.name, contig.len).as_bytes())
            .await
            .unwrap();
    }

    let mut total_written = 0;

    while let (Some(meta_batch), Some(score_batch)) = (meta_rx.recv().await, score_rx.recv().await)
    {
        let genome_ref = Arc::clone(&genome);

        // ВЕСЬ тяжелый процессинг отдаем пулу потоков Rayon внутри spawn_blocking!
        let sam_lines: Vec<String> =
            tokio::task::spawn_blocking(move || {
                let candidates = &meta_batch.candidates;
                let n_candidates = candidates.len();
                if n_candidates == 0 {
                    return Vec::new();
                }

                // ШАГ А: Быстро находим границы групп [start..end] для каждого read_id
                // (Кандидаты одного рида гарантированно идут подряд!)
                let mut ranges = Vec::new();
                let mut start = 0;

                for i in 1..n_candidates {
                    if candidates[i].read_id != candidates[start].read_id {
                        ranges.push(start..i);
                        start = i;
                    }
                }
                ranges.push(start..n_candidates);

                // ШАГ Б: Параллельно обрабатываем каждую группу на всех ядрах CPU
                ranges
                .into_par_iter()
                .map(|range| {
                    let mut best_idx = range.start;
                    let mut best_score = score_batch.scores_and_coords[range.start * 4];
                    let mut second_score = 0;

                    // Ищем лучший и второй лучший скор среди кандидатов этого рида
                    for idx in range.clone() {
                        let score = score_batch.scores_and_coords[idx * 4];
                        if idx == range.start {
                            continue;
                        }

                        if score > best_score {
                            second_score = best_score;
                            best_score = score;
                            best_idx = idx;
                        } else if score > second_score {
                            second_score = score;
                        }
                    }

                    let best_cand = &candidates[best_idx];
                    let meta = &best_cand.meta;

                    let meta_name = unsafe { std::str::from_utf8_unchecked(&meta.name) };
                    let meta_seq = unsafe { std::str::from_utf8_unchecked(&meta.seq) };
                    let meta_qual = unsafe { std::str::from_utf8_unchecked(&meta.qual) };

                    let flag = if meta.is_revcomp { 16 } else { 0 };

                    // Если скор слишком низкий — считаем невыровненным
                    if best_score < 200 {
                        return format!(
                            "{meta_name}\t4\t*\t0\t0\t*\t*\t0\t0\t{meta_seq}\t{meta_qual}\n"
                        );
                    }

                    // Вычисляем честный MAPQ
                    let mapq =
                        calculate_mapq(best_score, second_score, (READ_LEN as i32) * MATCH_SCORE);

                    // Декодируем CIGAR только для победителя!
                    let cigar_offset = best_idx * 32;
                    let cigar_str =
                        decode_cigar(&score_batch.cigar_bin[cigar_offset..cigar_offset + 32]);

                    // Координата старта на референсе
                    let align_start_j = score_batch.scores_and_coords[best_idx * 4 + 1] as usize;
                    let global_pos = meta.pos as usize + align_start_j;

                    // Разрешаем координаты в конкретную хромосому
                    if let Some((chrom_name, local_pos)) = genome_ref.resolve_coords(global_pos) {
                        format!(
                            "{meta_name}\t{flag}\t{chrom_name}\t{local_pos}\t{mapq}\t{cigar_str}\
                            \t*\t0\t0\t{meta_seq}\t{meta_qual}\tAS:i:{best_score}\n"
                        )
                    } else {
                        // Рид попал на технический разделитель N-ок
                        format!("{}\t4\t*\t0\t0\t*\t*\t0\t0\t{meta_seq}\t{meta_qual}\n", meta_name)
                    }
                })
                .collect()
            })
            .await
            .unwrap();

        // Запись готовых строк в файл
        let count = sam_lines.len();
        for line in sam_lines {
            writer.write_all(line.as_bytes()).await.unwrap();
            total_written += 1;
        }

        pb.inc(count as u64);
    }

    writer.flush().await.unwrap();
    if pb.is_hidden() {
        println!("📝 [Writer] Файл записан. Выровнено ридов: {total_written}");
    }
}
