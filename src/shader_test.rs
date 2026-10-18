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

use crate::constants::MATCH_SCORE;
#[cfg(test)]
mod tests {
    // Импортируем все функции и константы из родительского файла
    use super::*;
    use std::fmt::Write;

    // Вспомогательная функция для красивого вывода CIGAR
    fn format_cigar(cigar: &[CigarOp]) -> String {
        let mut s = String::new();
        for op in cigar.iter().rev() {
            // Внимание: если твой бектрекинг выдает задом наперед, возможно .rev() не нужен
            let _ = match op {
                CigarOp::Match(c) => write!(s, "{c}M"),
                CigarOp::Insert(c) => write!(s, "{c}I"),
                CigarOp::Delete(c) => write!(s, "{c}D"),
                CigarOp::SoftClip(c) => write!(s, "{c}S"),
            };
        }
        s
    }

    #[test] // Аннотация, превращающая функцию в тест
    fn test_perfect_match() {
        let ref_seq = format!("{}{}{}", "C".repeat(14), "A".repeat(100), "G".repeat(14));
        let read_seq = "A".repeat(100);

        let result = align_sw_gotoh(read_seq.as_bytes(), ref_seq.as_bytes());

        // Проверяем ожидания. Если они не совпадут - тест упадет с красной ошибкой.
        assert_eq!(result.max_score, 300, "Скор должен быть 300");
        assert_eq!(
            format_cigar(&result.cigar),
            "100M",
            "Ожидалось идеальное совпадение"
        );
        assert_eq!(result.ref_start, 14, "Смещение в референсе должно быть 14");
    }

    #[test]
    fn test_insertion_gotoh() {
        // Реф: 14 'C', 100 'A', 14 'G'
        let ref_seq = format!("{}{}{}", "C".repeat(14), "A".repeat(100), "G".repeat(14));

        // Рид: 40 'A', 5 'T', 55 'A'
        let read_seq = format!("{}TTTTT{}", "A".repeat(40), "A".repeat(55));

        let result = align_sw_gotoh(read_seq.as_bytes(), ref_seq.as_bytes());

        // Записываем то, что фактически выдал алгоритм (наш золотой стандарт)
        println!(
            "Gotoh CIGAR: {} Score {}",
            format_cigar(&result.cigar),
            result.max_score
        );
        assert_eq!(result.max_score, 274);
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CigarOp {
    Match(u32),    // 0: M
    Insert(u32),   // 1: I (в риде)
    Delete(u32),   // 2: D (в рефе)
    SoftClip(u32), // 3: S
}

pub struct AlignmentResult {
    pub max_score: i32,
    pub ref_start: usize,
    pub cigar: Vec<CigarOp>,
}

const MISMATCH_PENALTY: i32 = -3;
const GAP_OPEN: i32 = -6;
const GAP_EXTEND: i32 = -1;
const NEG_INF: i32 = -100_000;

pub fn align_sw_gotoh(read: &[u8], reference: &[u8]) -> AlignmentResult {
    let read_len = read.len();
    let ref_len = reference.len();

    let mut dp_m = vec![0i32; ref_len + 1];
    let mut dp_i = vec![NEG_INF; ref_len + 1];

    // directions: 0 = Stop, 1 = Diag, 2 = Up (I), 3 = Left (D)
    let mut directions = vec![vec![0u8; ref_len]; read_len];

    let mut max_score = 0;
    let mut max_i = 0;
    let mut max_j = 0;

    for i in 0..read_len {
        let char_read = read[i];
        let mut prev_diag = 0;
        let mut left_m = 0;
        let mut left_d = NEG_INF;

        for j in 0..ref_len {
            let char_ref = reference[j];
            let match_val = if char_read == char_ref {
                MATCH_SCORE
            } else {
                MISMATCH_PENALTY
            };
            let m_diag = prev_diag + match_val;

            // 1. Вставка (Идем сверху)
            let i_open = dp_m[j + 1] + GAP_OPEN + GAP_EXTEND;
            let i_ext = dp_i[j + 1] + GAP_EXTEND;
            let i_up = i_open.max(i_ext);
            // Запоминаем: это продление (1) или открытие (0)?
            let i_is_ext = if i_ext >= i_open { 1u8 } else { 0u8 };

            // 2. Делеция (Идем слева)
            let d_open = left_m + GAP_OPEN + GAP_EXTEND;
            let d_ext = left_d + GAP_EXTEND;
            let d_left = d_open.max(d_ext);
            // Запоминаем: это продление (1) или открытие (0)?
            let d_is_ext = if d_ext >= d_open { 1u8 } else { 0u8 };

            // 3. Максимум для M
            let current = 0.max(m_diag.max(i_up.max(d_left)));

            let dir_m = if current == 0 {
                0u8
            } else if current == m_diag {
                1u8
            } else if current == i_up {
                2u8
            } else {
                3u8
            };

            // УПАКОВКА В 4 БИТА:
            // биты 0..1 = dir_m, бит 2 = i_is_ext, бит 3 = d_is_ext
            directions[i][j] = dir_m | (i_is_ext << 2) | (d_is_ext << 3);

            // Сдвиг состояний
            prev_diag = dp_m[j + 1];
            dp_m[j + 1] = current;
            dp_i[j + 1] = i_up;
            left_m = current;
            left_d = d_left;

            if current > max_score {
                max_score = current;
                max_i = i;
                max_j = j;
            }
        }
    }

    // --- БЕКТРЕКИНГ ---
    let mut cigar = Vec::new();
    let mut cur_op: Option<u8> = None;
    let mut cur_count = 0u32;

    let mut append_op = |op: u8, count: u32, cigar: &mut Vec<CigarOp>| {
        if cur_op == Some(op) {
            cur_count += count;
        } else {
            if let Some(prev) = cur_op {
                cigar.push(match prev {
                    0 => CigarOp::Match(cur_count),    // 0: M
                    1 => CigarOp::Insert(cur_count),   // 1: I (в риде)
                    2 => CigarOp::Delete(cur_count),   // 2: D (в рефе)
                    3 => CigarOp::SoftClip(cur_count), // 3: S
                    _ => unreachable!(),
                });
            }
            cur_op = Some(op);
            cur_count = count;
        }
    };

    let clip_end = (read_len - 1 - max_i) as u32;
    if clip_end > 0 {
        append_op(3, clip_end, &mut cigar);
    }

    print_dp_matrix_window(&read, &reference, &directions, 38..46, 50..56, &[(40, 53)]);

    const STATE_M: u8 = 0;
    const STATE_I: u8 = 1;
    const STATE_D: u8 = 2;

    let mut state = STATE_M;
    let mut curr_i = max_i as i32;
    let mut curr_j = max_j as i32;
    let mut ref_start = max_j;

    while curr_i >= 0 && curr_j >= 0 {
        let cell = directions[curr_i as usize][curr_j as usize];
        let dir_m = cell & 0b0011;
        let i_ext = (cell >> 2) & 1;
        let d_ext = (cell >> 3) & 1;

        ref_start = curr_j as usize;

        match state {
            STATE_M => {
                if dir_m == 0 {
                    // Достигли нуля (начало локального выравнивания)
                    let clip_start = (curr_i + 1) as u32;
                    if clip_start > 0 {
                        append_op(3, clip_start, &mut cigar);
                    }
                    break;
                } else if dir_m == 1 {
                    // Диагональ (Match / Mismatch)
                    append_op(0, 1, &mut cigar);
                    if curr_i == 0 || curr_j == 0 {
                        if curr_i > 0 {
                            append_op(3, curr_i as u32, &mut cigar);
                        }
                        break;
                    }
                    curr_i -= 1;
                    curr_j -= 1;
                    // Остаемся в STATE_M
                } else if dir_m == 2 {
                    // Вход в инсерцию сверху
                    append_op(1, 1, &mut cigar);
                    if curr_i == 0 {
                        break;
                    }
                    curr_i -= 1;
                    // Если гэп продлевался, переходим в STATE_I, иначе остаемся в M
                    state = if i_ext == 1 { STATE_I } else { STATE_M };
                } else {
                    // dir_m == 3
                    // Вход в делецию слева
                    append_op(2, 1, &mut cigar);
                    if curr_j == 0 {
                        if curr_i > 0 {
                            append_op(3, (curr_i + 1) as u32, &mut cigar);
                        }
                        break;
                    }
                    curr_j -= 1;
                    // Если гэп продлевался, переходим в STATE_D, иначе остаемся в M
                    state = if d_ext == 1 { STATE_D } else { STATE_M };
                }
            }

            STATE_I => {
                // Мы внутри инсерции: игнорируем dir_m, просто идем вверх!
                append_op(1, 1, &mut cigar);
                if curr_i == 0 {
                    break;
                }
                curr_i -= 1;
                // Выходим из инсерции в M, когда дошли до точки её открытия (i_ext == 0)
                if i_ext == 0 {
                    state = STATE_M;
                }
            }

            STATE_D => {
                // Мы внутри делеции: игнорируем dir_m, просто идем влево!
                append_op(2, 1, &mut cigar);
                if curr_j == 0 {
                    if curr_i > 0 {
                        append_op(3, (curr_i + 1) as u32, &mut cigar);
                    }
                    break;
                }
                curr_j -= 1;
                // Выходим из делеции в M, когда дошли до точки её открытия (d_ext == 0)
                if d_ext == 0 {
                    state = STATE_M;
                }
            }

            _ => unreachable!(),
        }
    }

    if let Some(prev) = cur_op {
        cigar.push(match prev {
            0 => CigarOp::Match(cur_count),    // 0: M
            1 => CigarOp::Insert(cur_count),   // 1: I (в риде)
            2 => CigarOp::Delete(cur_count),   // 2: D (в рефе)
            3 => CigarOp::SoftClip(cur_count), // 3: S
            _ => unreachable!(),
        });
    }

    AlignmentResult {
        max_score,
        ref_start,
        cigar,
    }
}

pub fn print_dp_matrix_window(
    read: &[u8],
    reference: &[u8],
    directions: &[Vec<u8>],
    i_range: std::ops::Range<usize>,
    j_range: std::ops::Range<usize>,
    path: &[(usize, usize)], // координаты бектрекинга
) {
    print!("      ");
    for j in j_range.clone() {
        print!("  {:^3}", reference[j] as char);
    }
    println!();

    for i in i_range {
        print!("{:^3} {:^2} |", read[i] as char, i);
        for j in j_range.clone() {
            let dir = directions[i][j];
            let symbol = match dir {
                1 => " ↖ ",
                2 => " ↑ ",
                3 => " ← ",
                _ => " · ",
            };

            // Подсвечиваем путь трейсбека зеленым
            if path.contains(&(i, j)) {
                print!("\x1b[1;32m{:^5}\x1b[0m", symbol);
            } else {
                print!("{:^5}", symbol);
            }
        }
        println!();
    }
}
