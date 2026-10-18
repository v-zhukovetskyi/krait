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

struct Params {
    read_len: u32,
    ref_len: u32,
    match_score: i32,
    mismatch_penalty: i32,
    gap_open: i32,
    gap_extend: i32,
    z_drop: i32,
    batch_size: u32,
}

@group(0) @binding(0)
var<storage, read_write> scores_out: array<i32>;
@group(0) @binding(1)
var<storage, read> directions: array<vec4<u32>>;
@group(0) @binding(2)
var<storage, read_write> cigar_out: array<u32>;
@group(0) @binding(3)
var<uniform> params: Params;
@group(0) @binding(4)
var<storage, read> group_info: array<u32>;

const BAND_WORDS_4BIT: u32 = 4u; // 4 слова на строку вместо 16!

// Извлечение ячейки по координатам полосы (i, k)
fn get_saved_cell(idx: u32, batch_size: u32, i: u32, k: u32) -> u32 {
    let row_vec = directions[i * batch_size + idx];

    // Разделяем вектор на две половины по 64 бита (две u32) без динамических индексов
    // Если k >= 16, нам нужна вторая половина (z или w), иначе первая (x or y)
    let half_vec = select(row_vec.xy, row_vec.zw, k >= 16u);

    // Теперь внутри 64-битной половины выбираем конкретное u32 слово
    // (к & 15u) проверяет позицию внутри половины. Если >= 8, берем второй компонент
    let word_val = select(half_vec.x, half_vec.y, (k & 15u) >= 8u);

    // Вычисляем сдвиг внутри конкретного 32-битного слова (от 0 до 28 бит)
    let bit_offset = (k & 7u) << 2u;

    return (word_val >> bit_offset) & 15u;
}

fn pack_cigar_op(count: u32, op: u32) -> u32 {
    // Здесь ветвлений нет, это чистая битовая арифметика — работает мгновенно.
    return (count << 2u) | op;
}

fn append_cigar_op(
    cigar: ptr<function, array<u32, 32>>,
    cigar_idx: ptr<function, u32>,
    cur_op: ptr<function, u32>,
    cur_count: ptr<function, u32>,
    op: u32,
    count: u32,
) {
    if *cur_op == op {
        // Сценарий 1: Продолжаем текущую операцию
        *cur_count += count;
    } else {
        // Сценарий 2: Операция изменилась. 
        // Вместо "if (*cur_op != 0xFFu && *cigar_idx < 32u)" делаем branchless-выбор индекса:

        // Проверяем, нужно ли реально делать запись в массив
        let should_write = (*cur_op != 0xFFu) && (*cigar_idx < 32u);

        // Безопасный индекс для записи. Если писать не нужно, сбрасываем в 0u 
        // (запись произойдет в «мусорный» нулевой элемент, но индекс cigar_idx не увеличится).
        let safe_idx = select(0u, *cigar_idx, should_write);

        // Выполняем запись без ветвления
        (*cigar)[safe_idx] = pack_cigar_op(*cur_count, *cur_op);

        // Инкрементируем индекс только если запись была валидной
        *cigar_idx += select(0u, 1u, should_write);

        // Инициализируем новую операцию
        *cur_op = op;
        *cur_count = count;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch_size = params.batch_size;
    if idx >= batch_size {
        return;
    }

    let my_score = scores_out[idx * 4u + 0u];

    // Переменная-флаг: стоит ли выполнять тяжелый трейсбек?
    var should_run_traceback = true;

    // 1. Быстрый фильтр мусора (без return!)
    should_run_traceback = should_run_traceback && (my_score >= 200);

    // 2. GPU-ФИЛЬТР ПОБЕДИТЕЛЯ (без break и без return!)
    let info = group_info[idx];
    let group_count = info & 3u;
    let group_start = info >> 2u;

    for(var c = 0u; c < group_count; c = c + 1u) {
        let other_idx = group_start + c;
        let other_score = scores_out[other_idx * 4u + 0u];

        let lost_to_other = (other_score > my_score)
            || (other_score == my_score && other_idx < idx);
        let valid_check = lost_to_other && (other_idx != idx);

        // Сбрасываем флаг, если проиграли соседу
        should_run_traceback = select(should_run_traceback, false, valid_check);
    }

    // Инициализируем локальный массив cigar маской по умолчанию
    var cigar: array<u32, 32>;
    for(var k = 0u; k < 32u; k = k + 1u) {
        cigar[k] = 0xFFFFFFFFu;
    }

    // Переменная для сохранения финального стартового индекса j
    var final_align_start_j = scores_out[idx * 4u + 1u]; // по умолчанию старое значение

    // 3. ПРОВЕРКА НА ИДЕАЛЬНЫЙ МАТЧ (Выполняется только если поток еще «жив»)
    let perfect_score = params.read_len * u32(params.match_score);
    let max_j = u32(scores_out[idx * 4u + 2u]);

    let is_perfect = should_run_traceback && (u32(my_score) == perfect_score);

    if is_perfect {
        cigar[0u] = pack_cigar_op(params.read_len, 0u); // 100M
        let align_start_j = max_j - params.read_len + 1u;
        final_align_start_j = i32(align_start_j);

        // Трейсбек для идеального матча уже не нужен, отключаем
        should_run_traceback = false;
    }

    // 4. ТРЕЙСБЕК (Зайдут только реальные победители, прошедшие все фильтры!)
    if should_run_traceback {
        let max_i = u32(scores_out[idx * 4u + 1u]);
        var cigar_idx = 0u;
        var cur_op = 0xFFu;
        var cur_count = 0u;

        var curr_i = i32(max_i);
        var curr_k = i32(max_j) - curr_i;

        let clip_end = params.read_len - 1u - max_i;
        if clip_end > 0u {
            append_cigar_op(&cigar, &cigar_idx, &cur_op, &cur_count, 3u, clip_end);
        }

        const STATE_M: u32 = 0u;
        const STATE_I: u32 = 1u;
        const STATE_D: u32 = 2u;
        var state = STATE_M;

        while curr_i >= 0 && curr_k >= 0 && curr_k < 32 {
            let cell = get_saved_cell(idx, batch_size, u32(curr_i), u32(curr_k));
            let dir_m = cell & 3u;
            let i_ext = (cell >> 2u) & 1u;
            let d_ext = (cell >> 3u) & 1u;

            if state == STATE_M {
                if dir_m == 1u {
                    append_cigar_op(&cigar, &cigar_idx, &cur_op, &cur_count, 0u, 1u);
                    curr_i -= 1;
                } else if dir_m == 2u {
                    append_cigar_op(&cigar, &cigar_idx, &cur_op, &cur_count, 1u, 1u);
                    curr_i -= 1;
                    curr_k += 1;
                    state = select(STATE_M, STATE_I, i_ext == 1u);
                } else {
                    append_cigar_op(&cigar, &cigar_idx, &cur_op, &cur_count, 2u, 1u);
                    curr_k -= 1;
                    state = select(STATE_M, STATE_D, d_ext == 1u);
                }
            } else if state == STATE_I {
                append_cigar_op(&cigar, &cigar_idx, &cur_op, &cur_count, 1u, 1u);
                curr_i -= 1;
                curr_k += 1;
                if i_ext == 0u {
                    state = STATE_M;
                }
            } else if state == STATE_D {
                append_cigar_op(&cigar, &cigar_idx, &cur_op, &cur_count, 2u, 1u);
                curr_k -= 1;
                if d_ext == 0u {
                    state = STATE_M;
                }
            }
        }

        if curr_i >= 0 {
            append_cigar_op(&cigar, &cigar_idx, &cur_op, &cur_count, 3u, u32(curr_i + 1));
        }

        if cur_op != 0xFFu && cigar_idx < 32u {
            cigar[cigar_idx] = pack_cigar_op(cur_count, cur_op);
        }

        final_align_start_j = max(0, curr_i + curr_k + 1);
    }

    // 5. ЕДИНАЯ СИНХРОННАЯ ЗАПИСЬ ДЛЯ ВСЕХ ПОТОКОВ (Coalesced Memory Access)
    // Проигравшие запишут дефолтный массив с 0xFFFFFFFFu, который мы создали вначале.
    // Победители запишут посчитанный cigar.
    for(var k = 0u; k < 32u; k = k + 1u) {
        cigar_out[idx * 32u + k] = cigar[k];
    }

    scores_out[idx * 4u + 1u] = final_align_start_j;
}
