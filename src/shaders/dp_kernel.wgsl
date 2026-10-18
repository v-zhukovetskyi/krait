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
var<storage, read> seqs_a: array<u32>;
@group(0) @binding(1)
var<storage, read> seqs_b: array<u32>;
@group(0) @binding(2)
var<storage, read> mask_n_a: array<u32>;
@group(0) @binding(3)
var<storage, read> mask_n_b: array<u32>;
@group(0) @binding(4)
var<storage, read_write> scores_out: array<i32>;
@group(0) @binding(5)
var<storage, read_write> directions: array<vec4<u32>>;
@group(0) @binding(6)
var<uniform> params: Params;

const NEG_INF: i32 = -100000;
const BAND_WORDS_4BIT: u32 = 4u; // 32 клетки / 8 = 4 слова

fn get_nt(packed_val: u32, idx: u32) -> u32 {
    return (packed_val >> ((idx & 15u) << 1u)) & 3u;
}

fn is_n_base(packed_mask: u32, idx: u32) -> bool {
    return ((packed_mask >> (idx & 31u)) & 1u) != 0u;
}

// Безопасное чтение слова референса (0 при выходе за границы)
fn get_ref_word(packed: array<u32, 9>, w: u32) -> u32 {
    return packed[min(w, 8u)];
    // Если w >= 8, вызов min вернет 8, а packed[8] вернет 0u. Никаких select и if!
}

// Безопасное чтение N-маски
fn get_n_word(mask: array<u32, 5>, w: u32) -> u32 {
    // Если w >= 4, min вернет 4, и мы прочитаем законный 0u из конца массива
    return mask[min(w, 4u)];
}

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch_size = params.batch_size;
    if idx >= batch_size {
        return;
    }

    // 1. Загрузка данных рида и референса в локальные регистры
    var read_packed: array<u32, 7>;
    for(var w = 0u; w < 7u; w = w + 1u) {
        read_packed[w] = seqs_b[w * batch_size + idx];
    }

    var ref_packed: array<u32, 8>;
    for(var w = 0u; w < 8u; w = w + 1u) {
        ref_packed[w] = seqs_a[w * batch_size + idx];
    }

    var read_n_mask: array<u32, 4>;
    for(var w = 0u; w < 4u; w = w + 1u) {
        read_n_mask[w] = mask_n_b[w * batch_size + idx];
    }

    var ref_n_mask: array<u32, 4>;
    for(var w = 0u; w < 4u; w = w + 1u) {
        ref_n_mask[w] = mask_n_a[w * batch_size + idx];
    }

    // 2. Инициализация полосы (33 элемента в сверхбыстром L1)
    var dp_m: array<i32, 33>;
    var dp_i: array<i32, 33>;

    for(var k = 0u; k < 32u; k = k + 1u) {
        dp_m[k] = 0;
        dp_i[k] = NEG_INF;
    }
    dp_m[32] = NEG_INF;
    dp_i[32] = NEG_INF;

    var max_score: i32 = 0;
    var max_i: u32 = 0u;
    var max_j: u32 = 0u;

    // 3. Основной цикл по строкам рида
    for(var i = 0u; i < params.read_len; i = i + 1u) {
        let char_read = get_nt(read_packed[i >> 4u], i);
        let read_is_n = is_n_base(read_n_mask[i >> 5u], i);

        // =====================================================================
        // ПРЕДЗАГРУЗКА 32 БАЗ РЕФЕРЕНСА ДЛЯ СТРОКИ i (Вне внутреннего цикла!)
        // =====================================================================
        // Нам нужны базы ref[i .. i + 31]. Они пересекают максимум 3 слова ref_packed.
        let w_ref = i >> 4u; // слово начала (i / 16)
        let bit_ref = (i & 15u) << 1u; // битовое смещение внутри слова

        let rw0 = select(ref_packed[min(w_ref, 7u)], 0u, w_ref >= 8u);
        let rw1 = select(ref_packed[min(w_ref + 1u, 7u)], 0u, (w_ref + 1u) >= 8u);
        let rw2 = select(ref_packed[min(w_ref + 2u, 7u)], 0u, (w_ref + 2u) >= 8u);

        var ref_chunk0: u32; // Базы полосы 0..15 (32 бита)
        var ref_chunk1: u32; // Базы полосы 16..31 (32 бита)

        // 1. Вычисляем сдвинутые варианты заранее
        let r_shift = bit_ref;
        let l_shift = 32u - bit_ref;

        let shifted_chunk0 = (rw0 >> r_shift) | (rw1 << l_shift);
        let shifted_chunk1 = (rw1 >> r_shift) | (rw2 << l_shift);

        // 2. Убираем ветвление с помощью select
        let is_zero = bit_ref == 0u;
        ref_chunk0 = select(shifted_chunk0, rw0, is_zero);
        ref_chunk1 = select(shifted_chunk1, rw1, is_zero);

        // Предзагрузка 32 бит N-маски для полосы:
        let w_n = i >> 5u; // i / 32
        let bit_n = i & 31u;

        let nw0 = select(ref_n_mask[min(w_n, 3u)], 0u, w_n >= 4u);
        let nw1 = select(ref_n_mask[min(w_n + 1u, 3u)], 0u, (w_n + 1u) >= 4u);

        var ref_n_chunk: u32;
        // === 1. Первое ветвление (безопасный select) ===
        let shifted_chunk = (nw0 >> bit_n) | (nw1 << (32u - bit_n));
        ref_n_chunk = select(shifted_chunk, nw0, bit_n == 0u);

        // === 2. Второе ветвление (исправлено) ===
        let is_oob = (i + 32u) > params.ref_len;

        // Вычисляем сдвиг. Если мы вылетели за край, oob_bases будет от 0 до 31.
        // Если мы в безопасности, oob_bases нам не важен (select все равно выберет 0u).
        let oob_bases = params.ref_len - i;
        let safe_shift = clamp(oob_bases, 0u, 31u);

        // Создаем маску для случая, если вылет РЕАЛЬНО есть
        let oob_mask = 0xFFFFFFFFu << safe_shift;

        // Если вылет есть (is_oob == true) -> применяем oob_mask.
        // Если вылета нет -> применяем 0u (что при операции |= ничего не изменит).
        ref_n_chunk |= select(0u, oob_mask, is_oob);

        // =====================================================================
        // ВЫЧИСЛЕНИЕ ПОЛОСЫ (ЧИСТЫЕ СДВИГИ В РЕГИСТРАХ БЕЗ ДОСТУПА К ПАМЯТИ)
        // =====================================================================
        var left_m: i32 = NEG_INF;
        var left_d: i32 = NEG_INF;
        var row_max: i32 = NEG_INF;
        var k = 0u;

        var row_directions: vec4<u32>;

        for(var w = 0u; w < BAND_WORDS_4BIT; w = w + 1u) {
            var trace_word = 0u;

            // Выбираем 8 нуклеотидов (16 бит) для текущего блока w:
            // Для w = 0, 1 берем из chunk0, для w = 2, 3 — из chunk1
            var ref_bits = select(
                    (ref_chunk0 >> (w * 16u)) & 0xFFFFu,
                    (ref_chunk1 >> ((w - 2u) * 16u)) & 0xFFFFu,
                    w >= 2u,
                );

            // 8 бит N-маски:
            var n_bits = (ref_n_chunk >> (w * 8u)) & 0xFFu;

            for(var b = 0u; b < 8u; b = b + 1u) {
                // Извлечение за 1 такт АЛУ:
                let char_ref = ref_bits & 3u;
                ref_bits >>= 2u;

                let ref_is_n = (n_bits & 1u) != 0u;
                n_bits >>= 1u;

                let is_match = (char_read == char_ref) && (!read_is_n) && (!ref_is_n);
                let match_val = select(params.mismatch_penalty, params.match_score, is_match);

                let m_diag = dp_m[k] + match_val;

                let i_open = dp_m[k + 1u] + params.gap_open + params.gap_extend;
                let i_ext = dp_i[k + 1u] + params.gap_extend;
                let i_up = max(i_open, i_ext);
                let i_is_ext = select(0u, 1u, i_ext >= i_open);

                let d_open = left_m + params.gap_open + params.gap_extend;
                let d_ext = left_d + params.gap_extend;
                let d_left = max(d_open, d_ext);
                let d_is_ext = select(0u, 1u, d_ext >= d_open);

                let current = max(m_diag, max(i_up, d_left));

                let dir_m = select(select(3u, 2u, current == i_up), 1u, current == m_diag);
                let cell_4bit = dir_m | (i_is_ext << 2u) | (d_is_ext << 3u);
                trace_word = (trace_word >> 4u) | (cell_4bit << 28u);

                dp_m[k] = current;
                dp_i[k] = i_up;
                left_m = current;
                left_d = d_left;

                row_max = max(row_max, current);

                let is_greater = current > max_score;
                max_score = select(max_score, current, is_greater);
                max_i = select(max_i, i, is_greater);
                max_j = select(max_j, i + k, is_greater);

                k += 1u;
            }
            row_directions[w] = trace_word;
        }
        directions[i * batch_size + idx] = row_directions;

        if (max_score - row_max) > params.z_drop {
            break;
        }
    }

    scores_out[idx * 4u + 0u] = max_score;
    scores_out[idx * 4u + 1u] = i32(max_i);
    scores_out[idx * 4u + 2u] = i32(max_j);
    scores_out[idx * 4u + 3u] = 0;
}
