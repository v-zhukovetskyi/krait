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

// =============================================================================
// Banded Dynamic Programming (Smith-Waterman-Gotoh with Affine Gap Penalties)
// =============================================================================
// - Band Width (W): 32 cells (k = 0..31).
// - Target (Ref) coordinate: j = i + k.
// - Query (Read) coordinate: i.
// - Traceback storage: 4 bits per cell, packed into vec4<u32> (16 bytes/row).
//   - Bits [0..1]: Primary direction (1 = Match/Mismatch, 2 = Ins, 3 = Del).
//   - Bit [2]: Insertion state transition (0 = open, 1 = extend).
//   - Bit [3]: Deletion state transition (0 = open, 1 = extend).
// =============================================================================

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

/* @constants:start */
// Default dummy values for IDE / WGSL linter.
// Rust injects real values here at runtime.
const WORKGROUP_SIZE: u32 = 64u;
const BAND_SIZE: u32 = 32u;
const BAND_WORDS_4BIT: u32 = 4u;
const READ_WORDS: u32 = 7u;
const REF_WORDS: u32 = 8u;
const REF_N_MASK_WORDS: u32 = 4u;
const READ_N_MASK_WORDS: u32 = 4u;
const SCORES_PER_READ: u32 = 4u;
const NEG_INF: i32 = -100000;
const DIR_MATCH: u32 = 1u;
const DIR_INS: u32 = 2u;
const DIR_DEL: u32 = 3u;
/* @constants:end */

// Derived constants for bit manipulation
const CELLS_PER_WORD: u32 = BAND_SIZE / BAND_WORDS_4BIT; // 32 / 4 = 8 cells per word
const HALF_BAND_WORDS: u32 = BAND_WORDS_4BIT / 2u; // 4 / 2 = 2 words per 16-base chunk

/// Unpacks a 2-bit nucleotide base (0=A, 1=C, 2=G, 3=T) from a 32-bit word.
fn get_nt(packed_val: u32, idx: u32) -> u32 {
    return (packed_val >> ((idx & 15u) << 1u)) & 3u;
}

/// Checks if the base at the specified index is an ambiguous 'N' (1-bit mask).
fn is_n_base(packed_mask: u32, idx: u32) -> bool {
    return ((packed_mask >> (idx & 31u)) & 1u) != 0u;
}

/// Pre-loads and bit-aligns 32 reference bases (16 bases per u32 chunk) for row i.
/// Spans across up to three 16-base (32-bit) words in `ref_packed`.
fn extract_ref_band_bases(ref_packed: array<u32, REF_WORDS>, i: u32) -> vec2<u32> {
    let w_ref = i >> 4u; // Start word index (i / 16)
    let bit_ref = (i & 15u) << 1u; // Intra-word bit offset (2 bits per base)

    // Branchless clamped word loads using REF_WORDS boundary
    let max_w = REF_WORDS - 1u;
    let rw0 = select(ref_packed[min(w_ref, max_w)], 0u, w_ref >= REF_WORDS);
    let rw1 = select(ref_packed[min(w_ref + 1u, max_w)], 0u, (w_ref + 1u) >= REF_WORDS);
    let rw2 = select(ref_packed[min(w_ref + 2u, max_w)], 0u, (w_ref + 2u) >= REF_WORDS);

    let r_shift = bit_ref;
    let l_shift = 32u - bit_ref;

    let shifted0 = (rw0 >> r_shift) | (rw1 << l_shift);
    let shifted1 = (rw1 >> r_shift) | (rw2 << l_shift);

    let is_aligned = bit_ref == 0u;
    let chunk0 = select(shifted0, rw0, is_aligned); // Bases 0..15
    let chunk1 = select(shifted1, rw1, is_aligned); // Bases 16..31

    return vec2<u32>(chunk0, chunk1);
}

/// Pre-loads and aligns 32 bits of the reference N-mask for row i.
/// Automatically sets the 'N' mask for coordinates that exceed `ref_len` to prevent false matches.
fn extract_ref_band_mask(ref_n_mask: array<u32, REF_N_MASK_WORDS>, i: u32, ref_len: u32) -> u32 {
    let w_n = i >> 5u; // Word index (i / 32)
    let bit_n = i & 31u; // Bit offset within word

    let max_w = REF_N_MASK_WORDS - 1u;
    let nw0 = select(ref_n_mask[min(w_n, max_w)], 0u, w_n >= REF_N_MASK_WORDS);
    let nw1 = select(ref_n_mask[min(w_n + 1u, max_w)], 0u, (w_n + 1u) >= REF_N_MASK_WORDS);

    let shifted = (nw0 >> bit_n) | (nw1 << (32u - bit_n));
    var mask = select(shifted, nw0, bit_n == 0u);

    // Branchless padding: set mask bits to 1 for bases past ref_len
    let is_oob = (i + BAND_SIZE) > ref_len;
    let valid_bases = ref_len - i;
    let safe_shift = clamp(valid_bases, 0u, 31u);
    let oob_mask = 0xFFFFFFFFu << safe_shift;

    mask |= select(0u, oob_mask, is_oob);
    return mask;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch_size = params.batch_size;
    if idx >= batch_size {
        return;
    }

    // 1. Load packed sequences and N-masks into registers (SoA column-read)
    var ref_packed: array<u32, REF_WORDS>;
    for(var w = 0u; w < REF_WORDS; w = w + 1u) {
        ref_packed[w] = seqs_a[w * batch_size + idx];
    }

    var read_packed: array<u32, READ_WORDS>;
    for(var w = 0u; w < READ_WORDS; w = w + 1u) {
        read_packed[w] = seqs_b[w * batch_size + idx];
    }

    var ref_n_mask: array<u32, REF_N_MASK_WORDS>;
    for(var w = 0u; w < REF_N_MASK_WORDS; w = w + 1u) {
        ref_n_mask[w] = mask_n_a[w * batch_size + idx];
    }

    var read_n_mask: array<u32, READ_N_MASK_WORDS>;
    for(var w = 0u; w < READ_N_MASK_WORDS; w = w + 1u) {
        read_n_mask[w] = mask_n_b[w * batch_size + idx];
    }

    // 2. Initialize DP band in registers (BAND_SIZE + 1 elements: active cells + right boundary)
    var dp_m: array<i32, BAND_SIZE + 1u>;
    var dp_i: array<i32, BAND_SIZE + 1u>;

    for(var k = 0u; k < BAND_SIZE; k = k + 1u) {
        dp_m[k] = 0;
        dp_i[k] = NEG_INF;
    }
    dp_m[BAND_SIZE] = NEG_INF;
    dp_i[BAND_SIZE] = NEG_INF;

    var max_score: i32 = 0;
    var max_i: u32 = 0u;
    var max_j: u32 = 0u;

    // 3. Dynamic programming recurrence loop over read rows
    for(var i = 0u; i < params.read_len; i = i + 1u) {
        let char_read = get_nt(read_packed[i >> 4u], i);
        let read_is_n = is_n_base(read_n_mask[i >> 5u], i);

        // Preload reference bases and N-mask bits for row i
        let ref_chunks = extract_ref_band_bases(ref_packed, i);
        let ref_chunk0 = ref_chunks.x;
        let ref_chunk1 = ref_chunks.y;
        let ref_n_chunk = extract_ref_band_mask(ref_n_mask, i, params.ref_len);

        var left_m: i32 = NEG_INF;
        var left_d: i32 = NEG_INF;
        var row_max: i32 = NEG_INF;
        var k = 0u;

        var row_directions: vec4<u32>;

        // Process BAND_SIZE cells across BAND_WORDS_4BIT packed words
        for(var w = 0u; w < BAND_WORDS_4BIT; w = w + 1u) {
            var trace_word = 0u;

            // Extract 16 bits (8 nucleotides) from chunk0 or chunk1
            var ref_bits = select(
                    (ref_chunk0 >> (w * 16u)) & 0xFFFFu,
                    (ref_chunk1 >> ((w - HALF_BAND_WORDS) * 16u)) & 0xFFFFu,
                    w >= HALF_BAND_WORDS,
                );
            var n_bits = (ref_n_chunk >> (w * 8u)) & 0xFFu;

            for(var b = 0u; b < CELLS_PER_WORD; b = b + 1u) {
                let char_ref = ref_bits & 3u;
                ref_bits >>= 2u;

                let ref_is_n = (n_bits & 1u) != 0u;
                n_bits >>= 1u;

                // Match / mismatch scoring
                let is_match = (char_read == char_ref) && (!read_is_n) && (!ref_is_n);
                let match_val = select(params.mismatch_penalty, params.match_score, is_match);
                let m_diag = dp_m[k] + match_val;

                // Insertion recurrence: gap in target (ref)
                let i_open = dp_m[k + 1u] + params.gap_open + params.gap_extend;
                let i_ext = dp_i[k + 1u] + params.gap_extend;
                let i_up = max(i_open, i_ext);
                let i_is_ext = select(0u, 1u, i_ext >= i_open);

                // Deletion recurrence: gap in query (read)
                let d_open = left_m + params.gap_open + params.gap_extend;
                let d_ext = left_d + params.gap_extend;
                let d_left = max(d_open, d_ext);
                let d_is_ext = select(0u, 1u, d_ext >= d_open);

                // Optimal cell score
                let current = max(m_diag, max(i_up, d_left));

                // Encode 4-bit traceback record: [D_ext:1][I_ext:1][Direction:2]
                let dir_m = select(
                        select(DIR_DEL, DIR_INS, current == i_up),
                        DIR_MATCH,
                        current == m_diag,
                    );
                let cell_4bit = dir_m | (i_is_ext << 2u) | (d_is_ext << 3u);
                trace_word = (trace_word >> 4u) | (cell_4bit << 28u);

                // Update recurrence registers for next cell
                dp_m[k] = current;
                dp_i[k] = i_up;
                left_m = current;
                left_d = d_left;

                row_max = max(row_max, current);

                // Track global alignment maximum
                let is_greater = current > max_score;
                max_score = select(max_score, current, is_greater);
                max_i = select(max_i, i, is_greater);
                max_j = select(max_j, i + k, is_greater);

                k += 1u;
            }
            row_directions[w] = trace_word;
        }

        directions[i * batch_size + idx] = row_directions;

        // Suzuki-Kasahara Z-drop heuristic: terminate if score decays severely
        if (max_score - row_max) > params.z_drop {
            break;
        }
    }

    // Write final output scores and peak alignment coordinates
    let score_base = idx * SCORES_PER_READ;
    scores_out[score_base + 0u] = max_score;
    scores_out[score_base + 1u] = i32(max_i);
    scores_out[score_base + 2u] = i32(max_j);
    scores_out[score_base + 3u] = 0;
}
