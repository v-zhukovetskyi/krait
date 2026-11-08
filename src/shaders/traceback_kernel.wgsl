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
// Traceback & CIGAR Generation Shader
// =============================================================================
// State Machine:
//   - STATE_M (0): Diagonal transition (Match / Mismatch)
//   - STATE_I (1): Vertical transition (Insertion into query read)
//   - STATE_D (2): Horizontal transition (Deletion from reference)
//
// CIGAR Operation Encoding (2 bits op, 30 bits length):
//   - Op 0: 'M' (Alignment match / mismatch)
//   - Op 1: 'I' (Insertion into reference)
//   - Op 2: 'D' (Deletion from reference)
//   - Op 3: 'S' (Soft-clipping)
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
var<storage, read_write> scores_out: array<i32>;
@group(0) @binding(1)
var<storage, read> directions: array<vec4<u32>>;
@group(0) @binding(2)
var<storage, read_write> cigar_out: array<u32>;
@group(0) @binding(3)
var<uniform> params: Params;
@group(0) @binding(4)
var<storage, read> group_info: array<u32>;

/* @constants:start */
// Default dummy values for IDE / WGSL linter.
// Rust injects real values here at runtime.
const WORKGROUP_SIZE: u32 = 64u;
const BAND_SIZE: u32 = 32u;
const BAND_WORDS_4BIT: u32 = 4u;
const CIGAR_CAPACITY: u32 = 32u;
const CIGAR_SENTINEL: u32 = 0xFFFFFFFFu;
const MIN_SCORE_THRESHOLD: i32 = 200;
const SCORES_PER_READ: u32 = 4u;

const DIR_MATCH: u32 = 1u;
const DIR_INS: u32 = 2u;
const DIR_DEL: u32 = 3u;

const CIGAR_OP_M: u32 = 0u;
const CIGAR_OP_I: u32 = 1u;
const CIGAR_OP_D: u32 = 2u;
const CIGAR_OP_S: u32 = 3u;
const CIGAR_OP_EMPTY: u32 = 0xFFu;

const STATE_M: u32 = 0u;
const STATE_I: u32 = 1u;
const STATE_D: u32 = 2u;
/* @constants:end */

// Derived constants for coordinate and nibble indexing
const HALF_BAND_CELLS: u32 = BAND_SIZE / 2u; // 32 / 2 = 16 cells (xy vs zw)
const CELLS_PER_WORD: u32 = BAND_SIZE / BAND_WORDS_4BIT; // 32 / 4 = 8 cells per u32 word

/// Encapsulates local CIGAR string generation on the GPU stack.
struct CigarBuilder {
    cigar: array<u32, CIGAR_CAPACITY>,
    len: u32,
    cur_op: u32,
    cur_count: u32,
}

/// Initializes an empty CIGAR builder padded with sentinel values.
fn init_cigar_builder() -> CigarBuilder {
    var builder: CigarBuilder;
    builder.len = 0u;
    builder.cur_op = CIGAR_OP_EMPTY;
    builder.cur_count = 0u;

    for(var k = 0u; k < CIGAR_CAPACITY; k = k + 1u) {
        builder.cigar[k] = CIGAR_SENTINEL;
    }
    return builder;
}

/// Packs count and operation code into a single 32-bit word: [length:30][opcode:2].
fn pack_cigar_op(count: u32, op: u32) -> u32 {
    return (count << 2u) | op;
}

/// Appends or merges a run-length CIGAR operation without branching.
fn append_cigar(b: ptr<function, CigarBuilder>, op: u32, count: u32) {
    if (*b).cur_op == op {
        // Continue current run
        (*b).cur_count += count;
    } else {
        // Operation boundary: commit previous run if valid
        let should_write = ((*b).cur_op != CIGAR_OP_EMPTY) && ((*b).len < CIGAR_CAPACITY);
        let safe_idx = select(0u, (*b).len, should_write);

        (*b).cigar[safe_idx] = pack_cigar_op((*b).cur_count, (*b).cur_op);
        (*b).len += select(0u, 1u, should_write);

        (*b).cur_op = op;
        (*b).cur_count = count;
    }
}

/// Commits the final pending CIGAR operation run.
fn flush_cigar(b: ptr<function, CigarBuilder>, count: u32, op: u32) {
    if op != CIGAR_OP_EMPTY && (*b).len < CIGAR_CAPACITY {
        (*b).cigar[(*b).len] = pack_cigar_op(count, op);
        (*b).len += 1u;
    }
}

/// Extracts a 4-bit direction cell for band coordinates (i, k) without dynamic vector indexing.
fn get_saved_cell(idx: u32, batch_size: u32, i: u32, k: u32) -> u32 {
    let row_vec = directions[i * batch_size + idx];

    // Branchless 64-bit split: cells 0..15 in xy, cells 16..31 in zw
    let half_vec = select(row_vec.xy, row_vec.zw, k >= HALF_BAND_CELLS);

    // Select lower or upper 32-bit word inside the 64-bit lane
    let intra_half_mask = HALF_BAND_CELLS - 1u; // 15u
    let word_val = select(half_vec.x, half_vec.y, (k & intra_half_mask) >= CELLS_PER_WORD);

    // Intra-word 4-bit nibble shift (0, 4, 8, ..., 28)
    let intra_word_mask = CELLS_PER_WORD - 1u; // 7u
    let bit_offset = (k & intra_word_mask) << 2u;

    return (word_val >> bit_offset) & 15u;
}

/// Evaluates candidate competition within the SoA candidate family (Winner-Takes-All tournament).
fn is_group_winner(idx: u32, my_score: i32) -> bool {
    var is_winner = my_score >= MIN_SCORE_THRESHOLD;

    let info = group_info[idx];
    let group_count = info & 3u;
    let group_start = info >> 2u;

    for(var c = 0u; c < group_count; c = c + 1u) {
        let other_idx = group_start + c;
        let other_score = scores_out[other_idx * SCORES_PER_READ + 0u];

        // Strict score superiority or lower index tie-breaker
        let loses_to_other = (other_score > my_score)
            || (other_score == my_score && other_idx < idx);
        let competitor_valid = loses_to_other && (other_idx != idx);

        is_winner = select(is_winner, false, competitor_valid);
    }

    return is_winner;
}

/// Executes dynamic traceback along affine gap states, writing operations to `builder`.
fn run_traceback(
    idx: u32,
    batch_size: u32,
    max_i: u32,
    max_j: u32,
    read_len: u32,
    builder: ptr<function, CigarBuilder>,
) -> i32 {
    var curr_i = i32(max_i);
    var curr_k = i32(max_j) - curr_i;

    // 1. Right soft-clip: query bases past the alignment peak
    let clip_end = read_len - 1u - max_i;
    if clip_end > 0u {
        append_cigar(builder, CIGAR_OP_S, clip_end);
    }

    var state = STATE_M;

    // 2. Traceback recurrence loop within the band bounds [0, BAND_SIZE)
    while curr_i >= 0 && curr_k >= 0 && curr_k < i32(BAND_SIZE) {
        let cell = get_saved_cell(idx, batch_size, u32(curr_i), u32(curr_k));
        let dir_m = cell & 3u;
        let i_ext = (cell >> 2u) & 1u;
        let d_ext = (cell >> 3u) & 1u;

        if state == STATE_M {
            if dir_m == DIR_MATCH {
                append_cigar(builder, CIGAR_OP_M, 1u);
                curr_i -= 1;
            } else if dir_m == DIR_INS {
                append_cigar(builder, CIGAR_OP_I, 1u);
                curr_i -= 1;
                curr_k += 1;
                state = select(STATE_M, STATE_I, i_ext == 1u);
            } else {
                append_cigar(builder, CIGAR_OP_D, 1u);
                curr_k -= 1;
                state = select(STATE_M, STATE_D, d_ext == 1u);
            }
        } else if state == STATE_I {
            append_cigar(builder, CIGAR_OP_I, 1u);
            curr_i -= 1;
            curr_k += 1;
            if i_ext == 0u {
                state = STATE_M;
            }
        } else if state == STATE_D {
            append_cigar(builder, CIGAR_OP_D, 1u);
            curr_k -= 1;
            if d_ext == 0u {
                state = STATE_M;
            }
        }
    }

    // 3. Left soft-clip: unaligned query bases before alignment start
    if curr_i >= 0 {
        append_cigar(builder, CIGAR_OP_S, u32(curr_i + 1));
    }

    flush_cigar(builder, (*builder).cur_count, (*builder).cur_op);

    // Compute 0-based reference start coordinate: j = i + k + 1
    return max(0, curr_i + curr_k + 1);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch_size = params.batch_size;
    if idx >= batch_size {
        return;
    }

    let score_base = idx * SCORES_PER_READ;
    let my_score = scores_out[score_base + 0u];
    var builder = init_cigar_builder();
    var final_align_start_j = scores_out[score_base + 1u];

    // 1. Candidate tournament: determine if current candidate wins the read family
    let is_winner = is_group_winner(idx, my_score);

    if is_winner {
        let max_i = u32(scores_out[score_base + 1u]);
        let max_j = u32(scores_out[score_base + 2u]);
        let perfect_score = params.read_len * u32(params.match_score);

        // 2. Fast-path: perfect match (100M) skips matrix traversal entirely
        if u32(my_score) == perfect_score {
            builder.cigar[0u] = pack_cigar_op(params.read_len, CIGAR_OP_M);
            final_align_start_j = i32(max_j - params.read_len + 1u);
        } else {
            // 3. Full affine Gotoh traceback
            final_align_start_j = run_traceback(
                    idx,
                    batch_size,
                    max_i,
                    max_j,
                    params.read_len,
                    &builder,
                );
        }
    }

    // 4. Coalesced writeback: all 64 lane threads synchronously dump CIGAR_CAPACITY words
    // Losing candidates write pre-initialized CIGAR_SENTINEL buffers
    for(var k = 0u; k < CIGAR_CAPACITY; k = k + 1u) {
        cigar_out[idx * CIGAR_CAPACITY + k] = builder.cigar[k];
    }

    scores_out[score_base + 1u] = final_align_start_j;
}
