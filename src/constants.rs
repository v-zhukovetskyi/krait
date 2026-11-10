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
// Primary Biological & Alignment Parameters
// =============================================================================

/// Target read length in base pairs (Illumina standard).
pub const READ_LEN: usize = 100;

/// Genomic window flank allowance around candidate seed hits.
pub const FLANK: usize = 14;

/// Target reference window length: `read_len` + 2 * flank = 100 + 28 = 128 bp.
pub const REF_LEN: usize = READ_LEN + 2 * FLANK;

/// Inter-chromosomal 'N' padding gap in the concatenated reference genome.
pub const PADDING_LEN: usize = 256;

/// Minimum and maximum expected paired-end insert sizes.
pub const MIN_INSERT: usize = 100;
pub const MAX_INSERT: usize = 800;

/// Heuristic offset for rescuing an unmapped mate when only one end is anchored.
pub const RESCUE_OFFSET: usize = 380;
pub const RESCUE_TLEN: usize = 480;

/// Minimum Smith-Waterman alignment score required to declare a read mapped.
pub const MIN_MAPPED_SCORE: i32 = 200;

// =============================================================================
// GPU Hardware & Kernel Grid Parameters
// =============================================================================

/// GPU thread workgroup size (number of parallel alignments executed per workgroup).
pub const WORKGROUP_SIZE: usize = 64;

/// Hardware warp/wavefront size (32 threads for NVIDIA/Intel, multiple of 32 for AMD).
pub const WARP_SIZE: usize = 32;

/// Dynamic programming band width (32 cells). Must match `WARP_SIZE`.
pub const BAND_SIZE: usize = 32;

/// Maximum number of alternative candidate pair alignments evaluated per read.
pub const MAX_CANDIDATE_PAIRS: usize = 3;

/// Number of 32-bit score outputs per task: [`max_score`, `max_i`, `max_j`, `padding`].
pub const SCORES_PER_READ: usize = 4;

/// Maximum number of packed CIGAR operation words per alignment.
pub const CIGAR_CAPACITY: usize = 32;

/// Sentinel value indicating an empty or unaligned CIGAR slot.
pub const CIGAR_SENTINEL: u32 = 0xFFFF_FFFF;

/// Representation of negative infinity in fixed-point DP scoring.
pub const NEG_INF: i32 = -100_000;

// =============================================================================
// Computed Bit-Packing Layouts (Zero Manual Calculations)
// =============================================================================

/// 2-bit packed query read length: ceil(100 / 16) = 7 words.
pub const READ_WORDS: usize = READ_LEN.div_ceil(16);

/// 2-bit packed reference window length: ceil(128 / 16) = 8 words.
pub const REF_WORDS: usize = REF_LEN.div_ceil(16);

/// 1-bit ambiguous 'N' mask ref words: ceil(128 / 32) = 4 words.
pub const REF_N_MASK_WORDS: usize = REF_LEN.div_ceil(32);

/// 1-bit ambiguous 'N' mask read words: ceil(100 / 32) = 4 words.
pub const READ_N_MASK_WORDS: usize = READ_LEN.div_ceil(32);

/// 4-bit packed band cells: 32 cells / 8 nibbles = 4 words per row.
pub const BAND_WORDS_4BIT: usize = BAND_SIZE / 8;

/// Task stride in the contiguous compute arena:
/// 8 (ref) + 7 (read) + 4 (mask ref) + 4 (mask read) + 1 (group info) = 24 words.
pub const STRIDE: usize = REF_WORDS + READ_WORDS + REF_N_MASK_WORDS + READ_N_MASK_WORDS + 1;

/// Byte footprint of a single alignment task in the primary traceback direction buffer:
/// 100 rows * 4 words * 4 bytes = 1600 bytes.
pub const BYTES_PER_TASK_DIRECTIONS: usize = READ_LEN * BAND_WORDS_4BIT * 4;

/// I/O disk buffer capacity for the SAM file writer (1 MiB).
pub const WRITER_BUFFER_CAPACITY: usize = 1024 * 1024;

// =============================================================================
// Shared GPU-CPU Protocol Encodings
// =============================================================================

// Traceback matrix direction flags
pub const DIR_MATCH: u32 = 1;
pub const DIR_INS: u32 = 2;
pub const DIR_DEL: u32 = 3;

// CIGAR operation codes
pub const CIGAR_OP_M: u32 = 0;
pub const CIGAR_OP_I: u32 = 1;
pub const CIGAR_OP_D: u32 = 2;
pub const CIGAR_OP_S: u32 = 3;
pub const CIGAR_OP_EMPTY: u32 = 0xFF;

// Traceback state machine states
pub const STATE_M: u32 = 0;
pub const STATE_I: u32 = 1;
pub const STATE_D: u32 = 2;

// Standard SAM format flag bitmasks
pub mod sam_flags {
    pub const PAIRED: u16 = 0x1;
    pub const PROPER_PAIR: u16 = 0x2;
    pub const UNMAPPED: u16 = 0x4;
    pub const MATE_UNMAPPED: u16 = 0x8;
    pub const REVERSE: u16 = 0x10;
    pub const MATE_REVERSE: u16 = 0x20;
    pub const READ1: u16 = 0x40;
    pub const READ2: u16 = 0x80;
}

// =============================================================================
// Shader Header Generators
// =============================================================================

/// Common execution grid and protocol constants shared across pipelines.
fn common_shader_prelude() -> String {
    format!(
        r"
const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const BAND_SIZE: u32 = {BAND_SIZE}u;
const BAND_WORDS_4BIT: u32 = {BAND_WORDS_4BIT}u;

// Inter-shader protocol: direction bits stored by DP and consumed by Traceback
const DIR_MATCH: u32 = {DIR_MATCH}u;
const DIR_INS: u32 = {DIR_INS}u;
const DIR_DEL: u32 = {DIR_DEL}u;
"
    )
}

/// Generates the minimal compile-time header strictly for the DP kernel.
pub fn dp_shader_header() -> String {
    format!(
        r"{}
// DP Recurrence & Storage Layout
const READ_WORDS: u32 = {READ_WORDS}u;
const REF_WORDS: u32 = {REF_WORDS}u;
const REF_N_MASK_WORDS: u32 = {REF_N_MASK_WORDS}u;
const READ_N_MASK_WORDS: u32 = {READ_N_MASK_WORDS}u;
const SCORES_PER_READ: u32 = {SCORES_PER_READ}u;
const NEG_INF: i32 = {NEG_INF};
",
        common_shader_prelude()
    )
}

/// Generates the minimal compile-time header strictly for the Traceback kernel.
pub fn traceback_shader_header() -> String {
    format!(
        r"{}
// Traceback & CIGAR Construction
const CIGAR_CAPACITY: u32 = {CIGAR_CAPACITY}u;
const CIGAR_SENTINEL: u32 = 0x{CIGAR_SENTINEL:08X}u;
const MIN_SCORE_THRESHOLD: i32 = {MIN_MAPPED_SCORE};
const SCORES_PER_READ: u32 = {SCORES_PER_READ}u;

const CIGAR_OP_M: u32 = {CIGAR_OP_M}u;
const CIGAR_OP_I: u32 = {CIGAR_OP_I}u;
const CIGAR_OP_D: u32 = {CIGAR_OP_D}u;
const CIGAR_OP_S: u32 = {CIGAR_OP_S}u;
const CIGAR_OP_EMPTY: u32 = {CIGAR_OP_EMPTY}u;

const STATE_M: u32 = {STATE_M}u;
const STATE_I: u32 = {STATE_I}u;
const STATE_D: u32 = {STATE_D}u;
",
        common_shader_prelude()
    )
}
