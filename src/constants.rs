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


pub const MATCH_SCORE: i32 = 3;
pub const READ_LEN: usize = 100;
pub const REF_LEN: usize = 128;
pub const READ_WORDS: usize = 7;
pub const REF_WORDS: usize = 8;
pub const REF_WORDS_4BIT: usize = 16; // 128 нуклеотидов / 8 клеток на u32
pub const N_MASK_WORDS: usize = 4; // ceil(128 / 32) = 4 слова для масок 'N'
pub const SCORES_PER_READ: usize = 4; // max_score, max_i, max_j, actual_rows
pub const CIGAR_CAPACITY: usize = 32; // 32 операции вместо 16
pub const PADDING_LEN: usize = 256;

