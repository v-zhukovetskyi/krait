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


use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author = "Твое Имя", version = "1.0", about, long_about = None)]
pub struct Args {
    #[arg(short = 'r', long)]
    pub reference: String,

    /// Первый файл ридов (или единственный для Single-End)
    #[arg(short = 'i', short_alias = '1', long = "reads")]
    pub reads: String,

    /// Второй файл ридов для Paired-End (опционально)
    #[arg(short = '2', long = "reads2")]
    pub reads2: Option<String>,

    #[arg(short = 'o', long, default_value = "alignment.sam")]
    pub output: String,

    #[arg(short = 'b', long, default_value_t = 100_000)]
    pub batch_size: usize,

    #[arg(short = 'l', long, default_value_t = 0)]
    pub limit: usize,

    #[arg(short = 'p', long)]
    pub progress: bool,

    #[arg(short = 'k', long, default_value_t = 15)]
    pub kmer_size: usize,

    #[arg(short = 'f', long, default_value_t = 50)]
    pub max_freq: u32,
}
