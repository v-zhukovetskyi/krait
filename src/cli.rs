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

/// Fast GPU-accelerated short-read DNA sequence aligner.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "krait",
    author,
    version = "0.1",
    about = "High-performance GPU-accelerated short-read DNA aligner",
    long_about = None
)]
pub struct Args {
    // =========================================================================
    // Input / Output Options
    // =========================================================================
    #[arg(
        short = 'r',
        long = "reference",
        value_name = "FASTA",
        help_heading = "Input / Output",
        help = "Path to reference genome file (FASTA/GZ format)"
    )]
    pub reference: String,

    #[arg(
        short = '1',
        short_alias = 'i',
        long = "reads",
        visible_alias = "reads1",
        value_name = "FASTQ",
        help_heading = "Input / Output",
        help = "Single-end or first pair (R1) FASTQ file"
    )]
    pub reads: String,

    #[arg(
        short = '2',
        long = "reads2",
        value_name = "FASTQ",
        help_heading = "Input / Output",
        help = "Second pair (R2) FASTQ file for paired-end mode"
    )]
    pub reads2: Option<String>,

    #[arg(
        short = 'o',
        long,
        default_value = "-",
        value_name = "FILE",
        help_heading = "Input / Output",
        help = "Output alignment file path ('-' for stdout) [default: -]"
    )]
    pub output: String,

    // =========================================================================
    // Algorithm & Indexing Options
    // =========================================================================
    #[arg(
        short = 'k',
        long,
        default_value_t = 15,
        value_name = "INT",
        help_heading = "Algorithm & Indexing",
        help = "Length of k-mer seeds for index lookup [limit: 16]"
    )]
    pub kmer_size: usize,

    // TODO
    // #[arg(
    //     short = 'f',
    //     long = "frequency",
    //     default_value_t = 50,
    //     value_name = "INT",
    //     help_heading = "Algorithm & Indexing",
    //     help = "Maximum occurrence frequency threshold for seed anchors"
    // )]
    // pub max_freq: u32,

    // =========================================================================
    // Execution & Hardware Options
    // =========================================================================
    #[arg(
        short = 't',
        long = "threads",
        default_value_t = 0,
        value_name = "INT",
        help_heading = "Execution & Performance",
        help = "Number of CPU threads to use (0 = auto-detect all cores)"
    )]
    pub threads: usize,

    #[arg(
        short = 'b',
        long = "batch_size",
        default_value_t = 100_000,
        value_name = "INT",
        help_heading = "Execution & Performance",
        help = "Target batch size (number of reads/pairs processed per GPU slot)"
    )]
    pub batch_size: usize,

    #[arg(
        short = 'l',
        long = "limit",
        default_value_t = 0,
        value_name = "INT",
        help_heading = "Execution & Performance",
        help = "Limit total number of input reads to process (0 = unlimited)"
    )]
    pub limit: u64,

    #[arg(
        short = 'p',
        long = "progress",
        help_heading = "Execution & Performance",
        help = "Display live terminal progress bar and throughput metrics"
    )]
    pub progress: bool,

    #[arg(
        short,
        long = "telemetry",
        help_heading = "Execution & Performance",
        help = "Display live terminal progress bar and throughput metrics"
    )]
    pub telemetry: bool,
}
