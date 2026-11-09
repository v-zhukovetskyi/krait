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

mod actors;
mod cli;
mod constants;
mod init_gpu;
mod telemetry;

use clap::Parser;
use crossbeam::channel::bounded;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::actors::aggregator::run_aggregator_worker;
use crate::actors::gpu_actor::run_gpu_worker;
use crate::actors::producer::{Contig, ReaderInput, ReferenceGenome, run_producer_worker};

use crate::actors::reader::{open_flexible_reader, run_single_reader};
use crate::actors::writer::run_writer_worker;
use crate::cli::Args;
use crate::constants::PADDING_LEN;
use crate::init_gpu::init_gpu;
use crate::telemetry::print_pipeline_telemetry;

fn main() {
    let total_timer = Instant::now();
    let args = Args::parse();

    // 1. Determine thread count (explicit from CLI or auto-detected)
    let num_threads = if args.threads > 0 {
        args.threads
    } else {
        std::thread::available_parallelism().map_or(4, std::num::NonZero::get)
    };

    // 2. Initialize Rayon global thread pool strictly limited to num_threads
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .expect("Failed to initialize Rayon global thread pool");

    eprintln!("Threads allocated for compute: {num_threads}");

    // 3. Initialize GPU hardware context and compile WGSL pipelines
    let gpu_context = pollster::block_on(init_gpu(&args));

    // 4. Load and index the reference genome
    eprintln!("Loading reference genome from {}...", args.reference);
    let genome = load_reference(&args.reference);
    let genome_arc = Arc::new(genome);

    // 5. Topology of inter-stage communication channels
    // Bounded capacity provides backpressure across pipeline stages
    let (producer_tx, producer_rx) = bounded(3); // Producer -> Aggregator
    let (gpu_req_tx, gpu_req_rx) = bounded(1); // GPU Worker -> Aggregator (request channel)
    let (gpu_batch_tx, gpu_batch_rx) = bounded(1); // Aggregator -> GPU Worker (data channel)
    let (meta_tx, meta_rx) = bounded(3); // Producer -> Writer (SAM metadata)
    let (score_tx, score_rx) = bounded(3); // GPU Worker -> Writer (Alignment results)

    let progress_bar = init_progress_bar(args.progress);

    // 6. Spawn dedicated OS worker threads
    let (reader_threads, producer_input) = if let Some(reads2_path) = &args.reads2 {
        let (r1_tx, r1_rx) = crossbeam::channel::bounded(3); // Reader1 -> Producer
        let (r2_tx, r2_rx) = crossbeam::channel::bounded(3); // Reader2 -> Producer

        let t1 = run_single_reader(args.reads.clone(), gpu_context.max_size, r1_tx);
        let t2 = run_single_reader(reads2_path.clone(), gpu_context.max_size, r2_tx);

        (vec![t1, t2], ReaderInput::Paired(r1_rx, r2_rx))
    } else {
        let (tx, rx) = crossbeam::channel::bounded(3); // Reader -> Producer
        let t = run_single_reader(args.reads.clone(), gpu_context.max_size, tx);

        (vec![t], ReaderInput::Single(rx))
    };

    let producer_thread = run_producer_worker(
        args.clone(),
        Arc::clone(&genome_arc),
        producer_input,
        producer_tx,
        meta_tx,
    );

    let aggregator_thread =
        run_aggregator_worker(producer_rx, gpu_req_rx, gpu_batch_tx, gpu_context.max_size);

    let gpu_thread = run_gpu_worker(
        gpu_context.gpu,
        gpu_req_tx,
        gpu_batch_rx,
        score_tx,
        gpu_context.max_size,
    );

    let writer_thread = run_writer_worker(
        args.output.clone(),
        genome_arc,
        meta_rx,
        score_rx,
        progress_bar.clone(),
    );

    // 7. Synchronously join all OS worker threads and collect their telemetry
    let (r1_metrics, r2_metrics) = match reader_threads.len() {
        2 => {
            let mut it = reader_threads.into_iter();
            (
                it.next().unwrap().join().unwrap(),
                Some(it.next().unwrap().join().unwrap()),
            )
        }
        1 => (
            reader_threads.into_iter().next().unwrap().join().unwrap(),
            None,
        ),
        _ => unreachable!(),
    };

    let producer_metrics = producer_thread.join().expect("Producer panicked");
    let aggregator_metrics = aggregator_thread.join().expect("Aggregator panicked");
    let gpu_metrics = gpu_thread.join().expect("GPU worker panicked");
    let writer_metrics = writer_thread.join().expect("Writer panicked");

    if args.progress {
        progress_bar.finish_with_message("Done!");
    }

    if args.telemetry {
        // 8. Print high-level telemetry dashboard
        let mut report = vec![("Reader R1", r1_metrics)];
        if let Some(r2) = r2_metrics {
            report.push(("Reader R2", r2));
        }
        report.extend_from_slice(&[
            ("Producer (Rayon)", producer_metrics),
            ("Aggregator (SoA)", aggregator_metrics),
            ("GPU Worker", gpu_metrics),
            ("Writer (Disk)", writer_metrics),
        ]);

        print_pipeline_telemetry(&report);
    }

    eprintln!("Total execution time: {:.2?}", total_timer.elapsed());
}

/// Loads a FASTA reference genome into a single contiguous byte array,
/// inserting 'N' padding gaps between chromosomes to prevent cross-boundary alignments.
fn load_reference(path: &str) -> ReferenceGenome {
    let mut reader = open_flexible_reader(path, 2);

    // Preallocate vector capacity if file metadata is available
    let estimated_size =
        std::fs::metadata(path).map_or(4 * 1024 * 1024, |m| usize::try_from(m.len()).unwrap());

    let mut seq = Vec::with_capacity(estimated_size);
    let mut contigs = Vec::new();

    let mut current_name = String::new();
    let mut current_start = 0;
    let mut in_sequence = false;

    // Single reusable string buffer: avoids millions of heap allocations on large genomes!
    let mut line = String::with_capacity(128);

    while reader.read_line(&mut line).unwrap_or(0) > 0 {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            line.clear();
            continue;
        }

        if let Some(stripped) = trimmed.strip_prefix('>') {
            // Commit previous contig when entering a new header
            if in_sequence {
                let len = seq.len() - current_start;
                contigs.push(Contig {
                    name: current_name.clone(),
                    len,
                    global_start: current_start,
                });

                // Insert 'N' padding buffer between chromosomes
                seq.extend(std::iter::repeat_n(b'N', PADDING_LEN));
            }

            // Parse ID (first whitespace-delimited word after '>')
            current_name = stripped.split_whitespace().next().unwrap_or("").to_string();

            current_start = seq.len();
            in_sequence = true;
        } else if in_sequence {
            seq.extend_from_slice(trimmed.as_bytes());
        }

        line.clear();
    }

    // Commit the final contig in the file
    if in_sequence {
        let len = seq.len() - current_start;
        contigs.push(Contig {
            name: current_name,
            len,
            global_start: current_start,
        });

        // Trailing padding prevents out-of-bounds slicing when alignment
        // windows extend past the very last bases of the genome.
        seq.extend(std::iter::repeat_n(b'N', PADDING_LEN));
    }

    seq.shrink_to_fit();
    contigs.shrink_to_fit();

    ReferenceGenome { seq, contigs }
}

/// Initializes terminal progress indicator with steady spinner animation.
#[allow(clippy::literal_string_with_formatting_args)]
fn init_progress_bar(enabled: bool) -> ProgressBar {
    if !enabled {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new_spinner();
    let style = ProgressStyle::default_spinner()
        .template("{spinner:.green} [{elapsed_precise}] {msg} {pos} reads ({per_sec:0})")
        .expect("Valid progress bar template");

    pb.set_style(style);
    pb.set_message("Aligning...");
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}
