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


//#![allow(warnings)]
mod actors;
mod cli;
mod constants;
mod shader_test;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::Instant;

use crate::actors::gpu_actor::gpu_actor;
use crate::actors::producer::{load_reference, producer_task};
use crate::actors::reader::reader_task;
use crate::actors::writer::writer_task;
use crate::cli::Args;

// 3. Сам асинхронный писатель

#[tokio::main]
async fn main() {
    let index_start = Instant::now();
    let args = Args::parse();

    println!("📦 Загрузка генома из {}...", args.reference);

    let genome = load_reference(&args.reference);

    let genome_arc = Arc::new(genome);

    let (chunk_tx, chunk_rx) = mpsc::channel(3); // От Reader к Producer
    let (gpu_tx, gpu_rx) = mpsc::channel(3); // От Producer к GPU
    let (meta_tx, meta_rx) = mpsc::channel(3); // От Producer к Writer
    let (score_tx, score_rx) = mpsc::channel(3); // От GPU к Writer

    let pb = if args.progress {
        let pb = ProgressBar::new_spinner();
        let style = ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg} {pos} ридов ({per_sec})")
            .unwrap();
        pb.set_style(style);
        pb.set_message("Выравнивание...");
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    } else {
        ProgressBar::hidden()
    };

    let reader_handle = tokio::spawn(reader_task(
        args.reads.clone(),
        args.reads2.clone(),
        args.batch_size,
        chunk_tx,
    ));
    let producer_handle = tokio::spawn(producer_task(
        args.clone(),
        Arc::clone(&genome_arc),
        chunk_rx,
        gpu_tx,
        meta_tx,
    ));
    let gpu_handle = tokio::spawn(gpu_actor(args.clone(), gpu_rx, score_tx));
    let writer_handle = tokio::spawn(writer_task(
        args.output.clone(),
        genome_arc,
        meta_rx,
        score_rx,
        pb.clone(),
    ));

    let (r, p, g, w) = tokio::join!(reader_handle, producer_handle, gpu_handle, writer_handle);
    r.unwrap();
    p.unwrap();
    g.unwrap();
    w.unwrap();

    if args.progress {
        pb.finish_with_message("✅ Готово!");
    }
    println!("Общее время работы: {:?}", index_start.elapsed());
}
