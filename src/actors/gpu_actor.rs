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

use std::time::Instant;

use crate::{
    actors::{aggregator::GpuReadyBatch, writer::ScoreBatch},
    constants::{CIGAR_CAPACITY, READ_LEN, REF_LEN, SCORES_PER_READ, WORKGROUP_SIZE},
    init_gpu::{Gpu, GpuParams, GpuSlot},
    telemetry::StageMetrics,
};
use crossbeam::channel::{Receiver, Sender};

/// Spawns the dedicated GPU execution actor thread.
pub fn run_gpu_worker(
    gpu: Gpu,
    gpu_req_tx: Sender<usize>,
    gpu_batch_rx: Receiver<Option<GpuReadyBatch>>,
    score_tx: Sender<ScoreBatch>,
    max_size: usize,
) -> std::thread::JoinHandle<StageMetrics> {
    std::thread::Builder::new()
        .name("krait-gpu-actor".to_string())
        .spawn(move || {
            let mut metrics = StageMetrics::default();
            let timer = Instant::now();
            let mut slot_idx = 0;
            let mut pending_batch: Option<InFlight> = None;
            let mut batch_size = (max_size / 2).max(8192);

            // Prime the pipeline: send the initial request to the aggregator
            if gpu_req_tx.send(batch_size).is_err() {
                return metrics;
            }
            let mut t_idle = Instant::now();
            // 1. Block until the aggregator delivers the prepared SoA batch
            while let Ok(Some(batch)) = gpu_batch_rx.recv() {
                metrics.idle += t_idle.elapsed();
                let t_busy = Instant::now();

                // 2. Immediately request the NEXT batch so the aggregator works in parallel
                let _ = gpu_req_tx.send(batch_size);

                if batch.count == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }

                let slot = &gpu.slots[slot_idx];

                // 3. Upload input sequences and execution parameters to the active slot
                upload_batch_data(&gpu.queue, slot, &batch);

                let time_start = timer.elapsed().as_millis();

                // 4. Encode compute passes and submit commands to the GPU queue
                let (sub_idx, scores_size, cigar_size) = encode_and_submit(&gpu, slot, batch.count);

                // 5. Schedule asynchronous mapping of host-visible staging buffers
                initiate_staging_readback(slot, scores_size, cigar_size);

                // 6. Ping-Pong: wait for and drain the PREVIOUS slot while current slot runs on GPU

                match flush_in_flight(&gpu, pending_batch.take(), &score_tx, &timer) {
                    Ok(Some(elapsed_ms)) => {
                        batch_size = adapt_batch_size(batch_size, elapsed_ms, max_size);
                    }
                    Ok(None) => {} // Initial warm-up iteration
                    Err(()) => eprintln!("Error flushing in-flight batch from GPU"),
                }

                // Track the currently submitted batch as the pending in-flight unit
                pending_batch = Some(InFlight {
                    time_start,
                    sub_idx,
                    valid_reads: batch.count,
                    slot_idx,
                });

                // Toggle ping-pong double-buffering slot (0 <-> 1)
                slot_idx = 1 - slot_idx;

                metrics.busy += t_busy.elapsed();
                metrics.items += batch.count as u64;
                t_idle = Instant::now();
            }

            // 7. Final drain: harvest the very last batch remaining in the pipeline
            let _ = flush_in_flight(&gpu, pending_batch.take(), &score_tx, &timer);

            metrics
        })
        .unwrap()
}

/// Uploads packed `SoA` input buffers and execution parameters to the GPU.
fn upload_batch_data(queue: &wgpu::Queue, slot: &GpuSlot, batch: &GpuReadyBatch) {
    let arena = &batch.soa_arena;

    queue.write_buffer(
        &slot.dp.buffer_a,
        0,
        bytemuck::cast_slice(&arena[batch.soa_packed_a.clone()]),
    );
    queue.write_buffer(
        &slot.dp.buffer_b,
        0,
        bytemuck::cast_slice(&arena[batch.soa_packed_b.clone()]),
    );
    queue.write_buffer(
        &slot.dp.buffer_mask_n_a,
        0,
        bytemuck::cast_slice(&arena[batch.soa_mask_n_a.clone()]),
    );
    queue.write_buffer(
        &slot.dp.buffer_mask_n_b,
        0,
        bytemuck::cast_slice(&arena[batch.soa_mask_n_b.clone()]),
    );
    queue.write_buffer(
        &slot.tb.buffer_group_info,
        0,
        bytemuck::cast_slice(&arena[batch.soa_group_info.clone()]),
    );

    let params = GpuParams {
        read_len: u32::try_from(READ_LEN).unwrap(),
        ref_len: u32::try_from(REF_LEN).unwrap(),
        match_score: 3,
        mismatch_penalty: -3,
        gap_open: -6,
        gap_extend: -1,
        z_drop: 100,
        // Invariant column stride matching SoA batch count
        stride: u32::try_from(batch.count).unwrap(),
    };

    queue.write_buffer(&slot.dp.buffer_params, 0, bytemuck::bytes_of(&params));
}

/// Records compute commands, schedules readback copies, and submits to the command queue.
fn encode_and_submit(
    gpu: &Gpu,
    slot: &GpuSlot,
    count: usize,
) -> (
    wgpu::SubmissionIndex,
    wgpu::BufferAddress,
    wgpu::BufferAddress,
) {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_actor_encoder"),
        });

    let workgroups = u32::try_from(count.div_ceil(WORKGROUP_SIZE)).unwrap();

    // 1. Dynamic Programming compute pass
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dp_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&gpu.dp_pipeline);
        cpass.set_bind_group(0, &slot.dp.bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    // 2. Traceback compute pass
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tb_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&gpu.tb_pipeline);
        cpass.set_bind_group(0, &slot.tb.bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    let scores_size = (count * SCORES_PER_READ * 4) as wgpu::BufferAddress;
    let cigar_size = (count * CIGAR_CAPACITY * 4) as wgpu::BufferAddress;

    // 3. Copy results to host-visible staging buffers
    encoder.copy_buffer_to_buffer(
        &slot.dp.buffer_scores,
        0,
        &slot.dp.staging_scores,
        0,
        scores_size,
    );
    encoder.copy_buffer_to_buffer(
        &slot.tb.buffer_cigar,
        0,
        &slot.tb.staging_cigar,
        0,
        cigar_size,
    );

    let sub_idx = gpu.queue.submit(Some(encoder.finish()));
    (sub_idx, scores_size, cigar_size)
}

/// Schedules asynchronous buffer mapping for readback.
fn initiate_staging_readback(
    slot: &GpuSlot,
    scores_size: wgpu::BufferAddress,
    cigar_size: wgpu::BufferAddress,
) {
    slot.dp
        .staging_scores
        .slice(..scores_size)
        .map_async(wgpu::MapMode::Read, |_| {});
    slot.tb
        .staging_cigar
        .slice(..cigar_size)
        .map_async(wgpu::MapMode::Read, |_| {});
}

/// Represents an in-flight compute pass executing on the GPU hardware.
struct InFlight {
    time_start: u128,
    sub_idx: wgpu::SubmissionIndex,
    valid_reads: usize,
    slot_idx: usize,
}

fn flush_in_flight(
    gpu: &Gpu,
    in_flight: Option<InFlight>,
    score_tx: &crossbeam::channel::Sender<ScoreBatch>,
    timer: &Instant,
) -> Result<Option<u128>, ()> {
    let Some(prev) = in_flight else {
        return Ok(None);
    };

    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(prev.sub_idx),
            timeout: None,
        })
        .unwrap();

    let elapsed = timer.elapsed().as_millis().saturating_sub(prev.time_start);

    let prev_slot = &gpu.slots[prev.slot_idx];
    let p_scores_sz = (prev.valid_reads * SCORES_PER_READ * 4) as wgpu::BufferAddress;
    let p_cigar_sz = (prev.valid_reads * CIGAR_CAPACITY * 4) as wgpu::BufferAddress;

    let score_batch = {
        let d_score = prev_slot
            .dp
            .staging_scores
            .slice(..p_scores_sz)
            .get_mapped_range()
            .unwrap();
        let d_cigar = prev_slot
            .tb
            .staging_cigar
            .slice(..p_cigar_sz)
            .get_mapped_range()
            .unwrap();

        ScoreBatch {
            scores_and_coords: bytemuck::cast_slice(&d_score).to_vec(),
            cigar_bin: bytemuck::cast_slice(&d_cigar).to_vec(),
        }
    };

    prev_slot.dp.staging_scores.unmap();
    prev_slot.tb.staging_cigar.unmap();

    score_tx.send(score_batch).map_err(|_| ())?;

    Ok(Some(elapsed))
}

/// Adjusts batch size dynamically using an AIMD (Additive Increase, Multiplicative Decrease) policy.
fn adapt_batch_size(current_size: usize, elapsed_ms: u128, max_size: usize) -> usize {
    if elapsed_ms < 20 {
        (current_size + 4096).min(max_size)
    } else if elapsed_ms > 50 {
        ((current_size * 3 / 4) & !1).max(4096)
    } else {
        current_size
    }
}
