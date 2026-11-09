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

use crossbeam::channel::Receiver;
use std::{ops::Range, time::Instant};

use crate::{
    actors::producer::ComputeChunk,
    constants::{READ_N_MASK_WORDS, READ_WORDS, REF_N_MASK_WORDS, REF_WORDS, STRIDE},
    telemetry::StageMetrics,
};

pub struct GpuReadyBatch {
    pub soa_arena: Vec<u32>,
    pub soa_packed_a: Range<usize>,
    pub soa_packed_b: Range<usize>,
    pub soa_mask_n_a: Range<usize>,
    pub soa_mask_n_b: Range<usize>,
    pub soa_group_info: Range<usize>,
    pub count: usize,
}

/// Coordinates batch assembly from producer chunks to GPU-bound Structure-of-Arrays (`SoA`) layout.
pub fn run_aggregator_worker(
    producer_rx: crossbeam::channel::Receiver<ComputeChunk>,
    gpu_req_rx: crossbeam::channel::Receiver<usize>,
    gpu_batch_tx: crossbeam::channel::Sender<Option<GpuReadyBatch>>,
    max_alloc_size: usize,
) -> std::thread::JoinHandle<StageMetrics> {
    std::thread::Builder::new()
        .name("krait-aggregator".to_string())
        .spawn(move || {
            let mut metrics = StageMetrics::default();
            let mut state = AggregatorState::default();

            let mut t_idle = Instant::now();
            while let Ok(requested_size) = gpu_req_rx.recv() {
                metrics.idle += t_idle.elapsed();
                let t_busy = Instant::now();

                let batch_size = requested_size.min(max_alloc_size);
                let batch = state.assemble_batch(&producer_rx, batch_size);
                let is_empty = batch.is_none();

                if gpu_batch_tx.send(batch).is_err() {
                    break;
                }

                if is_empty && state.producer_finished {
                    break;
                }

                metrics.busy += t_busy.elapsed();
                metrics.items += requested_size as u64;
                t_idle = Instant::now();
            }
            metrics
        })
        .unwrap()
}

/// Internal state of the aggregator actor.
#[derive(Default)]
struct AggregatorState {
    current_cursor: Option<ChunkCursor>,
    producer_finished: bool,
}

impl AggregatorState {
    /// Assembles an ``SoA`` GPU batch up to `batch_size`, fetching new chunks as necessary.
    fn assemble_batch(
        &mut self,
        producer_rx: &Receiver<ComputeChunk>,
        batch_size: usize,
    ) -> Option<GpuReadyBatch> {
        let layout = SoaLayout::new(batch_size);
        let mut gpu_arena = vec![0u32; layout.total_words];
        let mut take_count = 0;

        while take_count < batch_size {
            // Refill chunk from channel if depleted
            if self.current_cursor.is_none() && !self.producer_finished {
                match producer_rx.recv().ok() {
                    Some(chunk) => {
                        self.current_cursor = Some(ChunkCursor::new(chunk));
                    }
                    None => {
                        self.producer_finished = true;
                    }
                }
            }

            if self.producer_finished && self.current_cursor.is_none() {
                break;
            }

            // Scoped extraction avoiding any borrow collisions with `self`
            let is_drained = if let Some(cursor) = &mut self.current_cursor {
                cursor.transcribe(&mut gpu_arena, &layout, &mut take_count, batch_size)
            } else {
                break;
            };

            if is_drained {
                self.current_cursor = None;
            } else {
                // Preserved candidate family boundary; finalize current batch
                break;
            }
        }

        if take_count == 0 && self.producer_finished {
            return None;
        }

        Some(GpuReadyBatch {
            soa_arena: gpu_arena,
            soa_packed_a: layout.seq_a,
            soa_packed_b: layout.seq_b,
            soa_mask_n_a: layout.ref_mask,
            soa_mask_n_b: layout.read_mask,
            soa_group_info: layout.group_info,
            count: take_count,
        })
    }
}

/// Computes range partitions and overall buffer dimensions for `SoA` columns.
struct SoaLayout {
    seq_a: Range<usize>,
    seq_b: Range<usize>,
    ref_mask: Range<usize>,
    read_mask: Range<usize>,
    group_info: Range<usize>,
    total_words: usize,
}

impl SoaLayout {
    const fn new(batch_size: usize) -> Self {
        let a_len = batch_size * REF_WORDS;
        let b_len = batch_size * READ_WORDS;
        let a_mask_len = batch_size * REF_N_MASK_WORDS;
        let b_mask_len = batch_size * READ_N_MASK_WORDS;
        let info_len = batch_size;

        let seq_a = 0..a_len;
        let seq_b = seq_a.end..seq_a.end + b_len;
        let ref_mask = seq_b.end..seq_b.end + a_mask_len;
        let read_mask = ref_mask.end..ref_mask.end + b_mask_len;
        let group_info = read_mask.end..read_mask.end + info_len;

        // Read total_words before group_info is moved into Self
        let total_words = group_info.end;

        Self {
            seq_a,
            seq_b,
            ref_mask,
            read_mask,
            group_info,
            total_words,
        }
    }
}

/// Maintains reading cursor and state over an active producer compute chunk.
struct ChunkCursor {
    chunk: ComputeChunk,
    task_idx: usize,
}

impl ChunkCursor {
    const fn new(chunk: ComputeChunk) -> Self {
        Self { chunk, task_idx: 0 }
    }

    /// Transcribes tasks from ``AoS`` layout into column-major ``SoA`` arrays.
    ///
    /// Returns `true` if the chunk was completely exhausted, or `false` if copying
    /// stopped early to prevent splitting a candidate family across batch boundaries.
    fn transcribe(
        &mut self,
        gpu_arena: &mut [u32],
        layout: &SoaLayout,
        take_count: &mut usize,
        batch_size: usize,
    ) -> bool {
        let task_arena = &self.chunk.task_arena;

        while self.task_idx < self.chunk.task_count && *take_count < batch_size {
            let src_base = self.task_idx * STRIDE;

            // Parse temporary candidate family info from the 24th task word
            let temp_info = task_arena[src_base + 23];
            let cand_offset = (temp_info & 3) as usize;
            let group_count = (temp_info >> 2) as usize;

            // Protect family integrity: defer entire group if it cannot fit in remaining batch capacity
            if cand_offset == 0 && (*take_count + group_count > batch_size) {
                return false;
            }

            let dst_idx = *take_count;

            // 1. Packed reference (words 0..8)
            for w in 0..REF_WORDS {
                gpu_arena[layout.seq_a.start + w * batch_size + dst_idx] = task_arena[src_base + w];
            }

            // 2. Packed read (words 8..15)
            for w in 0..READ_WORDS {
                gpu_arena[layout.seq_b.start + w * batch_size + dst_idx] =
                    task_arena[src_base + REF_WORDS + w];
            }

            // 3. Mask N reference (words 15..19)
            for w in 0..REF_N_MASK_WORDS {
                gpu_arena[layout.ref_mask.start + w * batch_size + dst_idx] =
                    task_arena[src_base + REF_WORDS + READ_WORDS + w];
            }

            // 4. Mask N read (words 19..23)
            for w in 0..READ_N_MASK_WORDS {
                gpu_arena[layout.read_mask.start + w * batch_size + dst_idx] =
                    task_arena[src_base + REF_WORDS + READ_WORDS + REF_N_MASK_WORDS + w];
            }

            // 5. Final group info calculation for GPU traceback (word 23)
            let group_start_gpu =
                (u32::try_from(dst_idx).unwrap()) - (u32::try_from(cand_offset).unwrap());

            let final_gpu_info = (group_start_gpu << 2) | (u32::try_from(group_count).unwrap() & 3);
            gpu_arena[layout.group_info.start + dst_idx] = final_gpu_info;

            self.task_idx += 1;
            *take_count += 1;
        }

        self.task_idx >= self.chunk.task_count
    }
}
