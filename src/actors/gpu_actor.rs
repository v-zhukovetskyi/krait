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

use bytemuck::{Pod, Zeroable};
use tokio::sync::mpsc;
use wgpu::PipelineCompilationOptions;

use crate::{
    actors::writer::ScoreBatch,
    cli::Args,
    constants::{
        CIGAR_CAPACITY, N_MASK_WORDS, READ_LEN, READ_WORDS, REF_LEN, REF_WORDS, REF_WORDS_4BIT,
        SCORES_PER_READ,
    },
};
// Подставьте ваши импорты Args, GpuBatch, ScoreBatch, PipelineCompilationOptions и wgpu
pub struct GpuBatch {
    pub id: usize,
    pub packed_a: Vec<u32>,
    pub packed_b: Vec<u32>,
    pub mask_n_a: Vec<u32>,
    pub mask_n_b: Vec<u32>,
    pub group_info: Vec<u32>, // <--- НОВОЕ: 1 число на кандидата (200 КБ)
    pub valid_reads: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuParams {
    pub read_len: u32,
    pub ref_len: u32,
    pub match_score: i32,
    pub mismatch_penalty: i32,
    pub gap_open: i32,
    pub gap_extend: i32,
    pub z_drop: i32,
    pub batch_size: u32,
}

// Изолированный транспортный слот
struct GpuSlot {
    buffer_a: wgpu::Buffer,
    buffer_b: wgpu::Buffer,
    buffer_mask_n_a: wgpu::Buffer,
    buffer_mask_n_b: wgpu::Buffer,
    buffer_scores: wgpu::Buffer,
    buffer_dirs: wgpu::Buffer,
    buffer_cigar: wgpu::Buffer,
    staging_scores: wgpu::Buffer,
    staging_cigar: wgpu::Buffer,
    bind_group_dp: wgpu::BindGroup,
    bind_group_tb: wgpu::BindGroup,
    buffer_group_info: wgpu::Buffer,
}

fn create_slot(
    device: &wgpu::Device,
    batch_size: usize,
    buffer_params: &wgpu::Buffer,
    dp_layout: &wgpu::BindGroupLayout,
    tb_layout: &wgpu::BindGroupLayout,
    slot_id: usize,
) -> GpuSlot {
    let buf_a_size = (batch_size * REF_WORDS * 4) as wgpu::BufferAddress;
    let buf_b_size = (batch_size * READ_WORDS * 4) as wgpu::BufferAddress;
    let mask_size = (batch_size * N_MASK_WORDS * 4) as wgpu::BufferAddress;
    let scores_size = (batch_size * SCORES_PER_READ * 4) as wgpu::BufferAddress;
    let dirs_size = (batch_size * READ_LEN * REF_WORDS_4BIT * 4) as wgpu::BufferAddress;
    let cigar_size = (batch_size * CIGAR_CAPACITY * 4) as wgpu::BufferAddress;
    let group_info_size = (batch_size * 4) as wgpu::BufferAddress;

    let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("slot_{slot_id}_seqs_a")),
        size: buf_a_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("slot_{slot_id}_seqs_b")),
        size: buf_b_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_mask_n_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("slot_{slot_id}_mask_n_a")),
        size: mask_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_mask_n_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("slot_{slot_id}_mask_n_b")),
        size: mask_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("slot_{slot_id}_scores")),
        size: scores_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let buffer_dirs = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("slot_{slot_id}_dirs")),
        size: dirs_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_cigar = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("slot_{slot_id}_cigar")),
        size: cigar_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let staging_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("slot_{slot_id}_staging_scores")),
        size: scores_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });

    let staging_cigar = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("slot_{slot_id}_staging_cigar")),
        size: cigar_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_group_info = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("slot_{slot_id}_group_info")),
        size: group_info_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Привязка для Шейдера 1 (DP)
    let bind_group_dp = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("slot_{slot_id}_bg_dp")),
        layout: dp_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_mask_n_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_mask_n_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffer_scores.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buffer_dirs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buffer_params.as_entire_binding(),
            },
        ],
    });

    // Привязка для Шейдера 2 (Traceback)
    let bind_group_tb = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("slot_{slot_id}_bg_tb")),
        layout: tb_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_scores.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_dirs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_cigar.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffer_group_info.as_entire_binding(),
            },
        ],
    });

    GpuSlot {
        buffer_a,
        buffer_b,
        buffer_mask_n_a,
        buffer_mask_n_b,
        buffer_scores,
        buffer_dirs,
        buffer_cigar,
        staging_scores,
        staging_cigar,
        bind_group_dp,
        bind_group_tb,
        buffer_group_info,
    }
}

// Данные, находящиеся в процессе вычислений на GPU
struct InFlight {
    sub_idx: wgpu::SubmissionIndex,
    rx_score: tokio::sync::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
    rx_cigar: tokio::sync::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
    valid_reads: usize,
    slot_idx: usize,
}

pub async fn gpu_actor(
    args: Args,
    mut rx: mpsc::Receiver<GpuBatch>,
    score_tx: mpsc::Sender<ScoreBatch>,
) {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .unwrap_or_else(|_| panic!("❌ Не удалось найти видеокарту!"));

    let info = adapter.get_info();
    println!("🚀 Используемый GPU: [{:?}] {}", info.backend, info.name);

    // Запрашиваем максимальные лимиты, доступные на адаптере
    let limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: (args.batch_size * READ_LEN * 16 * 4 * 2) as u64,
                max_buffer_size: (args.batch_size * READ_LEN * 16 * 4 * 2) as u64,
                ..limits
            },
            ..Default::default()
        })
        .await
        .unwrap();

    // 1. Компиляция ДВУХ раздельных шейдеров
    let dp_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/dp_kernel.wgsl"));
    let dp_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dp_pipeline"),
        layout: None,
        module: &dp_shader,
        entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let tb_shader =
        device.create_shader_module(wgpu::include_wgsl!("../shaders/traceback_kernel.wgsl"));
    let tb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("traceback_pipeline"),
        layout: None,
        module: &tb_shader,
        entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let dp_layout = dp_pipeline.get_bind_group_layout(0);
    let tb_layout = tb_pipeline.get_bind_group_layout(0);

    // Общий буфер параметров
    let buffer_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params_uniform"),
        size: std::mem::size_of::<GpuParams>() as wgpu::BufferAddress,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let default_params = GpuParams {
        read_len: u32::try_from(READ_LEN).unwrap(),
        ref_len: u32::try_from(REF_LEN).unwrap(),
        match_score: 3,
        mismatch_penalty: -3,
        gap_open: -6,
        gap_extend: -1,
        z_drop: 100,
        batch_size: u32::try_from(args.batch_size).unwrap(),
    };
    queue.write_buffer(&buffer_params, 0, bytemuck::bytes_of(&default_params));

    // 2. Создаем ДВА слота (Double-Buffering)
    let slots = [
        create_slot(
            &device,
            args.batch_size,
            &buffer_params,
            &dp_layout,
            &tb_layout,
            0,
        ),
        create_slot(
            &device,
            args.batch_size,
            &buffer_params,
            &dp_layout,
            &tb_layout,
            1,
        ),
    ];

    let mut slot_idx = 0;
    let mut in_flight: Option<InFlight> = None;

    while let Some(batch) = rx.recv().await {
        let slot = &slots[slot_idx];

        // Асинхронно заливаем данные в текущий слот
        queue.write_buffer(&slot.buffer_a, 0, bytemuck::cast_slice(&batch.packed_a));
        queue.write_buffer(&slot.buffer_b, 0, bytemuck::cast_slice(&batch.packed_b));
        queue.write_buffer(
            &slot.buffer_mask_n_a,
            0,
            bytemuck::cast_slice(&batch.mask_n_a),
        );
        queue.write_buffer(
            &slot.buffer_mask_n_b,
            0,
            bytemuck::cast_slice(&batch.mask_n_b),
        );
        queue.write_buffer(
            &slot.buffer_group_info,
            0,
            bytemuck::cast_slice(&batch.group_info),
        );

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        const WORKGROUP_SIZE: u32 = 64;
        let workgroups = (batch.valid_reads as f32 / WORKGROUP_SIZE as f32).ceil() as u32;

        // ПАСС 1: DP (Матрица Гото)
        {
            let mut cpass1 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DP_Pass"),
                timestamp_writes: None,
            });
            cpass1.set_pipeline(&dp_pipeline);
            cpass1.set_bind_group(0, &slot.bind_group_dp, &[]);
            cpass1.dispatch_workgroups(workgroups, 1, 1);
        }

        // ПАСС 2: Traceback (Драйвер GPU автоматически ставит барьер памяти между пассами)
        {
            let mut cpass2 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TB_Pass"),
                timestamp_writes: None,
            });
            cpass2.set_pipeline(&tb_pipeline);
            cpass2.set_bind_group(0, &slot.bind_group_tb, &[]);
            cpass2.dispatch_workgroups(workgroups, 1, 1);
        }

        let copy_scores_size = (batch.valid_reads * SCORES_PER_READ * 4) as wgpu::BufferAddress;
        let copy_cigar_size = (batch.valid_reads * CIGAR_CAPACITY * 4) as wgpu::BufferAddress;

        encoder.copy_buffer_to_buffer(
            &slot.buffer_scores,
            0,
            &slot.staging_scores,
            0,
            copy_scores_size,
        );
        encoder.copy_buffer_to_buffer(
            &slot.buffer_cigar,
            0,
            &slot.staging_cigar,
            0,
            copy_cigar_size,
        );

        // Отправляем оба пасса на исполнение одной транзакцией
        let sub_idx = queue.submit(Some(encoder.finish()));

        // Регистрируем async-чтение для текущего слота
        let (tx1, rx1) = tokio::sync::oneshot::channel();
        let (tx2, rx2) = tokio::sync::oneshot::channel();

        slot.staging_scores
            .slice(..copy_scores_size)
            .map_async(wgpu::MapMode::Read, move |v| {
                let _ = tx1.send(v);
            });
        slot.staging_cigar
            .slice(..copy_cigar_size)
            .map_async(wgpu::MapMode::Read, move |v| {
                let _ = tx2.send(v);
            });

        // Если есть батч, который GPU считал на ПРОШЛОМ шаге — забираем его
        if let Some(prev) = in_flight.take() {
            // Ждем ТОЛЬКО завершения предыдущей отправки (не блокируя текущую!)
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(prev.sub_idx),
                    timeout: None,
                })
                .unwrap();

            if prev.rx_score.await.is_ok() && prev.rx_cigar.await.is_ok() {
                let prev_slot = &slots[prev.slot_idx];
                let prev_scores_size =
                    (prev.valid_reads * SCORES_PER_READ * 4) as wgpu::BufferAddress;
                let prev_cigar_size =
                    (prev.valid_reads * CIGAR_CAPACITY * 4) as wgpu::BufferAddress;

                let data_score = prev_slot
                    .staging_scores
                    .slice(..prev_scores_size)
                    .get_mapped_range()
                    .unwrap();
                let data_cigar = prev_slot
                    .staging_cigar
                    .slice(..prev_cigar_size)
                    .get_mapped_range()
                    .unwrap();

                if score_tx
                    .send(ScoreBatch {
                        scores_and_coords: bytemuck::cast_slice(&data_score).to_vec(),
                        cigar_bin: bytemuck::cast_slice(&data_cigar).to_vec(),
                    })
                    .await
                    .is_err()
                {
                    break;
                }

                drop(data_score);
                drop(data_cigar);
                prev_slot.staging_scores.unmap();
                prev_slot.staging_cigar.unmap();
            }
        }

        // Запоминаем текущий батч как in_flight
        in_flight = Some(InFlight {
            sub_idx,
            rx_score: rx1,
            rx_cigar: rx2,
            valid_reads: batch.valid_reads,
            slot_idx,
        });

        // Переключаем слот (0 -> 1 -> 0 -> 1)
        slot_idx = 1 - slot_idx;
    }

    // Хвостовой батч (забираем самый последний результат после завершения цикла)
    if let Some(prev) = in_flight.take() {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(prev.sub_idx),
                timeout: None,
            })
            .unwrap();

        if prev.rx_score.await.is_ok() && prev.rx_cigar.await.is_ok() {
            let prev_slot = &slots[prev.slot_idx];
            let prev_scores_size = (prev.valid_reads * SCORES_PER_READ * 4) as wgpu::BufferAddress;
            let prev_cigar_size = (prev.valid_reads * CIGAR_CAPACITY * 4) as wgpu::BufferAddress;

            let data_score = prev_slot
                .staging_scores
                .slice(..prev_scores_size)
                .get_mapped_range()
                .unwrap();
            let data_cigar = prev_slot
                .staging_cigar
                .slice(..prev_cigar_size)
                .get_mapped_range()
                .unwrap();

            let _ = score_tx
                .send(ScoreBatch {
                    scores_and_coords: bytemuck::cast_slice(&data_score).to_vec(),
                    cigar_bin: bytemuck::cast_slice(&data_cigar).to_vec(),
                })
                .await;

            drop(data_score);
            drop(data_cigar);
            prev_slot.staging_scores.unmap();
            prev_slot.staging_cigar.unmap();
        }
    }
}

// async fn gpu_actor(
//     args: Args,
//     mut rx: mpsc::Receiver<GpuBatch>,
//     score_tx: mpsc::Sender<ScoreBatch>,
// ) {
//     let instance = wgpu::Instance::default();

//     let adapter = instance
//         .request_adapter(&wgpu::RequestAdapterOptions {
//             power_preference: wgpu::PowerPreference::HighPerformance,
//             force_fallback_adapter: false,
//             compatible_surface: None,
//             apply_limit_buckets: false,
//         })
//         .await
//         .unwrap_or_else(|_| panic!("❌ Не удалось найти дискретную видеокарту!"));

//     let info = adapter.get_info();
//     println!("🚀 Используемый GPU: [{:?}] {}", info.backend, info.name);

//     let (device, queue) = adapter
//         .request_device(&wgpu::DeviceDescriptor {
//             required_limits: wgpu::Limits {
//                 max_storage_buffer_binding_size: 1024 * 1024 * 1024,
//                 max_buffer_size: 1024 * 1024 * 1024,
//                 ..adapter.limits()
//             },
//             ..Default::default()
//         })
//         .await
//         .unwrap();

//     let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
//     let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//         label: None,
//         layout: None,
//         module: &shader,
//         entry_point: Some("main"),
//         compilation_options: PipelineCompilationOptions::default(),
//         cache: None,
//     });

//     // --- РАСЧЕТ РАЗМЕРОВ БУФЕРОВ ---
//     let buffer_a_size = (args.batch_size * REF_WORDS * 4) as wgpu::BufferAddress;
//     let buffer_b_size = (args.batch_size * READ_WORDS * 4) as wgpu::BufferAddress;
//     let mask_n_a_size = (args.batch_size * N_MASK_WORDS * 4) as wgpu::BufferAddress;
//     let mask_n_b_size = (args.batch_size * N_MASK_WORDS * 4) as wgpu::BufferAddress;
//     let scores_size = (args.batch_size * SCORES_PER_READ * 4) as wgpu::BufferAddress;
//     let dirs_size = (args.batch_size * READ_LEN * REF_WORDS_4BIT * 4) as wgpu::BufferAddress;
//     let cigar_size = (args.batch_size * CIGAR_CAPACITY * 4) as wgpu::BufferAddress;
//     let params_size = std::mem::size_of::<GpuParams>() as wgpu::BufferAddress;

//     // --- СОЗДАНИЕ STORAGE И UNIFORM БУФЕРОВ НА GPU ---
//     let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("seqs_a"),
//         size: buffer_a_size,
//         mapped_at_creation: false,
//         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//     });

//     let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("seqs_b"),
//         size: buffer_b_size,
//         mapped_at_creation: false,
//         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//     });

//     let buffer_mask_n_a = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("mask_n_a"),
//         size: mask_n_a_size,
//         mapped_at_creation: false,
//         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//     });

//     let buffer_mask_n_b = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("mask_n_b"),
//         size: mask_n_b_size,
//         mapped_at_creation: false,
//         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//     });

//     let buffer_scores = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("scores_out"),
//         size: scores_size,
//         mapped_at_creation: false,
//         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//     });

//     let buffer_dirs = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("directions"),
//         size: dirs_size,
//         mapped_at_creation: false,
//         usage: wgpu::BufferUsages::STORAGE,
//     });

//     let buffer_cigar = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("cigar_out"),
//         size: cigar_size,
//         mapped_at_creation: false,
//         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//     });

//     let buffer_params = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("params_uniform"),
//         size: params_size,
//         mapped_at_creation: false,
//         usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
//     });

//     // --- STAGING БУФЕРЫ ДЛЯ ЧТЕНИЯ НА CPU ---
//     let staging_scores = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("staging_scores"),
//         size: scores_size,
//         mapped_at_creation: false,
//         usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
//     });

//     let staging_cigar = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("staging_cigar"),
//         size: cigar_size,
//         mapped_at_creation: false,
//         usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
//     });

//     // Инициализируем параметры по умолчанию
//     let default_params = GpuParams {
//         read_len: u32::try_from(READ_LEN).unwrap(),
//         ref_len: u32::try_from(REF_LEN).unwrap(),
//         match_score: 3,
//         mismatch_penalty: -3,
//         gap_open: -6,
//         gap_extend: -1,
//         z_drop: 100,
//         batch_size: u32::try_from(args.batch_size).unwrap(),
//     };
//     queue.write_buffer(&buffer_params, 0, bytemuck::bytes_of(&default_params));

//     // На случай, если в батче пока нет N-масок — создаем буфер нулей
//     let zero_n_masks = vec![0u32; args.batch_size * N_MASK_WORDS];
//     queue.write_buffer(&buffer_mask_n_a, 0, bytemuck::cast_slice(&zero_n_masks));
//     queue.write_buffer(&buffer_mask_n_b, 0, bytemuck::cast_slice(&zero_n_masks));

//     let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

//     let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
//         label: Some("main_bind_group"),
//         layout: &bind_group_layout,
//         entries: &[
//             wgpu::BindGroupEntry {
//                 binding: 0,
//                 resource: buffer_a.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 1,
//                 resource: buffer_b.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 2,
//                 resource: buffer_mask_n_a.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 3,
//                 resource: buffer_mask_n_b.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 4,
//                 resource: buffer_scores.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 5,
//                 resource: buffer_dirs.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 6,
//                 resource: buffer_cigar.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 7,
//                 resource: buffer_params.as_entire_binding(),
//             },
//         ],
//     });

//     while let Some(batch) = rx.recv().await {
//         queue.write_buffer(&buffer_a, 0, bytemuck::cast_slice(&batch.packed_a));
//         queue.write_buffer(&buffer_b, 0, bytemuck::cast_slice(&batch.packed_b));

//         // Если вы добавили поля mask_n_a/mask_n_b в GpuBatch, раскомментируйте:
//         queue.write_buffer(&buffer_mask_n_a, 0, bytemuck::cast_slice(&batch.mask_n_a));
//         queue.write_buffer(&buffer_mask_n_b, 0, bytemuck::cast_slice(&batch.mask_n_b));

//         let mut encoder =
//             device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
//         {
//             let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
//                 label: None,
//                 timestamp_writes: None,
//             });

//             cpass.set_pipeline(&compute_pipeline);
//             cpass.set_bind_group(0, &bind_group, &[]);
//             cpass.dispatch_workgroups((batch.valid_reads as f32 / 64.0).ceil() as u32, 1, 1);
//         }

//         // Копируем точный объем данных валидных ридов
//         let copy_scores_size = (batch.valid_reads * SCORES_PER_READ * 4) as wgpu::BufferAddress;
//         let copy_cigar_size = (batch.valid_reads * CIGAR_CAPACITY * 4) as wgpu::BufferAddress;

//         encoder.copy_buffer_to_buffer(&buffer_scores, 0, &staging_scores, 0, copy_scores_size);
//         encoder.copy_buffer_to_buffer(&buffer_cigar, 0, &staging_cigar, 0, copy_cigar_size);

//         queue.submit(Some(encoder.finish()));

//         let (tx1, rx1) = tokio::sync::oneshot::channel();
//         let (tx2, rx2) = tokio::sync::oneshot::channel();

//         staging_scores
//             .slice(..copy_scores_size)
//             .map_async(wgpu::MapMode::Read, move |v| {
//                 tx1.send(v).unwrap();
//             });

//         staging_cigar
//             .slice(..copy_cigar_size)
//             .map_async(wgpu::MapMode::Read, move |v| {
//                 tx2.send(v).unwrap();
//             });

//         device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

//         if rx1.await.is_ok() && rx2.await.is_ok() {
//             let data_score = staging_scores
//                 .slice(..copy_scores_size)
//                 .get_mapped_range()
//                 .unwrap();

//             let data_cigar = staging_cigar
//                 .slice(..copy_cigar_size)
//                 .get_mapped_range()
//                 .unwrap();

//             if score_tx
//                 .send(ScoreBatch {
//                     scores_and_coords: bytemuck::cast_slice(&data_score).to_vec(),
//                     cigar_bin: bytemuck::cast_slice(&data_cigar).to_vec(),
//                 })
//                 .await
//                 .is_err()
//             {
//                 break;
//             }
//         }

//         staging_scores.unmap();
//         staging_cigar.unmap();
//     }
// }
