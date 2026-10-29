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
use wgpu::{ComputePipeline, Device, PipelineCompilationOptions, Queue};

use crate::{
    actors::{
        producer::{GpuBatch, PairedGpuBatch, PairedMeta, SingleGpuBatch},
        writer::ScoreBatch,
    },
    cli::Args,
    constants::{
        CIGAR_CAPACITY, N_MASK_WORDS, READ_LEN, READ_WORDS, REF_LEN, REF_WORDS, REF_WORDS_4BIT,
        SCORES_PER_READ,
    },
};
struct GPU {
    device: Device,
    queue: Queue,
    dp_pipeline: ComputePipeline,
    tb_pipeline: ComputePipeline,
    slots: [GpuSlot; 2],
}
// Подставьте ваши импорты Args, GpuBatch, ScoreBatch, PipelineCompilationOptions и wgp
async fn init_gpu(args: &Args) -> GPU {
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
    let slots: [GpuSlot; 2] = [
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

    GPU {
        device,
        queue,
        dp_pipeline,
        tb_pipeline,
        slots,
    }
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

struct InFlight {
    sub_idx: wgpu::SubmissionIndex,
    rx_score: tokio::sync::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
    rx_cigar: tokio::sync::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
    valid_reads: usize,
    slot_idx: usize,
}

async fn copy_and_send_data(
    gpu: &GPU,
    in_flight: &mut Option<InFlight>,
    count: &std::ops::Range<usize>,
) -> Option<ScoreBatch> {
    let prev = match in_flight.take() {
        Some(p) => p,
        None => return None,
    };

    // ОПТИМИЗАЦИЯ 1: Не блокируем весь асинхронный рантайм Tokio.
    // Выносим тяжелое ожидание GPU в block_in_place, чтобы Tokio переключил другие задачи на соседние потоки.
    tokio::task::block_in_place(|| {
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(prev.sub_idx),
                timeout: None,
            })
            .unwrap();
    });

    // Ожидаем коллбеки от маппинга wgpu
    if prev.rx_score.await.is_err() || prev.rx_cigar.await.is_err() {
        return None; // Или false, в зависимости от того, критична ли эта ошибка для цикла
    }

    let prev_slot = &gpu.slots[prev.slot_idx];
    let prev_scores_size = (prev.valid_reads * SCORES_PER_READ * 4) as wgpu::BufferAddress;
    let prev_cigar_size = (prev.valid_reads * CIGAR_CAPACITY * 4) as wgpu::BufferAddress;

    let mut result = None;

    // Ссылки на данные получаем в ограниченной области видимости
    let _ = {
        let data_score = prev_slot
            .staging_scores
            .slice(..prev_scores_size)
            .get_mapped_range();
        let data_cigar = prev_slot
            .staging_cigar
            .slice(..prev_cigar_size)
            .get_mapped_range();

        let (data_score, data_cigar) = match (data_score, data_cigar) {
            (Ok(s), Ok(c)) => (s, c),
            _ => {
                // Если маппинг не удался, обязательно размапливаем перед выходом
                prev_slot.staging_scores.unmap();
                prev_slot.staging_cigar.unmap();
                return None;
            }
        };

        // ОПТИМИЗАЦИЯ 2: Аллокация (.to_vec()) происходит ДО отправки.
        // Мы сразу освобождаем GPU буферы, чтобы сделать unmap() как можно раньше,
        // не дожидаясь пока `.send().await` разблокируется (если канал был полон).
        result = Some(ScoreBatch {
            count: count.clone(),
            scores_and_coords: bytemuck::cast_slice(&data_score).to_vec(),
            cigar_bin: bytemuck::cast_slice(&data_cigar).to_vec(),
        });
        // Здесь data_score и data_cigar автоматически дропаются (выходят из scope)
    };

    // ИСПРАВЛЕНИЕ БАГА: Размещаем unmap ДО отправки в канал.
    // Теперь, если канал заблокирован (канал заполнен) или закрыт, буферы GPU уже свободны для следующего кадра!
    prev_slot.staging_scores.unmap();
    prev_slot.staging_cigar.unmap();

    result
}

async fn send_task(
    gpu: &GPU,
    slot_idx: &mut usize,
    count: &std::ops::Range<usize>,
    soa_packed_a: &Vec<u32>,
    soa_packed_b: &Vec<u32>,
    soa_mask_n_a: &Vec<u32>,
    soa_mask_n_b: &Vec<u32>,
    soa_group_info: &Vec<u32>,
    in_flight: &mut Option<InFlight>,
) -> (Option<ScoreBatch>, Option<ScoreBatch>) {
    let slot = &gpu.slots[*slot_idx];

    // Асинхронно заливаем данные в текущий слот
    gpu.queue
        .write_buffer(&slot.buffer_a, 0, bytemuck::cast_slice(&soa_packed_a));

    gpu.queue
        .write_buffer(&slot.buffer_b, 0, bytemuck::cast_slice(&soa_packed_b));

    gpu.queue.write_buffer(
        &slot.buffer_mask_n_a,
        0,
        bytemuck::cast_slice(&soa_mask_n_a),
    );

    gpu.queue.write_buffer(
        &slot.buffer_mask_n_b,
        0,
        bytemuck::cast_slice(&soa_mask_n_b),
    );

    gpu.queue.write_buffer(
        &slot.buffer_group_info,
        0,
        bytemuck::cast_slice(&soa_group_info),
    );

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    const WORKGROUP_SIZE: u32 = 64;
    let workgroups = (count.len() as f32 / WORKGROUP_SIZE as f32).ceil() as u32;

    // ПАСС 1: DP (Матрица Гото)
    {
        let mut cpass1 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("DP_Pass"),
            timestamp_writes: None,
        });
        cpass1.set_pipeline(&gpu.dp_pipeline);
        cpass1.set_bind_group(0, &slot.bind_group_dp, &[]);
        cpass1.dispatch_workgroups(workgroups, 1, 1);
    }

    // ПАСС 2: Traceback (Драйвер GPU автоматически ставит барьер памяти между пассами)
    {
        let mut cpass2 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("TB_Pass"),
            timestamp_writes: None,
        });
        cpass2.set_pipeline(&gpu.tb_pipeline);
        cpass2.set_bind_group(0, &slot.bind_group_tb, &[]);
        cpass2.dispatch_workgroups(workgroups, 1, 1);
    }

    let copy_scores_size = (count.len() * SCORES_PER_READ * 4) as wgpu::BufferAddress;
    let copy_cigar_size = (count.len() * CIGAR_CAPACITY * 4) as wgpu::BufferAddress;

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
    let sub_idx = gpu.queue.submit(Some(encoder.finish()));

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
    let result1 = copy_and_send_data(gpu, in_flight, count).await;

    // Запоминаем текущий батч как in_flight
    *in_flight = Some(InFlight {
        sub_idx,
        rx_score: rx1,
        rx_cigar: rx2,
        valid_reads: count.len(),
        slot_idx: *slot_idx,
    });

    // Переключаем слот (0 -> 1 -> 0 -> 1)
    *slot_idx = 1 - *slot_idx;

    // Хвостовой батч (забираем самый последний результат после завершения цикла)
    let result2 = copy_and_send_data(gpu, in_flight, count).await;
    (result1, result2)
}
// Данные, находящиеся в процессе вычислений на GPU
struct SingleWriteButch {
    pub data: SingleGpuBatch,
    pub gpu_result: Vec<ScoreBatch>,
}
struct PairedWriteButch {
    pub data: PairedGpuBatch,
    pub gpu_result: Vec<ScoreBatch>,
}

enum WriteButch {
    Single(SingleWriteButch),
    Paired(PairedWriteButch),
}
pub async fn gpu_actor(
    args: Args,
    mut rx: mpsc::Receiver<GpuBatch>,
    score_tx: mpsc::Sender<WriteButch>,
) {
    let gpu = init_gpu(&args).await;

    let mut slot_idx = 0;
    let mut in_flight: Option<InFlight> = None;

    let mut soa_packed_a = Vec::with_capacity(args.batch_size * REF_WORDS);
    let mut soa_packed_b = Vec::with_capacity(args.batch_size * READ_WORDS);
    let mut soa_mask_n_a = Vec::with_capacity(args.batch_size * N_MASK_WORDS);
    let mut soa_mask_n_b = Vec::with_capacity(args.batch_size * N_MASK_WORDS);
    let mut soa_group_info = Vec::with_capacity(args.batch_size);

    while let Some(batch) = rx.recv().await {
        let mut count: usize = 0;
        let mut prev_count: usize = 0;

        // Вспомогательное микро-замыкание: кладет ОДНУ готовую задачу в SoA-колонки
        macro_rules! push_task {
            ($task:expr, $group_info:expr) => {
                // 1. Раскладываем по колонкам SoA для идеального коалесцинга памяти на GPU:
                for (w, offset) in (0..REF_WORDS).zip((0..).step_by(args.batch_size)) {
                    soa_packed_a[offset + count] = $task.packed_ref[w];
                }

                // Для soa_packed_b (размер READ_WORDS)
                for (w, offset) in (0..READ_WORDS).zip((0..).step_by(args.batch_size)) {
                    soa_packed_b[offset + count] = $task.packed_read[w];
                }

                // Для soa_mask_n_a (размер N_MASK_WORDS)
                for (w, offset) in (0..N_MASK_WORDS).zip((0..).step_by(args.batch_size)) {
                    soa_mask_n_a[offset + count] = $task.mask_n_ref[w];
                }

                // Для soa_mask_n_b (размер N_MASK_WORDS)
                for (w, offset) in (0..N_MASK_WORDS).zip((0..).step_by(args.batch_size)) {
                    soa_mask_n_b[offset + count] = $task.mask_n_read[w];
                }

                soa_group_info[count] = $group_info;
                }
            }

        // А ТЕПЕРЬ MATCH СТАНОВИТСЯ НАМНОГО ЧИЩЕ (не нужно писать &mut count и звёздочки):
        let mut result: Vec<ScoreBatch> = Vec::new();

        match batch {
            GpuBatch::Single(chunk) => {
                while count < chunk.single.len() {
                    let current_id = chunk.single[count].meta.id;
                    let mut j = count + 1;

                    while j < chunk.single.len() && chunk.single[j].meta.id == current_id {
                        j += 1;
                    }

                    if j >= args.batch_size {
                        let batch_tuple = send_task(
                            &gpu,
                            &mut slot_idx,
                            &(prev_count..count),
                            &soa_packed_a,
                            &soa_packed_b,
                            &soa_mask_n_a,
                            &soa_mask_n_b,
                            &soa_group_info,
                            &mut in_flight,
                        )
                        .await; // Здесь мы вышли из async-контекста и получили (Option, Option)

                        // 2. Распаковываем Some и добавляем все найденные элементы в вектор за один раз
                        result.extend([batch_tuple.0, batch_tuple.1].into_iter().flatten());
                        prev_count = count
                    }

                    let group_start_gpu = count as u32;
                    let group_count = (j - count) as u32;
                    let info = (group_start_gpu << 2) | (group_count & 3);

                    for k in count..j {
                        push_task!(&chunk.single[k], info); // Чистый вызов макроса через !
                    }

                    count = j;
                }

                let batch_tuple = send_task(
                    &gpu,
                    &mut slot_idx,
                    &(prev_count..count),
                    &soa_packed_a,
                    &soa_packed_b,
                    &soa_mask_n_a,
                    &soa_mask_n_b,
                    &soa_group_info,
                    &mut in_flight,
                )
                .await; // Здесь мы вышли из async-контекста и получили (Option, Option)

                // 2. Распаковываем Some и добавляем все найденные элементы в вектор за один раз
                result.extend([batch_tuple.0, batch_tuple.1].into_iter().flatten());

                if score_tx
                    .send(WriteButch::Single(SingleWriteButch {
                        data: chunk,
                        gpu_result: result,
                    }))
                    .await
                    .is_err()
                {
                    return; // Канал закрыт, выходим из внешнего цикла
                }
            }

            GpuBatch::Paired(chunk) => {
                for pair in &chunk.pairs {
                    if count + 2 > args.batch_size {
                        let batch_tuple = send_task(
                            &gpu,
                            &mut slot_idx,
                            &(prev_count..count),
                            &soa_packed_a,
                            &soa_packed_b,
                            &soa_mask_n_a,
                            &soa_mask_n_b,
                            &soa_group_info,
                            &mut in_flight,
                        )
                        .await; // Здесь мы вышли из async-контекста и получили (Option, Option)

                        // 2. Распаковываем Some и добавляем все найденные элементы в вектор за один раз
                        result.extend([batch_tuple.0, batch_tuple.1].into_iter().flatten());

                        prev_count = count
                    }

                    let info1 = ((count as u32) << 2) | 1u32;
                    push_task!(&pair.meta1, info1); // Неявная работа с count без ошибок заимствования!

                    let info2 = ((count as u32) << 2) | 1u32;
                    push_task!(&pair.meta2, info2);
                }
                let batch_tuple = send_task(
                    &gpu,
                    &mut slot_idx,
                    &(prev_count..count),
                    &soa_packed_a,
                    &soa_packed_b,
                    &soa_mask_n_a,
                    &soa_mask_n_b,
                    &soa_group_info,
                    &mut in_flight,
                )
                .await; // Здесь мы вышли из async-контекста и получили (Option, Option)

                // 2. Распаковываем Some и добавляем все найденные элементы в вектор за один раз
                result.extend([batch_tuple.0, batch_tuple.1].into_iter().flatten());

                if score_tx
                    .send(WriteButch::Paired(PairedWriteButch {
                        data: chunk,
                        gpu_result: result,
                    }))
                    .await
                    .is_err()
                {
                    return; // Канал закрыт, выходим из внешнего цикла
                }
            }
        }

        if count == 0 {
            break;
        }

        soa_packed_a.clear();
        soa_packed_b.clear();
        soa_mask_n_a.clear();
        soa_mask_n_b.clear();
        soa_group_info.clear();
    }
}
