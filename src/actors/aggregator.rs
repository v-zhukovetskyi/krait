use crate::{
    actors::producer::ComputeChunk,
    constants::{N_MASK_WORDS, READ_WORDS, REF_WORDS},
};
use std::collections::VecDeque;
use tokio::sync::mpsc;

// Сообщение-запрос от GPU-актора: "Дай мне batch_size задач"
pub struct GpuRequest {
    pub requested_size: usize,
    pub respond_to: tokio::sync::oneshot::Sender<Option<GpuReadyBatch>>,
}

pub struct GpuReadyBatch {
    pub soa_packed_a: Vec<u32>,
    pub soa_packed_b: Vec<u32>,
    pub soa_mask_n_a: Vec<u32>,
    pub soa_mask_n_b: Vec<u32>,
    pub soa_group_info: Vec<u32>,
    pub count: usize, // Реальное количество задач (<= запрошенного)
}
pub struct GpuReadyBatch {
    pub soa_arena: Vec<u32>,
    pub soa_packed_a: std::ops::Range<u64>,
    pub soa_packed_b: std::ops::Range<u64>,
    pub soa_mask_n_a: std::ops::Range<u64>,
    pub soa_mask_n_b: std::ops::Range<u64>,
    pub soa_group_info: std::ops::Range<u64>,
    pub count: usize, // Реальное количество задач (<= запрошенного)
}

pub async fn aggregator_task_1(
    mut producer_rx: mpsc::Receiver<ComputeChunk>,
    mut gpu_req_rx: mpsc::Receiver<GpuRequest>,
    writer_data_tx: mpsc::Sender<WriterDataChunk>,
    max_memory_tasks: usize, // Лимит на размер склада в ОЗУ (Backpressure)
) {
    // Внутренний склад задач
    let mut task_queue: VecDeque<GpuTask> = VecDeque::with_capacity(max_memory_tasks);

    // Очередь чанков с сырыми данными (ждут полной выработки)
    let mut chunks_tracker: VecDeque<(Vec<u8>, std::ops::Range<u64>, usize)> = VecDeque::new();
    // (сырые данные, диапазон id, сколько задач из этого чанка еще осталось на складе)

    let mut producer_finished = false;

    // Предвыделенные SoA-массивы максимального размера для быстрой сборки
    let max_alloc_size = 131_072; // Максимально возможный батч GPU
    let mut soa_packed_a = vec![0u32; max_alloc_size * REF_WORDS];
    let mut soa_packed_b = vec![0u32; max_alloc_size * READ_WORDS];
    let mut soa_mask_n_a = vec![0u32; max_alloc_size * N_MASK_WORDS];
    let mut soa_mask_n_b = vec![0u32; max_alloc_size * N_MASK_WORDS];
    let mut soa_group_info = vec![0u32; max_alloc_size];

    loop {
        tokio::select! {
            // 1. ПРИЕМ НОВЫХ ДАННЫХ ОТ ПРОДЮСЕРА (только если на складе есть место)
            maybe_chunk = producer_rx.recv(), if !producer_finished && task_queue.len() < max_memory_tasks => {
                match maybe_chunk {
                    Some(chunk) => {
                        let task_count = chunk.tasks.len();
                        if task_count > 0 {
                            // Запоминаем чанк в трекер
                            chunks_tracker.push_back((chunk.raw_data, chunk.read_id_range, task_count));
                            // Ссыпаем задачи на общий склад
                            task_queue.extend(chunk.tasks);
                        }
                    }
                    None => {
                        // Продюсер закончил чтение файла
                        producer_finished = true;
                    }
                }
            }

            // 2. ОБСЛУЖИВАНИЕ ЗАПРОСА ОТ GPU-АКТОРА
            Some(req) = gpu_req_rx.recv() => {
                let batch_size = req.requested_size.min(max_alloc_size);

                // Если задач на складе нет, и продюсер закончил — отправляем сигнал завершения
                if task_queue.is_empty() && producer_finished {
                    let _ = req.respond_to.send(None);
                    break;
                }

                // Забираем ровно столько, сколько есть, но не больше batch_size
                let take_count = task_queue.len().min(batch_size);

                if take_count > 0 {
                    // РАСКЛАДЫВАЕМ В SoA НАПРЯМУЮ В L1/L2 КЭШЕ
                    for count in 0..take_count {
                        let task = task_queue.pop_front().unwrap();

                        for w in 0..REF_WORDS {
                            soa_packed_a[w * take_count + count] = task.packed_ref[w];
                        }
                        for w in 0..READ_WORDS {
                            soa_packed_b[w * take_count + count] = task.packed_read[w];
                        }
                        for w in 0..N_MASK_WORDS {
                            soa_mask_n_a[w * take_count + count] = task.mask_n_ref[w];
                        }
                        for w in 0..N_MASK_WORDS {
                            soa_mask_n_b[w * take_count + count] = task.mask_n_read[w];
                        }
                        soa_group_info[count] = task.group_info;

                        // Уменьшаем счетчик оставшихся задач у самого старого чанка
                        if let Some(front_chunk) = chunks_tracker.front_mut() {
                            front_chunk.2 -= 1;
                            // Если чанк полностью выработан — отправляем его Райтеру!
                            if front_chunk.2 == 0 {
                                let (raw_data, id_range, _) = chunks_tracker.pop_front().unwrap();
                                let _ = writer_data_tx.send(WriterDataChunk {
                                    raw_data,
                                    read_id_range: id_range,
                                }).await;
                            }
                        }
                    }

                    // Формируем готовый GPU батч ровно на take_count задач
                    let batch = GpuReadyBatch {
                        soa_packed_a: soa_packed_a[..take_count * REF_WORDS].to_vec(),
                        soa_packed_b: soa_packed_b[..take_count * READ_WORDS].to_vec(),
                        soa_mask_n_a: soa_mask_n_a[..take_count * N_MASK_WORDS].to_vec(),
                        soa_mask_n_b: soa_mask_n_b[..take_count * N_MASK_WORDS].to_vec(),
                        soa_group_info: soa_group_info[..take_count].to_vec(),
                        count: take_count,
                    };

                    let _ = req.respond_to.send(Some(batch));
                } else {
                    // Если склад временно пуст, но продюсер еще работает, шлем пустой батч на ожидание
                    let _ = req.respond_to.send(Some(GpuReadyBatch {
                        soa_packed_a: Vec::new(),
                        soa_packed_b: Vec::new(),
                        soa_mask_n_a: Vec::new(),
                        soa_mask_n_b: Vec::new(),
                        soa_group_info: Vec::new(),
                        count: 0,
                    }));
                }
            }
        }
    }
}

pub async fn aggregator_task(
    mut producer_rx: mpsc::Receiver<ComputeChunk>,
    mut gpu_req_rx: mpsc::Receiver<GpuRequest>,
    max_alloc_size: usize,
) {
    // Храним только трекер для райтера и остатки задач из текущего чанка
    let mut current_chunk: Option<ComputeChunk> = None;
    let mut current_task_idx = 0;
    let mut producer_finished = false;

    // Цикл полностью просыпается ТОЛЬКО по запросу от GPU
    while let Some(req) = gpu_req_rx.recv().await {
        let batch_size = req.requested_size.min(max_alloc_size);

        // Выделяем SoA-батч сразу нужного размера
        let mut soa_packed_a = Vec::with_capacity(batch_size * REF_WORDS);
        // ... (остальные соа массивы)
        let mut take_count = 0;

        while take_count < batch_size {
            // Если текущий чанк закончился или еще не был взят, берем следующий из канала-буфера
            if current_chunk.is_none() && !producer_finished {
                // МГНОВЕННО забираем из канала, так как Продюсер заполнил его заранее!
                // Если канал пуст, мы асинхронно подождем (но GPU тут не проиграет, так как данных физически еще нет)
                match producer_rx.recv().await {
                    Some(chunk) => {
                        current_chunk = Some(chunk);
                        current_task_idx = 0;
                    }
                    None => {
                        producer_finished = true;
                    }
                }
            }

            // Если продюсер пуст и чанков больше нет — выходим из сборки батча
            if current_chunk.is_none() && producer_finished {
                break;
            }

            // Извлекаем задачи из текущего активного чанка
            if let Some(chunk) = &current_chunk {
                let task = &chunk.tasks[current_task_idx];

                // Прямая SoA-упаковка (мгновенно в кэш процессора)
                // soa_packed_a.push(...) и т.д.

                current_task_idx += 1;
                take_count += 1;

                // Если чанк полностью исчерпан
            }
        }

        // Финишный сценарий: батч пуст и продюсер завершился
        if take_count == 0 && producer_finished {
            let _ = req.respond_to.send(None);
            break;
        }

        // Отправляем сформированный батч на GPU
        let batch = GpuReadyBatch {
            /* ... */ count: take_count,
        };
        let _ = req.respond_to.send(Some(batch));
    }
}



const STRIDE: usize = REF_WORDS + READ_WORDS + (N_MASK_WORDS * 2) + 1;

let total_u32_size = take_count * STRIDE;

// Выделяем память ОДИН РАЗ под все 5 массивов сразу
let mut arena = vec![0u32; total_u32_size];

// Размечаем диапазоны (Range) в этой арене друг за другом
let a_start = 0;
let a_end = a_start + (take_count * REF_WORDS);

let b_start = a_end;
let b_end = b_start + (take_count * READ_WORDS);

let mask_a_start = b_end;
let mask_a_end = mask_a_start + (take_count * N_MASK_WORDS);

let mask_b_start = mask_a_end;
let mask_b_end = mask_b_start + (take_count * N_MASK_WORDS);

let info_start = mask_b_end;
let info_end = info_start + take_count;

for task_idx in 0..batch_size {
    let base = task_idx * STRIDE;
    
    // 0..8: packed_ref
    for w in 0..REF_WORDS { gpu_arena[a_start + w * batch_size + task_idx] = task_arena[base + w]; }
    
    // 8..15: packed_read
    for w in 0..READ_WORDS { gpu_arena[b_start + w * batch_size + task_idx] = task_arena[base + REF_WORDS + w]; }
    
    // 15..19: mask_n_ref
    for w in 0..N_MASK_WORDS { gpu_arena[mask_a_start + w * batch_size + task_idx] = task_arena[base + REF_WORDS + READ_WORDS + w]; }
    
    // 19..23: mask_n_read
    for w in 0..N_MASK_WORDS { gpu_arena[mask_b_start + w * batch_size + task_idx] = task_arena[base + REF_WORDS + READ_WORDS + N_MASK_WORDS + w]; }
    
    // 23: group_info
    gpu_arena[info_start + task_idx] = task_arena[base + REF_WORDS + READ_WORDS + N_MASK_WORDS * 2];
}

// Сборка батча (передаем саму арену и диапазоны)
let batch = GpuReadyBatch {
    soa_arena: arena, // Вектор перемещается (move), аллокации не происходит!
    soa_packed_a: a_start..a_end,
    soa_packed_b: b_start..b_end,
    soa_mask_n_a: mask_a_start..mask_a_end,
    soa_mask_n_b: mask_b_start..mask_b_end,
    soa_group_info: info_start..info_end,
    count: take_count,
};
