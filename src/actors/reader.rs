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

use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
};

use flate2::read::MultiGzDecoder;
use std::ops::Range;
use tokio::sync::mpsc;

pub fn open_flexible_reader(path: &str) -> BufReader<Box<dyn Read + Send>> {
    let file = File::open(path).unwrap_or_else(|_| panic!("❌ Не удалось открыть файл: {path}"));

    if std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
    {
        BufReader::new(Box::new(MultiGzDecoder::new(file)))
    } else {
        BufReader::new(Box::new(file))
    }
}

pub struct SingleOffsets {
    pub name: std::ops::Range<usize>,
    pub seq: std::ops::Range<usize>,
    pub qual: std::ops::Range<usize>,
}

// Единый буфер батча
pub struct SingleChunkBuffer {
    pub data: Vec<u8>,
    pub pairs: Vec<SingleOffsets>,
}

// Легковесная структура координат для пары ридов
pub struct PairOffsets {
    pub name: std::ops::Range<usize>,
    pub seq1: std::ops::Range<usize>,
    pub qual1: std::ops::Range<usize>,
    pub seq2: std::ops::Range<usize>,
    pub qual2: std::ops::Range<usize>,
}

// Единый буфер батча
pub struct PairChunkBuffer {
    pub data: Vec<u8>,
    pub pairs: Vec<PairOffsets>,
}

pub enum ChunkType {
    Single(SingleChunkBuffer),
    Paired(PairChunkBuffer),
}

fn trim_newline_slice(mut s: &[u8]) -> &[u8] {
    // Пока срез не пустой и заканчивается на перенос строки — уменьшаем его длину
    while !s.is_empty() && (s[s.len() - 1] == b'\n' || s[s.len() - 1] == b'\r') {
        s = &s[..s.len() - 1]; // Сужаем срез с конца на 1 байт
    }
    s
}

fn clean_fastq_name(mut name: &[u8]) -> &[u8] {
    // 1. Отрезаем @ в начале (просто сдвигаем указатель начала среза на 1 байт вперед)
    if name.starts_with(b"@") {
        name = &name[1..];
    }

    // 2. Отрезаем пробелы и метаданные (сужаем срез до позиции пробела)
    if let Some(pos) = name.iter().position(|&b| b == b' ' || b == b'\t') {
        name = &name[..pos];
    }

    // 3. Отрезаем суффиксы "/1" или "/2" на конце (сужаем срез с конца)
    if name.ends_with(b"/1") || name.ends_with(b"/2") {
        name = &name[..name.len() - 2];
    }

    name
}

fn read_name(reader: &mut impl BufRead, buffer: &mut Vec<u8>) -> Range<usize> {
    let start_pos = buffer.len();
    let res = reader.read_until(b'\n', buffer).unwrap();
    if res == 0 {
        return 0..0; // EOF
    }
    let end_pos = buffer.len();

    let raw = &buffer[start_pos..end_pos];
    let trimmed = trim_newline_slice(raw);
    let cleaned = clean_fastq_name(trimmed);

    // Смещение начала (если отрезали '@'):
    let front_offset = if trimmed.starts_with(b"@") { 1 } else { 0 };
    let clean_start = start_pos + front_offset;
    let clean_end = clean_start + cleaned.len();

    clean_start..clean_end
}

fn read_seq(reader: &mut impl BufRead, buffer: &mut Vec<u8>) -> Range<usize> {
    let start_pos = buffer.len();
    reader.read_until(b'\n', buffer).unwrap();
    let end_pos = buffer.len();

    let clean_len = trim_newline_slice(&buffer[start_pos..end_pos]).len();
    start_pos..(start_pos + clean_len)
}

/// Пропускает ровно одну строку (разделитель '+') вообще без копирования байт
fn skip_line(reader: &mut impl BufRead) -> bool {
    let available_bytes = reader.fill_buf().unwrap();

    // Если внутренний буфер пуст — мы гарантированно дошли до конца файла (EOF)
    if available_bytes.is_empty() {
        return false;
    }

    if let Some(pos) = available_bytes.iter().position(|&b| b == b'\n') {
        let bytes_to_consume = pos + 1;
        reader.consume(bytes_to_consume);
        true
    } else {
        // Если строка оказалась длиннее, чем внутренний буфер ридера,
        // или файл оборвался без \n на конце
        let mut fallback = Vec::new();
        let bytes_read = reader.read_until(b'\n', &mut fallback).unwrap();

        // Если read_until прочитал 0 байт, значит это честный EOF
        bytes_read > 0
    }
}

pub fn reader_task(
    path1: String,
    path2: Option<String>,
    batch_size: usize,
    chunk_tx: mpsc::Sender<ChunkType>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        let mut r1: BufReader<Box<dyn Read + Send>> = open_flexible_reader(&path1);

        // 1. Прощупываем первый рид из обоих файлов, чтобы узнать точные размеры
        let mut first_r1_bytes = Vec::with_capacity(2048);

        // Считываем по 4 строки для определения размера
        let r1_name = read_name(&mut r1, &mut first_r1_bytes);
        if r1_name.is_empty() {
            return;
        } // Честный конец файла
        let r1_seq = read_seq(&mut r1, &mut first_r1_bytes);
        if !skip_line(&mut r1) {
            return;
        };
        let r1_qual = read_seq(&mut r1, &mut first_r1_bytes);

        if let Some(p2) = path2 {
            println!("📦 [Reader] Режим: Paired-End...");

            let mut r2 = open_flexible_reader(&p2);
            let mut first_r2_bytes = Vec::with_capacity(2048);

            if !skip_line(&mut r2) {
                eprintln!("⚠️ Внимание: Файл R2 короче, чем R1! Возможна рассинхронизация пар.");
                return;
            }
            let r2_seq = read_seq(&mut r2, &mut first_r2_bytes);
            if !skip_line(&mut r2) {
                return;
            }
            let r2_qual = read_seq(&mut r2, &mut first_r2_bytes);

            // Вычисляем точный размер одной пары ридов в байтах (включая все заголовки и \n)
            // Добавляем к длине первой пары небольшой запас (например, 8 байт)
            // на случай, если у последующих ридов заголовки будут длиннее
            let pair_bytes_len = first_r1_bytes.len() + first_r2_bytes.len() + 8;

            // Идеальный размер буфера под данные всего батча
            let data_capacity = pair_bytes_len * batch_size;

            // Создаем первый чанк и сразу копируем туда данные первого рида, которые мы уже считали
            let mut current_chunk = PairChunkBuffer {
                data: Vec::with_capacity(data_capacity),
                pairs: Vec::with_capacity(batch_size),
            };

            // Переносим байты первого рида в основной буфер
            current_chunk.data.extend_from_slice(&first_r1_bytes);
            current_chunk.data.extend_from_slice(&first_r2_bytes);

            let r1_bytes_len = first_r1_bytes.len();

            current_chunk.pairs.push(PairOffsets {
                name: r1_name,
                seq1: r1_seq,
                qual1: r1_qual,
                seq2: (r2_seq.start + r1_bytes_len)..(r2_seq.end + r1_bytes_len),
                qual2: (r2_qual.start + r1_bytes_len)..(r2_qual.end + r1_bytes_len),
            });

            // Переиспользуемая функция/замыкание для создания нового чанка с точно рассчитанной емкостью
            let make_new_chunk = || PairChunkBuffer {
                data: Vec::with_capacity(data_capacity),
                pairs: Vec::with_capacity(batch_size),
            };

            loop {
                // Читаем R1
                let name_range = read_name(&mut r1, &mut current_chunk.data);
                if name_range.is_empty() {
                    break;
                } // Честный конец файла

                let seq1_range = read_seq(&mut r1, &mut current_chunk.data);
                if !skip_line(&mut r1) {
                    break;
                } // Защита: файл оборвался на строке '+'
                let qual1_range = read_seq(&mut r1, &mut current_chunk.data);

                // Читаем R2
                if !skip_line(&mut r2) {
                    eprintln!(
                        "⚠️ Внимание: Файл R2 короче, чем R1! Возможна рассинхронизация пар."
                    );
                    break;
                } // Защита: файл R2 закончился раньше времени!
                let seq2_range = read_seq(&mut r2, &mut current_chunk.data);
                if !skip_line(&mut r2) {
                    break;
                }
                let qual2_range = read_seq(&mut r2, &mut current_chunk.data);

                // Записываем точечные координаты (убирая служебные \n и символы @ / + при желании)
                current_chunk.pairs.push(PairOffsets {
                    name: name_range,
                    seq1: seq1_range,
                    qual1: qual1_range,
                    seq2: seq2_range,
                    qual2: qual2_range,
                });

                if current_chunk.pairs.len() == batch_size {
                    // Подменяем заполненный чанк новым, у которого емкость выставлена ИДЕАЛЬНО
                    let full_chunk = std::mem::replace(&mut current_chunk, make_new_chunk());
                    if chunk_tx
                        .blocking_send(ChunkType::Paired(full_chunk))
                        .is_err()
                    {
                        break;
                    }
                }
            }
            if !current_chunk.pairs.is_empty() {
                let _ = chunk_tx.blocking_send(ChunkType::Paired(current_chunk));
            }
        } else {
            println!("📦 [Reader] Режим: Single-End...");

            // Идеальный размер буфера под данные всего батча
            let data_capacity = batch_size * (first_r1_bytes.len() + 8);

            // Создаем первый чанк и сразу копируем туда данные первого рида, которые мы уже считали
            let mut current_chunk = SingleChunkBuffer {
                data: Vec::with_capacity(data_capacity),
                pairs: Vec::with_capacity(batch_size),
            };

            // Переносим байты первого рида в основной буфер
            current_chunk.data.extend_from_slice(&first_r1_bytes);

            current_chunk.pairs.push(SingleOffsets {
                name: r1_name,
                seq: r1_seq,
                qual: r1_qual,
            });

            // Переиспользуемая функция/замыкание для создания нового чанка с точно рассчитанной емкостью
            let make_new_chunk = || SingleChunkBuffer {
                data: Vec::with_capacity(data_capacity),
                pairs: Vec::with_capacity(batch_size),
            };

            loop {
                // Читаем R1
                let name_range = read_name(&mut r1, &mut current_chunk.data);
                if name_range.is_empty() {
                    break;
                } // Честный конец файла

                let seq1_range = read_seq(&mut r1, &mut current_chunk.data);
                if !skip_line(&mut r1) {
                    break;
                } // Защита: файл оборвался на строке '+'
                let qual1_range = read_seq(&mut r1, &mut current_chunk.data);

                // Записываем точечные координаты (убирая служебные \n и символы @ / + при желании)
                current_chunk.pairs.push(SingleOffsets {
                    name: name_range,
                    seq: seq1_range,
                    qual: qual1_range,
                });

                if current_chunk.pairs.len() == batch_size {
                    // Подменяем заполненный чанк новым, у которого емкость выставлена ИДЕАЛЬНО
                    let full_chunk = std::mem::replace(&mut current_chunk, make_new_chunk());
                    if chunk_tx
                        .blocking_send(ChunkType::Single(full_chunk))
                        .is_err()
                    {
                        break;
                    }
                }
            }
            if !current_chunk.pairs.is_empty() {
                let _ = chunk_tx.blocking_send(ChunkType::Single(current_chunk));
            }
        }
    })
}
