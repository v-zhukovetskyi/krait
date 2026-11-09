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
use wgpu::{ComputePipeline, Device, Limits, Queue};

use crate::{
    cli::Args,
    constants::{
        BAND_WORDS_4BIT, BYTES_PER_TASK_DIRECTIONS, CIGAR_CAPACITY, READ_LEN, READ_N_MASK_WORDS,
        READ_WORDS, REF_N_MASK_WORDS, REF_WORDS, SCORES_PER_READ, WARP_SIZE,
    },
};

pub struct Gpu {
    pub device: Device,
    pub queue: Queue,
    pub dp_pipeline: ComputePipeline,
    pub tb_pipeline: ComputePipeline,
    pub slots: [GpuSlot; 2],
}

pub struct GpuContext {
    pub gpu: Gpu,
    pub max_size: usize,
}

/// Initializes the GPU context, calibrating buffer limits and compiling compute kernels.
pub async fn init_gpu(args: &Args) -> GpuContext {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .unwrap_or_else(|_| panic!("Failed to find a compatible GPU adapter"));

    let info = adapter.get_info();
    let limits = adapter.limits();
    eprintln!("GPU in use: [{:?}] {}", info.backend, info.name);

    // 1. Calibrate maximum batch size according to driver and device constraints
    let max_size = calibrate_batch_size(&limits, info.device_type, args.batch_size);
    eprintln!("Calibrated max_size for GPU: {max_size} tasks per slot");

    // 2. Request logical device with calibrated storage capacities
    let dirs_buffer_size = (max_size * BYTES_PER_TASK_DIRECTIONS) as u64;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("krait_device"),
            required_limits: Limits {
                max_storage_buffer_binding_size: dirs_buffer_size
                    .max(limits.max_storage_buffer_binding_size),
                max_buffer_size: dirs_buffer_size.max(limits.max_buffer_size),
                ..limits
            },
            ..Default::default()
        })
        .await
        .expect("Failed to create GPU device and command queue");

    // 3. Compile WGSL compute pipelines
    let dp_pipeline = create_compute_pipeline(
        &device,
        "dp_pipeline",
        &dp_shader_header(),
        include_str!("shaders/dp_kernel.wgsl"),
    );

    let tb_pipeline = create_compute_pipeline(
        &device,
        "traceback_pipeline",
        &traceback_shader_header(),
        include_str!("shaders/traceback_kernel.wgsl"),
    );

    let dp_layout = dp_pipeline.get_bind_group_layout(0);
    let tb_layout = tb_pipeline.get_bind_group_layout(0);

    // 4. Initialize double-buffered execution slots
    let slots = [
        GpuSlot::new(&device, max_size, &dp_layout, &tb_layout, 0),
        GpuSlot::new(&device, max_size, &dp_layout, &tb_layout, 1),
    ];

    GpuContext {
        gpu: Gpu {
            device,
            queue,
            dp_pipeline,
            tb_pipeline,
            slots,
        },
        max_size,
    }
}

/// Calculates the optimal GPU batch capacity based on hardware binding limits and device profile.
fn calibrate_batch_size(
    limits: &Limits,
    device_type: wgpu::DeviceType,
    user_batch_size: usize,
) -> usize {
    // Maximum tasks that can physically fit within the storage buffer binding limit
    let max_tasks_by_hardware = usize::try_from(limits.max_storage_buffer_binding_size).unwrap()
        / BYTES_PER_TASK_DIRECTIONS;

    // Software ceiling to prevent excessive GPU dispatch latency
    let software_ceiling = if user_batch_size > 0 {
        user_batch_size
    } else {
        match device_type {
            wgpu::DeviceType::IntegratedGpu => 32_768,
            _ => 65_536,
        }
    };

    let calibrated = max_tasks_by_hardware.min(software_ceiling);

    // Align down to warp size (32) for uniform thread-group dispatch
    (calibrated / WARP_SIZE) * WARP_SIZE
}

use crate::constants::{dp_shader_header, traceback_shader_header};

fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    header: &str,
    raw_shader_src: &str,
) -> wgpu::ComputePipeline {
    // Replace the IDE-mocked block with actual Rust constants
    let final_shader_src = inject_shader_constants(raw_shader_src, header);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(final_shader_src.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Replaces the marked template block in the WGSL source with auto-generated Rust constants.
/// If no tags are found, gracefully prepends the header to the top of the file.
pub fn inject_shader_constants(shader_src: &str, header: &str) -> String {
    const START_TAG: &str = "/* @constants:start */";
    const END_TAG: &str = "/* @constants:end */";

    if let (Some(start), Some(end)) = (shader_src.find(START_TAG), shader_src.find(END_TAG)) {
        let after_end = end + END_TAG.len();
        format!(
            "{}\n{}\n{}",
            &shader_src[..start],
            header,
            &shader_src[after_end..]
        )
    } else {
        format!("{header}\n{shader_src}")
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
    pub stride: u32,
}

/// Holds resources and bind groups for a single GPU execution slot.
pub struct GpuSlot {
    pub dp: DpSlot,
    pub tb: TbSlot,
}

impl GpuSlot {
    /// Creates a complete execution slot containing both DP and Traceback stages.
    pub fn new(
        device: &wgpu::Device,
        batch_size: usize,
        dp_layout: &wgpu::BindGroupLayout,
        tb_layout: &wgpu::BindGroupLayout,
        slot_id: usize,
    ) -> Self {
        let dp = DpSlot::new(device, batch_size, dp_layout, slot_id);
        let tb = TbSlot::new(
            device,
            batch_size,
            tb_layout,
            slot_id,
            &dp.buffer_scores,
            &dp.buffer_dirs,
            &dp.buffer_params,
        );

        Self { dp, tb }
    }
}

/// GPU resources and bind group for the Dynamic Programming (DP) pass.
pub struct DpSlot {
    pub buffer_a: wgpu::Buffer,
    pub buffer_b: wgpu::Buffer,
    pub buffer_mask_n_a: wgpu::Buffer,
    pub buffer_mask_n_b: wgpu::Buffer,
    pub buffer_scores: wgpu::Buffer,
    pub buffer_dirs: wgpu::Buffer,
    pub buffer_params: wgpu::Buffer,
    pub staging_scores: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl DpSlot {
    #[allow(clippy::too_many_lines)]
    pub fn new(
        device: &wgpu::Device,
        batch_size: usize,
        layout: &wgpu::BindGroupLayout,
        slot_id: usize,
    ) -> Self {
        let storage_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let storage_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let staging_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        let ref_size = (batch_size * REF_WORDS * 4) as wgpu::BufferAddress;
        let read_size = (batch_size * READ_WORDS * 4) as wgpu::BufferAddress;
        let ref_mask_size = (batch_size * REF_N_MASK_WORDS * 4) as wgpu::BufferAddress;
        let read_mask_size = (batch_size * READ_N_MASK_WORDS * 4) as wgpu::BufferAddress;
        let scores_size = (batch_size * SCORES_PER_READ * 4) as wgpu::BufferAddress;
        let dirs_size = (batch_size * READ_LEN * BAND_WORDS_4BIT * 4) as wgpu::BufferAddress;
        let params_size = size_of::<GpuParams>() as wgpu::BufferAddress;

        let buffer_a = create_buffer(
            device,
            &format!("slot_{slot_id}_seqs_a"),
            ref_size,
            storage_dst,
        );
        let buffer_b = create_buffer(
            device,
            &format!("slot_{slot_id}_seqs_b"),
            read_size,
            storage_dst,
        );
        let buffer_mask_n_a = create_buffer(
            device,
            &format!("slot_{slot_id}_mask_n_a"),
            ref_mask_size,
            storage_dst,
        );
        let buffer_mask_n_b = create_buffer(
            device,
            &format!("slot_{slot_id}_mask_n_b"),
            read_mask_size,
            storage_dst,
        );
        let buffer_scores = create_buffer(
            device,
            &format!("slot_{slot_id}_scores"),
            scores_size,
            storage_src,
        );
        let buffer_dirs = create_buffer(
            device,
            &format!("slot_{slot_id}_dirs"),
            dirs_size,
            wgpu::BufferUsages::STORAGE,
        );
        let staging_scores = create_buffer(
            device,
            &format!("slot_{slot_id}_staging_scores"),
            scores_size,
            staging_usage,
        );
        let buffer_params = create_buffer(
            device,
            &format!("slot_{slot_id}_params_uniform"),
            params_size,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("slot_{slot_id}_bg_dp")),
            layout,
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

        Self {
            buffer_a,
            buffer_b,
            buffer_mask_n_a,
            buffer_mask_n_b,
            buffer_scores,
            buffer_dirs,
            buffer_params,
            staging_scores,
            bind_group,
        }
    }
}

/// GPU resources and bind group for the Traceback pass.
pub struct TbSlot {
    pub buffer_cigar: wgpu::Buffer,
    pub buffer_group_info: wgpu::Buffer,
    pub staging_cigar: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl TbSlot {
    pub fn new(
        device: &wgpu::Device,
        batch_size: usize,
        layout: &wgpu::BindGroupLayout,
        slot_id: usize,
        scores: &wgpu::Buffer,
        dirs: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> Self {
        let cigar_size = (batch_size * CIGAR_CAPACITY * 4) as wgpu::BufferAddress;
        let group_info_size = (batch_size * 4) as wgpu::BufferAddress;

        let buffer_cigar = create_buffer(
            device,
            &format!("slot_{slot_id}_cigar"),
            cigar_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let buffer_group_info = create_buffer(
            device,
            &format!("slot_{slot_id}_group_info"),
            group_info_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let staging_cigar = create_buffer(
            device,
            &format!("slot_{slot_id}_staging_cigar"),
            cigar_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("slot_{slot_id}_bg_tb")),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scores.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dirs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_cigar.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffer_group_info.as_entire_binding(),
                },
            ],
        });

        Self {
            buffer_cigar,
            buffer_group_info,
            staging_cigar,
            bind_group,
        }
    }
}

/// Helper function to allocate an unmapped GPU buffer.
fn create_buffer(
    device: &wgpu::Device,
    label: &str,
    size: wgpu::BufferAddress,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        mapped_at_creation: false,
        usage,
    })
}
