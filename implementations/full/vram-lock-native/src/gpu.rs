#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
#[path = "../../../../shared/gpu-timing/wgpu.rs"]
mod timing;

use crate::graph;
use anyhow::{ensure, Result};
use bytemuck::{Pod, Zeroable};
#[cfg(not(target_arch = "wasm32"))]
use experiment_common::OutputFormat;
use std::num::NonZeroU64;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

#[allow(dead_code)]
pub type GpuEdgeInfo = crate::graph::EdgeInfo;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub iteration: u32,
    pub num_pairs: u32,
    pub chunk_offset: u32,
    pub chunk_size: u32,
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub adapter_name: String,
    pub backend: String,
    pub timestamp_supported: bool,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct GpuRunResult {
    pub initial_positions: Vec<[f64; 2]>,
    pub positions: Vec<[f32; 2]>,
    pub method_setup_time: Duration,
    pub upload_time: Duration,
    pub preparation_time: Duration,
    pub time_to_result: Duration,
    pub result_ready_monotonic_ms: f64,
    pub iteration_times: Option<Vec<Duration>>,
    pub iteration_time: Duration,
    pub readback_time: Duration,
    pub attempted_updates: u64,
    pub completed_updates: u64,
    pub dispatches: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuContext {
    pub fn new() -> Result<Self> {
        Self::with_shader(include_str!("shader.wgsl"))
    }

    pub fn with_shader(shader_source: &str) -> Result<Self> {
        let (instance, adapter, device, queue) =
            experiment_common::gpu::init_gpu_context(wgpu::PowerPreference::HighPerformance)?;

        let info = adapter.get_info();
        let adapter_name = info.name;
        let backend = format!("{:?}", info.backend);
        let timestamp_supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Atomic Full SGD Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Atomic Full SGD Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(
                            NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Atomic Full SGD Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Atomic Full SGD Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sgd"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(GpuContext {
            instance,
            adapter,
            device,
            queue,
            pipeline,
            bind_group_layout,
            adapter_name,
            backend,
            timestamp_supported,
        })
    }

    pub fn execute_sgd(
        &self,
        params: graph::SgdParams,
        chunk_size_param: Option<u32>,
        verbose: bool,
        output_format: OutputFormat,
    ) -> Result<GpuRunResult> {
        let execution_started = Instant::now();
        let method_setup_started = Instant::now();

        ensure!(
            !params.pairs.is_empty() && !params.etas.is_empty(),
            "Empty Full workload"
        );
        ensure!(
            params.pairs.len() <= u32::MAX as usize,
            "Too many constraints"
        );
        ensure!(
            params.positions.len() <= u32::MAX as usize / 2,
            "Too many vertices"
        );

        let initial_positions = params.positions.clone();
        let gpu_etas: Vec<f32> = params.etas.into_iter().map(|e| e as f32).collect();
        let gpu_positions: Vec<[f32; 2]> = params
            .positions
            .into_iter()
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect();
        let gpu_pairs = params.pairs;

        let node_size = gpu_positions.len();
        let num_iterations = gpu_etas.len();
        let num_pairs = gpu_pairs.len() as u32;

        const MAX_PAIRS_PER_BANK: usize = 8_388_608;
        let num_banks = (gpu_pairs.len()).div_ceil(MAX_PAIRS_PER_BANK);
        let chunk_size = chunk_size_param.unwrap_or(0);

        let mut bank_pair_counts = Vec::with_capacity(num_banks);
        for b in 0..num_banks {
            let start = b * MAX_PAIRS_PER_BANK;
            let end = (start + MAX_PAIRS_PER_BANK).min(gpu_pairs.len());
            bank_pair_counts.push((end - start) as u32);
        }

        let mut dispatches_per_iter = 0usize;
        for &b_pairs in &bank_pair_counts {
            let step = if chunk_size == 0 { b_pairs } else { chunk_size };
            dispatches_per_iter += (b_pairs as usize).div_ceil(step as usize);
        }
        let total_dispatches = num_iterations * dispatches_per_iter;

        let method_setup_time = method_setup_started.elapsed();

        let preparation_started = Instant::now();
        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as usize;
        let slot_size = std::mem::size_of::<Uniforms>().div_ceil(alignment) * alignment;
        let total_slots = 1 + total_dispatches;
        let mut uniform_bytes = vec![0u8; slot_size * total_slots];

        // Slot 0: warmup dummy uniform with chunk_size: 0 (immediate return in shader)
        let warmup_uniforms = Uniforms {
            iteration: 0,
            num_pairs: 0,
            chunk_offset: 0,
            chunk_size: 0,
        };
        uniform_bytes[..std::mem::size_of::<Uniforms>()]
            .copy_from_slice(bytemuck::bytes_of(&warmup_uniforms));

        let mut dispatch_slot = 1;
        for iteration in 0..num_iterations {
            for &b_pairs in &bank_pair_counts {
                let step = if chunk_size == 0 { b_pairs } else { chunk_size };
                let mut offset = 0u32;
                while offset < b_pairs {
                    let this_chunk = step.min(b_pairs - offset);
                    let u = Uniforms {
                        iteration: iteration as u32,
                        num_pairs: b_pairs,
                        chunk_offset: offset,
                        chunk_size: this_chunk,
                    };
                    let byte_offset = dispatch_slot * slot_size;
                    uniform_bytes[byte_offset..byte_offset + std::mem::size_of::<Uniforms>()]
                        .copy_from_slice(bytemuck::bytes_of(&u));
                    dispatch_slot += 1;
                    offset += this_chunk;
                }
            }
        }
        assert_eq!(dispatch_slot, total_slots);

        let uniforms_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dynamic Uniforms Buffer"),
                contents: &uniform_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let zero_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Zero Buffer"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let etas_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Etas Buffer"),
                contents: bytemuck::cast_slice(&gpu_etas),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let positions_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Positions Buffer"),
                contents: bytemuck::cast_slice(&gpu_positions),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let locks_zero = vec![0u32; node_size];
        let lock_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Locks Buffer"),
                contents: bytemuck::cast_slice(&locks_zero),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let updated_count_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Updated Count Buffer"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let count_download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Count Download Buffer"),
            size: ((num_iterations * 4) as u64).max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Positions Download Buffer"),
            size: (node_size * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut pair_bank_buffers = Vec::with_capacity(num_banks);
        let mut bind_groups = Vec::with_capacity(num_banks);

        for b in 0..num_banks {
            let start = b * MAX_PAIRS_PER_BANK;
            let end = (start + MAX_PAIRS_PER_BANK).min(gpu_pairs.len());
            let bank_slice = &gpu_pairs[start..end];

            let bank_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Pairs Bank Buffer {b}")),
                    contents: bytemuck::cast_slice(bank_slice),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Atomic Full SGD Bind Group Bank {b}")),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: etas_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: positions_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bank_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: lock_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: updated_count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &uniforms_buffer,
                            offset: 0,
                            size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                        }),
                    },
                ],
            });

            pair_bank_buffers.push(bank_buffer);
            bind_groups.push(bg);
        }

        // Warmup PSO dispatch to isolate Mesa NIR->DXIL compilation overhead
        // Uses slot 0 (chunk_size: 0) with dynamic offset 0, returning immediately
        experiment_common::gpu::run_warmup_dispatch(
            &self.device,
            &self.queue,
            &self.pipeline,
            &bind_groups[0],
            &[0],
        )?;

        let preparation_time = preparation_started.elapsed();
        let upload_time = Duration::ZERO;

        let total_passes = total_dispatches;
        let mut pass_timer = timing::PassTimer::new(&self.device, total_passes);

        let iteration_start = Instant::now();
        let mut dispatches = 0u64;
        let mut current_slot = 1;

        for iteration in 0..num_iterations {
            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("atomic_iteration_encoder"),
                    });

            encoder.copy_buffer_to_buffer(&zero_buffer, 0, &updated_count_buffer, 0, 4);

            for (bank_idx, &b_pairs) in bank_pair_counts.iter().enumerate() {
                let step = if chunk_size == 0 { b_pairs } else { chunk_size };
                let mut offset = 0u32;
                while offset < b_pairs {
                    let this_chunk = step.min(b_pairs - offset);
                    let height = this_chunk.div_ceil(65_535);
                    let width = this_chunk.div_ceil(height);

                    let dynamic_offset = (current_slot * slot_size) as u32;
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("atomic_chunk_pass"),
                            timestamp_writes: pass_timer.next(),
                        });
                        pass.set_pipeline(&self.pipeline);
                        pass.set_bind_group(0, &bind_groups[bank_idx], &[dynamic_offset]);
                        pass.dispatch_workgroups(width, height, 1);
                    }
                    dispatches += 1;
                    current_slot += 1;
                    offset += this_chunk;
                }
            }

            encoder.copy_buffer_to_buffer(
                &updated_count_buffer,
                0,
                &count_download_buffer,
                (iteration as u64) * 4,
                4,
            );

            self.queue.submit(std::iter::once(encoder.finish()));
            pass_timer.end_iteration();
        }
        assert_eq!(current_slot, total_slots);

        let iteration_time = iteration_start.elapsed();

        // Read back final positions and counts asynchronously
        let readback_started = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("positions_readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &positions_buffer,
            0,
            &download_buffer,
            0,
            (node_size * std::mem::size_of::<[f32; 2]>()) as u64,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let count_slice = count_download_buffer.slice(..);
        count_slice.map_async(wgpu::MapMode::Read, |_| {});

        let pos_slice = download_buffer.slice(..);
        pos_slice.map_async(wgpu::MapMode::Read, |_| {});

        self.device.poll(wgpu::PollType::wait_indefinitely())?;

        let mut completed_updates = 0u64;
        {
            let view = count_slice.get_mapped_range();
            let counts: &[u32] = bytemuck::cast_slice(&view);
            for (iter_idx, &count) in counts.iter().enumerate() {
                ensure!(
                    count == num_pairs,
                    "Incomplete Full iteration {}: completed {} of {} pairs (no successful result emitted)",
                    iter_idx + 1, count, num_pairs
                );
                completed_updates += u64::from(count);

                if verbose {
                    let message = format!(
                        "Iteration {} - Updated {} pairs",
                        iter_idx + 1,
                        count
                    );
                    match output_format {
                        OutputFormat::Json => eprintln!("{message}"),
                        OutputFormat::Human => println!("{message}"),
                    }
                }
            }
            drop(view);
            count_download_buffer.unmap();
        }

        let final_positions: Vec<[f32; 2]> = {
            let view = pos_slice.get_mapped_range();
            let coords: &[[f32; 2]] = bytemuck::cast_slice(&view);
            let res = coords.to_vec();
            drop(view);
            download_buffer.unmap();
            res
        };

        ensure!(
            final_positions.iter().flatten().all(|v| v.is_finite()),
            "Non-finite final coordinates"
        );

        let iteration_times = pass_timer.finish(&self.device, &self.queue)?;
        let readback_time = readback_started.elapsed();
        let time_to_result = execution_started.elapsed();
        let result_ready_monotonic_ms = experiment_common::monotonic_time_ms();

        Ok(GpuRunResult {
            initial_positions,
            positions: final_positions,
            method_setup_time,
            upload_time,
            preparation_time,
            time_to_result,
            result_ready_monotonic_ms,
            iteration_times,
            iteration_time,
            readback_time,
            attempted_updates: (num_pairs as u64) * (num_iterations as u64),
            completed_updates,
            dispatches,
        })
    }
}

#[allow(dead_code)]
pub struct AtomicFullStepper {
    n: u32,
    initial_positions_flat: Vec<f32>,
    pipeline: wgpu::ComputePipeline,
    etas_buffer: wgpu::Buffer,
    positions_buffer: wgpu::Buffer,
    download_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    lock_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    updated_count_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    uniforms_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    pair_bank_buffers: Vec<wgpu::Buffer>,
    bind_groups: Vec<wgpu::BindGroup>,
    bank_pair_counts: Vec<u32>,
    slot_size: usize,
}

#[allow(dead_code)]
impl AtomicFullStepper {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: u32,
        dist_u16: &[u16],
        initial_positions_flat: &[f32],
        seed: u64,
    ) -> Result<Self> {
        ensure!(n >= 2, "AtomicFullStepper requires n >= 2");
        ensure!(
            initial_positions_flat.len() == (n as usize) * 2,
            "initial_positions_flat length must be 2 * n"
        );
        let pairs = graph::build_pairs_from_dist_u16(n as usize, dist_u16, seed);
        ensure!(!pairs.is_empty(), "Empty Full workload");

        let max_binding_bytes = device.limits().max_storage_buffer_binding_size as usize;
        let max_pairs_per_bank = (max_binding_bytes / std::mem::size_of::<GpuEdgeInfo>())
            .min(8_388_608)
            .max(1);
        let num_banks = pairs.len().div_ceil(max_pairs_per_bank);

        let mut bank_pair_counts = Vec::with_capacity(num_banks);
        for b in 0..num_banks {
            let start = b * max_pairs_per_bank;
            let end = (start + max_pairs_per_bank).min(pairs.len());
            bank_pair_counts.push((end - start) as u32);
        }

        // Remove storageBarrier() inside non-uniform control flow for WebGPU Tint compliance.
        let shader_src = include_str!("shader.wgsl").replace("storageBarrier();", "");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Atomic Full SGD Stepper Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Atomic Full SGD Stepper Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(
                            NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Atomic Full SGD Stepper Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Atomic Full SGD Stepper Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sgd"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let alignment = (device.limits().min_uniform_buffer_offset_alignment as usize).max(256);
        let slot_size = std::mem::size_of::<Uniforms>().div_ceil(alignment) * alignment;
        let mut uniform_bytes = vec![0u8; slot_size * num_banks];
        for (b, &b_pairs) in bank_pair_counts.iter().enumerate() {
            let u = Uniforms {
                iteration: 0,
                num_pairs: b_pairs,
                chunk_offset: 0,
                chunk_size: b_pairs,
            };
            let byte_offset = b * slot_size;
            uniform_bytes[byte_offset..byte_offset + std::mem::size_of::<Uniforms>()]
                .copy_from_slice(bytemuck::bytes_of(&u));
        }

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Atomic Full Stepper Uniforms Buffer"),
            contents: &uniform_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let etas_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Atomic Full Stepper Etas Buffer"),
            contents: bytemuck::bytes_of(&0.0f32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let positions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Atomic Full Stepper Positions Buffer"),
            contents: bytemuck::cast_slice(initial_positions_flat),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Atomic Full Stepper Download Buffer"),
            size: (initial_positions_flat.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let locks_zero = vec![0u32; n as usize];
        let lock_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Atomic Full Stepper Locks Buffer"),
            contents: bytemuck::cast_slice(&locks_zero),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let updated_count_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Atomic Full Stepper Updated Count Buffer"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mut pair_bank_buffers = Vec::with_capacity(num_banks);
        let mut bind_groups = Vec::with_capacity(num_banks);

        for b in 0..num_banks {
            let start = b * max_pairs_per_bank;
            let end = (start + max_pairs_per_bank).min(pairs.len());
            let bank_slice = &pairs[start..end];

            let bank_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Atomic Full Stepper Pairs Bank Buffer {b}")),
                contents: bytemuck::cast_slice(bank_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Atomic Full Stepper Bind Group Bank {b}")),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: etas_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: positions_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bank_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: lock_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: updated_count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &uniforms_buffer,
                            offset: 0,
                            size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                        }),
                    },
                ],
            });

            pair_bank_buffers.push(bank_buffer);
            bind_groups.push(bg);
        }

        let _ = queue;

        Ok(Self {
            n,
            initial_positions_flat: initial_positions_flat.to_vec(),
            pipeline,
            etas_buffer,
            positions_buffer,
            download_buffer,
            lock_buffer,
            updated_count_buffer,
            uniforms_buffer,
            pair_bank_buffers,
            bind_groups,
            bank_pair_counts,
            slot_size,
        })
    }

    pub fn step_epoch(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, eta: f32) {
        queue.write_buffer(&self.etas_buffer, 0, bytemuck::bytes_of(&eta));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("atomic_full_stepper_epoch"),
        });

        for (bank_idx, &b_pairs) in self.bank_pair_counts.iter().enumerate() {
            let height = b_pairs.div_ceil(65_535);
            let width = b_pairs.div_ceil(height);
            let dynamic_offset = (bank_idx * self.slot_size) as u32;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atomic_full_stepper_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_groups[bank_idx], &[dynamic_offset]);
            pass.dispatch_workgroups(width, height, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.positions_buffer,
            0,
            &self.download_buffer,
            0,
            (self.n as u64) * 8,
        );
        queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn reset(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.positions_buffer,
            0,
            bytemuck::cast_slice(&self.initial_positions_flat),
        );
    }

    pub fn download_buffer(&self) -> &wgpu::Buffer {
        &self.download_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> graph::SgdParams {
        graph::SgdParams {
            positions: vec![[0.0, 0.0], [4.0, 0.0]],
            pairs: vec![graph::EdgeInfo {
                u: 0,
                v: 1,
                dij: 2.0,
                wij: 0.25,
            }],
            etas: vec![4.0],
            center: false,
        }
    }

    #[test]
    fn completed_gpu_update_matches_known_solution() {
        let context = GpuContext::new().unwrap();
        let result = context
            .execute_sgd(params(), None, false, OutputFormat::Json)
            .unwrap();
        assert_eq!(result.completed_updates, result.attempted_updates);
        assert_eq!(result.positions, vec![[1.0, 0.0], [3.0, 0.0]]);
    }

    #[test]
    fn missing_gpu_updates_are_an_error() {
        let shader = include_str!("shader.wgsl")
            .replace("atomicAdd(&updated_count, 1u);", "/* update skipped */");
        assert_ne!(shader, include_str!("shader.wgsl"));
        let context = GpuContext::with_shader(&shader).unwrap();
        let error = context
            .execute_sgd(params(), None, false, OutputFormat::Json)
            .unwrap_err();
        assert!(
            error.to_string().contains("completed 0 of 1 pairs"),
            "{error}"
        );
    }

    #[test]
    fn test_chunked_dispatch_matches_single_dispatch() {
        let context = GpuContext::new().unwrap();
        let test_params = graph::SgdParams {
            positions: vec![[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]],
            pairs: vec![
                graph::EdgeInfo {
                    u: 0,
                    v: 1,
                    dij: 4.0,
                    wij: 0.0625,
                },
                graph::EdgeInfo {
                    u: 0,
                    v: 2,
                    dij: 3.0,
                    wij: 0.1111,
                },
                graph::EdgeInfo {
                    u: 1,
                    v: 2,
                    dij: 5.0,
                    wij: 0.0400,
                },
            ],
            etas: vec![2.0, 1.0],
            center: false,
        };
        let res_chunked = context
            .execute_sgd(test_params, Some(1), false, OutputFormat::Json)
            .unwrap();
        assert_eq!(res_chunked.completed_updates, 6);
        assert_eq!(res_chunked.completed_updates, res_chunked.attempted_updates);
        assert_eq!(res_chunked.dispatches, 6);
    }

    #[test]
    fn test_atomic_full_stepper_matches_known_solution_and_resets() {
        let context = GpuContext::new().unwrap();
        let dist_u16 = vec![0u16, 2u16, 2u16, 0u16];
        let initial_flat = vec![0.0f32, 0.0, 4.0, 0.0];
        let mut stepper = AtomicFullStepper::new(
            &context.device,
            &context.queue,
            2,
            &dist_u16,
            &initial_flat,
            0,
        )
        .unwrap();
        stepper.step_epoch(&context.device, &context.queue, 4.0);
        let slice = stepper.download_buffer().slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        context.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        {
            let view = slice.get_mapped_range();
            let coords: &[f32] = bytemuck::cast_slice(&view);
            assert_eq!(coords, &[1.0, 0.0, 3.0, 0.0]);
        }
        stepper.download_buffer().unmap();
        stepper.reset(&context.queue);

        let coincident_flat = vec![2.0f32, 3.0, 2.0, 3.0];
        let mut coincident_stepper = AtomicFullStepper::new(
            &context.device,
            &context.queue,
            2,
            &dist_u16,
            &coincident_flat,
            0,
        )
        .unwrap();
        coincident_stepper.step_epoch(&context.device, &context.queue, 4.0);
        let c_slice = coincident_stepper.download_buffer().slice(..);
        c_slice.map_async(wgpu::MapMode::Read, |_| {});
        context.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        {
            let view = c_slice.get_mapped_range();
            let coords: &[f32] = bytemuck::cast_slice(&view);
            assert!(coords.iter().all(|v| v.is_finite()));
            assert!((coords[0] - coords[2]).abs() > 0.1, "Coincident vertices did not separate: {coords:?}");
        }
        coincident_stepper.download_buffer().unmap();
    }
}
