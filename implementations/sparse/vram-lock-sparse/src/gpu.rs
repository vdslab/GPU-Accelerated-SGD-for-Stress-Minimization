#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
#[path = "../../../../shared/gpu-timing/wgpu.rs"]
mod timing;

#[cfg(not(target_arch = "wasm32"))]
use crate::graph::{center_inplace, SgdParams};
use anyhow::{ensure, Result};
use bytemuck::{Pod, Zeroable};
#[cfg(not(target_arch = "wasm32"))]
use experiment_common::OutputFormat;
#[cfg(not(target_arch = "wasm32"))]
use rand::{seq::SliceRandom, Rng};
use std::mem;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const MAX_PAIRS_PER_BANK: usize = 6_710_880; // 127.9999 MiB (20 B/pair, within 128 MiB max_storage_buffer_binding_size)
const DYNAMIC_UNIFORM_STRIDE: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuPair {
    pub u: u32,
    pub v: u32,
    pub dij: f32,
    pub weight_u: f32,
    pub weight_v: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Uniforms {
    chunk_start: u32,
    chunk_count: u32,
    epoch: u32,
    eta: f32,
    pair_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    pub verify_pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub device_name: String,
    pub backend: String,
}

#[cfg(not(target_arch = "wasm32"))]
pub struct GpuRunResult {
    pub initial_positions: Vec<[f64; 2]>,
    pub positions: Vec<[f64; 2]>,
    pub method_setup_time: Duration,
    pub upload_time: Duration,
    pub preparation_time: Duration,
    pub time_to_result: Duration,
    pub result_ready_monotonic_ms: f64,
    pub iteration_times: Option<Vec<Duration>>,
    pub iteration_time: Duration,
    pub readback_time: Duration,
    pub postprocess_time: Duration,
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
        let (_instance, adapter, device, queue) =
            experiment_common::gpu::init_gpu_context(wgpu::PowerPreference::HighPerformance)?;
        let info = adapter.get_info();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sparse Atomic SGD Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sparse Atomic bind group layout"),
            entries: &[
                // binding 0: positions
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: pairs
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: locks
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: pending
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: completed
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
                // binding 5: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<Uniforms>() as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
                // binding 6: incomplete_counts
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sparse Atomic pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sparse Atomic compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sparse_sgd"),
            compilation_options: Default::default(),
            cache: None,
        });

        let verify_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sparse Atomic verify_epoch compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("verify_epoch"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            verify_pipeline,
            bind_group_layout,
            device_name: info.name,
            backend: format!("{:?}", info.backend),
        })
    }

    pub fn execute_sgd<R: Rng + ?Sized>(
        &self,
        params: SgdParams,
        rng: &mut R,
        verbose: bool,
        format: OutputFormat,
    ) -> Result<GpuRunResult> {
        self.execute_sgd_with_cpu_shuffle(params, rng, verbose, format, false)
    }

    pub fn execute_sgd_with_cpu_shuffle<R: Rng + ?Sized>(
        &self,
        params: SgdParams,
        rng: &mut R,
        verbose: bool,
        format: OutputFormat,
        cpu_shuffle: bool,
    ) -> Result<GpuRunResult> {
        let initial_positions = params.positions;
        let pairs: Vec<GpuPair> = params
            .pairs
            .iter()
            .map(|p| {
                ensure!(
                    p.u < p.v && p.v < initial_positions.len(),
                    "Invalid Sparse endpoints"
                );
                ensure!(
                    (p.dij as f32).is_finite() && p.dij as f32 > 0.0,
                    "Invalid target distance"
                );
                Ok(GpuPair {
                    u: p.u as u32,
                    v: p.v as u32,
                    dij: p.dij as f32,
                    weight_u: p.weight_u as f32,
                    weight_v: p.weight_v as f32,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.execute_sgd_prepared(
            pairs,
            initial_positions,
            params.etas,
            params.center,
            rng,
            verbose,
            format,
            cpu_shuffle,
        )
    }

    pub fn execute_sgd_prepared<R: Rng + ?Sized>(
        &self,
        mut pairs: Vec<GpuPair>,
        initial_positions: Vec<[f64; 2]>,
        etas_f64: Vec<f64>,
        center: bool,
        rng: &mut R,
        verbose: bool,
        format: OutputFormat,
        cpu_shuffle: bool,
    ) -> Result<GpuRunResult> {
        let execution_started = Instant::now();
        let setup = Instant::now();
        ensure!(
            !pairs.is_empty() && !etas_f64.is_empty(),
            "Empty Sparse workload"
        );
        ensure!(
            initial_positions.len() <= (u32::MAX as usize / 2),
            "Too many vertices"
        );
        ensure!(
            pairs.len() <= u32::MAX as usize,
            "Too many constraints"
        );
        ensure!(
            initial_positions
                .iter()
                .flatten()
                .all(|x| (*x as f32).is_finite()),
            "Invalid initial coordinates"
        );

        if !cpu_shuffle {
            pairs.shuffle(rng);
        }

        let etas: Vec<f32> = etas_f64.iter().map(|&e| e as f32).collect();
        let initial_coords: Vec<[f32; 2]> = initial_positions
            .iter()
            .map(|&[x, y]| [x as f32, y as f32])
            .collect();
        let total_pairs = pairs.len();
        let num_banks = total_pairs.div_ceil(MAX_PAIRS_PER_BANK);
        let num_iterations = etas.len();
        let method_setup_time = setup.elapsed();

        let preparation_started = Instant::now();
        let position_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Positions buffer"),
                contents: bytemuck::cast_slice(&initial_coords),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let pair_bank_buffers: Vec<wgpu::Buffer> = (0..num_banks)
            .map(|b| {
                let start = b * MAX_PAIRS_PER_BANK;
                let end = (start + MAX_PAIRS_PER_BANK).min(total_pairs);
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Pairs bank buffer {}", b)),
                    contents: bytemuck::cast_slice(&pairs[start..end]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })
            })
            .collect();

        let locks = vec![0u32; initial_positions.len()];
        let lock_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Locks buffer"),
                contents: bytemuck::cast_slice(&locks),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let mut bank_pendings: Vec<Vec<u32>> = (0..num_banks)
            .map(|b| {
                let start = b * MAX_PAIRS_PER_BANK;
                let end = (start + MAX_PAIRS_PER_BANK).min(total_pairs);
                (0..(end - start) as u32).collect()
            })
            .collect();

        let pending_bank_buffers: Vec<wgpu::Buffer> = bank_pendings
            .iter()
            .enumerate()
            .map(|(b, pending)| {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Pending bank buffer {}", b)),
                    contents: bytemuck::cast_slice(pending),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })
            })
            .collect();

        let completed_bank_buffers: Vec<wgpu::Buffer> = (0..num_banks)
            .map(|b| {
                let start = b * MAX_PAIRS_PER_BANK;
                let end = (start + MAX_PAIRS_PER_BANK).min(total_pairs);
                let count = end - start;
                let initial_completed = vec![0u32; count];
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Completed bank buffer {}", b)),
                    contents: bytemuck::cast_slice(&initial_completed),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                })
            })
            .collect();

        let initial_incomplete_counts = vec![0u32; num_iterations];
        let incomplete_counts_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Incomplete counts buffer"),
                contents: bytemuck::cast_slice(&initial_incomplete_counts),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let total_uniform_slots = 1 + num_iterations * num_banks;
        let mut uniform_bytes = vec![0u8; total_uniform_slots * DYNAMIC_UNIFORM_STRIDE];

        // Slot 0: warmup dummy uniform with chunk_count: 0, pair_count: 0 (immediate return in shader)
        let warmup_uniforms = Uniforms {
            chunk_start: 0,
            chunk_count: 0,
            epoch: 0,
            eta: 0.0,
            pair_count: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        uniform_bytes[..mem::size_of::<Uniforms>()]
            .copy_from_slice(bytemuck::bytes_of(&warmup_uniforms));

        for iteration in 0..num_iterations {
            let epoch = (iteration + 1) as u32;
            let eta = etas[iteration];
            for bank in 0..num_banks {
                let start = bank * MAX_PAIRS_PER_BANK;
                let end = (start + MAX_PAIRS_PER_BANK).min(total_pairs);
                let bank_pairs = (end - start) as u32;
                let uniforms = Uniforms {
                    chunk_start: 0,
                    chunk_count: bank_pairs,
                    epoch,
                    eta,
                    pair_count: bank_pairs,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                let slot = 1 + iteration * num_banks + bank;
                let byte_offset = slot * DYNAMIC_UNIFORM_STRIDE;
                let slice = bytemuck::bytes_of(&uniforms);
                uniform_bytes[byte_offset..byte_offset + slice.len()].copy_from_slice(slice);
            }
        }

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dynamic Uniform buffer"),
                contents: &uniform_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_groups: Vec<wgpu::BindGroup> = (0..num_banks)
            .map(|b| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Sparse Atomic bind group bank {}", b)),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: position_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: pair_bank_buffers[b].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: lock_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pending_bank_buffers[b].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: completed_bank_buffers[b].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &uniform_buffer,
                                offset: 0,
                                size: Some(
                                    std::num::NonZeroU64::new(mem::size_of::<Uniforms>() as u64)
                                        .unwrap(),
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: incomplete_counts_buffer.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        // Warm up Mesa NIR->DXIL compilation
        experiment_common::gpu::run_warmup_dispatch(
            &self.device,
            &self.queue,
            &self.pipeline,
            &bind_groups[0],
            &[0],
        )?;
        experiment_common::gpu::run_warmup_dispatch(
            &self.device,
            &self.queue,
            &self.verify_pipeline,
            &bind_groups[0],
            &[0],
        )?;

        let preparation_time = preparation_started.elapsed();
        let upload_time = Duration::ZERO; // Staging included in init

        let total_passes = num_iterations * num_banks;
        let mut pass_timer = timing::PassTimer::new(&self.device, total_passes);

        let mut completed_updates = 0u64;
        let mut dispatches = 0u64;
        let start = Instant::now();

        for iteration in 0..num_iterations {
            if cpu_shuffle {
                for (b, pending) in bank_pendings.iter_mut().enumerate() {
                    pending.shuffle(rng);
                    self.queue.write_buffer(
                        &pending_bank_buffers[b],
                        0,
                        bytemuck::cast_slice(pending),
                    );
                }
            }

            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!("Sparse Atomic iteration {} encoder", iteration + 1)),
                    });

            for bank in 0..num_banks {
                let bank_start = bank * MAX_PAIRS_PER_BANK;
                let bank_end = (bank_start + MAX_PAIRS_PER_BANK).min(total_pairs);
                let bank_pairs = (bank_end - bank_start) as u32;

                let slot = 1 + iteration * num_banks + bank;
                let dynamic_offset = (slot * DYNAMIC_UNIFORM_STRIDE) as u32;

                // 1. SGD dispatch for bank
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Sparse Atomic SGD pass b{}", bank)),
                        timestamp_writes: pass_timer.next(),
                    });
                    cpass.set_pipeline(&self.pipeline);
                    cpass.set_bind_group(0, &bind_groups[bank], &[dynamic_offset]);
                    let height = bank_pairs.div_ceil(65_535);
                    let width = bank_pairs.div_ceil(height);
                    cpass.dispatch_workgroups(width, height, 1);
                }
                dispatches += 1;

                // 2. verify_epoch dispatch for bank
                {
                    let mut vpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Sparse verify epoch pass b{}", bank)),
                        timestamp_writes: None,
                    });
                    vpass.set_pipeline(&self.verify_pipeline);
                    vpass.set_bind_group(0, &bind_groups[bank], &[dynamic_offset]);
                    let verify_workgroups = bank_pairs.div_ceil(256);
                    vpass.dispatch_workgroups(verify_workgroups, 1, 1);
                }
            }

            // Submit per-iteration command buffer (anti-freeze / WDDM TDR compliance)
            // No device.poll here! Everything continues asynchronously!
            self.queue.submit(Some(encoder.finish()));
            completed_updates += total_pairs as u64;

            pass_timer.end_iteration();

            if verbose {
                let message = format!(
                    "Iteration {} - Updated {} pairs, dispatches={}",
                    iteration + 1,
                    total_pairs,
                    num_banks
                );
                match format {
                    OutputFormat::Json => eprintln!("{message}"),
                    OutputFormat::Human => println!("{message}"),
                }
            }
        }

        let iteration_time = start.elapsed();

        // Read back incomplete_counts and final positions in a single device.poll
        let readback = Instant::now();
        let incomplete_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Incomplete counts staging buffer"),
            size: (num_iterations * mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let position_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position staging buffer"),
            size: (initial_positions.len() * 2 * mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &incomplete_counts_buffer,
            0,
            &incomplete_staging,
            0,
            (num_iterations * mem::size_of::<u32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &position_buffer,
            0,
            &position_staging,
            0,
            (initial_positions.len() * 2 * mem::size_of::<f32>()) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        let inc_slice = incomplete_staging.slice(..);
        inc_slice.map_async(wgpu::MapMode::Read, |_| ());

        let pos_slice = position_staging.slice(..);
        pos_slice.map_async(wgpu::MapMode::Read, |_| ());

        self.device.poll(wgpu::PollType::wait_indefinitely())?;

        // 1. Check completion verification
        {
            let inc_data = inc_slice.get_mapped_range();
            let counts: &[u32] = bytemuck::cast_slice(&inc_data);
            for (t, &dropped) in counts.iter().enumerate() {
                let epoch = (t + 1) as u32;
                ensure!(
                    dropped == 0,
                    "Incomplete Sparse iteration {} (Quality Gate 4 violation: update dropped)",
                    epoch
                );
            }
        }
        incomplete_staging.unmap();

        // 2. Extract final positions
        let mut final_positions: Vec<[f64; 2]> = {
            let data = pos_slice.get_mapped_range();
            let coords: &[[f32; 2]] = bytemuck::cast_slice(&data);
            coords.iter().map(|&[x, y]| [x as f64, y as f64]).collect()
        };
        position_staging.unmap();
        let readback_time = readback.elapsed();

        let iteration_times = pass_timer.finish(&self.device, &self.queue)?;
        let time_to_result = execution_started.elapsed();
        let result_ready_monotonic_ms = experiment_common::monotonic_time_ms();

        let postprocess = Instant::now();
        if center {
            center_inplace(&mut final_positions);
        }
        let postprocess_time = postprocess.elapsed();

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
            postprocess_time,
            attempted_updates: total_pairs as u64 * etas.len() as u64,
            completed_updates,
            dispatches,
        })
    }
}

const STEPPER_SHADER_SRC: &str = r#"
struct Pair {
    u: u32,
    v: u32,
    dij: f32,
    weight_u: f32,
    weight_v: f32,
};

struct Uniforms {
    chunk_start: u32,
    chunk_count: u32,
    epoch: u32,
    eta: f32,
    pair_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> locks: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read_write> updated_count: atomic<u32>;

fn acquire_pair_locks(u: u32, v: u32) -> bool {
    let first = min(u, v);
    let second = max(u, v);
    if (atomicExchange(&locks[first], 1u) != 0u) {
        return false;
    }
    if (atomicExchange(&locks[second], 1u) == 0u) {
        return true;
    }
    atomicExchange(&locks[first], 0u);
    return false;
}

fn release_pair_locks(u: u32, v: u32) {
    let first = min(u, v);
    let second = max(u, v);
    atomicExchange(&locks[second], 0u);
    atomicExchange(&locks[first], 0u);
}

fn update_pair(p: Pair) {
    let u = p.u;
    let v = p.v;

    // Read positions atomically via atomicAdd(0u) ensuring L2 cache coherence across workgroups
    let u_x = atomicAdd(&positions[2u * u], 0u);
    let u_y = atomicAdd(&positions[2u * u + 1u], 0u);
    let pos_u = vec2<f32>(bitcast<f32>(u_x), bitcast<f32>(u_y));

    let v_x = atomicAdd(&positions[2u * v], 0u);
    let v_y = atomicAdd(&positions[2u * v + 1u], 0u);
    let pos_v = vec2<f32>(bitcast<f32>(v_x), bitcast<f32>(v_y));

    var diff = pos_v - pos_u;
    var norm = length(diff);

    if (norm < 1e-12) {
        // Deterministic tiny fallback matching native Atomic SGD
        diff = vec2<f32>(1e-6, 1e-6);
        norm = length(diff);
    }

    let r = ((norm - p.dij) / (2.0 * norm)) * diff;
    let du = min(uniforms.eta * p.weight_u, 1.0) * r;
    let dv = min(uniforms.eta * p.weight_v, 1.0) * r;

    let new_pos_u = pos_u + du;
    let new_pos_v = pos_v - dv;

    // Write back updated coordinates atomically ensuring L2 cache coherence across workgroups
    atomicExchange(&positions[2u * u], bitcast<u32>(new_pos_u.x));
    atomicExchange(&positions[2u * u + 1u], bitcast<u32>(new_pos_u.y));
    atomicExchange(&positions[2u * v], bitcast<u32>(new_pos_v.x));
    atomicExchange(&positions[2u * v + 1u], bitcast<u32>(new_pos_v.y));
}

@compute @workgroup_size(1, 1, 1)
fn sparse_sgd(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>,
) {
    let linear_idx = group_id.y * num_wgs.x + group_id.x;
    if (linear_idx >= uniforms.chunk_count) {
        return;
    }
    let slot = uniforms.chunk_start + linear_idx;
    if (slot >= uniforms.pair_count) {
        return;
    }

    let p = pairs[slot];
    if (p.u == p.v) {
        atomicAdd(&updated_count, 1u);
        return;
    }

    loop {
        if (acquire_pair_locks(p.u, p.v)) {
            update_pair(p);
            release_pair_locks(p.u, p.v);
            atomicAdd(&updated_count, 1u);
            break;
        }
    }
}
"#;

pub struct AtomicSparseStepper {
    n: usize,
    initial_positions_flat: Vec<f32>,
    pipeline: wgpu::ComputePipeline,
    position_buffer: wgpu::Buffer,
    download_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    lock_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    updated_count_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    pair_bank_buffers: Vec<wgpu::Buffer>,
    bind_groups: Vec<wgpu::BindGroup>,
    bank_pair_counts: Vec<u32>,
    slot_size: usize,
    epoch: u32,
}

impl AtomicSparseStepper {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: usize,
        pairs: &[GpuPair],
        initial_positions_flat: &[f32],
    ) -> Result<Self> {
        ensure!(n >= 2, "AtomicSparseStepper requires n >= 2");
        ensure!(!pairs.is_empty(), "Empty Sparse workload");
        ensure!(
            initial_positions_flat.len() == n * 2,
            "initial_positions_flat length must be 2 * n"
        );

        let max_binding_bytes = device.limits().max_storage_buffer_binding_size as usize;
        let max_pairs_per_bank = (max_binding_bytes / mem::size_of::<GpuPair>())
            .min(MAX_PAIRS_PER_BANK)
            .max(1);
        let total_pairs = pairs.len();
        let num_banks = total_pairs.div_ceil(max_pairs_per_bank);

        let mut bank_pair_counts = Vec::with_capacity(num_banks);
        for b in 0..num_banks {
            let start = b * max_pairs_per_bank;
            let end = (start + max_pairs_per_bank).min(total_pairs);
            bank_pair_counts.push((end - start) as u32);
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sparse Atomic Stepper Shader"),
            source: wgpu::ShaderSource::Wgsl(STEPPER_SHADER_SRC.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sparse Atomic Stepper bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<Uniforms>() as u64).unwrap(),
                        ),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sparse Atomic Stepper pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sparse Atomic Stepper compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sparse_sgd"),
            compilation_options: Default::default(),
            cache: None,
        });

        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sparse Atomic Stepper Positions buffer"),
            contents: bytemuck::cast_slice(initial_positions_flat),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Atomic Stepper Download buffer"),
            size: (initial_positions_flat.len() * mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let pair_bank_buffers: Vec<wgpu::Buffer> = (0..num_banks)
            .map(|b| {
                let start = b * max_pairs_per_bank;
                let end = (start + max_pairs_per_bank).min(total_pairs);
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Sparse Atomic Stepper Pairs bank buffer {}", b)),
                    contents: bytemuck::cast_slice(&pairs[start..end]),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();

        let locks = vec![0u32; n];
        let lock_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sparse Atomic Stepper Locks buffer"),
            contents: bytemuck::cast_slice(&locks),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let updated_count_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sparse Atomic Stepper Updated Count buffer"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let alignment =
            (device.limits().min_uniform_buffer_offset_alignment as usize).max(DYNAMIC_UNIFORM_STRIDE);
        let slot_size = mem::size_of::<Uniforms>().div_ceil(alignment) * alignment;
        let uniform_bytes = vec![0u8; num_banks * slot_size];
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sparse Atomic Stepper Uniform buffer"),
            contents: &uniform_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_groups: Vec<wgpu::BindGroup> = (0..num_banks)
            .map(|b| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Sparse Atomic Stepper bind group bank {}", b)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: position_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: pair_bank_buffers[b].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: lock_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &uniform_buffer,
                                offset: 0,
                                size: Some(
                                    std::num::NonZeroU64::new(mem::size_of::<Uniforms>() as u64)
                                        .unwrap(),
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: updated_count_buffer.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        let _ = queue;

        Ok(Self {
            n,
            initial_positions_flat: initial_positions_flat.to_vec(),
            pipeline,
            position_buffer,
            download_buffer,
            lock_buffer,
            updated_count_buffer,
            uniform_buffer,
            pair_bank_buffers,
            bind_groups,
            bank_pair_counts,
            slot_size,
            epoch: 0,
        })
    }

    pub fn step_epoch(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, eta: f32) {
        self.epoch = self.epoch.wrapping_add(1);
        let num_banks = self.bank_pair_counts.len();
        let mut uniform_bytes = vec![0u8; num_banks * self.slot_size];
        for (bank, &bank_pairs) in self.bank_pair_counts.iter().enumerate() {
            let uniforms = Uniforms {
                chunk_start: 0,
                chunk_count: bank_pairs,
                epoch: self.epoch,
                eta,
                pair_count: bank_pairs,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let byte_offset = bank * self.slot_size;
            let slice = bytemuck::bytes_of(&uniforms);
            uniform_bytes[byte_offset..byte_offset + slice.len()].copy_from_slice(slice);
        }
        queue.write_buffer(&self.uniform_buffer, 0, &uniform_bytes);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("atomic_sparse_stepper_epoch"),
        });

        for (bank, &bank_pairs) in self.bank_pair_counts.iter().enumerate() {
            let dynamic_offset = (bank * self.slot_size) as u32;
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atomic_sparse_stepper_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_groups[bank], &[dynamic_offset]);
            let height = bank_pairs.div_ceil(65_535);
            let width = bank_pairs.div_ceil(height);
            cpass.dispatch_workgroups(width, height, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.position_buffer,
            0,
            &self.download_buffer,
            0,
            (self.n as u64) * 8,
        );
        queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn reset(&mut self, queue: &wgpu::Queue) {
        self.epoch = 0;
        queue.write_buffer(
            &self.position_buffer,
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
    use crate::{cpu_reference::apply_constraint, graph::EdgeInfo};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn stale_completion_from_previous_iteration_is_rejected() {
        let source = include_str!("shader.wgsl").replace(
            "completed[slot] = uniforms.epoch;",
            "if (uniforms.epoch == 1u) { completed[slot] = uniforms.epoch; }",
        );
        assert_ne!(source, include_str!("shader.wgsl"));
        let context = GpuContext::with_shader(&source).unwrap();
        let result = context.execute_sgd(
            SgdParams {
                positions: vec![[0.0, 0.0], [4.0, 0.0]],
                pairs: vec![EdgeInfo {
                    u: 0,
                    v: 1,
                    dij: 2.0,
                    weight_u: 0.25,
                    weight_v: 0.25,
                }],
                etas: vec![4.0, 2.0],
                pivots: vec![0],
                center: false,
            },
            &mut StdRng::seed_from_u64(0),
            false,
            OutputFormat::Json,
        );
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("Incomplete Sparse iteration 2"));
    }

    #[test]
    fn moving_shared_endpoint_tracks_serial_recurrence() {
        let context = GpuContext::new().unwrap();
        let count = 4096;
        let eta = 0.0001;
        let mut expected = 0.0f32;
        for _ in 0..count {
            let diff = 3.0 - expected;
            let r = ((diff - 1.0) / (2.0 * diff)) * diff;
            expected += eta as f32 * r;
        }

        for seed in [0, 1, 42] {
            let mut positions = vec![[0.0, 0.0]];
            positions.extend((0..count).map(|_| [3.0, 0.0]));
            let pairs: Vec<_> = (1..=count)
                .map(|v| EdgeInfo {
                    u: 0,
                    v,
                    dij: 1.0,
                    weight_u: 1.0,
                    weight_v: 0.0,
                })
                .collect();
            let r = context
                .execute_sgd_with_cpu_shuffle(
                    SgdParams {
                        positions,
                        pairs,
                        etas: vec![eta],
                        pivots: vec![0],
                        center: false,
                    },
                    &mut StdRng::seed_from_u64(seed),
                    false,
                    OutputFormat::Json,
                    false,
                )
                .unwrap();
            assert_eq!(r.completed_updates, count as u64);
            assert!(
                (r.positions[0][0] - expected as f64).abs() < 0.0001,
                "{} != {}",
                r.positions[0][0],
                expected
            );
            assert_eq!(r.positions[0][1], 0.0);
            assert!(r.positions[1..].iter().all(|&p| p == [3.0, 0.0]));
        }
    }

    #[test]
    fn gpu_directional_updates_match_cpu_reference() {
        let context = GpuContext::new().unwrap();
        for (wu, wv, eta) in [
            (1.0, 1.0, 0.25),
            (0.2, 0.8, 0.5),
            (0.0, 1.0, 0.5),
            (1.0, 0.0, 0.5),
            (10.0, 0.2, 0.5),
        ] {
            let pair = EdgeInfo {
                u: 0,
                v: 1,
                dij: 2.0,
                weight_u: wu,
                weight_v: wv,
            };
            let initial = vec![[0.0, 0.0], [4.0, 1.0]];
            let mut expected = initial.clone();
            apply_constraint(
                &mut expected,
                pair,
                eta,
                &mut StdRng::seed_from_u64(0),
            );
            let run = context
                .execute_sgd(
                    SgdParams {
                        positions: initial,
                        pairs: vec![pair],
                        etas: vec![eta],
                        pivots: vec![0],
                        center: false,
                    },
                    &mut StdRng::seed_from_u64(0),
                    false,
                    OutputFormat::Json,
                )
                .unwrap();
            for (&actual, &expected) in run
                .positions
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
            {
                assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
            }
            assert_eq!(run.completed_updates, 1);
        }
    }

    #[test]
    fn contention_completes_each_pair_once_and_preserves_fixed_endpoint() {
        let context = GpuContext::new().unwrap();
        let n = 4097;
        let run = context
            .execute_sgd(
                SgdParams {
                    positions: (0..n).map(|i| [i as f64 + 1.0, 0.0]).collect(),
                    pairs: (1..n)
                        .map(|v| EdgeInfo {
                            u: 0,
                            v,
                            dij: 1.0,
                            weight_u: 0.0,
                            weight_v: 1.0,
                        })
                        .collect(),
                    etas: vec![0.5, 0.5, 0.5],
                    pivots: vec![0],
                    center: false,
                },
                &mut StdRng::seed_from_u64(1234),
                false,
                OutputFormat::Json,
            )
            .unwrap();
        assert_eq!(run.completed_updates, (n - 1) as u64 * 3);
        assert_eq!(run.positions[0], [1.0, 0.0]);
        for (v, p) in run.positions.iter().enumerate().skip(1) {
            let expected = 2.0 + (v as f64 - 1.0) * 0.75f64.powi(3);
            assert!(
                (p[0] - expected).abs() < 0.001,
                "pair {v}: {} != {expected}",
                p[0]
            );
        }
    }

    #[test]
    fn two_dimensional_dispatch_updates_every_pair_including_last() {
        let context = GpuContext::new().unwrap();
        let count = 65536;
        let run = context
            .execute_sgd(
                SgdParams {
                    positions: (0..count).flat_map(|_| [[0.0, 0.0], [4.0, 0.0]]).collect(),
                    pairs: (0..count)
                        .map(|i| EdgeInfo {
                            u: 2 * i,
                            v: 2 * i + 1,
                            dij: 2.0,
                            weight_u: 0.25,
                            weight_v: 0.25,
                        })
                        .collect(),
                    etas: vec![4.0, 4.0, 4.0],
                    pivots: vec![0],
                    center: false,
                },
                &mut StdRng::seed_from_u64(0),
                false,
                OutputFormat::Json,
            )
            .unwrap();
        assert_eq!(run.completed_updates, count as u64 * 3);
        assert_eq!(run.dispatches, 3);
        for pair in run.positions.chunks_exact(2) {
            assert_eq!(pair, &[[1.0, 0.0], [3.0, 0.0]]);
        }
    }

    #[test]
    fn invalid_endpoints_and_non_finite_input_are_rejected_before_dispatch() {
        let context = GpuContext::new().unwrap();
        for (positions, v) in [
            (vec![[0.0, 0.0], [1.0, 0.0]], 0),
            (vec![[f64::NAN, 0.0], [1.0, 0.0]], 1),
        ] {
            let result = context.execute_sgd(
                SgdParams {
                    positions,
                    pairs: vec![EdgeInfo {
                        u: 0,
                        v,
                        dij: 1.0,
                        weight_u: 1.0,
                        weight_v: 1.0,
                    }],
                    etas: vec![1.0],
                    pivots: vec![0],
                    center: false,
                },
                &mut StdRng::seed_from_u64(0),
                false,
                OutputFormat::Json,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn gpu_handles_coincident_positions_and_centers_output() {
        let context = GpuContext::new().unwrap();
        let run = context
            .execute_sgd(
                SgdParams {
                    positions: vec![[2.0, 3.0]; 2],
                    pairs: vec![EdgeInfo {
                        u: 0,
                        v: 1,
                        dij: 1.0,
                        weight_u: 1.0,
                        weight_v: 1.0,
                    }],
                    etas: vec![0.5],
                    pivots: vec![0],
                    center: true,
                },
                &mut StdRng::seed_from_u64(0),
                false,
                OutputFormat::Json,
            )
            .unwrap();
        assert_eq!(run.completed_updates, 1);
        let center_x = (run.positions[0][0] + run.positions[1][0]) / 2.0;
        let center_y = (run.positions[0][1] + run.positions[1][1]) / 2.0;
        assert!(center_x.abs() < 1e-5, "{center_x}");
        assert!(center_y.abs() < 1e-5, "{center_y}");
    }

    #[test]
    fn atomic_sparse_stepper_matches_cpu_reference_and_resets() {
        let context = GpuContext::new().unwrap();
        let pair = GpuPair {
            u: 0,
            v: 1,
            dij: 2.0,
            weight_u: 0.25,
            weight_v: 0.25,
        };
        let initial_flat = vec![0.0f32, 0.0, 4.0, 0.0];
        let mut stepper = AtomicSparseStepper::new(
            &context.device,
            &context.queue,
            2,
            &[pair],
            &initial_flat,
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
        let mut coincident_stepper = AtomicSparseStepper::new(
            &context.device,
            &context.queue,
            2,
            &[GpuPair {
                u: 0,
                v: 1,
                dij: 1.0,
                weight_u: 1.0,
                weight_v: 1.0,
            }],
            &coincident_flat,
        )
        .unwrap();
        coincident_stepper.step_epoch(&context.device, &context.queue, 0.5);
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
