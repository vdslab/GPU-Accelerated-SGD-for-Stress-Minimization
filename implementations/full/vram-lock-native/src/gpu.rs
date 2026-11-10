#[allow(dead_code)]
#[path = "../../../../shared/gpu-timing/wgpu.rs"]
mod timing;

use crate::graph;
use anyhow::{ensure, Result};
use bytemuck::{Pod, Zeroable};
use experiment_common::OutputFormat;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuEdgeInfo {
    pub u: u32,
    pub v: u32,
    pub dij: f32,
    pub wij: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub iteration: u32,
    pub num_pairs: u32,
    pub chunk_offset: u32,
    pub chunk_size: u32,
}

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
}

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
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
        let gpu_pairs: Vec<GpuEdgeInfo> = params
            .pairs
            .into_iter()
            .map(|p| GpuEdgeInfo {
                u: p.u as u32,
                v: p.v as u32,
                dij: p.dij as f32,
                wij: p.wij as f32,
            })
            .collect();

        let node_size = gpu_positions.len();
        let num_iterations = gpu_etas.len();
        let num_pairs = gpu_pairs.len() as u32;

        let effective_chunk_size = match chunk_size_param {
            Some(0) => num_pairs,
            Some(size) => size,
            None => 65536u32,
        };

        let method_setup_time = method_setup_started.elapsed();

        let preparation_started = Instant::now();
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

        let pairs_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pairs Buffer"),
                contents: bytemuck::cast_slice(&gpu_pairs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let locks_zero = vec![0u32; node_size];
        let lock_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Locks Buffer"),
                contents: bytemuck::cast_slice(&locks_zero),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let updated_count_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Updated Count Buffer"),
                    contents: bytemuck::bytes_of(&0u32),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });

        let initial_uniforms = Uniforms {
            iteration: 0,
            num_pairs,
            chunk_offset: 0,
            chunk_size: 0,
        };
        let uniforms_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniforms Buffer"),
                contents: bytemuck::bytes_of(&initial_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let count_download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Count Download Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Positions Download Buffer"),
            size: (node_size * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Atomic Full SGD Bind Group"),
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
                    resource: pairs_buffer.as_entire_binding(),
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
                    resource: uniforms_buffer.as_entire_binding(),
                },
            ],
        });

        // Warmup PSO dispatch to isolate Mesa NIR->DXIL compilation overhead
        experiment_common::gpu::run_warmup_dispatch_simple(
            &self.device,
            &self.queue,
            &self.pipeline,
            &bind_group,
        )?;

        let preparation_time = preparation_started.elapsed();
        let upload_time = Duration::ZERO;

        let total_chunks_per_iter = num_pairs.div_ceil(effective_chunk_size);
        let total_passes = (num_iterations as u32 * total_chunks_per_iter) as usize;
        let mut pass_timer = timing::PassTimer::new(&self.device, total_passes);

        let iteration_start = Instant::now();
        let mut completed_updates = 0u64;

        for iteration in 0..num_iterations {
            self.queue
                .write_buffer(&updated_count_buffer, 0, bytemuck::bytes_of(&0u32));

            let mut offset = 0u32;
            while offset < num_pairs {
                let this_chunk = effective_chunk_size.min(num_pairs - offset);
                let uniforms = Uniforms {
                    iteration: iteration as u32,
                    num_pairs,
                    chunk_offset: offset,
                    chunk_size: this_chunk,
                };
                self.queue
                    .write_buffer(&uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

                let dispatch_x = this_chunk.min(65535);
                let dispatch_y = this_chunk.div_ceil(65535);

                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("atomic_chunk_encoder"),
                        });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("atomic_chunk_pass"),
                        timestamp_writes: pass_timer.next(),
                    });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }
                self.queue.submit(std::iter::once(encoder.finish()));
                self.device.poll(wgpu::PollType::wait_indefinitely())?;

                offset += this_chunk;
            }

            // Read back updated count and assert 100% completion
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("count_readback_encoder"),
                });
            encoder.copy_buffer_to_buffer(&updated_count_buffer, 0, &count_download_buffer, 0, 4);
            self.queue.submit(std::iter::once(encoder.finish()));

            let count_slice = count_download_buffer.slice(..);
            count_slice.map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
            let updated_count_val = {
                let view = count_slice.get_mapped_range();
                let val = *bytemuck::from_bytes::<u32>(&view);
                drop(view);
                count_download_buffer.unmap();
                val
            };

            ensure!(
                updated_count_val == num_pairs,
                "Incomplete Full iteration {}: completed {} of {} pairs (no successful result emitted)",
                iteration + 1, updated_count_val, num_pairs
            );
            completed_updates += u64::from(updated_count_val);

            pass_timer.end_iteration();

            if verbose {
                let message = format!(
                    "Iteration {} - Updated {} pairs",
                    iteration + 1,
                    updated_count_val
                );
                match output_format {
                    OutputFormat::Json => eprintln!("{message}"),
                    OutputFormat::Human => println!("{message}"),
                }
            }
        }

        let iteration_time = iteration_start.elapsed();

        // Read back final positions
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

        let slice = download_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        let final_positions: Vec<[f32; 2]> = {
            let view = slice.get_mapped_range();
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
        })
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
    }
}
