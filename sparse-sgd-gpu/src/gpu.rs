use crate::embedding::MIN_DELTA;
use crate::schedule::{Pair, ScheduledPairs};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub n: u32,
    pub embed_dim: u32,
    pub pair_start: u32,
    pub pair_len: u32,
    pub eta: f32,
    pub min_delta: f32,
    _pad: [u32; 2],
}

pub struct GpuSgdParams {
    pub positions: Vec<[f32; 2]>,
    pub embedding: Vec<f32>,
    pub embed_dim: usize,
    pub schedule: ScheduledPairs,
    pub etas: Vec<f32>,
    pub seed: u64,
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub module: wgpu::ShaderModule,
}

impl GpuContext {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| anyhow::anyhow!("GPU adapter not found: {:?}", e))?;

        println!(
            "GPU: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );
        let limits = adapter.limits();
        println!(
            "max_compute_invocations_per_workgroup: {}",
            limits.max_compute_invocations_per_workgroup
        );

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| anyhow::anyhow!("device creation failed: {}", e))?;

        let module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        Ok(Self {
            device,
            queue,
            module,
        })
    }

    pub fn execute_sgd(&self, params: GpuSgdParams) -> Result<(Vec<[f32; 2]>, Vec<[f32; 2]>)> {
        let n = params.positions.len() as u32;
        let embed_dim = params.embed_dim as u32;
        let initial_positions = params.positions.clone();

        println!(
            "GPU SparseSGD: n={}, embed_dim={}, pairs={}, ranges={}, iterations={}",
            n,
            embed_dim,
            params.schedule.packed_pairs.len(),
            params.schedule.dispatch_ranges.len(),
            params.etas.len()
        );

        let positions_flat: Vec<f32> = params.positions.iter().flat_map(|p| [p[0], p[1]]).collect();

        let positions_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("positions"),
                contents: bytemuck::cast_slice(&positions_flat),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let pairs_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pairs"),
                contents: bytemuck::cast_slice::<Pair, u8>(&params.schedule.packed_pairs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let weights_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pair_weights"),
                contents: bytemuck::cast_slice(&params.schedule.packed_weights),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let embedding_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("embedding"),
                contents: bytemuck::cast_slice(&params.embedding),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let uniforms_init = Uniforms {
            n,
            embed_dim,
            pair_start: 0,
            pair_len: 0,
            eta: params.etas.first().copied().unwrap_or(0.0),
            min_delta: MIN_DELTA,
            _pad: [0; 2],
        };
        let uniforms_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::bytes_of(&uniforms_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download"),
            size: positions_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64).unwrap(),
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(8).unwrap()),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(8).unwrap()),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
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
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: embedding_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &self.module,
                entry_point: Some("sparse_sgd"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let mut rng = StdRng::seed_from_u64(params.seed ^ 0x5eed_5eed);
        let mut range_order: Vec<usize> = (0..params.schedule.dispatch_ranges.len()).collect();

        for (iter, &eta) in params.etas.iter().enumerate() {
            range_order.shuffle(&mut rng);

            for &range_index in &range_order {
                let range = params.schedule.dispatch_ranges[range_index];
                let uniforms = Uniforms {
                    n,
                    embed_dim,
                    pair_start: range.start,
                    pair_len: range.len,
                    eta,
                    min_delta: MIN_DELTA,
                    _pad: [0; 2],
                };
                self.queue
                    .write_buffer(&uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(range.len.div_ceil(WORKGROUP_SIZE), 1, 1);
                }
                self.queue.submit([encoder.finish()]);
                self.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
            }

            println!(
                "iter {}/{} complete (eta={:.6})",
                iter + 1,
                params.etas.len(),
                eta
            );
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &positions_buffer,
            0,
            &download_buffer,
            0,
            positions_buffer.size(),
        );
        self.queue.submit([encoder.finish()]);

        let buf_slice = download_buffer.slice(..);
        buf_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let data = buf_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let final_positions = floats.chunks(2).map(|c| [c[0], c[1]]).collect();

        Ok((initial_positions, final_positions))
    }
}
