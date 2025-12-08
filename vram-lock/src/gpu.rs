use anyhow::Result;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuEdgeInfo {
    pub u: u32,
    pub v: u32,
    pub dij: f32,
    pub wij: f32,
}

#[derive(Debug)]
pub struct GpuGraphParams {
    pub etas: Vec<f32>,
    pub positions: Vec<[f32; 2]>,
    pub pairs: Vec<GpuEdgeInfo>,
}

#[derive(Debug)]
pub struct GpuPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,

    // Buffers
    pub positions_buffer: wgpu::Buffer,
    pub download_buffer: wgpu::Buffer,
    pub iteration_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    pub lock_buffer: wgpu::Buffer,  // Used by GPU shader for atomic locks
    pub node_size: u32,
    pub num_iterations: u32,
    pub num_pairs: u32,
}

#[derive(Debug)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub module: wgpu::ShaderModule,
}

impl GpuContext {
    // Initialize GPU context
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("Failed to create adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .expect("Failed to create device");

        // LOG: graphics card info
        // println!("Running on Adapter: {:#?}", adapter.get_info());
        // println!(
        //     "thread limit per workgroup: {:#?}",
        //     adapter.limits().max_compute_invocations_per_workgroup
        // );

        let module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        Ok(GpuContext {
            device,
            queue,
            module,
        })
    }

    pub fn setup_compute_pipeline(&self, params: GpuGraphParams) -> Result<GpuPipeline> {
        let etas_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Etas Buffer"),
                contents: bytemuck::cast_slice(&params.etas),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let positions_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Positions Buffer"),
                contents: bytemuck::cast_slice(&params.positions),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let pairs_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pairs Buffer"),
                contents: bytemuck::cast_slice(&params.pairs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // NOTE: Only use this if you need to read the data on the CPU.
        let download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: positions_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let iteration_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Iteration Buffer"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Lock buffer (initialized to 0 = unlocked for all nodes)
        let lock_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Lock Buffer"),
                contents: bytemuck::cast_slice(&vec![0u32; params.positions.len()]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            });

        // NOTE: Bind group
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        // Etas buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                                has_dynamic_offset: false,
                            },
                            count: None,
                        },
                        // Positions buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                min_binding_size: Some(NonZeroU64::new(8).unwrap()),
                                has_dynamic_offset: false,
                            },
                            count: None,
                        },
                        // Pairs buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                                has_dynamic_offset: false,
                            },
                            count: None,
                        },
                        // Iteration buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                                has_dynamic_offset: false,
                            },
                            count: None,
                        },
                        // Lock buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                                has_dynamic_offset: false,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
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
                    resource: pairs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: iteration_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: lock_buffer.as_entire_binding(),
                },
            ],
        });

        // NOTE: Pipeline
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &self.module,
                entry_point: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(GpuPipeline {
            pipeline,
            bind_group,
            positions_buffer,
            download_buffer,
            iteration_buffer,
            lock_buffer,
            node_size: params.positions.len() as u32,
            num_iterations: params.etas.len() as u32,
            num_pairs: params.pairs.len() as u32,
        })
    }

    pub fn execute_compute_pipeline(&self, p: GpuPipeline) -> Result<Vec<[f32; 2]>> {
        // @workgroup_size(32,1,1): Each workgroup = 32 threads (= 1 warp)
        // Each workgroup processes one pair (only local_id.x == 0 does work)
        // Use 2D dispatch to handle more pairs (up to 65535 * 65535)
        let max_x = 65535u32;
        let workgroup_count_x = p.num_pairs.min(max_x);
        let workgroup_count_y = (p.num_pairs + max_x - 1) / max_x;
        
        println!("Dispatching {}x{} workgroups (1 WG per pair, 32 threads per WG) for {} pairs on {} nodes", workgroup_count_x, workgroup_count_y, p.num_pairs, p.node_size);
        
        for iteration in 0..p.num_iterations {
            // Update iteration buffer
            self.queue.write_buffer(&p.iteration_buffer, 0, bytemuck::cast_slice(&[iteration]));
            
            let mut encoder =
                self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { 
                    label: Some(&format!("SGD Iteration {}", iteration)) 
                });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("SGD Pass {}", iteration)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&p.pipeline);
            compute_pass.set_bind_group(0, &p.bind_group, &[]);
            
            // Dispatch workgroups in 2D (x, y)
            compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);

            drop(compute_pass);

            self.queue.submit([encoder.finish()]);

            // Wait for GPU to complete this iteration before printing
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            
            println!("Iteration {}", iteration);
        }
        
        // NOTE: Download final results
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Copy positions_buffer (the one actually updated) to download_buffer
        encoder.copy_buffer_to_buffer(
            &p.positions_buffer,
            0,
            &p.download_buffer,
            0,
            p.positions_buffer.size(),
        );

        let command_buffer = encoder.finish();

        self.queue.submit([command_buffer]);

        let buffer_slice = p.download_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        // Wait for the GPU to finish working on the submitted work.
        // Note: poll() works on native (desktop) environments, but NOT on Web (wasm/browser).
        // On Web, you must use the callback to know when the buffer is mapped.
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        // We can now read the data from the buffer.
        let data = buffer_slice.get_mapped_range();
        // Convert the data to Vec<[f32; 2]>
        let positions_data: &[[f32; 2]] = bytemuck::cast_slice(&data);
        let result: Vec<[f32; 2]> = positions_data.to_vec();

        Ok(result)
    }
}
