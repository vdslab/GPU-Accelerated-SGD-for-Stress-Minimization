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
    pub output_data_buffer: wgpu::Buffer,
    pub download_buffer: wgpu::Buffer,
    pub debug_info_buffer: wgpu::Buffer,
    pub debug_download_buffer: wgpu::Buffer,
    pub debug_pairs_buffer: wgpu::Buffer,
    pub debug_pairs_download_buffer: wgpu::Buffer,
    pub iteration_buffer: wgpu::Buffer,
    pub node_size: u32,
    pub num_iterations: u32,
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
                usage: wgpu::BufferUsages::STORAGE,
            });

        let pairs_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pairs Buffer"),
                contents: bytemuck::cast_slice(&params.pairs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_data_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output"),
            size: positions_buffer.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // NOTE: Only use this if you need to read the data on the CPU.
        let download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: positions_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let debug_info_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Info Buffer"),
            size: 12, // 3 x f32 = 12 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // NOTE: Only use this if you need to read the data on the CPU.
        let debug_download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Download Buffer"),
            size: 12,
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
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Debug pairs buffer (to store node 2's pair partners)
        let debug_pairs_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Pairs Buffer"),
            size: (params.positions.len() * 4) as u64, // Max possible pairs
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let debug_pairs_download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Pairs Download Buffer"),
            size: (params.positions.len() * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
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
                        // Debug info buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                min_binding_size: Some(NonZeroU64::new(12).unwrap()),
                                has_dynamic_offset: false,
                            },
                            count: None,
                        },
                        // Iteration buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                                has_dynamic_offset: false,
                            },
                            count: None,
                        },
                        // Debug pairs buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
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
                    resource: debug_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: iteration_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: lock_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: debug_pairs_buffer.as_entire_binding(),
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
            output_data_buffer,
            download_buffer,
            debug_info_buffer,
            debug_download_buffer,
            debug_pairs_buffer,
            debug_pairs_download_buffer,
            iteration_buffer,
            node_size: params.positions.len() as u32,
            num_iterations: params.etas.len() as u32,
        })
    }

    pub fn execute_compute_pipeline(&self, p: GpuPipeline) -> Result<Vec<[f32; 2]>> {
        // @workgroup_size(32,32,1) = 1024 threads per workgroup
        let workgroup_size = 32u32;
        let workgroup_x = (p.node_size + workgroup_size - 1) / workgroup_size;
        let workgroup_y = (p.node_size + workgroup_size - 1) / workgroup_size;
        
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
            
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);

            drop(compute_pass);

            encoder.copy_buffer_to_buffer(
                &p.debug_info_buffer,
                0,
                &p.debug_download_buffer,
                0,
                p.debug_info_buffer.size(),
            );

            encoder.copy_buffer_to_buffer(
                &p.debug_pairs_buffer,
                0,
                &p.debug_pairs_download_buffer,
                0,
                p.debug_pairs_buffer.size(),
            );

            self.queue.submit([encoder.finish()]);

            let debug_slice = p.debug_download_buffer.slice(..);
            debug_slice.map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let debug_data = debug_slice.get_mapped_range();
            let debug_floats: &[f32] = bytemuck::cast_slice(&debug_data);
            
            // Read debug pairs
            let pairs_slice = p.debug_pairs_download_buffer.slice(..);
            pairs_slice.map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let pairs_data = pairs_slice.get_mapped_range();
            let pairs_u32: &[u32] = bytemuck::cast_slice(&pairs_data);
            let node2_partners: Vec<u32> = pairs_u32.to_vec();
            
            println!("Debug info: Iteration {}: val1={},node2 partners={:?}", iteration, debug_floats[0], node2_partners);
            
            drop(debug_data);
            drop(pairs_data);
            p.debug_download_buffer.unmap();
            p.debug_pairs_download_buffer.unmap();
        }
        
        // NOTE: Download final results
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(
            &p.output_data_buffer,
            0,
            &p.download_buffer,
            0,
            p.output_data_buffer.size(),
        );

        encoder.copy_buffer_to_buffer(
            &p.debug_info_buffer,
            0,
            &p.debug_download_buffer,
            0,
            p.debug_info_buffer.size(),
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

        // Read debug info
        let debug_slice = p.debug_download_buffer.slice(..);
        debug_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let debug_data = debug_slice.get_mapped_range();
        // Convert the data back to a slice of f32.
        let debug_floats: &[f32] = bytemuck::cast_slice(&debug_data);
        
        if debug_floats.len() >= 3 {
            println!("Debug info: val1={}, val2={}, val3={}", debug_floats[0], debug_floats[1], debug_floats[2]);
        }

        Ok(result)
    }
}
