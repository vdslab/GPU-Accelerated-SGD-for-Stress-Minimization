mod graph;

use std::num::NonZeroU64;
use std::path::Path;
use wgpu::util::DeviceExt;

fn main() {
    env_logger::init();

    let mtx_path = Path::new("../data/lesmis_pattern.mtx");
    let graph = graph::Graph::from_mtx(mtx_path).expect("Failed to load matrix");

    // LOG: Print graph information
    // println!("Node size: {:?}", graph.node_size);
    // println!("Edge size: {:?}", graph.edge_size);
    // println!("Edge src: {:?}", graph.edge_src);
    // println!("Edge dst: {:?}", graph.edge_dst);

    let dist = graph::Graph::calc_dist_matrix(&graph);

    // LOG: Print distance matrix
    // println!("Dist matrix: {:?}", dist);

    let (pairs, wmin, wmax) = graph::Graph::calc_edge_info(&graph, &dist);

    let etas = graph::calc_learning_rate(15, wmin, wmax, 0.1);

    // LOG: Print pairs and etas
    println!("pairs: {:?}", pairs);
    println!("Etas: {:?}", etas);



    // GPU setup
    let width = 64u32;
    let height = 64u32;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];

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

    // NOTE: uniform buffer is a small GPU buffer used to store constant data
    let uniform_data = [width];
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::cast_slice(&uniform_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let input_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output"),
        size: input_data_buffer.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // NOTE: Only use this if you need to read the data on the CPU.
    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: input_data_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // NOTE: Bind group
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            // Uniform buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    // read only?
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Input buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    // This is the size of a single element in the buffer.
                    min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    has_dynamic_offset: false,
                },
                count: None,
            },
            // Output buffer
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    // This is the size of a single element in the buffer.
                    min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    has_dynamic_offset: false,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_data_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_data_buffer.as_entire_binding(),
            },
        ],
    });

    // NOTE: Pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // NOTE: Command encoder
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    let workgroup_x = width.div_ceil(16);
    let workgroup_y = height.div_ceil(16);
    compute_pass.dispatch_workgroups(workgroup_x as u32, workgroup_y as u32, 1);

    // NOTE: Rust borrow rules
    drop(compute_pass);

    encoder.copy_buffer_to_buffer(
        &output_data_buffer,
        0,
        &download_buffer,
        0,
        output_data_buffer.size(),
    );

    let command_buffer = encoder.finish();

    queue.submit([command_buffer]);

    let buffer_slice = download_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {
        // In this case we know exactly when the mapping will be finished,
        // so we don't need to do anything in the callback.
    });

    // Wait for the GPU to finish working on the submitted work. This doesn't work on WebGPU, so we would need
    // to rely on the callback to know when the buffer is mapped.
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    // We can now read the data from the buffer.
    let data = buffer_slice.get_mapped_range();
    // Convert the data back to a slice of f32.
    let result: &[f32] = bytemuck::cast_slice(&data);

    // Print out the result.
    println!("Result: {result:?}");
}
