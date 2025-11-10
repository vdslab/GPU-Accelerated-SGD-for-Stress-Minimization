use anyhow::Result;

#[derive(Debug)]
pub struct GpuParams {
    pub data: Vec<f32>,
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
}
