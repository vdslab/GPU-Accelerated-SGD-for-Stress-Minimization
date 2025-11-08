fn main() {
  // GPU setup
  let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

  let adapter =
  pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
      .expect("Failed to create adapter");

  let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
      label: None,
      required_features: wgpu::Features::empty(),
      required_limits: wgpu::Limits::downlevel_defaults(),
      experimental_features: wgpu::ExperimentalFeatures::disabled(),
      memory_hints: wgpu::MemoryHints::MemoryUsage,
      trace: wgpu::Trace::Off,
  }))
  .expect("Failed to create device");

  // NOTE: graphics card info
  println!("Running on Adapter: {:#?}", adapter.get_info());
}