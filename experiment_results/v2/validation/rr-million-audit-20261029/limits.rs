fn main() {
 let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
 let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
 let l = adapter.limits();
 println!("adapter={:?} max_buffer_size={} max_storage_buffer_binding_size={} max_compute_workgroups_per_dimension={}", adapter.get_info().name, l.max_buffer_size, l.max_storage_buffer_binding_size, l.max_compute_workgroups_per_dimension);
}
