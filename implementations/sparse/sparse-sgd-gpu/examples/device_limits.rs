fn main() -> anyhow::Result<()> {
    let context = sparse_sgd_gpu::gpu::GpuContext::new_with_pivot_shuffle()?;
    let limits = context.device.limits();
    println!("{}", serde_json::json!({
        "adapter": context.adapter_name,
        "backend": context.backend,
        "max_buffer_size": limits.max_buffer_size,
        "max_storage_buffer_binding_size": limits.max_storage_buffer_binding_size,
        "max_compute_workgroups_per_dimension": limits.max_compute_workgroups_per_dimension,
        "workgroup_size": 256
    }));
    Ok(())
}
