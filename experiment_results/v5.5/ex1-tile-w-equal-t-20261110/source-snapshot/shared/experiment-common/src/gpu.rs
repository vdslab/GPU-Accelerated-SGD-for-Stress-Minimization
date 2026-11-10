//! WebGPU adapter discovery, strict native GPU enforcement, and PSO warmup.
//!
//! Enforces execution on native hardware under WSL2 (D3D12 NVIDIA GeForce RTX 3050
//! via Backend::Gl) while hard-rejecting CPU/software rasterizers (llvmpipe, lavapipe).
//! Provides warmup dispatch to isolate deferred Mesa NIR->DXIL compilation from timed loops.

use anyhow::{bail, ensure, Context, Result};
use std::sync::Mutex;
use wgpu::{
    Adapter, AdapterInfo, Backends, ComputePipeline, Device, DeviceDescriptor, Features,
    Instance, InstanceDescriptor, InstanceFlags, MemoryHints, PowerPreference, Queue, Trace,
};

/// Global retainer to keep wgpu instances alive on Linux, preventing premature EGL teardown.
static INSTANCE_RETAINER: Mutex<Vec<Instance>> = Mutex::new(Vec::new());

/// Detects whether running inside a WSL2 Linux environment (or `/dev/dxg` is present).
pub fn is_wsl2_environment() -> bool {
    std::path::Path::new("/dev/dxg").exists()
        || std::fs::read_to_string("/proc/version")
            .map(|v| {
                let v_lower = v.to_lowercase();
                v_lower.contains("microsoft") || v_lower.contains("wsl")
            })
            .unwrap_or(false)
}

/// Configures environment variables for Mesa Gallium D3D12 under WSL2 before `wgpu::Instance` is created.
///
/// In WSL2, hardware graphics is exposed via the DirectX Graphics Kernel bridge (`/dev/dxg`).
/// Setting `GALLIUM_DRIVER=d3d12` and `MESA_LOADER_DRIVER_OVERRIDE=d3d12` directs Mesa's
/// Gallium driver subsystem to use the D3D12 backend, exposing the host GPU (NVIDIA GeForce RTX 3050).
pub fn configure_wsl2_environment_hints() {
    if is_wsl2_environment() {
        if std::env::var("GALLIUM_DRIVER").is_err() {
            log::info!("WSL2 environment detected: configuring GALLIUM_DRIVER=d3d12");
            #[allow(unused_unsafe)]
            unsafe {
                std::env::set_var("GALLIUM_DRIVER", "d3d12");
            }
        }
        if std::env::var("MESA_LOADER_DRIVER_OVERRIDE").is_err() {
            log::info!("WSL2 environment detected: configuring MESA_LOADER_DRIVER_OVERRIDE=d3d12");
            #[allow(unused_unsafe)]
            unsafe {
                std::env::set_var("MESA_LOADER_DRIVER_OVERRIDE", "d3d12");
            }
        }
    }
}

/// Creates a `wgpu::Instance` configured for all backends and non-compliant adapters.
pub fn create_gpu_instance() -> Instance {
    configure_wsl2_environment_hints();
    let descriptor = InstanceDescriptor {
        backends: Backends::all(),
        flags: InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
        ..Default::default()
    };
    let instance = Instance::new(&descriptor);

    if let Ok(mut guard) = INSTANCE_RETAINER.lock() {
        guard.push(instance.clone());
    }

    instance
}

/// Returns true if an adapter is a software CPU emulator or unaccelerated fallback rasterizer.
pub fn is_software_or_cpu_adapter(info: &AdapterInfo) -> bool {
    if info.device_type == wgpu::DeviceType::Cpu {
        return true;
    }
    let name_lower = info.name.to_lowercase();
    let driver_lower = info.driver.to_lowercase();
    let driver_info_lower = info.driver_info.to_lowercase();

    const REJECTED_KEYWORDS: &[&str] = &[
        "llvmpipe",
        "lavapipe",
        "softpipe",
        "swiftshader",
        "microsoft basic render driver",
        "software",
    ];

    for &kw in REJECTED_KEYWORDS {
        if name_lower.contains(kw) || driver_lower.contains(kw) || driver_info_lower.contains(kw) {
            return true;
        }
    }
    false
}

/// Scores an adapter based on discrete hardware capabilities and host platform preferences.
pub fn score_adapter(info: &AdapterInfo) -> i32 {
    if is_software_or_cpu_adapter(info) {
        return -100_000;
    }

    let mut score = 0i32;
    let name_lower = info.name.to_lowercase();
    let driver_info_lower = info.driver_info.to_lowercase();

    // Device type tier
    match info.device_type {
        wgpu::DeviceType::DiscreteGpu => score += 10_000,
        wgpu::DeviceType::Other => score += 5_000,
        wgpu::DeviceType::IntegratedGpu => score += 3_000,
        wgpu::DeviceType::VirtualGpu => score += 1_000,
        wgpu::DeviceType::Cpu => score -= 50_000,
    }

    // Hardware vendor and architecture keywords
    let is_hardware_keyword = name_lower.contains("nvidia")
        || name_lower.contains("geforce")
        || name_lower.contains("rtx")
        || name_lower.contains("gtx")
        || name_lower.contains("quadro")
        || name_lower.contains("radeon")
        || name_lower.contains("d3d12")
        || driver_info_lower.contains("nvidia")
        || driver_info_lower.contains("geforce")
        || driver_info_lower.contains("d3d12");

    if is_hardware_keyword {
        score += 10_000;
    }

    // Host platform priority (NVIDIA GeForce RTX 3050)
    if name_lower.contains("rtx 3050") {
        score += 5_000;
    }

    // Backend tie-breaker under WSL2 (Mesa Gallium D3D12 exposes Backend::Gl)
    match info.backend {
        wgpu::Backend::Gl => score += 2_000,
        wgpu::Backend::Vulkan => score += 1_000,
        wgpu::Backend::Dx12 => score += 1_000,
        wgpu::Backend::Metal => score += 1_000,
        _ => {}
    }

    score
}

/// Enumerates and selects the highest-scoring native GPU adapter, hard-rejecting software rasterizers.
///
/// Under WSL2 (`/dev/dxg` present or WSL detected), strictly validates that the selected adapter is native D3D12/RTX 3050
/// with `Backend::Gl` (Quality Gate 1).
pub fn select_native_adapter(instance: &Instance) -> Result<Adapter> {
    let mut adapters = instance.enumerate_adapters(Backends::all());
    if adapters.is_empty() {
        bail!("No WebGPU adapters found on system");
    }

    adapters.retain(|a| !is_software_or_cpu_adapter(&a.get_info()));
    if adapters.is_empty() {
        bail!(
            "All available WebGPU adapters were rejected as CPU/software emulators (llvmpipe/lavapipe). \
             Ensure Mesa Gallium D3D12 is enabled with GALLIUM_DRIVER=d3d12."
        );
    }

    adapters.sort_by_key(|a| std::cmp::Reverse(score_adapter(&a.get_info())));
    let selected = adapters.into_iter().next().unwrap();
    let info = selected.get_info();

    if is_wsl2_environment() {
        let name_lower = info.name.to_lowercase();
        let is_d3d12_or_rtx = name_lower.contains("d3d12") || name_lower.contains("rtx 3050");
        ensure!(
            is_d3d12_or_rtx && info.backend == wgpu::Backend::Gl,
            "Gate 1 Verification Failed: Under WSL2 (/dev/dxg), expected 'D3D12 (NVIDIA GeForce RTX 3050)' with Backend::Gl, but got '{}' with {:?}",
            info.name,
            info.backend
        );
    }

    log::info!(
        "Selected native GPU adapter: '{}' (backend: {:?}, type: {:?})",
        info.name,
        info.backend,
        info.device_type
    );

    Ok(selected)
}

/// Initializes the GPU context (Instance, Adapter, Device, Queue) with strict native selection.
pub fn init_gpu_context(
    power_preference: PowerPreference,
) -> Result<(Instance, Adapter, Device, Queue)> {
    init_gpu_context_with_features(power_preference, Features::empty())
}

/// Initializes the GPU context with caller-specified additional required features.
pub fn init_gpu_context_with_features(
    _power_preference: PowerPreference,
    extra_features: Features,
) -> Result<(Instance, Adapter, Device, Queue)> {
    configure_wsl2_environment_hints();
    let instance = create_gpu_instance();
    let adapter = select_native_adapter(&instance)?;

    let supported_features = adapter.features();
    let mut required_features = extra_features;
    if supported_features.contains(Features::TIMESTAMP_QUERY) {
        required_features |= Features::TIMESTAMP_QUERY;
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("Native GPU Device"),
        required_features,
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: MemoryHints::MemoryUsage,
        trace: Trace::Off,
    }))
    .with_context(|| {
        format!(
            "Failed to create GPU device from adapter '{}'",
            adapter.get_info().name
        )
    })?;

    Ok((instance, adapter, device, queue))
}

/// Dispatches 1 dummy workgroup to trigger Mesa Gallium NIR->DXIL compilation
/// ahead of timed iteration loops.
///
/// In Mesa Gallium D3D12, shader NIR is compiled to DXIL bytecode and PSO (Pipeline
/// State Object) generation is deferred until the first dispatch of that pipeline.
/// Executing this warmup dispatch before starting timed iteration loops isolates this
/// compilation overhead (50ms - 2000ms) to initialization time (`runtime_init_time_ms`).
///
/// For pipelines with dynamic uniform/storage offsets, pass them in `dynamic_offsets`.
/// For pipelines without dynamic offsets, pass `&[]`.
pub fn run_warmup_dispatch(
    device: &Device,
    queue: &Queue,
    pipeline: &ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dynamic_offsets: &[u32],
) -> Result<()> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("warmup_dispatch_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("warmup_dispatch_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, dynamic_offsets);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely())?;
    Ok(())
}

/// Convenience wrapper for `run_warmup_dispatch` when no dynamic buffer offsets are required.
pub fn run_warmup_dispatch_simple(
    device: &Device,
    queue: &Queue,
    pipeline: &ComputePipeline,
    bind_group: &wgpu::BindGroup,
) -> Result<()> {
    run_warmup_dispatch(device, queue, pipeline, bind_group, &[])
}
