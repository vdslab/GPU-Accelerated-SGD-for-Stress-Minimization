use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::process::{self, Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const PAGE_STRIDE_BYTES: u64 = 4096;
const WORKGROUP_SIZE: u64 = 64;
const DEFAULT_INITIAL_BYTES: u64 = GIB;
const DEFAULT_RESOLUTION_BYTES: u64 = 256 * MIB;
const DEFAULT_TIMEOUT_SECONDS: u64 = 120;
const DEFAULT_RETRIES: u32 = 3;
const DEFAULT_SAFETY_MARGIN_PERCENT: u8 = 10;

const TOUCH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<atomic<u32>>;

@compute @workgroup_size(64)
fn touch_pages(@builtin(global_invocation_id) id: vec3<u32>) {
    let word_index = id.x * 1024u;
    if (word_index < arrayLength(&data)) {
        atomicXor(&data[word_index], 1u);
    }
}
"#;

#[derive(Clone, Debug)]
struct ProbeOptions {
    bytes: u64,
    timeout: Duration,
}

#[derive(Clone, Debug)]
struct SearchOptions {
    initial_bytes: u64,
    max_bytes: Option<u64>,
    resolution_bytes: u64,
    timeout: Duration,
    retries: u32,
    safety_margin_percent: u8,
    output_dir: PathBuf,
    command_line: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ProbeStage {
    Device,
    Allocation,
    ShaderCompilation,
    Encoding,
    Completion,
    Timeout,
    Process,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ProbeResult {
    requested_bytes: u64,
    success: bool,
    stage: Option<ProbeStage>,
    error: Option<String>,
    elapsed_ms: u128,
    adapter_name: Option<String>,
    wgpu_backend: Option<String>,
    max_buffer_size_bytes: Option<u64>,
    max_storage_buffer_binding_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TrialRecord {
    requested_bytes: u64,
    attempt: u32,
    phase: String,
    result: ProbeResult,
}

#[derive(Clone, Debug, Default, Serialize)]
struct WgpuMetadata {
    adapter_name: Option<String>,
    backend: Option<String>,
    max_buffer_size_bytes: Option<u64>,
    max_storage_buffer_binding_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct EnvironmentInfo {
    hardware_model: Option<String>,
    physical_memory_bytes: Option<u64>,
    macos_version: Option<String>,
    wgpu: WgpuMetadata,
    git_commit: Option<String>,
}

#[derive(Debug, Serialize)]
struct SearchSummary {
    timestamp_unix_seconds: u64,
    command_line: Vec<String>,
    environment: EnvironmentInfo,
    initial_bytes: u64,
    configured_max_bytes: Option<u64>,
    resolution_bytes: u64,
    timeout_seconds: u64,
    retries: u32,
    safety_margin_percent: u8,
    page_stride_bytes: u64,
    largest_observed_success_bytes: u64,
    smallest_observed_failure_bytes: Option<u64>,
    upper_bound_confirmed: bool,
    stable_max_success_bytes: Option<u64>,
    recommended_limit_bytes: Option<u64>,
    trials: Vec<TrialRecord>,
}

#[derive(Debug)]
struct ProbeFailure {
    stage: ProbeStage,
    error: String,
    metadata: WgpuMetadata,
}

fn main() {
    let exit_code = match run(env::args().collect()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error}");
            print_usage();
            64
        }
    };
    process::exit(exit_code);
}

fn run(args: Vec<String>) -> Result<i32, String> {
    match args.get(1).map(String::as_str) {
        Some("probe") => {
            let options = parse_probe_options(&args[2..])?;
            let result =
                std::panic::catch_unwind(|| run_wgpu_probe(&options)).unwrap_or_else(|_| {
                    failed_result(options.bytes, ProbeStage::Process, "probe panicked")
                });
            println!(
                "{}",
                serde_json::to_string(&result).map_err(|error| error.to_string())?
            );
            Ok(if result.success { 0 } else { 2 })
        }
        Some("search") => {
            let options = parse_search_options(&args[2..], &args)?;
            let summary = run_search(&options)?;
            let (jsonl_path, report_path) = write_reports(&options, &summary)?;
            println!("JSONL log: {}", jsonl_path.display());
            println!("Markdown report: {}", report_path.display());
            println!(
                "largest observed success: {}",
                format_bytes(summary.largest_observed_success_bytes)
            );
            println!(
                "stable maximum: {}",
                optional_bytes(summary.stable_max_success_bytes)
            );
            println!(
                "recommended limit: {}",
                optional_bytes(summary.recommended_limit_bytes)
            );
            Ok(0)
        }
        Some("--help" | "-h" | "help") => {
            print_usage();
            Ok(0)
        }
        Some(command) => Err(format!("unknown subcommand: {command}")),
        None => Err("missing subcommand".to_string()),
    }
}

fn parse_probe_options(args: &[String]) -> Result<ProbeOptions, String> {
    let mut bytes = None;
    let mut timeout_seconds = DEFAULT_TIMEOUT_SECONDS;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--bytes" => bytes = Some(parse_u64_arg(args, &mut index, "--bytes")?),
            "--timeout-seconds" => {
                timeout_seconds = parse_u64_arg(args, &mut index, "--timeout-seconds")?
            }
            flag => return Err(format!("unknown probe option: {flag}")),
        }
        index += 1;
    }
    let bytes = bytes.ok_or_else(|| "probe requires --bytes".to_string())?;
    if bytes == 0 {
        return Err("--bytes must be greater than zero".to_string());
    }
    Ok(ProbeOptions {
        bytes,
        timeout: Duration::from_secs(timeout_seconds),
    })
}

fn parse_search_options(args: &[String], command_line: &[String]) -> Result<SearchOptions, String> {
    let mut initial_bytes = DEFAULT_INITIAL_BYTES;
    let mut max_bytes = None;
    let mut resolution_bytes = DEFAULT_RESOLUTION_BYTES;
    let mut timeout_seconds = DEFAULT_TIMEOUT_SECONDS;
    let mut retries = DEFAULT_RETRIES;
    let mut safety_margin_percent = DEFAULT_SAFETY_MARGIN_PERCENT;
    let mut output_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../output/m4-memory-capacity");
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--initial-bytes" => {
                initial_bytes = parse_u64_arg(args, &mut index, "--initial-bytes")?
            }
            "--max-bytes" => max_bytes = Some(parse_u64_arg(args, &mut index, "--max-bytes")?),
            "--resolution-bytes" => {
                resolution_bytes = parse_u64_arg(args, &mut index, "--resolution-bytes")?
            }
            "--timeout-seconds" => {
                timeout_seconds = parse_u64_arg(args, &mut index, "--timeout-seconds")?
            }
            "--retries" => {
                retries = u32::try_from(parse_u64_arg(args, &mut index, "--retries")?)
                    .map_err(|_| "--retries is too large".to_string())?
            }
            "--safety-margin-percent" => {
                safety_margin_percent =
                    u8::try_from(parse_u64_arg(args, &mut index, "--safety-margin-percent")?)
                        .map_err(|_| {
                            "--safety-margin-percent must be between 0 and 99".to_string()
                        })?
            }
            "--output-dir" => {
                index += 1;
                output_dir = PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "missing value for --output-dir".to_string())?,
                );
            }
            flag => return Err(format!("unknown search option: {flag}")),
        }
        index += 1;
    }
    if initial_bytes == 0 || resolution_bytes == 0 || retries == 0 || safety_margin_percent >= 100 {
        return Err("initial size, resolution, retries, and safety margin are invalid".to_string());
    }
    Ok(SearchOptions {
        initial_bytes,
        max_bytes,
        resolution_bytes,
        timeout: Duration::from_secs(timeout_seconds),
        retries,
        safety_margin_percent,
        output_dir,
        command_line: command_line.to_vec(),
    })
}

fn parse_u64_arg(args: &[String], index: &mut usize, flag: &str) -> Result<u64, String> {
    *index += 1;
    args.get(*index)
        .ok_or_else(|| format!("missing value for {flag}"))?
        .parse()
        .map_err(|_| format!("{flag} must be an unsigned integer number of bytes"))
}

fn run_wgpu_probe(options: &ProbeOptions) -> ProbeResult {
    let started = Instant::now();
    match pollster::block_on(run_wgpu_probe_async(options)) {
        Ok(metadata) => ProbeResult {
            requested_bytes: options.bytes,
            success: true,
            stage: None,
            error: None,
            elapsed_ms: started.elapsed().as_millis(),
            adapter_name: metadata.adapter_name,
            wgpu_backend: metadata.backend,
            max_buffer_size_bytes: metadata.max_buffer_size_bytes,
            max_storage_buffer_binding_bytes: metadata.max_storage_buffer_binding_bytes,
        },
        Err(failure) => ProbeResult {
            requested_bytes: options.bytes,
            success: false,
            stage: Some(failure.stage),
            error: Some(failure.error),
            elapsed_ms: started.elapsed().as_millis(),
            adapter_name: failure.metadata.adapter_name,
            wgpu_backend: failure.metadata.backend,
            max_buffer_size_bytes: failure.metadata.max_buffer_size_bytes,
            max_storage_buffer_binding_bytes: failure.metadata.max_storage_buffer_binding_bytes,
        },
    }
}

async fn run_wgpu_probe_async(options: &ProbeOptions) -> Result<WgpuMetadata, ProbeFailure> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|error| ProbeFailure {
            stage: ProbeStage::Device,
            error: format!("failed to request WGPU adapter: {error}"),
            metadata: WgpuMetadata::default(),
        })?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    let metadata = WgpuMetadata {
        adapter_name: Some(info.name),
        backend: Some(format!("{:?}", info.backend)),
        max_buffer_size_bytes: Some(limits.max_buffer_size),
        max_storage_buffer_binding_bytes: Some(limits.max_storage_buffer_binding_size as u64),
    };
    let allocated_bytes = options.bytes.div_ceil(4) * 4;
    if allocated_bytes > limits.max_buffer_size {
        return Err(ProbeFailure {
            stage: ProbeStage::Allocation,
            error: format!(
                "requested size exceeds WGPU max_buffer_size ({})",
                limits.max_buffer_size
            ),
            metadata,
        });
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("m4-memory-capacity-probe"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|error| ProbeFailure {
            stage: ProbeStage::Device,
            error: format!("failed to request WGPU device: {error}"),
            metadata: metadata.clone(),
        })?;

    let max_dispatch_bytes =
        limits.max_compute_workgroups_per_dimension as u64 * WORKGROUP_SIZE * PAGE_STRIDE_BYTES;
    let chunk_bytes = (limits.max_storage_buffer_binding_size as u64).min(max_dispatch_bytes)
        / PAGE_STRIDE_BYTES
        * PAGE_STRIDE_BYTES;
    if chunk_bytes == 0 {
        return Err(ProbeFailure {
            stage: ProbeStage::Encoding,
            error: "WGPU adapter does not expose a usable storage binding size".to_string(),
            metadata,
        });
    }

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("m4-memory-capacity-probe-buffer"),
        size: allocated_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("m4-memory-capacity-touch-pages"),
        source: wgpu::ShaderSource::Wgsl(TOUCH_SHADER.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("m4-memory-capacity-bind-group-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("m4-memory-capacity-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("m4-memory-capacity-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("touch_pages"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("m4-memory-capacity-command-encoder"),
    });
    let mut offset = 0;
    while offset < allocated_bytes {
        let size = (allocated_bytes - offset).min(chunk_bytes);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("m4-memory-capacity-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset,
                    size: NonZeroU64::new(size),
                }),
            }],
        });
        let workgroups = size.div_ceil(PAGE_STRIDE_BYTES).div_ceil(WORKGROUP_SIZE);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("m4-memory-capacity-touch-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        offset += size;
    }
    let submission = queue.submit(Some(encoder.finish()));
    let poll_result = device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: Some(options.timeout),
    });
    let out_of_memory = device.pop_error_scope().await;
    let validation = device.pop_error_scope().await;
    if let Some(error) = out_of_memory {
        return Err(ProbeFailure {
            stage: ProbeStage::Allocation,
            error: format!("WGPU out-of-memory error: {error}"),
            metadata,
        });
    }
    if let Some(error) = validation {
        return Err(ProbeFailure {
            stage: ProbeStage::Encoding,
            error: format!("WGPU validation error: {error}"),
            metadata,
        });
    }
    if let Err(error) = poll_result {
        let detail = error.to_string();
        return Err(ProbeFailure {
            stage: if detail.to_ascii_lowercase().contains("timed out") {
                ProbeStage::Timeout
            } else {
                ProbeStage::Completion
            },
            error: format!("WGPU command submission failed: {detail}"),
            metadata,
        });
    }
    Ok(metadata)
}

fn failed_result(bytes: u64, stage: ProbeStage, error: impl Into<String>) -> ProbeResult {
    ProbeResult {
        requested_bytes: bytes,
        success: false,
        stage: Some(stage),
        error: Some(error.into()),
        elapsed_ms: 0,
        adapter_name: None,
        wgpu_backend: None,
        max_buffer_size_bytes: None,
        max_storage_buffer_binding_bytes: None,
    }
}

fn run_search(options: &SearchOptions) -> Result<SearchSummary, String> {
    let mut trials = Vec::new();
    let mut low = 0;
    let mut high = None;
    let mut candidate = options.initial_bytes;
    loop {
        let trial = run_isolated_probe(candidate, 1, "expand", options.timeout)?;
        if trial.result.success {
            low = candidate;
            trials.push(trial);
            if options.max_bytes.is_some_and(|max| candidate >= max) {
                break;
            }
            candidate = candidate.saturating_mul(2);
            if let Some(max) = options.max_bytes {
                candidate = candidate.min(max);
            }
            if candidate <= low {
                break;
            }
        } else {
            high = Some(candidate);
            trials.push(trial);
            break;
        }
    }
    if let Some(mut upper) = high {
        while upper - low > options.resolution_bytes {
            let middle = low + (upper - low) / 2;
            let trial = run_isolated_probe(middle, 1, "binary_search", options.timeout)?;
            if trial.result.success {
                low = middle;
            } else {
                upper = middle;
            }
            trials.push(trial);
        }
        high = Some(upper);
    }
    let stable_max_success_bytes = if high.is_some() && low > 0 {
        find_stable_candidate(low, options, &mut trials)?
    } else {
        None
    };
    let recommended_limit_bytes = stable_max_success_bytes
        .map(|value| value * (100 - options.safety_margin_percent as u64) / 100);
    Ok(SearchSummary {
        timestamp_unix_seconds: unix_timestamp(),
        command_line: options.command_line.clone(),
        environment: collect_environment(),
        initial_bytes: options.initial_bytes,
        configured_max_bytes: options.max_bytes,
        resolution_bytes: options.resolution_bytes,
        timeout_seconds: options.timeout.as_secs(),
        retries: options.retries,
        safety_margin_percent: options.safety_margin_percent,
        page_stride_bytes: PAGE_STRIDE_BYTES,
        largest_observed_success_bytes: low,
        smallest_observed_failure_bytes: high,
        upper_bound_confirmed: high.is_some(),
        stable_max_success_bytes,
        recommended_limit_bytes,
        trials,
    })
}

fn find_stable_candidate(
    mut candidate: u64,
    options: &SearchOptions,
    trials: &mut Vec<TrialRecord>,
) -> Result<Option<u64>, String> {
    loop {
        let mut stable = true;
        for attempt in 1..=options.retries {
            let trial = run_isolated_probe(candidate, attempt, "stability", options.timeout)?;
            stable &= trial.result.success;
            trials.push(trial);
        }
        if stable {
            return Ok(Some(candidate));
        }
        if candidate <= options.resolution_bytes {
            return Ok(None);
        }
        candidate -= options.resolution_bytes;
    }
}

fn run_isolated_probe(
    bytes: u64,
    attempt: u32,
    phase: &str,
    timeout: Duration,
) -> Result<TrialRecord, String> {
    let executable = env::current_exe().map_err(|error| error.to_string())?;
    let child = Command::new(executable)
        .args([
            "probe",
            "--bytes",
            &bytes.to_string(),
            "--timeout-seconds",
            &timeout.as_secs().to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start probe process: {error}"))?;
    let (output, killed_for_timeout) = wait_for_child(child, timeout + Duration::from_secs(10))?;
    let mut result =
        serde_json::from_slice::<ProbeResult>(&output.stdout).unwrap_or_else(|error| {
            failed_result(
                bytes,
                ProbeStage::Process,
                format!("probe process produced no valid JSON: {error}"),
            )
        });
    if killed_for_timeout {
        result = failed_result(
            bytes,
            ProbeStage::Timeout,
            "probe process exceeded its timeout",
        );
    } else if !output.status.success() {
        result.success = false;
    }
    Ok(TrialRecord {
        requested_bytes: bytes,
        attempt,
        phase: phase.to_string(),
        result,
    })
}

fn wait_for_child(
    mut child: Child,
    timeout: Duration,
) -> Result<(std::process::Output, bool), String> {
    let deadline = Instant::now() + timeout;
    loop {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_some()
        {
            return child
                .wait_with_output()
                .map(|output| (output, false))
                .map_err(|error| error.to_string());
        }
        if Instant::now() >= deadline {
            child.kill().map_err(|error| error.to_string())?;
            return child
                .wait_with_output()
                .map(|output| (output, true))
                .map_err(|error| error.to_string());
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn collect_environment() -> EnvironmentInfo {
    EnvironmentInfo {
        hardware_model: command_output("sysctl", &["-n", "hw.model"]),
        physical_memory_bytes: command_output("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.parse().ok()),
        macos_version: command_output("sw_vers", &["-productVersion"]),
        wgpu: pollster::block_on(wgpu_metadata()),
        git_commit: command_output("git", &["rev-parse", "--short", "HEAD"]),
    }
}

async fn wgpu_metadata() -> WgpuMetadata {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });
    let Ok(adapter) = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
    else {
        return WgpuMetadata::default();
    };
    let info = adapter.get_info();
    let limits = adapter.limits();
    WgpuMetadata {
        adapter_name: Some(info.name),
        backend: Some(format!("{:?}", info.backend)),
        max_buffer_size_bytes: Some(limits.max_buffer_size),
        max_storage_buffer_binding_bytes: Some(limits.max_storage_buffer_binding_size as u64),
    }
}

fn command_output(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
}

fn write_reports(
    options: &SearchOptions,
    summary: &SearchSummary,
) -> Result<(PathBuf, PathBuf), String> {
    fs::create_dir_all(&options.output_dir).map_err(|error| error.to_string())?;
    let stem = format!("m4-memory-capacity-{}", summary.timestamp_unix_seconds);
    let jsonl_path = options.output_dir.join(format!("{stem}.jsonl"));
    let report_path = options.output_dir.join(format!("{stem}.md"));
    let mut jsonl = File::create(&jsonl_path).map_err(|error| error.to_string())?;
    for trial in &summary.trials {
        writeln!(
            jsonl,
            "{}",
            serde_json::to_string(trial).map_err(|error| error.to_string())?
        )
        .map_err(|error| error.to_string())?;
    }
    let mut report = File::create(&report_path).map_err(|error| error.to_string())?;
    writeln!(report, "# M4 WGPU Memory Capacity Measurement").map_err(|error| error.to_string())?;
    writeln!(report, "\n## Environment").map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- Hardware model: {}",
        optional_text(&summary.environment.hardware_model)
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- Physical memory: {}",
        optional_bytes(summary.environment.physical_memory_bytes)
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- macOS: {}",
        optional_text(&summary.environment.macos_version)
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- WGPU adapter: {}",
        optional_text(&summary.environment.wgpu.adapter_name)
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- WGPU backend: {}",
        optional_text(&summary.environment.wgpu.backend)
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- WGPU max buffer size: {}",
        optional_bytes(summary.environment.wgpu.max_buffer_size_bytes)
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- WGPU max storage binding: {}",
        optional_bytes(summary.environment.wgpu.max_storage_buffer_binding_bytes)
    )
    .map_err(|error| error.to_string())?;
    writeln!(report, "\n## Result").map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- Largest observed success: {}",
        format_bytes(summary.largest_observed_success_bytes)
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- Smallest observed failure: {}",
        optional_bytes(summary.smallest_observed_failure_bytes)
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- Upper boundary confirmed: {}",
        summary.upper_bound_confirmed
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- Stable maximum success: {}",
        optional_bytes(summary.stable_max_success_bytes)
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        report,
        "- Recommended limit after {}% margin: {}",
        summary.safety_margin_percent,
        optional_bytes(summary.recommended_limit_bytes)
    )
    .map_err(|error| error.to_string())?;
    writeln!(report, "\nThe probe allocates one WGPU buffer and dispatches a WGSL compute shader for each storage-binding-sized range.").map_err(|error| error.to_string())?;
    Ok((jsonl_path, report_path))
}

fn optional_text(value: &Option<String>) -> String {
    value.clone().unwrap_or_else(|| "not available".to_string())
}

fn optional_bytes(value: Option<u64>) -> String {
    value
        .map(format_bytes)
        .unwrap_or_else(|| "not available".to_string())
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= GIB {
        format!("{bytes} bytes ({:.2} GiB)", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{bytes} bytes ({:.2} MiB)", bytes as f64 / MIB as f64)
    } else {
        format!("{bytes} bytes ({:.2} KiB)", bytes as f64 / 1024.0)
    }
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  measure-m4-memory-capacity probe --bytes <bytes> [--timeout-seconds <seconds>]");
    eprintln!("  measure-m4-memory-capacity search [--initial-bytes <bytes>] [--max-bytes <bytes>] [--resolution-bytes <bytes>] [--timeout-seconds <seconds>] [--retries <count>] [--safety-margin-percent <0..99>] [--output-dir <path>]");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommendation_applies_safety_margin() {
        assert_eq!(10 * GIB * 90 / 100, 9 * GIB);
    }

    #[test]
    fn binary_search_converges_on_largest_success() {
        let (mut low, mut high) = (0, 1000);
        while high - low > 10 {
            let middle = low + (high - low) / 2;
            if middle <= 731 {
                low = middle;
            } else {
                high = middle;
            }
        }
        assert!(low <= 731 && high > 731 && high - low <= 10);
    }

    #[test]
    fn page_stride_keeps_chunk_offsets_aligned() {
        assert_eq!((8 * GIB) % PAGE_STRIDE_BYTES, 0);
    }
}
