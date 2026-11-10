#[cfg(not(target_arch = "wasm32"))]
#[path = "../../../../shared/gpu-timing/wgpu.rs"]
mod timing;
#[cfg(not(target_arch = "wasm32"))]
use crate::graph::{center_inplace, SgdParams};
#[cfg(not(target_arch = "wasm32"))]
use crate::schedule::Schedule;
#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context;
use anyhow::{ensure, Result};
use bytemuck::{Pod, Zeroable};
#[cfg(not(target_arch = "wasm32"))]
use experiment_common::OutputFormat;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::num::NonZeroU64;
use std::ops::Range;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(test)]
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 128;
const BANK_ENTRIES: usize = 33_554_432; // 256 MiB / 8 bytes; matches shader BANK_SHIFT/MASK
const MAX_DISPATCHES_PER_SUBMISSION: usize = 256;
const CANARY: [f32; 2] = [123_456.0, -654_321.0];

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    n: u32,
    start: u32,
    count: u32,
    iteration: u32,
    eta: f32,
    seed: u32,
    pivot_count: u32,
    _pad: u32,
}

#[cfg(not(target_arch = "wasm32"))]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
    shader: wgpu::ShaderModule,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct GpuRunResult {
    pub initial_positions: Vec<[f64; 2]>,
    pub positions: Vec<[f64; 2]>,
    pub upload_time: Duration,
    pub preparation_time: Duration,
    pub data_setup_time: Duration,
    pub time_to_result: Duration,
    pub result_ready_monotonic_ms: f64,
    pub iteration_times: Option<Vec<Duration>>,
    pub compute_time: Duration,
    pub readback_time: Duration,
    pub postprocess_time: Duration,
    pub gpu_device_time: Option<Duration>,
    /// Per-iteration capacity bound for delayed diagnostics; exact for production.
    pub dispatches_per_iteration: usize,
    /// Per-iteration capacity bound for delayed diagnostics; exact for production.
    pub submissions_per_iteration: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuContext {
    /// Independent per-vertex pivot permutations used by the production CLI.
    pub fn new_with_pivot_shuffle() -> Result<Self> {
        let mut context = Self::new()?;
        let source = include_str!("shader.wgsl").replace(
            "(k * stride + rotation) % uniforms.pivot_count",
            "shuffle_pivot(k, vertex ^ uniforms.seed ^ (uniforms.iteration * 0x9e3779b9u), uniforms.pivot_count)",
        );
        let source = format!("{source}\n{}", include_str!("shader_permutation.wgsl"));
        context.shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("independent pivot permutation"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        Ok(context)
    }

    pub fn new() -> Result<Self> {
        let (_instance, adapter, device, queue) =
            experiment_common::gpu::init_gpu_context(wgpu::PowerPreference::HighPerformance)?;
        let info = adapter.get_info();
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        Ok(Self {
            device,
            queue,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            shader,
        })
    }

    pub fn execute(
        &self,
        params: SgdParams,
        schedule: &Schedule,
        seed: u64,
        verbose: bool,
        output_format: OutputFormat,
    ) -> Result<GpuRunResult> {
        self.execute_with_phase_chunks(params, schedule, seed, verbose, output_format, 1)
    }

    /// Diagnostic control: each constraint is still visited exactly once per epoch.
    /// Chunk count is global, never selected by graph family or degree.
    pub fn execute_with_phase_chunks(
        &self,
        params: SgdParams,
        schedule: &Schedule,
        seed: u64,
        verbose: bool,
        output_format: OutputFormat,
        phase_chunks: usize,
    ) -> Result<GpuRunResult> {
        self.execute_diagnostic(
            params,
            schedule,
            seed,
            verbose,
            output_format,
            phase_chunks,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_diagnostic(
        &self,
        params: SgdParams,
        schedule: &Schedule,
        seed: u64,
        verbose: bool,
        output_format: OutputFormat,
        phase_chunks: usize,
        vertex_stride: bool,
    ) -> Result<GpuRunResult> {
        self.execute_quality_diagnostic(
            params,
            schedule,
            seed,
            verbose,
            output_format,
            phase_chunks,
            vertex_stride,
            0,
        )
    }

    /// Delay interleaving until a fixed iteration, identically for all graphs.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_quality_diagnostic(
        &self,
        params: SgdParams,
        schedule: &Schedule,
        seed: u64,
        verbose: bool,
        output_format: OutputFormat,
        phase_chunks: usize,
        vertex_stride: bool,
        interleave_start: usize,
    ) -> Result<GpuRunResult> {
        let execution_started = Instant::now();
        let n = params.positions.len();
        let h = params.pivots.len();
        ensure!(
            n > 0 && h > 0 && n <= u32::MAX as usize && h <= u32::MAX as usize,
            "グラフが大きすぎます"
        );
        ensure!(
            h <= 256,
            "pivot count ({h}) exceeds workgroup shared memory capacity (256)"
        );
        ensure!(
            schedule.one_sided.len() == n * h,
            "one-sided配列の長さが不正です"
        );

        // Constraints are already encoded in schedule; release before GPU allocation.
        drop(params.pairs);
        let limits = self.device.limits();
        ensure!(
            schedule.one_sided.len() <= 4 * BANK_ENTRIES,
            "one-sided entries exceeds 4-bank 1 GiB limit"
        );
        for (name, bytes) in [
            (
                "one-sided bank",
                (schedule.one_sided.len().min(BANK_ENTRIES).max(1)
                    * std::mem::size_of::<crate::schedule::OneSidedEntry>()) as u64,
            ),
            (
                "two-sided",
                std::mem::size_of_val(schedule.two_sided.as_slice())
                    .max(std::mem::size_of::<crate::schedule::GpuPair>()) as u64,
            ),
            ("positions", (n as u64 + 2) * 8),
        ] {
            ensure!(
                bytes <= limits.max_buffer_size
                    && bytes <= u64::from(limits.max_storage_buffer_binding_size),
                "{name} requires {bytes} bytes, exceeding this adapter's storage buffer limit"
            );
        }
        ensure!(
            n.div_ceil(WORKGROUP_SIZE as usize)
                <= limits.max_compute_workgroups_per_dimension as usize,
            "vertex dispatch exceeds adapter workgroup limit"
        );
        // The experimental affine permutation uses a u32 product.
        ensure!(
            !vertex_stride || h <= 65535,
            "diagnostic stride requires at most 65535 pivots"
        );
        let initial_positions = params.positions;
        let mut position_data: Vec<[f32; 2]> = initial_positions
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect();
        position_data.push(CANARY);
        position_data.push(CANARY);

        let data_setup_time = execution_started.elapsed();
        let preparation_started = Instant::now();
        let positions_buffer = timing::buffer(
            &self.device,
            Some("positions with canaries"),
            (bytemuck::cast_slice::<_, u8>(&position_data)).len(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let get_bank_slice = |bank_idx: usize| -> &[crate::schedule::OneSidedEntry] {
            let start = (bank_idx * BANK_ENTRIES).min(schedule.one_sided.len());
            let end = ((bank_idx + 1) * BANK_ENTRIES).min(schedule.one_sided.len());
            &schedule.one_sided[start..end]
        };
        let bank0 = get_bank_slice(0);
        let bank1 = get_bank_slice(1);
        let bank2 = get_bank_slice(2);
        let bank3 = get_bank_slice(3);

        let bank0_buffer = timing::buffer(
            &self.device,
            Some("one-sided bank 0"),
            (bytemuck::cast_slice::<_, u8>(bank0)).len().max(8),
            wgpu::BufferUsages::STORAGE,
        );
        let bank1_buffer = timing::buffer(
            &self.device,
            Some("one-sided bank 1"),
            (bytemuck::cast_slice::<_, u8>(bank1)).len().max(8),
            wgpu::BufferUsages::STORAGE,
        );
        let bank2_buffer = timing::buffer(
            &self.device,
            Some("one-sided bank 2"),
            (bytemuck::cast_slice::<_, u8>(bank2)).len().max(8),
            wgpu::BufferUsages::STORAGE,
        );
        let bank3_buffer = timing::buffer(
            &self.device,
            Some("one-sided bank 3"),
            (bytemuck::cast_slice::<_, u8>(bank3)).len().max(8),
            wgpu::BufferUsages::STORAGE,
        );
        let dummy_pair = [crate::schedule::GpuPair::default()];
        let pair_data = if schedule.two_sided.is_empty() {
            &dummy_pair[..]
        } else {
            &schedule.two_sided[..]
        };
        let pair_buffer = timing::buffer(
            &self.device,
            Some("two-sided pairs"),
            (bytemuck::cast_slice::<_, u8>(pair_data)).len(),
            wgpu::BufferUsages::STORAGE,
        );
        let permutation_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pivot permutation"),
            size: (2 * h.max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pivot_ids: Vec<u32> = params.pivots.iter().map(|&p| p as u32).collect();

        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as usize;
        let slot_size = std::mem::size_of::<Uniforms>().div_ceil(alignment) * alignment;
        ensure!(phase_chunks > 0, "phase chunks must be positive");
        let phase_chunks = phase_chunks.min(h);
        let invocation_count = schedule.round_count() + phase_chunks;
        let submission_ranges = submission_ranges(invocation_count);
        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dynamic iteration uniforms"),
            size: (slot_size * invocation_count.max(1)) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("positions readback"),
            size: positions_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sparse SGD layout"),
                entries: &[
                    buffer_layout(0, wgpu::BufferBindingType::Uniform, true, 32),
                    buffer_layout(
                        1,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                        8,
                    ),
                    buffer_layout(
                        2,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                        8,
                    ),
                    buffer_layout(
                        3,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                        4,
                    ),
                    buffer_layout(
                        4,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                        20,
                    ),
                    buffer_layout(
                        5,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                        8,
                    ),
                    buffer_layout(
                        6,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                        8,
                    ),
                    buffer_layout(
                        7,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                        8,
                    ),
                ],
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sparse SGD bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bank0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: permutation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pair_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bank1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bank2_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: bank3_buffer.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sparse SGD pipeline layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
        let one_pipeline = self.create_pipeline(&pipeline_layout, "one_sided_phase");
        let two_pipeline = self.create_pipeline(&pipeline_layout, "two_sided_round");

        // Warm up Mesa NIR->DXIL compilation for both pipelines before timed iteration loops
        experiment_common::gpu::run_warmup_dispatch(
            &self.device,
            &self.queue,
            &one_pipeline,
            &bind_group,
            &[0],
        )?;
        experiment_common::gpu::run_warmup_dispatch(
            &self.device,
            &self.queue,
            &two_pipeline,
            &bind_group,
            &[0],
        )?;
        self.queue
            .write_buffer(&positions_buffer, 0, bytemuck::cast_slice(&position_data));
        self.queue.submit([]);
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        let mut timer =
            timing::PassTimer::new(&self.device, params.etas.len() * submission_ranges.len());
        let preparation_time = preparation_started.elapsed();
        let upload_started = Instant::now();
        self.queue
            .write_buffer(&positions_buffer, 0, bytemuck::cast_slice(&position_data));
        self.queue
            .write_buffer(&bank0_buffer, 0, bytemuck::cast_slice(bank0));
        if !bank1.is_empty() {
            self.queue
                .write_buffer(&bank1_buffer, 0, bytemuck::cast_slice(bank1));
        }
        if !bank2.is_empty() {
            self.queue
                .write_buffer(&bank2_buffer, 0, bytemuck::cast_slice(bank2));
        }
        if !bank3.is_empty() {
            self.queue
                .write_buffer(&bank3_buffer, 0, bytemuck::cast_slice(bank3));
        }
        self.queue
            .write_buffer(&pair_buffer, 0, bytemuck::cast_slice(pair_data));
        self.queue.write_buffer(
            &permutation_buffer,
            (h * 4) as u64,
            bytemuck::cast_slice(&pivot_ids),
        );
        self.queue.submit([]);
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        let upload_time = upload_started.elapsed();

        let mut trace = experiment_common::trace::IterationTrace::from_env();
        if let Some(trace) = &mut trace {
            trace.record(0, Duration::ZERO, || {
                position_data[..n]
                    .iter()
                    .map(|p| [p[0] as f64, p[1] as f64])
                    .collect()
            });
        }
        let initial_trace_overhead = trace.as_ref().map(|t| t.overhead()).unwrap_or_default();
        let compute_start = Instant::now();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x4750_555f_5350_4152);
        let mut pivot_permutation: Vec<u32> = (0..h as u32).collect();
        let mut round_order: Vec<usize> = (0..schedule.round_count()).collect();
        let mut invocations = Vec::with_capacity(invocation_count);
        let mut uniform_bytes = vec![0u8; slot_size * invocation_count];
        for (iteration, &eta) in params.etas.iter().enumerate() {
            let trace_started = trace.as_ref().map(|_| Instant::now());
            if verbose {
                let message = iteration_log(iteration);
                match output_format {
                    OutputFormat::Json => eprintln!("{message}"),
                    OutputFormat::Human => println!("{message}"),
                }
            }
            pivot_permutation.shuffle(&mut rng);
            round_order.shuffle(&mut rng);
            self.queue.write_buffer(
                &permutation_buffer,
                0,
                bytemuck::cast_slice(&pivot_permutation),
            );

            let one_first = rng.random_bool(0.5);
            invocations.clear();
            let phase_chunks = if iteration < interleave_start {
                1
            } else {
                phase_chunks
            };
            for chunk in 0..phase_chunks {
                let start = chunk * h / phase_chunks;
                let end = (chunk + 1) * h / phase_chunks;
                let one = (None, start, end - start);
                if one_first {
                    invocations.push(one);
                }
                let begin_round = chunk * round_order.len() / phase_chunks;
                let end_round = (chunk + 1) * round_order.len() / phase_chunks;
                invocations.extend(
                    round_order[begin_round..end_round]
                        .iter()
                        .copied()
                        .map(|r| (Some(r), 0, 0)),
                );
                if !one_first {
                    invocations.push(one);
                }
            }

            for (slot, (round, pivot_start, pivot_len)) in invocations.iter().enumerate() {
                let (start, count) = round
                    .map(|ri| {
                        let range = schedule.round_range(ri);
                        (range.start as u32, range.len() as u32)
                    })
                    .unwrap_or((*pivot_start as u32, *pivot_len as u32));
                let uniforms = Uniforms {
                    n: n as u32,
                    start,
                    count,
                    iteration: iteration as u32,
                    eta: eta as f32,
                    seed: seed as u32,
                    pivot_count: h as u32,
                    _pad: u32::from(vertex_stride),
                };
                uniform_bytes[slot * slot_size..slot * slot_size + 32]
                    .copy_from_slice(bytemuck::bytes_of(&uniforms));
            }
            self.queue.write_buffer(&uniform_buffer, 0, &uniform_bytes);

            for batch_range in &submission_ranges {
                if batch_range.start >= invocations.len() {
                    break;
                }
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Sparse SGD invocation batch"),
                        });
                {
                    // Dispatches retain their order and storage dependencies in
                    // a compute pass; no per-round pass/encoder boundary is needed.
                    // Keep batches bounded to limit command recording overhead.
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Sparse SGD ordered dispatches"),
                        timestamp_writes: timer.next(),
                    });
                    for (slot, invocation) in invocations
                        .iter()
                        .enumerate()
                        .take(batch_range.end)
                        .skip(batch_range.start)
                    {
                        let round = invocation.0;
                        let count = round.map(|ri| schedule.round_range(ri).len()).unwrap_or(n);
                        if count == 0 {
                            continue;
                        }
                        pass.set_pipeline(if round.is_some() {
                            &two_pipeline
                        } else {
                            &one_pipeline
                        });
                        pass.set_bind_group(0, &bind_group, &[(slot * slot_size) as u32]);
                        pass.dispatch_workgroups((count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
                    }
                }
                self.queue.submit([encoder.finish()]);
            }
            // Queue submissions are ordered; retain the bounded command buffers,
            // but wait once per epoch before reusing its uniforms/permutation.
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .context("GPU iteration failed")?;
            timer.end_iteration();
            if let (Some(trace), Some(started)) = (&mut trace, trace_started) {
                trace.record(iteration + 1, started.elapsed(), || {
                    trace_positions(
                        &self.device,
                        &self.queue,
                        &positions_buffer,
                        &download_buffer,
                        n,
                    )
                });
            }
        }
        let compute_time = compute_start.elapsed().saturating_sub(
            trace
                .as_ref()
                .map(|t| t.overhead().saturating_sub(initial_trace_overhead))
                .unwrap_or_default(),
        );

        let readback_start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sparse SGD readback"),
            });
        encoder.copy_buffer_to_buffer(
            &positions_buffer,
            0,
            &download_buffer,
            0,
            positions_buffer.size(),
        );
        self.queue.submit([encoder.finish()]);
        let slice = download_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let mapped = slice.get_mapped_range();
        let values: &[[f32; 2]] = bytemuck::cast_slice(&mapped);
        ensure!(
            values[n] == CANARY && values[n + 1] == CANARY,
            "GPU境界外書き込みを検出しました"
        );
        let mut positions: Vec<[f64; 2]> = values[..n]
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64])
            .collect();
        drop(mapped);
        download_buffer.unmap();
        ensure!(
            positions.iter().flatten().all(|v| v.is_finite()),
            "GPU結果にNaNまたはInfがあります"
        );
        let readback_time = readback_start.elapsed();
        let time_to_result = execution_started.elapsed();
        let result_ready_monotonic_ms = experiment_common::monotonic_time_ms();
        let iteration_times = timer.finish(&self.device, &self.queue)?;
        let gpu_device_time = iteration_times
            .as_ref()
            .map(|times| times.iter().copied().sum());
        let postprocess_start = Instant::now();
        if params.center {
            center_inplace(&mut positions);
        }
        let postprocess_time = postprocess_start.elapsed();
        Ok(GpuRunResult {
            initial_positions,
            positions,
            upload_time,
            preparation_time,
            data_setup_time,
            time_to_result,
            result_ready_monotonic_ms,
            iteration_times,
            compute_time,
            readback_time,
            postprocess_time,
            gpu_device_time,
            dispatches_per_iteration: invocation_count,
            submissions_per_iteration: submission_ranges.len(),
        })
    }

    fn create_pipeline(
        &self,
        layout: &wgpu::PipelineLayout,
        entry_point: &'static str,
    ) -> wgpu::ComputePipeline {
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(layout),
                module: &self.shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }
}

fn submission_ranges(invocation_count: usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::with_capacity(invocation_count.div_ceil(MAX_DISPATCHES_PER_SUBMISSION));
    let mut start = 0;
    while start < invocation_count {
        let end = start
            .saturating_add(MAX_DISPATCHES_PER_SUBMISSION)
            .min(invocation_count);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

#[cfg(not(target_arch = "wasm32"))]
fn iteration_log(iteration: usize) -> String {
    format!("Iteration: {}", iteration + 1)
}

fn buffer_layout(
    binding: u32,
    ty: wgpu::BufferBindingType,
    dynamic: bool,
    min_size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: dynamic,
            min_binding_size: NonZeroU64::new(min_size),
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, SgdParams};
    use crate::schedule::build_schedule;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn uniform_has_wgsl_compatible_size() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);
    }

    #[test]
    fn iteration_log_is_one_based() {
        assert_eq!(iteration_log(0), "Iteration: 1");
        assert_eq!(iteration_log(2), "Iteration: 3");
    }

    #[test]
    fn submission_ranges_are_contiguous_and_bounded() {
        for (invocations, expected_batches) in [(1, 1), (256, 1), (257, 2), (38_626, 151)] {
            let ranges = submission_ranges(invocations);
            assert_eq!(ranges.len(), expected_batches);
            assert_eq!(ranges.first().unwrap().start, 0);
            assert_eq!(ranges.last().unwrap().end, invocations);
            for (index, range) in ranges.iter().enumerate() {
                assert!(!range.is_empty());
                assert!(range.len() <= MAX_DISPATCHES_PER_SUBMISSION);
                if index > 0 {
                    assert_eq!(ranges[index - 1].end, range.start);
                }
            }
        }
        assert!(submission_ranges(0).is_empty());
    }

    #[test]
    fn shader_declares_both_entry_points_and_directional_weights() {
        let shader = include_str!("shader.wgsl");
        assert!(shader.contains("fn one_sided_phase"));
        assert!(shader.contains("fn two_sided_round"));
        assert!(shader.contains("pair.weight_u"));
        assert!(shader.contains("pair.weight_v"));
    }

    fn full_stress(graph: &Graph, positions: &[[f64; 2]]) -> f64 {
        let mut stress = 0.0;
        for u in 0..graph.node_size {
            let distances = graph.shortest_path_distances(u);
            for v in u + 1..graph.node_size {
                let dx = positions[u][0] - positions[v][0];
                let dy = positions[u][1] - positions[v][1];
                let actual = (dx * dx + dy * dy).sqrt();
                let target = distances[v] as f64;
                stress += (actual - target).powi(2) / target.powi(2);
            }
        }
        stress
    }

    #[test]
    fn delayed_interleaving_does_not_execute_unused_dispatch_slots() {
        let graph = Graph::try_from_edges(
            8,
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 0),
            ],
        )
        .unwrap();
        let seed = 91;
        let params = graph
            .prepare_sgd_params(3, 0.1, 4, true, &mut StdRng::seed_from_u64(seed))
            .unwrap();
        let schedule = build_schedule(&graph, &params, seed).unwrap();
        let copy = || SgdParams {
            positions: params.positions.clone(),
            etas: params.etas.clone(),
            pairs: params.pairs.clone(),
            pivots: params.pivots.clone(),
            center: params.center,
        };
        let context = GpuContext::new().unwrap();
        let baseline = context
            .execute(copy(), &schedule, seed, false, OutputFormat::Json)
            .unwrap();
        // Allocates extra slots but must never execute them before the switch.
        let delayed = context
            .execute_quality_diagnostic(
                copy(),
                &schedule,
                seed,
                false,
                OutputFormat::Json,
                4,
                false,
                3,
            )
            .unwrap();
        assert_eq!(baseline.positions, delayed.positions);
        let immediate = context
            .execute_quality_diagnostic(
                copy(),
                &schedule,
                seed,
                false,
                OutputFormat::Json,
                4,
                false,
                0,
            )
            .unwrap();
        let original = context
            .execute_with_phase_chunks(copy(), &schedule, seed, false, OutputFormat::Json, 4)
            .unwrap();
        assert_eq!(immediate.positions, original.positions);
        let switched = context
            .execute_quality_diagnostic(
                copy(),
                &schedule,
                seed,
                false,
                OutputFormat::Json,
                4,
                false,
                1,
            )
            .unwrap();
        assert!(switched
            .positions
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn real_gpu_pipeline_matches_cpu_quality_and_preserves_guards() {
        let cases = [
            (
                8,
                vec![
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 0),
                    (1, 5),
                ],
            ),
            (
                7,
                vec![
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 0),
                    (0, 3),
                ],
            ),
        ];
        let context = GpuContext::new().unwrap();
        for (case, (node_count, edges)) in cases.into_iter().enumerate() {
            let graph = Graph::try_from_edges(node_count, &edges).unwrap();
            let seed = 31 + case as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let params = graph
                .prepare_sgd_params(15, 0.1, 4, true, &mut rng)
                .unwrap();
            let cpu_params = SgdParams {
                etas: params.etas.clone(),
                positions: params.positions.clone(),
                pairs: params.pairs.clone(),
                pivots: params.pivots.clone(),
                center: params.center,
            };
            let cpu_positions = crate::cpu_reference::execute_sgd(cpu_params, &mut rng);
            let schedule = build_schedule(&graph, &params, seed).unwrap();
            let gpu_positions = context
                .execute(params, &schedule, seed, false, OutputFormat::Human)
                .unwrap()
                .positions;
            let cpu_stress = full_stress(&graph, &cpu_positions);
            let gpu_stress = full_stress(&graph, &gpu_positions);
            assert!(gpu_positions.iter().flatten().all(|v| v.is_finite()));
            assert!(
                (gpu_stress - cpu_stress).abs() / cpu_stress <= 0.10,
                "case={case}, CPU stress={cpu_stress}, GPU stress={gpu_stress}"
            );
        }
    }
    fn run_storage_probe(
        source: String,
        data: &[f32],
        n: u32,
        pivots: u32,
        groups: u32,
    ) -> Vec<f32> {
        let context = GpuContext::new().unwrap();
        let shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pivot permutation probe"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("diagnostic_probe"),
                compilation_options: Default::default(),
                cache: None,
            });
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of_val(data) as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        let uniforms = Uniforms {
            n,
            start: 0,
            count: n,
            iteration: 0,
            eta: 0.0,
            seed: 0,
            pivot_count: pivots,
            _pad: 0,
        };
        let uniform = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = context.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buffer, 0, &output, 0, buffer.size());
        context.queue.submit([encoder.finish()]);
        output.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let mapped = output.slice(..).get_mapped_range();
        bytemuck::cast_slice::<u8, f32>(&mapped).to_vec()
    }

    #[test]
    fn gpu_cycle_walk_visits_every_pivot_once() {
        let source = format!(
            "{}\n{}\n{}",
            include_str!("shader.wgsl"),
            include_str!("shader_permutation.wgsl"),
            r#"
@compute @workgroup_size(256)
fn diagnostic_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= uniforms.n { return; }
    positions[gid.x] = vec2<f32>(f32(shuffle_pivot(gid.x, 0u, uniforms.pivot_count)), f32(shuffle_pivot(gid.x, 0xffffffffu, uniforms.pivot_count)));
}
"#
        );
        for count in [1u32, 2, 3, 199, 200, 256, 257] {
            let actual = run_storage_probe(
                source.clone(),
                &vec![0.0; count as usize * 2],
                count,
                count,
                count.div_ceil(256),
            );
            for coordinate in 0..2 {
                let mut permutation: Vec<_> = actual
                    .chunks_exact(2)
                    .map(|p| p[coordinate] as u32)
                    .collect();
                permutation.sort_unstable();
                assert_eq!(permutation, (0..count).collect::<Vec<_>>());
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn trace_positions(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    positions: &wgpu::Buffer,
    download: &wgpu::Buffer,
    node_count: usize,
) -> Vec<[f64; 2]> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("convergence checkpoint readback"),
    });
    encoder.copy_buffer_to_buffer(positions, 0, download, 0, positions.size());
    queue.submit([encoder.finish()]);
    let slice = download.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("convergence readback poll");
    let mapped = slice.get_mapped_range();
    let values: &[[f32; 2]] = bytemuck::cast_slice(&mapped);
    let result = values[..node_count]
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64])
        .collect();
    drop(mapped);
    download.unmap();
    result
}

/// Interactive / step-by-epoch RoundRobin Sparse SGD GPU engine (shared by native and WebAssembly).
#[allow(dead_code)]
pub struct SparseStepper {
    one_pipeline: wgpu::ComputePipeline,
    two_pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>,
    positions_buffer: wgpu::Buffer,
    download_buffer: wgpu::Buffer,
    permutation_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    n: usize,
    h: usize,
    seed: u64,
    phase_chunks: usize,
    slot_size: usize,
    pair_chunk_capacity: usize,
    round_offsets: Vec<u32>,
    rng: StdRng,
    pivot_permutation: Vec<u32>,
    round_order: Vec<usize>,
    invocations: Vec<(Option<(usize, u32, u32)>, usize, usize)>,
    uniform_bytes: Vec<u8>,
    initial_position_data: Vec<[f32; 2]>,
    iteration: u32,
}

#[allow(dead_code)]
impl SparseStepper {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: usize,
        pivots: &[u32],
        one_sided: &[crate::schedule::OneSidedEntry],
        two_sided: &[crate::schedule::GpuPair],
        round_offsets: &[u32],
        initial_positions_flat: &[f32],
        seed: u64,
        phase_chunks: usize,
    ) -> Result<Self> {
        let h = pivots.len();
        ensure!(
            n > 0 && h > 0 && n <= u32::MAX as usize && h <= 1024,
            "Invalid vertex count ({n}) or pivot count ({h}); pivot count must be 1..=1024"
        );
        ensure!(
            one_sided.len() == n * h,
            "one-sided array length mismatch: got {}, expected {}",
            one_sided.len(),
            n * h
        );
        ensure!(
            one_sided.len() <= 4 * BANK_ENTRIES,
            "one-sided entries exceed 4-bank 1 GiB limit"
        );
        ensure!(phase_chunks > 0, "phase_chunks must be positive");
        let phase_chunks = phase_chunks.min(h);

        let limits = device.limits();
        let max_binding_bytes = u64::from(limits.max_storage_buffer_binding_size)
            .min(limits.max_buffer_size) as usize;
        ensure!(
            one_sided.len().min(BANK_ENTRIES).max(1)
                * std::mem::size_of::<crate::schedule::OneSidedEntry>()
                <= max_binding_bytes,
            "one-sided bank exceeds adapter storage buffer limit"
        );
        ensure!(
            (n + 2) * 8 <= max_binding_bytes,
            "positions buffer exceeds adapter storage buffer limit"
        );

        // Build shader source with independent Feistel pivot shuffle, dynamic shared memory capacity
        // (256 for h <= 256, 1024 for 256 < h <= 1024), and 2D workgroup dispatch support.
        let wg_pivot_cap = if h <= 256 { 256 } else { 1024 };
        let source = include_str!("shader.wgsl")
            .replace(
                "var<workgroup> wg_pivot_perm: array<u32, 256>;\nvar<workgroup> wg_pivot_id: array<u32, 256>;\nvar<workgroup> wg_pivot_pos: array<vec2<f32>, 256>;",
                &format!(
                    "var<workgroup> wg_pivot_perm: array<u32, {wg_pivot_cap}>;\nvar<workgroup> wg_pivot_id: array<u32, {wg_pivot_cap}>;\nvar<workgroup> wg_pivot_pos: array<vec2<f32>, {wg_pivot_cap}>;"
                ),
            )
            .replace(
                "let vertex = gid.x;",
                "let vertex = gid.x + gid.y * 65535u * WORKGROUP_SIZE;",
            )
            .replace(
                "if gid.x >= uniforms.count {\n        return;\n    }\n    let pair_index = uniforms.start + gid.x;",
                "let flat_id = gid.x + gid.y * 65535u * WORKGROUP_SIZE;\n    if flat_id >= uniforms.count {\n        return;\n    }\n    let pair_index = uniforms.start + flat_id;",
            )
            .replace(
                "(k * stride + rotation) % uniforms.pivot_count",
                "shuffle_pivot(k, vertex ^ uniforms.seed ^ (uniforms.iteration * 0x9e3779b9u), uniforms.pivot_count)",
            );
        let source = format!("{source}\n{}", include_str!("shader_permutation.wgsl"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SparseStepper shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let mut initial_position_data: Vec<[f32; 2]> = initial_positions_flat
            .chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect();
        initial_position_data.push(CANARY);
        initial_position_data.push(CANARY);

        let make_buf = |label: Option<&str>, byte_len: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label,
                size: byte_len.max(16) as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let positions_buffer = make_buf(
            Some("positions with canaries"),
            initial_position_data.len() * 8,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let get_bank_slice = |bank_idx: usize| -> &[crate::schedule::OneSidedEntry] {
            let start = (bank_idx * BANK_ENTRIES).min(one_sided.len());
            let end = ((bank_idx + 1) * BANK_ENTRIES).min(one_sided.len());
            &one_sided[start..end]
        };
        let bank0 = get_bank_slice(0);
        let bank1 = get_bank_slice(1);
        let bank2 = get_bank_slice(2);
        let bank3 = get_bank_slice(3);

        let bank0_buffer = make_buf(
            Some("one-sided bank 0"),
            std::mem::size_of_val(bank0).max(8),
            wgpu::BufferUsages::STORAGE,
        );
        let bank1_buffer = make_buf(
            Some("one-sided bank 1"),
            std::mem::size_of_val(bank1).max(8),
            wgpu::BufferUsages::STORAGE,
        );
        let bank2_buffer = make_buf(
            Some("one-sided bank 2"),
            std::mem::size_of_val(bank2).max(8),
            wgpu::BufferUsages::STORAGE,
        );
        let bank3_buffer = make_buf(
            Some("one-sided bank 3"),
            std::mem::size_of_val(bank3).max(8),
            wgpu::BufferUsages::STORAGE,
        );

        let pair_size = std::mem::size_of::<crate::schedule::GpuPair>();
        let pair_chunk_capacity = (max_binding_bytes / pair_size).max(1);
        let dummy_pair = [crate::schedule::GpuPair::default()];
        let pair_slices: Vec<&[crate::schedule::GpuPair]> = if two_sided.is_empty() {
            vec![&dummy_pair[..]]
        } else {
            two_sided.chunks(pair_chunk_capacity).collect()
        };
        let mut pair_buffers = Vec::with_capacity(pair_slices.len());
        for slice in &pair_slices {
            let buf = make_buf(
                Some("two-sided pairs"),
                std::mem::size_of_val(*slice).max(pair_size),
                wgpu::BufferUsages::STORAGE,
            );
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(slice));
            pair_buffers.push(buf);
        }

        let permutation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pivot permutation"),
            size: (2 * h.max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let round_count = round_offsets.len().saturating_sub(1);
        let max_invocations = round_count + pair_buffers.len() + phase_chunks + 4;
        let alignment = limits.min_uniform_buffer_offset_alignment as usize;
        let slot_size = std::mem::size_of::<Uniforms>().div_ceil(alignment) * alignment;

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dynamic iteration uniforms"),
            size: (slot_size * max_invocations.max(1)) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("positions readback"),
            size: (n * 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sparse SGD layout"),
            entries: &[
                buffer_layout(0, wgpu::BufferBindingType::Uniform, true, 32),
                buffer_layout(
                    1,
                    wgpu::BufferBindingType::Storage { read_only: false },
                    false,
                    8,
                ),
                buffer_layout(
                    2,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                    8,
                ),
                buffer_layout(
                    3,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                    4,
                ),
                buffer_layout(
                    4,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                    20,
                ),
                buffer_layout(
                    5,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                    8,
                ),
                buffer_layout(
                    6,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                    8,
                ),
                buffer_layout(
                    7,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                    8,
                ),
            ],
        });

        let bind_groups: Vec<wgpu::BindGroup> = pair_buffers
            .iter()
            .map(|pair_buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Sparse SGD bind group"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &uniform_buffer,
                                offset: 0,
                                size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: positions_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: bank0_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: permutation_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: pair_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: bank1_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: bank2_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: bank3_buffer.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sparse SGD pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let one_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("one_sided_phase"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("one_sided_phase"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let two_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("two_sided_round"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("two_sided_round"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        queue.write_buffer(
            &positions_buffer,
            0,
            bytemuck::cast_slice(&initial_position_data),
        );
        if !bank0.is_empty() {
            queue.write_buffer(&bank0_buffer, 0, bytemuck::cast_slice(bank0));
        }
        if !bank1.is_empty() {
            queue.write_buffer(&bank1_buffer, 0, bytemuck::cast_slice(bank1));
        }
        if !bank2.is_empty() {
            queue.write_buffer(&bank2_buffer, 0, bytemuck::cast_slice(bank2));
        }
        if !bank3.is_empty() {
            queue.write_buffer(&bank3_buffer, 0, bytemuck::cast_slice(bank3));
        }
        queue.write_buffer(
            &permutation_buffer,
            (h * 4) as u64,
            bytemuck::cast_slice(pivots),
        );
        queue.submit([]);

        let rng = StdRng::seed_from_u64(seed ^ 0x4750_555f_5350_4152);
        let pivot_permutation = (0..h as u32).collect();
        let round_order = (0..round_count).collect();

        Ok(Self {
            one_pipeline,
            two_pipeline,
            bind_groups,
            positions_buffer,
            download_buffer,
            permutation_buffer,
            uniform_buffer,
            n,
            h,
            seed,
            phase_chunks,
            slot_size,
            pair_chunk_capacity,
            round_offsets: round_offsets.to_vec(),
            rng,
            pivot_permutation,
            round_order,
            invocations: Vec::with_capacity(max_invocations),
            uniform_bytes: vec![0u8; slot_size * max_invocations.max(1)],
            initial_position_data,
            iteration: 0,
        })
    }

    pub fn reset(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.positions_buffer,
            0,
            bytemuck::cast_slice(&self.initial_position_data),
        );
        queue.submit([]);
        self.rng = StdRng::seed_from_u64(self.seed ^ 0x4750_555f_5350_4152);
        self.pivot_permutation.clear();
        self.pivot_permutation.extend(0..self.h as u32);
        let round_count = self.round_offsets.len().saturating_sub(1);
        self.round_order.clear();
        self.round_order.extend(0..round_count);
        self.iteration = 0;
    }

    pub fn step_epoch(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, eta: f32) {
        let iteration = self.iteration;
        self.iteration += 1;

        self.pivot_permutation.shuffle(&mut self.rng);
        self.round_order.shuffle(&mut self.rng);
        queue.write_buffer(
            &self.permutation_buffer,
            0,
            bytemuck::cast_slice(&self.pivot_permutation),
        );

        let one_first = self.rng.random_bool(0.5);
        self.invocations.clear();
        let phase_chunks = self.phase_chunks;
        for chunk in 0..phase_chunks {
            let start = chunk * self.h / phase_chunks;
            let end = (chunk + 1) * self.h / phase_chunks;
            let one = (None, start, end - start);
            if one_first {
                self.invocations.push(one);
            }
            let begin_round = chunk * self.round_order.len() / phase_chunks;
            let end_round = (chunk + 1) * self.round_order.len() / phase_chunks;
            for &ri in &self.round_order[begin_round..end_round] {
                let mut r_start = self.round_offsets[ri] as usize;
                let r_end = self.round_offsets[ri + 1] as usize;
                while r_start < r_end {
                    let pair_chunk = r_start / self.pair_chunk_capacity;
                    let chunk_end = ((pair_chunk + 1) * self.pair_chunk_capacity).min(r_end);
                    let local_start = (r_start - pair_chunk * self.pair_chunk_capacity) as u32;
                    let count = (chunk_end - r_start) as u32;
                    self.invocations
                        .push((Some((pair_chunk, local_start, count)), 0, 0));
                    r_start = chunk_end;
                }
            }
            if !one_first {
                self.invocations.push(one);
            }
        }

        let needed_bytes = self.invocations.len() * self.slot_size;
        if self.uniform_bytes.len() < needed_bytes {
            self.uniform_bytes.resize(needed_bytes, 0);
        }

        for (slot, (round_slice, pivot_start, pivot_len)) in self.invocations.iter().enumerate() {
            let (start, count) = round_slice
                .map(|(_, local_start, count)| (local_start, count))
                .unwrap_or((*pivot_start as u32, *pivot_len as u32));
            let uniforms = Uniforms {
                n: self.n as u32,
                start,
                count,
                iteration,
                eta,
                seed: self.seed as u32,
                pivot_count: self.h as u32,
                _pad: 0,
            };
            self.uniform_bytes[slot * self.slot_size..slot * self.slot_size + 32]
                .copy_from_slice(bytemuck::bytes_of(&uniforms));
        }
        if !self.invocations.is_empty() {
            queue.write_buffer(&self.uniform_buffer, 0, &self.uniform_bytes[..needed_bytes]);
        }

        let batch_ranges = submission_ranges(self.invocations.len());
        let num_batches = batch_ranges.len();
        for (batch_idx, batch_range) in batch_ranges.iter().enumerate() {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sparse SGD invocation batch"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Sparse SGD ordered dispatches"),
                    timestamp_writes: None,
                });
                for slot in batch_range.clone() {
                    let invocation = self.invocations[slot];
                    let (is_two, bind_idx, count) = match invocation.0 {
                        Some((pair_chunk, _, count)) => (true, pair_chunk, count as usize),
                        None => (false, 0, self.n),
                    };
                    if count == 0 {
                        continue;
                    }
                    pass.set_pipeline(if is_two {
                        &self.two_pipeline
                    } else {
                        &self.one_pipeline
                    });
                    pass.set_bind_group(
                        0,
                        &self.bind_groups[bind_idx],
                        &[(slot * self.slot_size) as u32],
                    );
                    let total_wgs = (count as u32).div_ceil(WORKGROUP_SIZE);
                    if total_wgs <= 65_535 {
                        pass.dispatch_workgroups(total_wgs, 1, 1);
                    } else {
                        pass.dispatch_workgroups(65_535, total_wgs.div_ceil(65_535), 1);
                    }
                }
            }
            if batch_idx + 1 == num_batches {
                encoder.copy_buffer_to_buffer(
                    &self.positions_buffer,
                    0,
                    &self.download_buffer,
                    0,
                    (self.n * 8) as u64,
                );
            }
            queue.submit([encoder.finish()]);
        }
    }

    pub fn download_buffer(&self) -> &wgpu::Buffer {
        &self.download_buffer
    }
}
