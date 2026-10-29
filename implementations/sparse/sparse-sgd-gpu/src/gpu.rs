use crate::graph::{center_inplace, SgdParams};
use crate::schedule::Schedule;
use anyhow::{ensure, Context, Result};
use bytemuck::{Pod, Zeroable};
use experiment_common::OutputFormat;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::num::NonZeroU64;
use std::ops::Range;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;
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
    round_mask: u32,
    _reserved: [u32; 3],
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
    shader: wgpu::ShaderModule,
    compensated: bool,
    weight_residuals: Option<Vec<f32>>,
    recenter: bool,
}

#[derive(Debug)]
pub struct GpuRunResult {
    pub initial_positions: Vec<[f64; 2]>,
    pub positions: Vec<[f64; 2]>,
    pub upload_time: Duration,
    pub compute_time: Duration,
    pub readback_time: Duration,
    pub postprocess_time: Duration,
    pub gpu_device_time: Option<Duration>,
    /// Per-iteration capacity bound for delayed diagnostics; exact for production.
    pub dispatches_per_iteration: usize,
    /// Per-iteration capacity bound for delayed diagnostics; exact for production.
    pub submissions_per_iteration: usize,
}

impl GpuContext {
    pub fn new() -> Result<Self> {
        Self::new_with_compensation(false)
    }

    /// Diagnostic global upper bound on each directional step, independent of graph.
    pub fn new_with_step_cap(cap: f32) -> Result<Self> {
        ensure!(cap > 0.0 && cap <= 1.0, "step cap must be in (0, 1]");
        let mut context = Self::new()?;
        let source = include_str!("shader.wgsl").replace(", 1.0);", &format!(", {cap:?});"));
        context.shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("global step cap diagnostic"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        Ok(context)
    }

    /// Diagnostic translation after each epoch; exact arithmetic distances are invariant.
    pub fn new_with_recentering() -> Result<Self> {
        let mut context = Self::new()?;
        let source = format!(
            "{}\n{}",
            include_str!("shader.wgsl"),
            include_str!("shader_recenter.wgsl")
        );
        context.shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("f32 recenter diagnostic"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        context.recenter = true;
        Ok(context)
    }

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

    /// Experimental f32 high/low coordinate storage; preserves the RR schedule.
    pub fn new_with_compensation(compensated: bool) -> Result<Self> {
        Self::new_with_arithmetic(u32::from(compensated))
    }

    /// Diagnostic double-single arithmetic using only f32 shader operations.
    pub fn new_with_extended_precision() -> Result<Self> {
        Self::new_with_arithmetic(2)
    }

    pub fn new_with_extended_coefficients(residuals: Vec<f32>) -> Result<Self> {
        let mut context = Self::new_with_arithmetic(3)?;
        context.weight_residuals = Some(residuals);
        Ok(context)
    }

    fn new_with_arithmetic(mode: u32) -> Result<Self> {
        let compensated = mode > 0;
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| anyhow::anyhow!("GPUアダプタが見つかりません: {e:?}"))?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Sparse SGD GPU device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .context("GPUデバイスを作成できません")?;
        let shader = device.create_shader_module(if mode == 3 {
            wgpu::include_wgsl!("shader_extended_coefficients.wgsl")
        } else if mode == 2 {
            wgpu::include_wgsl!("shader_extended.wgsl")
        } else if compensated {
            wgpu::include_wgsl!("shader_compensated.wgsl")
        } else {
            wgpu::include_wgsl!("shader.wgsl")
        });
        Ok(Self {
            device,
            queue,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            shader,
            compensated,
            weight_residuals: None,
            recenter: false,
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
        let upload_start = Instant::now();
        let position_width = if self.compensated { 4 } else { 2 };
        let position_padding = if self.recenter { 3 } else { 2 };
        let n = params.positions.len();
        let h = params.pivots.len();
        ensure!(
            n > 0 && h > 0 && n <= u32::MAX as usize && h <= u32::MAX as usize,
            "グラフが大きすぎます"
        );
        ensure!(
            schedule.one_sided.len() == n * h,
            "one-sided配列の長さが不正です"
        );

        // Constraints are already encoded in schedule; release before GPU allocation.
        drop(params.pairs);
        let limits = self.device.limits();
        for (name, bytes) in [
            (
                "one-sided",
                std::mem::size_of_val(schedule.one_sided.as_slice()) as u64,
            ),
            (
                "two-sided",
                std::mem::size_of_val(schedule.two_sided.as_slice())
                    .max(std::mem::size_of::<crate::schedule::GpuPair>()) as u64,
            ),
            (
                "positions",
                (n as u64 + position_padding) * position_width as u64 * 4,
            ),
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
        let mut position_data = Vec::<f32>::with_capacity((n + 2) * position_width);
        for p in &initial_positions {
            let hi = [p[0] as f32, p[1] as f32];
            position_data.extend_from_slice(&hi);
            if self.compensated {
                position_data.extend_from_slice(&[
                    (p[0] - f64::from(hi[0])) as f32,
                    (p[1] - f64::from(hi[1])) as f32,
                ]);
            }
        }
        for _ in 0..2 {
            position_data.extend_from_slice(&CANARY);
            if self.compensated {
                position_data.extend_from_slice(&[0.0, 0.0]);
            }
        }

        if self.recenter {
            position_data.extend_from_slice(&[0.0, 0.0]);
        }
        let positions_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("positions with canaries"),
                contents: bytemuck::cast_slice(&position_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let one_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("one-sided dense CSR"),
                contents: bytemuck::cast_slice(&schedule.one_sided),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let dummy_pair = [crate::schedule::GpuPair::default()];
        let pair_data = if schedule.two_sided.is_empty() {
            &dummy_pair[..]
        } else {
            &schedule.two_sided[..]
        };
        let pair_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("two-sided pairs"),
                contents: bytemuck::cast_slice(pair_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let permutation_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pivot permutation"),
            size: (2 * h.max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pivot_ids: Vec<u32> = params.pivots.iter().map(|&p| p as u32).collect();
        self.queue.write_buffer(
            &permutation_buffer,
            (h * 4) as u64,
            bytemuck::cast_slice(&pivot_ids),
        );

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

        let residual_buffer = if let Some(values) = &self.weight_residuals {
            ensure!(
                values.len() == n * h + 2 * schedule.two_sided.len(),
                "coefficient residual count mismatch"
            );
            ensure!(
                !vertex_stride,
                "extended coefficients do not support diagnostic stride"
            );
            let bytes = std::mem::size_of_val(values.as_slice()) as u64;
            ensure!(
                bytes <= limits.max_buffer_size
                    && bytes <= u64::from(limits.max_storage_buffer_binding_size),
                "coefficient residual buffer exceeds device limit"
            );
            Some(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("coefficient residuals"),
                        contents: bytemuck::cast_slice(values),
                        usage: wgpu::BufferUsages::STORAGE,
                    }),
            )
        } else {
            None
        };
        let mut layout_entries = vec![
            buffer_layout(0, wgpu::BufferBindingType::Uniform, true, 48),
            buffer_layout(
                1,
                wgpu::BufferBindingType::Storage { read_only: false },
                false,
                (position_width * 4) as u64,
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
                24,
            ),
        ];
        if residual_buffer.is_some() {
            layout_entries.push(buffer_layout(
                5,
                wgpu::BufferBindingType::Storage { read_only: true },
                false,
                4,
            ));
        }
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sparse SGD layout"),
                entries: &layout_entries,
            });
        let mut bind_entries = vec![
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
                resource: one_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: permutation_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: pair_buffer.as_entire_binding(),
            },
        ];
        if let Some(buffer) = &residual_buffer {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: buffer.as_entire_binding(),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sparse SGD bind group"),
            layout: &layout,
            entries: &bind_entries,
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
        let recenter_pipelines = self.recenter.then(|| {
            (
                self.create_pipeline(&pipeline_layout, "reduce_center"),
                self.create_pipeline(&pipeline_layout, "subtract_center"),
            )
        });
        let upload_time = upload_start.elapsed();

        let compute_start = Instant::now();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x4750_555f_5350_4152);
        let mut pivot_permutation: Vec<u32> = (0..h as u32).collect();
        let mut round_order: Vec<usize> = (0..schedule.round_count()).collect();
        let mut invocations = Vec::with_capacity(invocation_count);
        let mut uniform_bytes = vec![0u8; slot_size * invocation_count];
        for (iteration, &eta) in params.etas.iter().enumerate() {
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
                    round_mask: u32::MAX,
                    _reserved: [0; 3],
                    _pad: if self.weight_residuals.is_some() {
                        ((eta - f64::from(eta as f32)) as f32).to_bits()
                    } else {
                        u32::from(vertex_stride)
                    },
                };
                uniform_bytes[slot * slot_size..slot * slot_size + std::mem::size_of::<Uniforms>()]
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
                        timestamp_writes: None,
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
            if let Some((reduce, subtract)) = &recenter_pipelines {
                let mut encoder = self.device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_bind_group(0, &bind_group, &[0]);
                    pass.set_pipeline(reduce);
                    pass.dispatch_workgroups(1, 1, 1);
                    pass.set_pipeline(subtract);
                    pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
                }
                self.queue.submit([encoder.finish()]);
            }
            // Queue submissions are ordered; retain the bounded command buffers,
            // but wait once per epoch before reusing its uniforms/permutation.
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .context("GPU iteration failed")?;
        }
        let compute_time = compute_start.elapsed();

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
        let values: &[f32] = bytemuck::cast_slice(&mapped);
        ensure!(
            values[n * position_width..(n + 2) * position_width]
                == position_data[n * position_width..(n + 2) * position_width],
            "GPU境界外書き込みを検出しました"
        );
        let mut positions: Vec<[f64; 2]> = values[..n * position_width]
            .chunks_exact(position_width)
            .map(|p| {
                let low = if self.compensated {
                    [f64::from(p[2]), f64::from(p[3])]
                } else {
                    [0.0, 0.0]
                };
                [f64::from(p[0]) + low[0], f64::from(p[1]) + low[1]]
            })
            .collect();
        drop(mapped);
        download_buffer.unmap();
        ensure!(
            positions.iter().flatten().all(|v| v.is_finite()),
            "GPU結果にNaNまたはInfがあります"
        );
        let readback_time = readback_start.elapsed();
        let postprocess_start = Instant::now();
        if params.center {
            center_inplace(&mut positions);
        }
        let postprocess_time = postprocess_start.elapsed();
        Ok(GpuRunResult {
            initial_positions,
            positions,
            upload_time,
            compute_time,
            readback_time,
            postprocess_time,
            gpu_device_time: None,
            dispatches_per_iteration: invocation_count + if self.recenter { 2 } else { 0 },
            submissions_per_iteration: submission_ranges.len() + usize::from(self.recenter),
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
        assert_eq!(std::mem::size_of::<Uniforms>(), 48);
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
                label: Some("runtime arithmetic probe"),
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
            round_mask: u32::MAX,
            _reserved: [0; 3],
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
    fn compensated_shader_retains_unit_below_f32_spacing() {
        let source = format!(
            "{}\n{}",
            include_str!("shader_compensated.wgsl"),
            r#"
@compute @workgroup_size(1)
fn diagnostic_probe() {
    let initial = positions[0];
    let first = compensated_add(initial, positions[1].xy);
    positions[0] = first;
    positions[1] = compensated_add(first, -initial.xy);
}
"#
        );
        let actual = run_storage_probe(
            source,
            &[16777216.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            2,
            1,
            1,
        );
        assert_eq!(actual, &[16777216.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn extended_shader_preserves_runtime_sum_and_product_residuals() {
        let source = format!(
            "{}\n{}",
            include_str!("shader_extended.wgsl"),
            r#"
@compute @workgroup_size(1)
fn diagnostic_probe() {
    let a = positions[0];
    let b = positions[1];
    let first = point_add(a, vec4<f32>(b.x, 0.0, 0.0, 0.0));
    let product = ds_mul(a.yw, b.yw);
    positions[0] = first;
    positions[1] = vec4<f32>(point_add(first, -a).x, product.x, product.y, 0.0);
}
"#
        );
        let factor = f32::from_bits(0x3f800001);
        let actual = run_storage_probe(
            source,
            &[16777216.0, factor, 0.0, 0.0, 1.0, factor, 0.0, 0.0],
            2,
            1,
            1,
        );
        assert_eq!(actual[2], 1.0, "TwoSum residual lost: {actual:?}");
        assert_eq!(actual[4], 1.0);
        assert!(
            (f64::from(actual[5]) + f64::from(actual[6]) - f64::from(factor).powi(2)).abs() < 1e-15,
            "product residual lost: {actual:?}"
        );
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

    #[test]
    fn recenter_translates_all_vertices_and_preserves_guards() {
        let graph =
            Graph::try_from_edges(257, &(1..257).map(|v| (v - 1, v)).collect::<Vec<_>>()).unwrap();
        let mut params = graph
            .prepare_sgd_params(15, 0.1, 4, false, &mut StdRng::seed_from_u64(0))
            .unwrap();
        params.positions = (0..257).map(|v| [1000.0 + v as f64, 3.0]).collect();
        params.etas = vec![0.0];
        let schedule = build_schedule(&graph, &params, 0).unwrap();
        let result = GpuContext::new_with_recentering()
            .unwrap()
            .execute(params, &schedule, 0, false, OutputFormat::Json)
            .unwrap();
        for (v, point) in result.positions.iter().enumerate() {
            assert_eq!(*point, [v as f64 - 128.0, 0.0]);
        }
    }

    #[test]
    fn step_cap_controls_displacement_and_one_preserves_original() {
        let graph = Graph::try_from_edges(2, &[(0, 1)]).unwrap();
        let params = || SgdParams {
            positions: vec![[0.0, 0.0], [4.0, 0.0]],
            etas: vec![100.0],
            pairs: vec![crate::graph::EdgeInfo {
                u: 0,
                v: 1,
                dij: 1.0,
                weight_u: 1.0,
                weight_v: 1.0,
            }],
            pivots: vec![0, 1],
            center: false,
        };
        let schedule = build_schedule(&graph, &params(), 0).unwrap();
        let original = GpuContext::new()
            .unwrap()
            .execute(params(), &schedule, 0, false, OutputFormat::Json)
            .unwrap();
        let unchanged = GpuContext::new_with_step_cap(1.0)
            .unwrap()
            .execute(params(), &schedule, 0, false, OutputFormat::Json)
            .unwrap();
        assert_eq!(original.positions, unchanged.positions);
        let capped = GpuContext::new_with_step_cap(0.9)
            .unwrap()
            .execute(params(), &schedule, 0, false, OutputFormat::Json)
            .unwrap();
        assert!((capped.positions[0][0] - 1.35).abs() < 1e-6);
        assert!((capped.positions[1][0] - 2.65).abs() < 1e-6);
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
}
