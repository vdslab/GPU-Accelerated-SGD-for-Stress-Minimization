use crate::graph::{center_inplace, SgdParams};
use crate::schedule::Schedule;
use anyhow::{ensure, Context, Result};
use bytemuck::{Pod, Zeroable};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::num::NonZeroU64;
use std::ops::Range;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;
const MAX_PASSES_PER_SUBMISSION: usize = 256;
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

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    shader: wgpu::ShaderModule,
}

#[derive(Debug)]
pub struct GpuRunResult {
    pub initial_positions: Vec<[f64; 2]>,
    pub positions: Vec<[f64; 2]>,
    pub upload_time: Duration,
    pub compute_time: Duration,
    pub readback_time: Duration,
    pub dispatches_per_iteration: usize,
    pub submissions_per_iteration: usize,
}

impl GpuContext {
    pub fn new() -> Result<Self> {
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
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        Ok(Self {
            device,
            queue,
            adapter_name: format!("{} ({:?})", info.name, info.backend),
            shader,
        })
    }

    pub fn execute(
        &self,
        params: SgdParams,
        schedule: &Schedule,
        seed: u64,
    ) -> Result<GpuRunResult> {
        let upload_start = Instant::now();
        let n = params.positions.len();
        let h = params.pivots.len();
        ensure!(
            n <= u32::MAX as usize && h <= u32::MAX as usize,
            "グラフが大きすぎます"
        );
        ensure!(
            schedule.one_sided.len() == n * h,
            "one-sided配列の長さが不正です"
        );

        let initial_positions = params.positions.clone();
        let mut position_data: Vec<[f32; 2]> = params
            .positions
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect();
        position_data.push(CANARY);
        position_data.push(CANARY);

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
        let pair_data = if schedule.two_sided.is_empty() {
            vec![crate::schedule::GpuPair::default()]
        } else {
            schedule.two_sided.clone()
        };
        let pair_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("two-sided pairs"),
                contents: bytemuck::cast_slice(&pair_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let permutation_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pivot permutation"),
            size: (h.max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as usize;
        let slot_size = std::mem::size_of::<Uniforms>().div_ceil(alignment) * alignment;
        let invocation_count = schedule.round_count() + 1;
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
                        16,
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
        let upload_time = upload_start.elapsed();

        let compute_start = Instant::now();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x4750_555f_5350_4152);
        let mut pivot_permutation: Vec<u32> = (0..h as u32).collect();
        let mut round_order: Vec<usize> = (0..schedule.round_count()).collect();
        for (iteration, &eta) in params.etas.iter().enumerate() {
            println!("{}", iteration_log(iteration));
            pivot_permutation.shuffle(&mut rng);
            round_order.shuffle(&mut rng);
            self.queue.write_buffer(
                &permutation_buffer,
                0,
                bytemuck::cast_slice(&pivot_permutation),
            );

            let one_first = rng.random_bool(0.5);
            let mut invocations = Vec::with_capacity(invocation_count);
            if one_first {
                invocations.push(None);
            }
            invocations.extend(round_order.iter().copied().map(Some));
            if !one_first {
                invocations.push(None);
            }

            let mut uniform_bytes = vec![0u8; slot_size * invocation_count];
            for (slot, round) in invocations.iter().enumerate() {
                let (start, count) = round
                    .map(|ri| {
                        let range = schedule.round_range(ri);
                        (range.start as u32, range.len() as u32)
                    })
                    .unwrap_or((0, n as u32));
                let uniforms = Uniforms {
                    n: n as u32,
                    start,
                    count,
                    iteration: iteration as u32,
                    eta: eta as f32,
                    seed: seed as u32,
                    pivot_count: h as u32,
                    _pad: 0,
                };
                uniform_bytes[slot * slot_size..slot * slot_size + 32]
                    .copy_from_slice(bytemuck::bytes_of(&uniforms));
            }
            self.queue.write_buffer(&uniform_buffer, 0, &uniform_bytes);

            for batch_range in &submission_ranges {
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Sparse SGD invocation batch"),
                        });
                for slot in batch_range.clone() {
                    let round = invocations[slot];
                    let count = round.map(|ri| schedule.round_range(ri).len()).unwrap_or(n);
                    if count == 0 {
                        continue;
                    }
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(if round.is_some() {
                            "two-sided matching"
                        } else {
                            "one-sided owners"
                        }),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(if round.is_some() {
                        &two_pipeline
                    } else {
                        &one_pipeline
                    });
                    pass.set_bind_group(0, &bind_group, &[(slot * slot_size) as u32]);
                    pass.dispatch_workgroups((count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
                }
                self.queue.submit([encoder.finish()]);
                self.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
            }
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
        if params.center {
            center_inplace(&mut positions);
        }
        let readback_time = readback_start.elapsed();
        Ok(GpuRunResult {
            initial_positions,
            positions,
            upload_time,
            compute_time,
            readback_time,
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
    let mut ranges = Vec::with_capacity(invocation_count.div_ceil(MAX_PASSES_PER_SUBMISSION));
    let mut start = 0;
    while start < invocation_count {
        let end = start
            .saturating_add(MAX_PASSES_PER_SUBMISSION)
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
                assert!(range.len() <= MAX_PASSES_PER_SUBMISSION);
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
            let gpu_positions = context.execute(params, &schedule, seed).unwrap().positions;
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
