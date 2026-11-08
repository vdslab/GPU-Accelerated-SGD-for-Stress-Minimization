use crate::graph::{center_inplace, SgdParams};
use anyhow::{bail, ensure, Result};
use experiment_common::OutputFormat;
use metal::*;
use rand::{seq::SliceRandom, Rng};
use std::mem;
use std::time::{Duration, Instant};

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuPair {
    u: u32,
    v: u32,
    dij: f32,
    weight_u: f32,
    weight_v: f32,
}

pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    pub device_name: String,
}

pub struct MetalRunResult {
    pub initial_positions: Vec<[f64; 2]>,
    pub positions: Vec<[f64; 2]>,
    pub method_setup_time: Duration,
    pub upload_time: Duration,
    pub iteration_time: Duration,
    pub readback_time: Duration,
    pub postprocess_time: Duration,
    /// Planned constraints across iterations, excluding resubmitted lock attempts.
    pub attempted_updates: u64,
    pub completed_updates: u64,
    pub dispatches: u64,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
        let library = device
            .new_library_with_source(include_str!("shader.metal"), &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("Failed to compile Sparse shader: {e}"))?;
        let kernel = library
            .get_function("sparse_sgd", None)
            .map_err(|e| anyhow::anyhow!("Failed to get Sparse kernel: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| anyhow::anyhow!("Failed to create Sparse pipeline: {e}"))?;
        Ok(Self {
            device_name: device.name().to_owned(),
            queue: device.new_command_queue(),
            device,
            pipeline,
        })
    }

    fn buffer<T>(&self, data: &[T]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn execute_sgd<R: Rng + ?Sized>(
        &self,
        params: SgdParams,
        rng: &mut R,
        verbose: bool,
        format: OutputFormat,
    ) -> Result<MetalRunResult> {
        self.execute_sgd_with_cpu_shuffle(params, rng, verbose, format, false)
    }

    /// Explicit CPU permutation is opt-in; by default lock acquisition determines update order.
    pub fn execute_sgd_with_cpu_shuffle<R: Rng + ?Sized>(
        &self,
        params: SgdParams,
        rng: &mut R,
        verbose: bool,
        format: OutputFormat,
        cpu_shuffle: bool,
    ) -> Result<MetalRunResult> {
        let setup = Instant::now();
        ensure!(
            !params.pairs.is_empty() && !params.etas.is_empty(),
            "Empty Sparse workload"
        );
        ensure!(
            params.positions.len() <= (u32::MAX as usize / 2),
            "Too many vertices"
        );
        ensure!(
            params.pairs.len() <= u32::MAX as usize,
            "Too many constraints"
        );
        let initial_positions = params.positions;
        ensure!(
            initial_positions
                .iter()
                .flatten()
                .all(|x| (*x as f32).is_finite()),
            "Invalid initial coordinates"
        );
        let pairs = params
            .pairs
            .iter()
            .map(|p| {
                ensure!(
                    p.u < p.v && p.v < initial_positions.len(),
                    "Invalid Sparse endpoints"
                );
                ensure!(
                    (p.dij as f32).is_finite() && p.dij as f32 > 0.0,
                    "Invalid target distance"
                );
                ensure!(
                    [p.weight_u, p.weight_v]
                        .iter()
                        .all(|w| (*w as f32).is_finite() && *w >= 0.0),
                    "Invalid directional weight"
                );
                Ok(GpuPair {
                    u: p.u as u32,
                    v: p.v as u32,
                    dij: p.dij as f32,
                    weight_u: p.weight_u as f32,
                    weight_v: p.weight_v as f32,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let etas = params
            .etas
            .iter()
            .map(|&eta| eta as f32)
            .collect::<Vec<_>>();
        ensure!(
            etas.iter().all(|e| e.is_finite() && *e > 0.0),
            "Invalid learning rate"
        );
        let positions = initial_positions
            .iter()
            .flatten()
            .map(|&x| x as f32)
            .collect::<Vec<_>>();
        let mut all_pending = (0..pairs.len() as u32).collect::<Vec<_>>();
        let method_setup_time = setup.elapsed();

        let upload = Instant::now();
        let position_buffer = self.buffer(&positions);
        let pair_buffer = self.buffer(&pairs);
        let lock_buffer = self.buffer(&vec![0u32; initial_positions.len()]);
        let pending_buffer = self.buffer(&all_pending);
        let completed_buffer = self.buffer(&vec![0u32; pairs.len()]);
        let upload_time = upload.elapsed();

        let mut trace = experiment_common::trace::IterationTrace::from_env();
        if let Some(trace) = &mut trace {
            trace.record(0, Duration::ZERO, || {
                positions
                    .chunks_exact(2)
                    .map(|p| [p[0] as f64, p[1] as f64])
                    .collect()
            });
        }
        let initial_trace_overhead = trace.as_ref().map(|t| t.overhead()).unwrap_or_default();
        let start = Instant::now();
        let mut completed_updates = 0u64;
        let mut dispatches = 0u64;
        for (iteration, eta) in etas.iter().enumerate() {
            let trace_started = trace.as_ref().map(|_| Instant::now());
            // Fixed submission order by default, matching Atomic Full. GPU lock
            // acquisition determines update order; it is not a uniform shuffle.
            if cpu_shuffle {
                // Preserve the historical in-place CPU permutation for comparisons.
                all_pending.shuffle(rng);
            }
            let mut pending = all_pending.clone();
            let mut passes = 0;
            while !pending.is_empty() {
                // The previous command is complete before shared buffers are overwritten.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        pending.as_ptr(),
                        pending_buffer.contents().cast::<u32>(),
                        pending.len(),
                    );
                    std::ptr::write_bytes(
                        completed_buffer.contents().cast::<u32>(),
                        0,
                        pending.len(),
                    );
                }
                let count = pending.len() as u32;
                let command = self.queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.pipeline);
                for (slot, buffer) in [
                    &position_buffer,
                    &pair_buffer,
                    &lock_buffer,
                    &pending_buffer,
                    &completed_buffer,
                ]
                .iter()
                .enumerate()
                {
                    encoder.set_buffer(slot as u64, Some(buffer), 0);
                }
                encoder.set_bytes(
                    5,
                    mem::size_of::<u32>() as u64,
                    (&count as *const u32).cast(),
                );
                encoder.set_bytes(6, mem::size_of::<f32>() as u64, (eta as *const f32).cast());
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: (count as u64).min(65535),
                        height: (count as u64).div_ceil(65535),
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                ensure!(
                    command.status() == MTLCommandBufferStatus::Completed,
                    "Metal command failed at iteration {}",
                    iteration + 1
                );
                dispatches += 1;
                let flags = unsafe {
                    std::slice::from_raw_parts(
                        completed_buffer.contents().cast::<u32>(),
                        pending.len(),
                    )
                };
                let before = pending.len();
                pending = pending
                    .iter()
                    .zip(flags)
                    .filter_map(|(&pair, &done)| (done == 0).then_some(pair))
                    .collect();
                completed_updates += (before - pending.len()) as u64;
                passes += 1;
                if !pending.is_empty() && (pending.len() == before || passes >= 1024) {
                    bail!("Sparse lock retries stalled: iteration={}, remaining={} (no successful result emitted)", iteration + 1, pending.len());
                }
            }
            if let (Some(trace), Some(started)) = (&mut trace, trace_started) {
                trace.record(iteration + 1, started.elapsed(), || unsafe {
                    std::slice::from_raw_parts(
                        position_buffer.contents().cast::<f32>(),
                        positions.len(),
                    )
                    .chunks_exact(2)
                    .map(|p| [p[0] as f64, p[1] as f64])
                    .collect()
                });
            }
            if verbose {
                let message = format!(
                    "Iteration {} - Updated {} pairs, dispatches={passes}",
                    iteration + 1,
                    pairs.len()
                );
                match format {
                    OutputFormat::Json => eprintln!("{message}"),
                    OutputFormat::Human => println!("{message}"),
                }
            }
        }
        let iteration_time = start.elapsed().saturating_sub(
            trace
                .as_ref()
                .map(|t| t.overhead().saturating_sub(initial_trace_overhead))
                .unwrap_or_default(),
        );
        let readback = Instant::now();
        let raw = unsafe {
            std::slice::from_raw_parts(position_buffer.contents().cast::<f32>(), positions.len())
        };
        let mut positions = raw
            .chunks_exact(2)
            .map(|p| [p[0] as f64, p[1] as f64])
            .collect::<Vec<_>>();
        ensure!(
            positions.iter().flatten().all(|x| x.is_finite()),
            "Non-finite final coordinates"
        );
        let readback_time = readback.elapsed();
        let postprocess = Instant::now();
        if params.center {
            center_inplace(&mut positions);
        }
        let postprocess_time = postprocess.elapsed();
        Ok(MetalRunResult {
            initial_positions,
            positions,
            method_setup_time,
            upload_time,
            iteration_time,
            readback_time,
            postprocess_time,
            attempted_updates: pairs.len() as u64 * etas.len() as u64,
            completed_updates,
            dispatches,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cpu_reference::apply_constraint, graph::EdgeInfo};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn moving_shared_endpoint_tracks_serial_recurrence() {
        let context = MetalContext::new().unwrap();
        let count = 4096;
        let eta = 0.0001;
        let mut expected = 0.0f32;
        for _ in 0..count {
            let diff = 3.0 - expected;
            let r = ((diff - 1.0) / (2.0 * diff)) * diff;
            expected += eta as f32 * r;
        }
        for seed in 0..20 {
            let mut positions = vec![[3.0, 0.0]; count + 1];
            positions[0] = [0.0, 0.0];
            let pairs = (1..=count)
                .map(|v| EdgeInfo {
                    u: 0,
                    v,
                    dij: 1.0,
                    weight_u: 1.0,
                    weight_v: 0.0,
                })
                .collect();
            let params = SgdParams {
                positions,
                pairs,
                etas: vec![eta],
                pivots: vec![0],
                center: false,
            };
            let r = context
                .execute_sgd_with_cpu_shuffle(
                    params,
                    &mut StdRng::seed_from_u64(seed),
                    false,
                    OutputFormat::Json,
                    seed % 2 == 1,
                )
                .unwrap();
            assert_eq!(r.completed_updates, count as u64);
            assert!(
                (r.positions[0][0] - expected as f64).abs() < 0.0001,
                "{} != {}",
                r.positions[0][0],
                expected
            );
            assert_eq!(r.positions[0][1], 0.0);
            assert!(r.positions[1..].iter().all(|&p| p == [3.0, 0.0]));
        }
    }
    #[test]
    fn gpu_directional_updates_match_cpu_reference() {
        let context = MetalContext::new().unwrap();
        for (wu, wv, eta) in [
            (1.0, 1.0, 0.25),
            (0.2, 0.8, 0.5),
            (0.0, 1.0, 0.5),
            (1.0, 0.0, 0.5),
            (10.0, 0.2, 0.5),
        ] {
            let pair = EdgeInfo {
                u: 0,
                v: 1,
                dij: 2.0,
                weight_u: wu,
                weight_v: wv,
            };
            let initial = vec![[0.0, 0.0], [4.0, 1.0]];
            let mut expected = initial.clone();
            apply_constraint(&mut expected, pair, eta, &mut StdRng::seed_from_u64(0));
            let run = context
                .execute_sgd(
                    SgdParams {
                        positions: initial,
                        pairs: vec![pair],
                        etas: vec![eta],
                        pivots: vec![0],
                        center: false,
                    },
                    &mut StdRng::seed_from_u64(0),
                    false,
                    OutputFormat::Json,
                )
                .unwrap();
            for (actual, expected) in run
                .positions
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
            {
                assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
            }
            assert_eq!(run.completed_updates, 1);
        }
    }

    #[test]
    fn contention_completes_each_pair_once_and_preserves_fixed_endpoint() {
        let context = MetalContext::new().unwrap();
        let n = 4097;
        let run = context
            .execute_sgd(
                SgdParams {
                    positions: (0..n).map(|i| [i as f64 + 1.0, 0.0]).collect(),
                    pairs: (1..n)
                        .map(|v| EdgeInfo {
                            u: 0,
                            v,
                            dij: 1.0,
                            weight_u: 0.0,
                            weight_v: 1.0,
                        })
                        .collect(),
                    etas: vec![0.5; 3],
                    pivots: vec![0],
                    center: false,
                },
                &mut StdRng::seed_from_u64(0),
                false,
                OutputFormat::Json,
            )
            .unwrap();
        assert_eq!(run.attempted_updates, 4096 * 3);
        assert_eq!(run.completed_updates, run.attempted_updates);
        assert_eq!(run.positions[0], [1.0, 0.0]);
        for (v, p) in run.positions.iter().enumerate().skip(1) {
            // The same fixed pivot is read each time: distance-1 contracts by 3/4.
            let expected = 2.0 + (v as f64 - 1.0) * 0.75f64.powi(3);
            assert!(
                (p[0] - expected).abs() < 0.001,
                "pair {v}: {} != {expected}",
                p[0]
            );
        }
    }

    #[test]
    fn two_dimensional_dispatch_updates_every_pair_including_last() {
        let context = MetalContext::new().unwrap();
        let count = 65536;
        let run = context
            .execute_sgd(
                SgdParams {
                    positions: (0..count).flat_map(|_| [[0.0, 0.0], [4.0, 0.0]]).collect(),
                    pairs: (0..count)
                        .map(|i| EdgeInfo {
                            u: 2 * i,
                            v: 2 * i + 1,
                            dij: 2.0,
                            weight_u: 1.0,
                            weight_v: 1.0,
                        })
                        .collect(),
                    etas: vec![1.0],
                    pivots: vec![0],
                    center: false,
                },
                &mut StdRng::seed_from_u64(0),
                false,
                OutputFormat::Json,
            )
            .unwrap();
        assert_eq!(run.completed_updates, count as u64);
        assert_eq!(run.dispatches, 1);
        for pair in run.positions.chunks_exact(2) {
            assert_eq!(pair, &[[1.0, 0.0], [3.0, 0.0]]);
        }
    }

    #[test]
    fn invalid_endpoints_and_non_finite_input_are_rejected_before_dispatch() {
        let context = MetalContext::new().unwrap();
        for (positions, v) in [
            (vec![[0.0, 0.0], [1.0, 0.0]], 0),
            (vec![[f64::NAN, 0.0], [1.0, 0.0]], 1),
        ] {
            let result = context.execute_sgd(
                SgdParams {
                    positions,
                    pairs: vec![EdgeInfo {
                        u: 0,
                        v,
                        dij: 1.0,
                        weight_u: 1.0,
                        weight_v: 1.0,
                    }],
                    etas: vec![1.0],
                    pivots: vec![0],
                    center: false,
                },
                &mut StdRng::seed_from_u64(0),
                false,
                OutputFormat::Json,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn gpu_handles_coincident_positions_and_centers_output() {
        let context = MetalContext::new().unwrap();
        let run = context
            .execute_sgd(
                SgdParams {
                    positions: vec![[2.0, 3.0]; 2],
                    pairs: vec![EdgeInfo {
                        u: 0,
                        v: 1,
                        dij: 1.0,
                        weight_u: 1.0,
                        weight_v: 1.0,
                    }],
                    etas: vec![1.0],
                    pivots: vec![0],
                    center: true,
                },
                &mut StdRng::seed_from_u64(0),
                false,
                OutputFormat::Json,
            )
            .unwrap();
        assert!(run.positions.iter().flatten().all(|x| x.is_finite()));
        for axis in 0..2 {
            assert!((run.positions[0][axis] + run.positions[1][axis]).abs() < 1e-6);
        }
        assert!((run.positions[0][0] - run.positions[1][0]).abs() > 0.1);
    }
}
