#[path = "../../../../shared/gpu-timing/metal.rs"]
mod timing;
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
    pub preparation_time: Duration,
    pub time_to_result: Duration,
    pub result_ready_monotonic_ms: f64,
    pub iteration_times: Option<Vec<Duration>>,
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
            .new_library_with_source(&std::fs::read_to_string(std::env::var("ATOMIC_PROBE_SHADER")?)?, &CompileOptions::new())
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
        let execution_started = Instant::now();
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

        let locks = vec![0u32; initial_positions.len()];
        let completed = vec![0u32; pairs.len()];
        let preparation_started = Instant::now();
        let position_buffer = self.device.new_buffer(
            mem::size_of_val(positions.as_slice()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let pair_buffer = self.device.new_buffer(
            mem::size_of_val(pairs.as_slice()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let lock_buffer = self.device.new_buffer(
            mem::size_of_val(locks.as_slice()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let pending_buffer = self.device.new_buffer(
            mem::size_of_val(all_pending.as_slice()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let completed_buffer = self.device.new_buffer(
            mem::size_of_val(completed.as_slice()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let preparation_time = preparation_started.elapsed();
        let upload = Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(
                positions.as_ptr().cast::<u8>(),
                position_buffer.contents().cast::<u8>(),
                mem::size_of_val(positions.as_slice()),
            );
            std::ptr::copy_nonoverlapping(
                pairs.as_ptr().cast::<u8>(),
                pair_buffer.contents().cast::<u8>(),
                mem::size_of_val(pairs.as_slice()),
            );
            std::ptr::copy_nonoverlapping(
                locks.as_ptr().cast::<u8>(),
                lock_buffer.contents().cast::<u8>(),
                mem::size_of_val(locks.as_slice()),
            );
            std::ptr::copy_nonoverlapping(
                all_pending.as_ptr().cast::<u8>(),
                pending_buffer.contents().cast::<u8>(),
                mem::size_of_val(all_pending.as_slice()),
            );
            std::ptr::copy_nonoverlapping(
                completed.as_ptr().cast::<u8>(),
                completed_buffer.contents().cast::<u8>(),
                mem::size_of_val(completed.as_slice()),
            );
        }
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
        let mut iteration_times = Some(Vec::with_capacity(etas.len()));
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
            let mut gpu_time = Duration::ZERO;
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
                if let Some(duration) = timing::command_time(command) {
                    gpu_time += duration;
                } else {
                    iteration_times = None;
                }
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
            if let Some(times) = &mut iteration_times {
                times.push(gpu_time);
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
        let time_to_result = execution_started.elapsed();
        let result_ready_monotonic_ms = experiment_common::monotonic_time_ms();
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
            preparation_time,
            time_to_result,
            result_ready_monotonic_ms,
            iteration_times,
            iteration_time,
            readback_time,
            postprocess_time,
            attempted_updates: pairs.len() as u64 * etas.len() as u64,
            completed_updates,
            dispatches,
        })
    }
}

