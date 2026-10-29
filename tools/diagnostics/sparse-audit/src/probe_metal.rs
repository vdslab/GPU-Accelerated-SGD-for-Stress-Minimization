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
            .new_library_with_source(&shader_source(), &CompileOptions::new())
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
        let trace = std::env::var("PROBE_TRACE").as_deref() == Ok("1");
        let batch: usize = std::env::var("PROBE_BATCH")
            .unwrap_or_else(|_| pairs.len().to_string())
            .parse()?;
        let history = self.buffer(&vec![u32::MAX; initial_positions.len()]);
        let logs = self.device.new_buffer(
            if trace { pairs.len() as u64 * 40 } else { 40 },
            MTLResourceOptions::StorageModeShared,
        );
        let initial_flat = positions.clone();
        let mut prior = initial_flat;

        let start = Instant::now();
        let mut completed_updates = 0u64;
        let mut dispatches = 0u64;
        for (iteration, eta) in etas.iter().enumerate() {
            // Fixed submission order by default, matching Atomic Full. GPU lock
            // acquisition determines update order; it is not a uniform shuffle.
            if cpu_shuffle {
                // Preserve the historical in-place CPU permutation for comparisons.
                all_pending.shuffle(rng);
            }
            unsafe {
                std::ptr::write_bytes(
                    history.contents().cast::<u32>(),
                    255,
                    initial_positions.len(),
                );
            }
            let mut iteration_passes = 0;
            let mut edge_pass_sum = 0u64;
            let mut sparse_pass_sum = 0u64;
            let mut edge_count = 0u64;
            let mut sparse_count = 0u64;
            for chunk in all_pending.chunks(batch) {
                let mut pending = chunk.to_vec();
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
                    if trace {
                        encoder.set_buffer(7, Some(&history), 0);
                        encoder.set_buffer(8, Some(&logs), 0);
                    }
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
                    for (&id, &done) in pending.iter().zip(flags) {
                        if done != 0 {
                            let p = pairs[id as usize];
                            if p.dij == 1.0 {
                                edge_pass_sum += passes;
                                edge_count += 1;
                            } else {
                                sparse_pass_sum += passes;
                                sparse_count += 1;
                            }
                        }
                    }
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
                iteration_passes += passes;
            }
            let raw = unsafe {
                std::slice::from_raw_parts(position_buffer.contents().cast::<f32>(), prior.len())
            };
            if trace {
                // Negative control changes only a completed diagnostic log, never GPU coordinates.
                if iteration == 0 && std::env::var("PROBE_CORRUPT_LOG").as_deref() == Ok("1") {
                    unsafe {
                        (*logs.contents().cast::<TraceRecord>()).before[0] += 1.0;
                    }
                }
                let records = unsafe {
                    std::slice::from_raw_parts(logs.contents().cast::<TraceRecord>(), pairs.len())
                };
                let last = unsafe {
                    std::slice::from_raw_parts(
                        history.contents().cast::<u32>(),
                        initial_positions.len(),
                    )
                };
                let mut mismatches = 0u64;
                let mut checked = 0u64;
                let mut formula_outliers = 0u64;
                let mut formula_max_scaled_error = 0.0f64;
                for (id, rec) in records.iter().enumerate() {
                    let p = pairs[id];
                    let a = rec.before.map(|v| v as f64);
                    let mut dx = a[2] - a[0];
                    let mut dy = a[3] - a[1];
                    let mut norm = (dx * dx + dy * dy).sqrt();
                    if norm < 1e-12 {
                        dx = 1e-6;
                        dy = 1e-6;
                        norm = (dx * dx + dy * dy).sqrt();
                    }
                    let rx = (norm - p.dij as f64) / (2.0 * norm) * dx;
                    let ry = (norm - p.dij as f64) / (2.0 * norm) * dy;
                    let mu = (*eta as f64 * p.weight_u as f64).min(1.0);
                    let mv = (*eta as f64 * p.weight_v as f64).min(1.0);
                    let delta = [mu * rx, mu * ry, -mv * rx, -mv * ry];
                    for j in 0..4 {
                        let err = (rec.after[j] as f64 - a[j] - delta[j]).abs()
                            / (1.0 + a[j].abs() + delta[j].abs());
                        formula_max_scaled_error = formula_max_scaled_error.max(err);
                        formula_outliers += (err > 2e-5) as u64;
                    }
                    for (side, vertex) in [p.u, p.v].into_iter().enumerate() {
                        let pred = rec.pred[side];
                        for xy in 0..2 {
                            let expected = if pred == u32::MAX {
                                prior[vertex as usize * 2 + xy]
                            } else {
                                ensure!((pred as usize) < pairs.len());
                                let pp = pairs[pred as usize];
                                ensure!(pp.u == vertex || pp.v == vertex);
                                records[pred as usize].after
                                    [if pp.u == vertex { xy } else { 2 + xy }]
                            };
                            mismatches +=
                                (rec.before[side * 2 + xy].to_bits() != expected.to_bits()) as u64;
                            checked += 1;
                        }
                    }
                }
                let mut final_mismatches = 0;
                let mut chain_links = 0usize;
                for (vertex, &lastid) in last.iter().enumerate() {
                    let mut id = lastid;
                    if id != u32::MAX {
                        let p = pairs[id as usize];
                        let side = if p.u as usize == vertex { 0 } else { 1 };
                        for xy in 0..2 {
                            final_mismatches += (raw[vertex * 2 + xy].to_bits()
                                != records[id as usize].after[side * 2 + xy].to_bits())
                                as u64;
                        }
                    }
                    let mut steps = 0;
                    while id != u32::MAX {
                        ensure!((id as usize) < pairs.len());
                        let p = pairs[id as usize];
                        ensure!(p.u as usize == vertex || p.v as usize == vertex);
                        let side = if p.u as usize == vertex { 0 } else { 1 };
                        id = records[id as usize].pred[side];
                        steps += 1;
                        ensure!(steps <= pairs.len(), "cycle");
                    }
                    chain_links += steps;
                }
                ensure!(chain_links == pairs.len() * 2, "missing chain links");
                eprintln!(
                    "TRACE {}",
                    serde_json::json!({"iteration":iteration+1,"reads_checked":checked,"read_mismatches":mismatches,"final_mismatches":final_mismatches,"chain_links":chain_links,"formula_outliers":formula_outliers,"formula_max_scaled_error":formula_max_scaled_error})
                );
                ensure!(
                    mismatches == 0 && final_mismatches == 0 && formula_outliers == 0,
                    "inconsistent position history"
                );
            }
            prior.copy_from_slice(raw);
            eprintln!(
                "ORDER {}",
                serde_json::json!({"iteration":iteration+1,"dispatches":iteration_passes,"edge_mean_retry_pass":edge_pass_sum as f64 / edge_count as f64,"other_mean_retry_pass":sparse_pass_sum as f64 / sparse_count as f64})
            );
            if verbose {
                let message = format!(
                    "Iteration {} - Updated {} pairs, dispatches={iteration_passes}",
                    iteration + 1,
                    pairs.len()
                );
                match format {
                    OutputFormat::Json => eprintln!("{message}"),
                    OutputFormat::Human => println!("{message}"),
                }
            }
        }
        let iteration_time = start.elapsed();
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

#[repr(C)]
struct TraceRecord {
    pred: [u32; 2],
    before: [f32; 4],
    after: [f32; 4],
}
fn shader_source() -> String {
    let mut s =
        include_str!("../../../../implementations/sparse/vram-lock-sparse/src/shader.metal")
            .to_string();
    if std::env::var("PROBE_FENCE").as_deref() == Ok("0") {
        s=s.replace("atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);", "");
    }
    if std::env::var("PROBE_TRACE").as_deref() == Ok("1") {
        s=s.replace("struct SparsePair {", "struct TraceRecord { uint pred[2]; float before[4]; float after[4]; };\nstruct SparsePair {");
        s=s.replace("uint3 local_id [[", "device atomic_uint* history [[buffer(7)]], device TraceRecord* logs [[buffer(8)]], uint3 local_id [[");
        s=s.replace("float2 diff = b - a;", "uint idx=pending[slot]; logs[idx].pred[0]=atomic_exchange_explicit(&history[pair.u],idx,memory_order_relaxed); logs[idx].pred[1]=atomic_exchange_explicit(&history[pair.v],idx,memory_order_relaxed); logs[idx].before[0]=a.x; logs[idx].before[1]=a.y; logs[idx].before[2]=b.x; logs[idx].before[3]=b.y; float2 diff = b - a;");
        s=s.replace("// Release fence:", "uint vertices[2]={pair.u,pair.v}; for(uint side=0;side<2;++side) for(uint xy=0;xy<2;++xy) logs[idx].after[side*2+xy]=atomic_load_explicit(&positions[vertices[side]*2+xy],memory_order_relaxed);\n// Release fence:");
    }
    s
}
