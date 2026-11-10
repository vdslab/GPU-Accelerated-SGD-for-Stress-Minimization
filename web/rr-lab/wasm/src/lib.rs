use experiment_common::seed::{rng_for_stream, seeded_positions, FULL_UPDATE_STREAM};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rr_gpu::gpu::{build_schedule, FullStepper, KernelConfig, MemoryMode};
use sparse_sgd_gpu::gpu::SparseStepper;
use sparse_sgd_gpu::preprocess::{prepare_with_backend, BfsBackend, Distances};
use sparse_sgd_gpu::schedule::{build_prepared_schedule, GpuPair, OneSidedEntry};
use wasm_bindgen::prelude::*;

fn to_js_err(err: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&err.to_string())
}

#[wasm_bindgen]
pub fn wasm_initial_positions(n: usize, seed: u32) -> Vec<f32> {
    let pos = seeded_positions(n, u64::from(seed), true);
    let mut flat = Vec::with_capacity(n * 2);
    for [x, y] in pos {
        flat.push(x as f32);
        flat.push(y as f32);
    }
    flat
}

#[wasm_bindgen]
pub struct WasmFullPrepared {
    dist_u16: Vec<u16>,
    dmin: f64,
    dmax: f64,
    constraint_count: usize,
}

#[wasm_bindgen]
impl WasmFullPrepared {
    pub fn take_dist_u16(&mut self) -> Vec<u16> {
        std::mem::take(&mut self.dist_u16)
    }

    pub fn dmin(&self) -> f64 {
        self.dmin
    }

    pub fn dmax(&self) -> f64 {
        self.dmax
    }

    pub fn constraint_count(&self) -> usize {
        self.constraint_count
    }
}

#[wasm_bindgen]
pub fn wasm_prepare_full(n: usize, edge_data: &[u32]) -> Result<WasmFullPrepared, JsValue> {
    let m = edge_data.len() / 2;
    let mut edge_src = Vec::with_capacity(m);
    let mut edge_dst = Vec::with_capacity(m);
    for pair in edge_data.chunks_exact(2) {
        let u = pair[0] as usize;
        let v = pair[1] as usize;
        if u != v && u < n && v < n {
            edge_src.push(u);
            edge_dst.push(v);
        }
    }
    let graph = rr_gpu::graph::Graph::from_edges(n, edge_src, edge_dst);
    let mut outcome =
        rr_gpu::graph::bfs::compute_apsp_cpu(&graph).map_err(to_js_err)?;
    outcome.dist_u16.truncate(n * n);
    Ok(WasmFullPrepared {
        dist_u16: outcome.dist_u16,
        dmin: if outcome.dmin.is_finite() {
            outcome.dmin
        } else {
            1.0
        },
        dmax: outcome.dmax,
        constraint_count: outcome.constraint_count,
    })
}

#[wasm_bindgen]
pub struct WasmSparsePrepared {
    pivots: Vec<u32>,
    one_sided: Vec<f32>,
    pairs: Vec<u8>,
    round_offsets: Vec<u32>,
    wmin: f64,
    wmax: f64,
    one_sided_count: usize,
    two_sided_count: usize,
    constraint_count: usize,
}

#[wasm_bindgen]
impl WasmSparsePrepared {
    pub fn take_pivots(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.pivots)
    }

    pub fn take_one_sided(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.one_sided)
    }

    pub fn take_pairs(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.pairs)
    }

    pub fn take_round_offsets(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.round_offsets)
    }

    pub fn wmin(&self) -> f64 {
        self.wmin
    }

    pub fn wmax(&self) -> f64 {
        self.wmax
    }

    pub fn one_sided_count(&self) -> usize {
        self.one_sided_count
    }

    pub fn two_sided_count(&self) -> usize {
        self.two_sided_count
    }

    pub fn constraint_count(&self) -> usize {
        self.constraint_count
    }
}

#[wasm_bindgen]
pub fn wasm_prepare_sparse(
    n: usize,
    edge_data: &[u32],
    pivots: usize,
    seed: u32,
) -> Result<WasmSparsePrepared, JsValue> {
    let edges: Vec<(usize, usize)> = edge_data
        .chunks_exact(2)
        .map(|c| (c[0] as usize, c[1] as usize))
        .collect();
    let graph = sparse_sgd_gpu::graph::Graph::try_from_edges(n, &edges).map_err(to_js_err)?;
    let mut rng = StdRng::seed_from_u64(u64::from(seed));
    let prepared = prepare_with_backend(&graph, 15, 0.1, pivots, true, &mut rng, BfsBackend::Cpu)
        .map_err(to_js_err)?;
    let pivots_u32: Vec<u32> = prepared.params.pivots.iter().map(|&p| p as u32).collect();
    let schedule = build_prepared_schedule(
        &graph,
        &prepared.params,
        prepared.one_sided,
        prepared.one_sided_count,
        u64::from(seed),
    )
    .map_err(to_js_err)?;

    let two_sided_count = schedule.two_sided.len();
    let one_sided_f32: Vec<f32> = bytemuck::cast_slice(&schedule.one_sided).to_vec();
    let pairs_u8: Vec<u8> = if schedule.two_sided.is_empty() {
        vec![0u8; std::mem::size_of::<GpuPair>()]
    } else {
        bytemuck::cast_slice(&schedule.two_sided).to_vec()
    };

    Ok(WasmSparsePrepared {
        pivots: pivots_u32,
        one_sided: one_sided_f32,
        pairs: pairs_u8,
        round_offsets: schedule.round_offsets,
        wmin: prepared.wmin,
        wmax: prepared.wmax,
        one_sided_count: prepared.one_sided_count,
        two_sided_count,
        constraint_count: prepared.constraint_count,
    })
}

#[wasm_bindgen]
pub struct WasmSelectedPivots {
    pivots: Vec<u32>,
    flat_distances: Vec<u32>,
}

#[wasm_bindgen]
impl WasmSelectedPivots {
    pub fn take_pivots(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.pivots)
    }

    pub fn take_flat_distances(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.flat_distances)
    }
}

#[wasm_bindgen]
pub fn wasm_select_pivots(
    n: usize,
    edge_data: &[u32],
    pivots: usize,
    seed: u32,
) -> Result<WasmSelectedPivots, JsValue> {
    let edges: Vec<(usize, usize)> = edge_data
        .chunks_exact(2)
        .map(|c| (c[0] as usize, c[1] as usize))
        .collect();
    let graph = sparse_sgd_gpu::graph::Graph::try_from_edges(n, &edges).map_err(to_js_err)?;
    let mut rng = StdRng::seed_from_u64(u64::from(seed));
    let distances =
        Distances::new_with_backend(&graph, pivots, &mut rng, BfsBackend::Cpu).map_err(to_js_err)?;
    Ok(WasmSelectedPivots {
        pivots: distances.pivots.into_iter().map(|p| p as u32).collect(),
        flat_distances: distances.flat_distances,
    })
}

#[wasm_bindgen]
pub struct WasmGpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[wasm_bindgen]
pub async fn create_wasm_gpu_context(
    max_storage_buffer_binding_size: u32,
    max_buffer_size: f64,
) -> Result<WasmGpuContext, JsValue> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(to_js_err)?;

    let mut limits = adapter.limits();
    if max_storage_buffer_binding_size > 0 {
        limits.max_storage_buffer_binding_size = limits
            .max_storage_buffer_binding_size
            .min(max_storage_buffer_binding_size);
    }
    if max_buffer_size > 0.0 {
        limits.max_buffer_size = limits.max_buffer_size.min(max_buffer_size as u64);
    }

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("rr-lab-wasm device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(to_js_err)?;

    Ok(WasmGpuContext { device, queue })
}

async fn readback_f32_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    float_count: usize,
) -> Result<js_sys::Float32Array, JsValue> {
    let slice = buffer.slice(..(float_count * 4) as u64);
    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    let _ = device.poll(wgpu::PollType::Poll);
    receiver
        .await
        .map_err(|e| to_js_err(format!("Buffer map channel cancelled: {e}")))?
        .map_err(to_js_err)?;

    let data = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);
    let out = js_sys::Float32Array::from(&floats[..float_count]);
    drop(data);
    buffer.unmap();
    Ok(out)
}

#[wasm_bindgen]
pub struct WasmFullEngine {
    stepper: FullStepper,
    n: usize,
}

#[wasm_bindgen]
impl WasmFullEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(
        ctx: &WasmGpuContext,
        n: u32,
        dist_u16: &[u16],
        initial_positions: &[f32],
        seed: u32,
    ) -> Result<WasmFullEngine, JsValue> {
        let mut padded_dist: Vec<u16>;
        let dist_slice: &[u16] = if dist_u16.len().is_multiple_of(2) {
            dist_u16
        } else {
            padded_dist = dist_u16.to_vec();
            padded_dist.push(0);
            &padded_dist
        };
        let kernel = KernelConfig {
            tile_size: 128,
            workgroup_size: 64,
            memory_mode: MemoryMode::Shared,
            outer_shuffle: true,
            inner_shuffle: true,
            phase_chunks: 1,
        };
        let stepper = FullStepper::new(
            &ctx.device,
            &ctx.queue,
            kernel,
            n,
            dist_slice,
            initial_positions,
            u64::from(seed),
        )
        .map_err(to_js_err)?;
        Ok(WasmFullEngine {
            stepper,
            n: n as usize,
        })
    }

    pub async fn epoch(
        &mut self,
        ctx: &WasmGpuContext,
        eta: f32,
    ) -> Result<js_sys::Float32Array, JsValue> {
        self.stepper.step_epoch(&ctx.device, &ctx.queue, eta);
        readback_f32_buffer(&ctx.device, self.stepper.download_buffer(), self.n * 2).await
    }

    pub fn reset(&mut self, ctx: &WasmGpuContext) {
        self.stepper.reset(&ctx.queue);
    }
}

#[wasm_bindgen]
pub struct WasmSparseEngine {
    stepper: SparseStepper,
    n: usize,
}

#[wasm_bindgen]
impl WasmSparseEngine {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &WasmGpuContext,
        n: usize,
        pivots: &[u32],
        one_sided_f32: &[f32],
        pairs_u8: &[u8],
        round_offsets: &[u32],
        initial_positions: &[f32],
        seed: u32,
    ) -> Result<WasmSparseEngine, JsValue> {
        let one_sided: &[OneSidedEntry] = bytemuck::try_cast_slice(one_sided_f32)
            .map_err(|e| to_js_err(format!("Invalid one_sided slice: {e}")))?;
        let pair_count = round_offsets.last().copied().unwrap_or(0) as usize;
        let pair_bytes = &pairs_u8[..pair_count * std::mem::size_of::<GpuPair>()];
        let two_sided: &[GpuPair] = bytemuck::try_cast_slice(pair_bytes)
            .map_err(|e| to_js_err(format!("Invalid pairs slice: {e}")))?;

        let stepper = SparseStepper::new(
            &ctx.device,
            &ctx.queue,
            n,
            pivots,
            one_sided,
            two_sided,
            round_offsets,
            initial_positions,
            u64::from(seed),
            1,
        )
        .map_err(to_js_err)?;
        Ok(WasmSparseEngine { stepper, n })
    }

    pub async fn epoch(
        &mut self,
        ctx: &WasmGpuContext,
        eta: f32,
    ) -> Result<js_sys::Float32Array, JsValue> {
        self.stepper.step_epoch(&ctx.device, &ctx.queue, eta);
        readback_f32_buffer(&ctx.device, self.stepper.download_buffer(), self.n * 2).await
    }

    pub fn reset(&mut self, ctx: &WasmGpuContext) {
        self.stepper.reset(&ctx.queue);
    }
}

#[wasm_bindgen]
pub fn wasm_hash32(value: u32) -> u32 {
    let mut x = value;
    x = ((x >> 16) ^ x).wrapping_mul(0x045d_9f3b);
    x = ((x >> 16) ^ x).wrapping_mul(0x045d_9f3b);
    (x >> 16) ^ x
}

#[wasm_bindgen]
pub fn wasm_shuffle_pivot(index: u32, key: u32, count: u32) -> u32 {
    if count <= 1 {
        return 0;
    }
    let mut remaining = count - 1;
    let mut bits = 0u32;
    while remaining != 0 {
        bits += 1;
        remaining >>= 2;
    }
    let mask = (1u32 << bits) - 1;
    let mut value = index;
    loop {
        let mut left = value >> bits;
        let mut right = value & mask;
        for round in 0..4u32 {
            let next = left
                ^ (wasm_hash32(right ^ key ^ round.wrapping_add(1).wrapping_mul(0x85eb_ca6b))
                    & mask);
            left = right;
            right = next;
        }
        value = (left << bits) | right;
        if value < count {
            return value;
        }
    }
}

fn self_round_count(length: u32) -> u32 {
    if length < 2 {
        0
    } else {
        length + length % 2 - 1
    }
}

#[wasm_bindgen]
pub fn wasm_reference_full(
    n: u32,
    dist_u16: &[u16],
    diameter: f64,
    iterations: usize,
    seed: u32,
    epsilon: f64,
) -> Vec<f32> {
    let mut p = wasm_initial_positions(n as usize, seed);
    let tile_size = 128u32;
    let blocks = n.div_ceil(tile_size);
    let block_len = |b: u32| tile_size.min(n - b * tile_size);
    let last_len = block_len(blocks - 1);
    let schedule = build_schedule(blocks);
    let mut round_order: Vec<usize> = (0..schedule.len()).collect();
    let mut rng = rng_for_stream(u64::from(seed), FULL_UPDATE_STREAM);

    let mut update = |u: usize, v: usize, eta: f32| {
        let d = f32::from(dist_u16[u * n as usize + v]);
        if d < 0.5 {
            return;
        }
        let dx = p[2 * v] - p[2 * u];
        let dy = p[2 * v + 1] - p[2 * u + 1];
        let len = (dx * dx + dy * dy).sqrt().max(1e-12);
        let mu = ((1.0 / (d * d)) * eta).min(1.0);
        let factor = mu * ((len - d) * 0.5) / len;
        let x = dx * factor;
        let y = dy * factor;
        p[2 * u] += x;
        p[2 * u + 1] += y;
        p[2 * v] -= x;
        p[2 * v + 1] -= y;
    };

    for epoch in 0..iterations {
        let eta = if iterations == 1 {
            (diameter * diameter) as f32
        } else {
            let d2 = diameter * diameter;
            (d2 * (-(d2 / epsilon).ln() * (epoch as f64) / ((iterations - 1) as f64)).exp()) as f32
        };
        round_order.shuffle(&mut rng);
        for &ri in &round_order {
            let mut cross: Vec<u32> = (0..tile_size).collect();
            cross.shuffle(&mut rng);
            let mut full: Vec<u32> = (0..self_round_count(tile_size)).collect();
            full.shuffle(&mut rng);
            let mut last: Vec<u32> = (0..self_round_count(last_len)).collect();
            last.shuffle(&mut rng);

            for &(i, j) in &schedule[ri] {
                let li = block_len(i);
                let lj = block_len(j);
                if i != j {
                    for &r in &cross {
                        for u in 0..li {
                            let v = (u + r) % tile_size;
                            if v < lj {
                                update(
                                    (i * tile_size + u) as usize,
                                    (j * tile_size + v) as usize,
                                    eta,
                                );
                            }
                        }
                    }
                } else {
                    let effective = li + (li % 2);
                    let m = effective.saturating_sub(1);
                    let active = if li == tile_size { &full } else { &last };
                    for &r in active {
                        for k in 0..(effective / 2) {
                            let (u, v) = if k == 0 {
                                (effective - 1, r)
                            } else {
                                ((r + k) % m, (r + m - k) % m)
                            };
                            if u < li && v < li {
                                update(
                                    (i * tile_size + u) as usize,
                                    (i * tile_size + v) as usize,
                                    eta,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    p
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn wasm_reference_sparse(
    n: usize,
    pivots: &[u32],
    one_sided_f32: &[f32],
    pairs_u8: &[u8],
    round_offsets: &[u32],
    weight_min: f64,
    weight_max: f64,
    iterations: usize,
    seed: u32,
    epsilon: f64,
) -> Vec<f32> {
    let mut p = wasm_initial_positions(n, seed);
    let h = pivots.len();
    let one_sided: &[OneSidedEntry] = bytemuck::cast_slice(one_sided_f32);
    let pair_count = round_offsets.last().copied().unwrap_or(0) as usize;
    let two_sided: &[GpuPair] =
        bytemuck::cast_slice(&pairs_u8[..pair_count * std::mem::size_of::<GpuPair>()]);
    let round_count = round_offsets.len().saturating_sub(1);

    let mut rng = StdRng::seed_from_u64(u64::from(seed) ^ 0x4750_555f_5350_4152);
    let mut pivot_permutation: Vec<u32> = (0..h as u32).collect();
    let mut round_order: Vec<usize> = (0..round_count).collect();

    let delta = |ux: f32, uy: f32, vx: f32, vy: f32, d: f32, salt: u32| -> [f32; 2] {
        let mut dx = vx - ux;
        let mut dy = vy - uy;
        let mut norm = (dx * dx + dy * dy).sqrt();
        if norm < 1e-12 {
            let angle = ((wasm_hash32(salt) % 6283) as f32) * 0.001;
            dx = angle.cos() * 1e-6;
            dy = angle.sin() * 1e-6;
            norm = (dx * dx + dy * dy).sqrt();
        }
        let factor = (norm - d) / (2.0 * norm);
        [dx * factor, dy * factor]
    };

    for iteration in 0..iterations {
        let eta = if iterations == 1 {
            (1.0 / weight_min) as f32
        } else {
            let eta_max = 1.0 / weight_min;
            let eta_min = epsilon / weight_max;
            let lambda = (eta_max / eta_min).ln() / ((iterations - 1) as f64);
            (eta_max * (-lambda * (iteration as f64)).exp()) as f32
        };

        pivot_permutation.shuffle(&mut rng);
        round_order.shuffle(&mut rng);
        let one_first = rng.random_bool(0.5);

        let run_one = |p: &mut [f32]| {
            let iter_u32 = iteration as u32;
            for v in 0..n {
                let mut x = p[2 * v];
                let mut y = p[2 * v + 1];
                let mut changed = false;
                let key = (v as u32) ^ seed ^ iter_u32.wrapping_mul(0x9e37_79b9);
                for k in 0..h as u32 {
                    let slot = pivot_permutation[wasm_shuffle_pivot(k, key, h as u32) as usize]
                        as usize;
                    let entry = one_sided[v * h + slot];
                    if entry.weight > 0.0 {
                        let pivot = pivots[slot] as usize;
                        let d = delta(
                            x,
                            y,
                            p[2 * pivot],
                            p[2 * pivot + 1],
                            entry.dij,
                            (v as u32) ^ (pivot as u32) ^ seed ^ iter_u32,
                        );
                        let mu = (entry.weight * eta).min(1.0);
                        x += mu * d[0];
                        y += mu * d[1];
                        changed = true;
                    }
                }
                if changed {
                    p[2 * v] = x;
                    p[2 * v + 1] = y;
                }
            }
        };

        let run_two = |p: &mut [f32]| {
            let iter_u32 = iteration as u32;
            for &r in &round_order {
                let start = round_offsets[r] as usize;
                let end = round_offsets[r + 1] as usize;
                for pair in &two_sided[start..end] {
                    let u = pair.u as usize;
                    let v = pair.v as usize;
                    let d = delta(
                        p[2 * u],
                        p[2 * u + 1],
                        p[2 * v],
                        p[2 * v + 1],
                        pair.dij,
                        pair.u ^ pair.v ^ seed ^ iter_u32,
                    );
                    let mu_u = (pair.weight_u * eta).min(1.0);
                    let mu_v = (pair.weight_v * eta).min(1.0);
                    p[2 * u] += mu_u * d[0];
                    p[2 * u + 1] += mu_u * d[1];
                    p[2 * v] -= mu_v * d[0];
                    p[2 * v + 1] -= mu_v * d[1];
                }
            }
        };

        if one_first {
            run_one(&mut p);
            run_two(&mut p);
        } else {
            run_two(&mut p);
            run_one(&mut p);
        }
    }
    p
}
