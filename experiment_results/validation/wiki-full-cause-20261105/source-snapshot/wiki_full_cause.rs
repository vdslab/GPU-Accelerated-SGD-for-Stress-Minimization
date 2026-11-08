//! Controlled Full SGD diagnostics. Production update paths remain in src/.
#[path = "wiki_full_cause/gpu.rs"]
mod gpu;
#[path = "../src/graph.rs"]
mod graph;
#[path = "../../baseline-sgd-non-gpu/src/algorithm.rs"]
mod cpu;

use anyhow::{ensure, Result};
use experiment_common::seed::{rng_for_stream, FULL_UPDATE_STREAM};
use experiment_common::OutputFormat;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde_json::json;
use std::path::Path;
use std::time::Instant;

fn self_rounds(length: usize) -> usize {
    if length < 2 { 0 } else { length + length % 2 - 1 }
}

// Match production RNG consumption, including unused partial-block permutations.
fn inner_permutation(tile: usize, last: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut permutation = Vec::new();
    for count in [tile, self_rounds(tile), self_rounds(last)] {
        let start = permutation.len();
        permutation.extend(0..count);
        permutation[start..].shuffle(rng);
    }
    permutation
}

fn visit_matching(mut visit: impl FnMut(usize, usize), n: usize, tile: usize,
                  i: usize, j: usize, r: usize) {
    let li = tile.min(n - i * tile);
    let lj = tile.min(n - j * tile);
    if i != j {
        for local in 0..li {
            let other = (local + r) % tile;
            if other < lj { visit(i * tile + local, j * tile + other); }
        }
    } else {
        let effective = li + li % 2;
        let rounds = effective - 1;
        for slot in 0..effective / 2 {
            let (u, v) = if slot == 0 { (effective - 1, r) }
                else { ((r + slot) % rounds, (r + rounds - slot) % rounds) };
            if u < li && v < li { visit(i * tile + u, i * tile + v); }
        }
    }
}

// chunks=1 reproduces the production order. More chunks interleave the SAME
// matching sequences across outer rounds, preserving every pair and every eta.
fn run_rr(mut update: impl FnMut(usize, usize, f64), n: usize, etas: &[f64],
          seed: u64, tile: usize, chunks: usize, remap: &str) {
    let blocks = n.div_ceil(tile);
    let last = n - (blocks - 1) * tile;
    let schedule = gpu::build_schedule(blocks as u32);
    let mut order: Vec<_> = (0..schedule.len()).collect();
    let mut rng = rng_for_stream(seed, FULL_UPDATE_STREAM);
    let mut map_rng = StdRng::seed_from_u64(seed ^ 0x766572746578);
    let mut vertices: Vec<_> = (0..n).collect();
    if remap == "fixed" { vertices.shuffle(&mut map_rng); }
    for &eta in etas {
        if remap == "epoch" { vertices.shuffle(&mut map_rng); }
        order.shuffle(&mut rng);
        let permutations: Vec<_> = order.iter()
            .map(|_| inner_permutation(tile, last, &mut rng)).collect();
        for chunk in 0..chunks {
            for (&round, permutation) in order.iter().zip(&permutations) {
                for &(i, j) in &schedule[round] {
                    let (i, j) = (i as usize, j as usize);
                    let li = tile.min(n - i * tile);
                    let (offset, count) = if i != j { (0, tile) }
                        else if li == tile { (tile, self_rounds(li)) }
                        else { (tile + self_rounds(tile), self_rounds(li)) };
                    let start = chunk * count / chunks;
                    let end = (chunk + 1) * count / chunks;
                    for &r in &permutation[offset + start..offset + end] {
                        visit_matching(|u, v| update(vertices[u], vertices[v], eta),
                                       n, tile, i, j, r);
                    }
                }
            }
        }
    }
}

macro_rules! update_function {
    ($name:ident, $float:ty) => {
        fn $name(positions: &mut [[$float; 2]], u: usize, v: usize,
                 d: $float, eta: $float) -> bool {
            let dx = positions[v][0] - positions[u][0];
            let dy = positions[v][1] - positions[u][1];
            let distance = (dx * dx + dy * dy).sqrt();
            let near_zero = distance < 1e-12;
            let current = distance.max(1e-12);
            let mu = ((1.0 / (d * d)) * eta).min(1.0);
            let rx = ((current - d) / 2.0) * (dx / current);
            let ry = ((current - d) / 2.0) * (dy / current);
            positions[u][0] += mu * rx;
            positions[u][1] += mu * ry;
            positions[v][0] -= mu * rx;
            positions[v][1] -= mu * ry;
            near_zero
        }
    }
}
update_function!(update_f32, f32);
update_function!(update_f64, f64);

fn stress(positions: &[[f64; 2]], distances: &[f32]) -> f64 {
    let n = positions.len();
    let mut sum = 0.0;
    for u in 0..n {
        for v in u + 1..n {
            let dx = positions[v][0] - positions[u][0];
            let dy = positions[v][1] - positions[u][1];
            let d = distances[u * n + v] as f64;
            let relative = ((dx * dx + dy * dy).sqrt() - d) / d;
            sum += relative * relative;
        }
    }
    sum
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let input = &args[1];
    let mode = &args[2];
    let seed: u64 = args[3].parse()?;
    let tile: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(1024);
    let chunks: usize = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(1);
    let remap = args.get(6).map(String::as_str).unwrap_or("none");
    ensure!(tile > 0 && chunks > 0, "tile and chunks must be positive");
    let graph = graph::Graph::from_mtx(Path::new(input))?;
    let params = graph.prepare_sgd_params(15, 0.1, true, seed);
    let n = params.positions.len();
    let initial_sha = experiment_common::positions_sha256_f64(&params.positions);
    let mut distances = vec![0.0_f32; n * n];
    for pair in &params.pairs {
        distances[pair.u * n + pair.v] = pair.dij as f32;
        distances[pair.v * n + pair.u] = pair.dij as f32;
    }
    ensure!(params.pairs.len() == n * (n - 1) / 2, "input must be connected");
    let mut zero_distances = 0u64;
    let mut updates = 0u64;
    let started = Instant::now();
    let positions: Vec<[f64; 2]> = match mode.as_str() {
        "cpu64" => cpu::execute_sgd(params, seed, false, OutputFormat::Json),
        "gpu" | "gpu-global" => {
            let config = gpu::KernelConfig {
                tile_size: tile as u32, workgroup_size: 1024,
                phase_chunks: chunks as u32,
                memory_mode: if mode == "gpu-global" { gpu::MemoryMode::Global }
                    else { gpu::MemoryMode::Shared },
                ..gpu::KernelConfig::default()
            };
            let context = gpu::GpuContext::new(config)?;
            context.execute_sgd(params, &gpu::build_schedule(n.div_ceil(tile) as u32),
                                seed, false, OutputFormat::Json)?.positions.into_iter()
                .map(|p| [p[0] as f64, p[1] as f64]).collect()
        }
        "rr64" => {
            let mut positions = params.positions.clone();
            run_rr(|u, v, eta| {
                zero_distances += update_f64(&mut positions, u, v,
                                            distances[u * n + v] as f64, eta) as u64;
                updates += 1;
            }, n, &params.etas, seed, tile, chunks, remap);
            positions
        }
        "audit" => {
            let mut counts = vec![0u8; n * n];
            run_rr(|u, v, _| {
                assert_ne!(u, v);
                let (u, v) = (u.min(v), u.max(v));
                counts[u * n + v] += 1;
                updates += 1;
            }, n, &params.etas[..1], seed, tile, chunks, remap);
            for u in 0..n { for v in u + 1..n { ensure!(counts[u*n+v] == 1,
                "pair ({u},{v}) visited {} times", counts[u*n+v]); } }
            params.positions.clone()
        }
        "rr32" | "cpu32" => {
            let mut positions: Vec<_> = params.positions.iter()
                .map(|p| [p[0] as f32, p[1] as f32]).collect();
            let mut update = |u: usize, v: usize, eta: f64| {
                zero_distances += update_f32(&mut positions, u, v,
                                            distances[u * n + v], eta as f32) as u64;
                updates += 1;
            };
            if mode == "rr32" {
                run_rr(&mut update, n, &params.etas, seed, tile, chunks, remap);
            } else {
                let mut pairs = params.pairs;
                let mut rng = rng_for_stream(seed, FULL_UPDATE_STREAM);
                for &eta in &params.etas {
                    pairs.shuffle(&mut rng);
                    for pair in &pairs { update(pair.u, pair.v, eta); }
                }
            }
            positions.into_iter().map(|p| [p[0] as f64, p[1] as f64]).collect()
        }
        _ => anyhow::bail!("unknown mode"),
    };
    let elapsed = started.elapsed().as_secs_f64();
    println!("{}", json!({"mode":mode,"seed":seed,"tile":tile,"chunks":chunks,
        "remap":remap,"n":n,"initial_sha":initial_sha,"updates":updates,
        "near_zero_updates":zero_distances,"stress":stress(&positions,&distances),
        "diagnostic_seconds":elapsed}));
    Ok(())
}
