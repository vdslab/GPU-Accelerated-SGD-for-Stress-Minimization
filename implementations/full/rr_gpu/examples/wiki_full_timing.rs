//! Paired iteration timing of production Full RR and the 16-chunk prototype.
#[path = "../src/graph.rs"]
mod graph;
#[path = "../src/gpu.rs"]
mod production;
#[path = "wiki_full_cause/gpu.rs"]
mod prototype;

use anyhow::Result;
use experiment_common::OutputFormat;
use serde_json::json;
use std::path::Path;

fn copy_params(params: &graph::SgdParams) -> graph::SgdParams {
    params.clone()
}

fn exact_stress(positions: &[[f32; 2]], pairs: &[graph::EdgeInfo]) -> f64 {
    pairs.iter().map(|pair| {
        let dx = positions[pair.u][0] as f64 - positions[pair.v][0] as f64;
        let dy = positions[pair.u][1] as f64 - positions[pair.v][1] as f64;
        let error = (dx * dx + dy * dy).sqrt() - pair.dij;
        pair.wij * error * error
    }).sum()
}

fn main() -> Result<()> {
    let input = std::env::args().nth(1).expect("input path required");
    let graph = graph::Graph::from_mtx(Path::new(&input))?;
    let current = production::GpuContext::new(production::KernelConfig::default())?;
    let split = prototype::GpuContext::new(prototype::KernelConfig {
        phase_chunks: 16,
        ..prototype::KernelConfig::default()
    })?;
    let schedule = production::build_schedule((graph.node_size as u32).div_ceil(1024));
    for seed in 0..3 {
        let params = graph.prepare_sgd_params(15, 0.1, true, seed);
        let initial_sha = experiment_common::positions_sha256_f64(&params.positions);
        // First pair warms both execution paths. Subsequent pairs alternate order.
        for repetition in 0..6 {
            let order = if (repetition + seed) % 2 == 0 { [1, 16] } else { [16, 1] };
            for chunks in order {
                let input_params = copy_params(&params);
                let (positions, iteration_time, device_time, dispatches) = if chunks == 1 {
                    let result = current.execute_sgd(input_params, &schedule, seed,
                                                     false, OutputFormat::Json)?;
                    (result.positions, result.iteration_time,
                     result.gpu_device_time, result.dispatches)
                } else {
                    let result = split.execute_sgd(input_params, &schedule, seed,
                                                   false, OutputFormat::Json)?;
                    (result.positions, result.iteration_time,
                     result.gpu_device_time, result.dispatches)
                };
                println!("{}", json!({
                    "seed": seed, "repetition": repetition, "warmup": repetition == 0,
                    "chunks": chunks, "initial_sha": initial_sha,
                    "sgd_total_ms": iteration_time.as_secs_f64() * 1000.0,
                    "mean_iteration_ms": iteration_time.as_secs_f64() * 1000.0 / 15.0,
                    "gpu_device_total_ms": device_time.map(|t| t.as_secs_f64() * 1000.0),
                    "dispatches": dispatches,
                    "stress": exact_stress(&positions, &params.pairs),
                    "adapter": current.adapter_name,
                }));
            }
        }
    }
    Ok(())
}
