//! Isolate CPU update-order variation with fixed preprocessing and exact stress.
use anyhow::{ensure, Result};
use experiment_common::{
    evaluate_f64, positions_sha256_f64, OutputFormat, StressMode, StressResult,
};
use rand::{rngs::StdRng, SeedableRng};
use sparse_sgd_gpu::{
    gpu::GpuContext,
    graph::{self, Graph, SgdParams},
    preprocess,
    schedule::build_prepared_schedule,
};
use std::path::Path;
#[allow(dead_code)]
#[path = "../../baseline-sparse-sgd-non-gpu/src/algorithm.rs"]
mod cpu;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "usage: cpu_shuffle_probe GRAPH PREP_SEED SHUFFLE_TRIALS"
    );
    let seed: u64 = args[2].parse()?;
    let trials: u64 = args[3].parse()?;
    ensure!(trials > 0, "trials must be positive");
    let graph = Graph::from_mtx(Path::new(&args[1]))?.largest_connected_component()?;
    ensure!(
        graph.node_size <= 10000,
        "exact diagnostic limited to 10000 vertices"
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let params = graph.prepare_sgd_params(15, 0.1, 200, true, &mut rng)?;
    let mut prepared =
        preprocess::prepare(&graph, 15, 0.1, 200, true, &mut StdRng::seed_from_u64(seed))?;
    assert_eq!(params.positions, prepared.params.positions);
    assert_eq!(params.etas, prepared.params.etas);
    assert_eq!(params.pivots, prepared.params.pivots);
    let initial = positions_sha256_f64(&params.positions);
    let schedule = build_prepared_schedule(
        &graph,
        &prepared.params,
        std::mem::take(&mut prepared.one_sided),
        prepared.one_sided_count,
        seed,
    )?;
    let gpu = GpuContext::new_with_pivot_shuffle()?.execute_with_phase_chunks(
        prepared.params,
        &schedule,
        seed,
        false,
        OutputFormat::Json,
        4,
    )?;
    let edges: Vec<_> = graph
        .edge_src
        .iter()
        .copied()
        .zip(graph.edge_dst.iter().copied())
        .collect();
    let emit = |method: &str, shuffle_seed: Option<u64>, positions: &[[f64; 2]]| {
        let StressResult::Exact(stress) = evaluate_f64(StressMode::Exact, positions, &edges, 64, 0)
        else {
            unreachable!()
        };
        println!(
            "{}",
            serde_json::json!({"graph":args[1], "nodes":graph.node_size,
            "preparation_seed":seed, "shuffle_seed":shuffle_seed, "method":method,
            "iterations":15, "pivots":200, "stress_kind":"exact", "stress":stress,
            "initial_sha256":initial, "final_sha256":positions_sha256_f64(positions)})
        );
    };
    emit("gpu_shuffle4", None, &gpu.positions);
    for trial in 0..=trials {
        let input = SgdParams {
            positions: params.positions.clone(),
            pairs: params.pairs.clone(),
            pivots: params.pivots.clone(),
            etas: params.etas.clone(),
            center: params.center,
        };
        // Trial 0 reproduces production's RNG state after preprocessing.
        // Others vary only the update RNG, keeping the entire prepared input fixed.
        let mut update_rng = if trial == 0 {
            rng.clone()
        } else {
            StdRng::seed_from_u64(trial - 1)
        };
        let positions = cpu::execute_sgd(input, &mut update_rng);
        emit(
            if trial == 0 {
                "cpu_production"
            } else {
                "cpu_order_trial"
            },
            trial.checked_sub(1),
            &positions,
        );
    }
    Ok(())
}
