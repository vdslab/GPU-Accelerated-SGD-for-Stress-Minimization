//! Reproducible diagnostic with common initial positions and fixed stress sources.
use experiment_common::{evaluate_f64, positions_sha256_f64, OutputFormat, StressMode};
use rand::{rngs::StdRng, SeedableRng};
use sparse_sgd_gpu::{gpu::GpuContext, graph, preprocess, schedule::build_prepared_schedule};
use std::{path::Path, time::Instant};
#[path = "../../baseline-sparse-sgd-non-gpu/src/algorithm.rs"]
mod cpu;
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 4,
        "usage: quality_probe GRAPH PIVOTS SEED [cpu|no-cpu] [phase-chunks,...] [stride]"
    );
    let graph = graph::Graph::from_mtx(Path::new(&args[1]))?.largest_connected_component()?;
    let h: usize = args[2].parse()?;
    let seed: u64 = args[3].parse()?;
    let mut rng = StdRng::seed_from_u64(seed);
    let t = Instant::now();
    let p = preprocess::prepare(&graph, 15, 0.1, h, true, &mut rng)?;
    let prep = t.elapsed().as_secs_f64();
    let initial_hash = positions_sha256_f64(&p.params.positions);
    let t = Instant::now();
    let schedule =
        build_prepared_schedule(&graph, &p.params, p.one_sided, p.one_sided_count, seed)?;
    let setup = t.elapsed().as_secs_f64();
    let context = GpuContext::new()?;
    let chunks: Vec<usize> = args
        .get(5)
        .map(String::as_str)
        .unwrap_or("1")
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let edges: Vec<_> = graph
        .edge_src
        .iter()
        .copied()
        .zip(graph.edge_dst.iter().copied())
        .collect();
    for chunks in chunks {
        let params = graph::SgdParams {
            etas: p.params.etas.clone(),
            positions: p.params.positions.clone(),
            pairs: vec![],
            pivots: p.params.pivots.clone(),
            center: p.params.center,
        };
        let stride = args.get(6).is_some_and(|s| s == "stride");
        let run = context.execute_diagnostic(
            params,
            &schedule,
            seed,
            false,
            OutputFormat::Json,
            chunks,
            stride,
        )?;
        let stress = evaluate_f64(StressMode::Sampled, &run.positions, &edges, 256, 0);
        println!(
            "{}",
            serde_json::json!({"graph":args[1],"nodes":graph.node_size,"edges":graph.edge_size,"stress_kind":"sampled","stress_samples":256,"seed":seed,"pivots":h,"chunks":chunks,"stride":stride,"preprocess_sha256":p.fingerprint,"initial_sha256":initial_hash,"final_sha256":positions_sha256_f64(&run.positions),"preprocess_seconds":prep,"setup_seconds":setup,"iteration_seconds":run.compute_time.as_secs_f64(),"rounds":schedule.round_count(),"stress":value(stress)})
        );
    }
    if args.get(4).is_some_and(|v| v == "cpu") {
        let mut rng = StdRng::seed_from_u64(seed);
        let params = graph.prepare_sgd_params(15, 0.1, h, true, &mut rng)?;
        assert_eq!(positions_sha256_f64(&params.positions), initial_hash);
        assert_eq!(sparse_preprocess_sha256(&params), p.fingerprint);
        let t = Instant::now();
        let positions = cpu::execute_sgd(params, &mut rng);
        let time = t.elapsed().as_secs_f64();
        println!(
            "{}",
            serde_json::json!({"method":"cpu","graph":args[1],"nodes":graph.node_size,"edges":graph.edge_size,"seed":seed,"pivots":h,"initial_sha256":initial_hash,"preprocess_sha256":p.fingerprint,"stress_kind":"sampled","stress_samples":256,"final_sha256":positions_sha256_f64(&positions),"iteration_seconds":time,"stress":value(evaluate_f64(StressMode::Sampled, &positions, &edges, 256, 0))})
        );
    }
    Ok(())
}

fn value(result: experiment_common::StressResult) -> f64 {
    match result {
        experiment_common::StressResult::Exact(v)
        | experiment_common::StressResult::Sampled { value: v, .. } => v,
        _ => panic!("missing stress"),
    }
}

fn sparse_preprocess_sha256(params: &graph::SgdParams) -> String {
    let mut fingerprint = experiment_common::FingerprintBuilder::new("sparse-preprocess-v1");
    fingerprint.usize(params.pivots.len());
    for &pivot in &params.pivots {
        fingerprint.usize(pivot);
    }
    fingerprint.usize(params.etas.len());
    for &eta in &params.etas {
        fingerprint.f64(eta);
    }
    fingerprint.usize(params.pairs.len());
    for pair in &params.pairs {
        fingerprint
            .usize(pair.u)
            .usize(pair.v)
            .f64(pair.dij)
            .f64(pair.weight_u)
            .f64(pair.weight_v);
    }
    fingerprint.finish()
}
