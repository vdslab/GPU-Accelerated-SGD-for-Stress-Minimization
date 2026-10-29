//! Screening-only schedule controls. All variants preserve the constraint visits.
use anyhow::Result;
use experiment_common::{
    evaluate_f64, positions_sha256_f64, OutputFormat, StressMode, StressResult,
};
use rand::{rngs::StdRng, SeedableRng};
use sparse_sgd_gpu::{
    gpu::GpuContext,
    graph::{Graph, SgdParams},
    preprocess,
    schedule::build_prepared_schedule,
};
use std::{path::Path, time::Instant};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len()>=4,"usage: quality_search GRAPH SEED VARIANTS [SAMPLES] [PIVOTS]; VARIANTS = chunks@start,... (zero-based start)");
    let seed = args[2].parse()?;
    let samples = args.get(4).map(|x| x.parse()).transpose()?.unwrap_or(64);
    let h = args.get(5).map(|x| x.parse()).transpose()?.unwrap_or(200);
    let g = Graph::from_mtx(Path::new(&args[1]))?.largest_connected_component()?;
    let p = preprocess::prepare(&g, 15, 0.1, h, true, &mut StdRng::seed_from_u64(seed))?;
    let s = build_prepared_schedule(&g, &p.params, p.one_sided, p.one_sided_count, seed)?;
    let context = GpuContext::new()?;
    let edges: Vec<_> = g
        .edge_src
        .iter()
        .copied()
        .zip(g.edge_dst.iter().copied())
        .collect();
    for variant in args[3].split(',') {
        let (chunks, start) = variant
            .split_once('@')
            .ok_or_else(|| anyhow::anyhow!("variant needs @"))?;
        let gp = SgdParams {
            positions: p.params.positions.clone(),
            etas: p.params.etas.clone(),
            pairs: vec![],
            pivots: p.params.pivots.clone(),
            center: true,
        };
        let t = Instant::now();
        let run = context.execute_quality_diagnostic(
            gp,
            &s,
            seed,
            false,
            OutputFormat::Json,
            chunks.parse()?,
            false,
            start.parse()?,
        )?;
        let wall = t.elapsed().as_secs_f64();
        let stress = match evaluate_f64(StressMode::Sampled, &run.positions, &edges, samples, 0) {
            StressResult::Sampled { value, .. } => value,
            _ => unreachable!(),
        };
        println!(
            "{}",
            serde_json::json!({"graph":args[1],"nodes":g.node_size,"edges":g.edge_size,"seed":seed,"pivots":h,"variant":variant,"samples":samples,"stress":stress,"iteration_seconds":run.compute_time.as_secs_f64(),"gpu_run_wall_seconds":wall,"initial_sha256":positions_sha256_f64(&p.params.positions),"preprocess_sha256":p.fingerprint,"final_sha256":positions_sha256_f64(&run.positions)})
        );
    }
    Ok(())
}
