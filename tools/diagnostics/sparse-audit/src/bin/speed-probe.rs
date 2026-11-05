use anyhow::Result;
pub use vram_lock_sparse::graph;
#[path = "../../../../../implementations/sparse/baseline-sparse-sgd-non-gpu/src/algorithm.rs"]
mod cpu;
#[path = "../speed_metal.rs"]
mod metal;
use experiment_common::{
    stress::{evaluate_f64, StressMode},
    OutputFormat,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::{path::Path, time::Instant};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let seed: u64 = a[2].parse()?;
    let samples: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(256);
    let modes = a[3].split(',');
    let g = graph::Graph::from_mtx(Path::new(&a[1]))?.largest_connected_component()?;
    let mut rng = StdRng::seed_from_u64(seed);
    let t = Instant::now();
    let p = g.prepare_sgd_params(15, 0.1, 200, true, &mut rng)?;
    let preprocess_ms = t.elapsed().as_secs_f64() * 1000.;
    let edges: Vec<_> = g
        .edge_src
        .iter()
        .copied()
        .zip(g.edge_dst.iter().copied())
        .collect();
    for mode in modes {
        anyhow::ensure!(
            [
                "cpu",
                "production",
                "v3",
                "v3-each",
                "v3-once",
                "full",
                "full-once",
                "old",
                "old-once"
            ]
            .contains(&mode),
            "Unknown mode: {mode}"
        );
        let mut r = rng.clone();
        let mut params = graph::SgdParams {
            positions: p.positions.clone(),
            pairs: p.pairs.clone(),
            etas: p.etas.clone(),
            pivots: p.pivots.clone(),
            center: true,
        };
        let setup = Instant::now();
        if mode.contains("once") {
            params.pairs.shuffle(&mut r);
        }
        let order_ms = setup.elapsed().as_secs_f64() * 1000.;
        let (x, ms, gpu_ms, dispatches, completed, setup_ms) = if mode == "cpu" {
            let (x, times) = cpu::execute_sgd_iterations_timed(params, &mut r);
            (
                x,
                times.iter().map(|t| t.as_secs_f64() * 1000.).sum::<f64>(),
                0.,
                0,
                0,
                0.,
            )
        } else {
            let shader = if mode.starts_with("old") {
                "shader-before.metal"
            } else if mode.starts_with("v3") {
                "shader-v3.metal"
            } else {
                "shader-full.metal"
            };
            std::env::set_var(
                "ATOMIC_PROBE_SHADER",
                format!("experiment_results/validation/atomic-sparse-20261106/sources/{shader}"),
            );
            macro_rules! metrics {
                ($out:expr) => {{
                    let out = $out;
                    anyhow::ensure!(out.attempted_updates == out.completed_updates);
                    (
                        out.positions,
                        out.iteration_time.as_secs_f64() * 1000.,
                        out.iteration_times
                            .unwrap()
                            .iter()
                            .map(|t| t.as_secs_f64() * 1000.)
                            .sum::<f64>(),
                        out.dispatches,
                        out.completed_updates,
                        out.method_setup_time.as_secs_f64() * 1000.
                            + out.upload_time.as_secs_f64() * 1000.
                            + out.preparation_time.as_secs_f64() * 1000.,
                    )
                }};
            }
            if mode == "production" {
                let ctx = vram_lock_sparse::metal::MetalContext::new()?;
                metrics!(ctx.execute_sgd(params, &mut r, false, OutputFormat::Json)?)
            } else {
                let ctx = metal::MetalContext::new()?;
                metrics!(ctx.execute_sgd_with_cpu_shuffle(
                    params,
                    &mut r,
                    false,
                    OutputFormat::Json,
                    mode.ends_with("each")
                )?)
            }
        };
        let stress = match evaluate_f64(StressMode::Sampled, &x, &edges, samples, 0) {
            experiment_common::stress::StressResult::Sampled { value, .. } => value,
            _ => unreachable!(),
        };
        println!(
            "{}",
            serde_json::json!({"graph":Path::new(&a[1]).file_stem().unwrap().to_str().unwrap(),"seed":seed,"mode":mode,"nodes":g.node_size,"constraints":p.pairs.len(),"preprocess_ms":preprocess_ms,"iteration_ms":ms,"gpu_ms":gpu_ms,"dispatches":dispatches,"completed":completed,"setup_ms":setup_ms+order_ms,"stress_samples":samples,"stress":stress})
        );
    }
    Ok(())
}
