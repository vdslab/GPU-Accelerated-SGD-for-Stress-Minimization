//! Screening-only schedule controls. All variants preserve the constraint visits.
use anyhow::Result;
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
use std::{
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
#[allow(dead_code)]
#[path = "../../baseline-sparse-sgd-non-gpu/src/algorithm.rs"]
mod cpu;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len()>=4,"usage: quality_search GRAPH SEED VARIANTS [SAMPLES] [PIVOTS] [standard|compensated|extended|coefficients|cap=VALUE] [cpu|no-cpu] [ITERATIONS] [POSITIONS_DIR] [REPEATS]; VARIANTS = chunks@start,... (zero-based start)");
    let seed = args[2].parse()?;
    let repeats: usize = args.get(10).map(|x| x.parse()).transpose()?.unwrap_or(1);
    anyhow::ensure!(repeats > 0, "repeats must be positive");
    let iterations = args.get(8).map(|x| x.parse()).transpose()?.unwrap_or(15);
    let samples = args.get(4).map(|x| x.parse()).transpose()?.unwrap_or(64);
    let h = args.get(5).map(|x| x.parse()).transpose()?.unwrap_or(200);
    let mode = args.get(6).map(String::as_str).unwrap_or("standard");
    anyhow::ensure!(
        matches!(
            mode,
            "standard" | "compensated" | "extended" | "coefficients" | "recenter" | "shuffle"
        ) || mode.starts_with("cap="),
        "unknown arithmetic mode: {mode}"
    );
    let g = Graph::from_mtx(Path::new(&args[1]))?.largest_connected_component()?;
    let coefficients = args.get(6).is_some_and(|x| x == "coefficients");
    let mut p = preprocess::prepare_with_residuals(
        &g,
        iterations,
        0.1,
        h,
        true,
        &mut StdRng::seed_from_u64(seed),
        coefficients,
    )?;
    let s = build_prepared_schedule(&g, &p.params, p.one_sided, p.one_sided_count, seed)?;
    let compensated = args.get(6).is_some_and(|x| x == "compensated");
    let extended = args.get(6).is_some_and(|x| x == "extended");
    let cap = args
        .get(6)
        .and_then(|x| x.strip_prefix("cap="))
        .map(str::parse::<f32>)
        .transpose()?;
    let context = if mode == "shuffle" {
        GpuContext::new_with_pivot_shuffle()?
    } else if mode == "recenter" {
        GpuContext::new_with_recentering()?
    } else if let Some(cap) = cap {
        GpuContext::new_with_step_cap(cap)?
    } else if coefficients {
        let lookup: std::collections::HashMap<_, _> = p
            .params
            .pairs
            .iter()
            .map(|pair| {
                (
                    (pair.u as u32, pair.v as u32),
                    (pair.weight_u, pair.weight_v),
                )
            })
            .collect();
        for pair in &s.two_sided {
            let (wu, wv) = lookup[&(pair.u, pair.v)];
            p.weight_residuals
                .push((wu - f64::from(pair.weight_u)) as f32);
            p.weight_residuals
                .push((wv - f64::from(pair.weight_v)) as f32);
        }
        GpuContext::new_with_extended_coefficients(p.weight_residuals)?
    } else if extended {
        GpuContext::new_with_extended_precision()?
    } else {
        GpuContext::new_with_compensation(compensated)?
    };
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
        let mut timings = Vec::with_capacity(repeats);
        let mut wall_times = Vec::with_capacity(repeats);
        let mut last_run: Option<sparse_sgd_gpu::gpu::GpuRunResult> = None;
        for _ in 0..repeats {
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
            wall_times.push(t.elapsed().as_secs_f64());
            timings.push(run.compute_time.as_secs_f64());
            if let Some(previous) = &last_run {
                assert_eq!(run.positions, previous.positions);
            }
            last_run = Some(run);
        }
        let run = last_run.unwrap();
        let wall = median(&wall_times);
        let positions_path = if let Some(directory) = args.get(9) {
            std::fs::create_dir_all(directory)?;
            let name = Path::new(&args[1]).file_stem().unwrap().to_string_lossy();
            let path =
                Path::new(directory).join(format!("{name}-s{seed}-{mode}-{variant}.positions.txt"));
            let mut out = BufWriter::new(std::fs::File::create(&path)?);
            writeln!(out, "# Positions (x y)")?;
            for point in &run.positions {
                writeln!(out, "{} {}", point[0], point[1])?;
            }
            out.flush()?;
            Some(path.display().to_string())
        } else {
            None
        };
        let stress = match evaluate_f64(StressMode::Sampled, &run.positions, &edges, samples, 0) {
            StressResult::Sampled { value, .. } => value,
            _ => unreachable!(),
        };
        println!(
            "{}",
            serde_json::json!({"positions_path":positions_path,"arithmetic_mode":mode,"iterations":iterations,"step_cap":cap,"coefficients":coefficients,"compensated":compensated,"extended":extended,"graph":args[1],"nodes":g.node_size,"edges":g.edge_size,"seed":seed,"pivots":h,"variant":variant,"samples":samples,"stress":stress,"repeats":repeats,"iteration_samples_seconds":timings,"iteration_seconds":median(&timings),"gpu_run_wall_seconds":wall,"initial_sha256":positions_sha256_f64(&p.params.positions),"preprocess_sha256":p.fingerprint,"final_sha256":positions_sha256_f64(&run.positions)})
        );
    }
    if args.get(7).is_some_and(|x| x == "cpu") {
        let mut rng = StdRng::seed_from_u64(seed);
        let params = g.prepare_sgd_params(iterations, 0.1, h, true, &mut rng)?;
        assert_eq!(
            positions_sha256_f64(&params.positions),
            positions_sha256_f64(&p.params.positions)
        );
        assert_eq!(sparse_preprocess_sha256(&params), p.fingerprint);
        let mut timings = Vec::with_capacity(repeats);
        let mut positions: Option<Vec<[f64; 2]>> = None;
        for _ in 0..repeats {
            let input = SgdParams {
                positions: params.positions.clone(),
                etas: params.etas.clone(),
                pairs: params.pairs.clone(),
                pivots: params.pivots.clone(),
                center: params.center,
            };
            let mut repeat_rng = rng.clone();
            let start = Instant::now();
            let mut result = cpu::execute_sgd_iterations(input, &mut repeat_rng);
            timings.push(start.elapsed().as_secs_f64());
            if params.center {
                graph::center_inplace(&mut result);
            }
            if let Some(previous) = &positions {
                assert_eq!(&result, previous);
            }
            positions = Some(result);
        }
        let positions = positions.unwrap();
        let seconds = median(&timings);
        let stress = match evaluate_f64(StressMode::Sampled, &positions, &edges, samples, 0) {
            StressResult::Sampled { value, .. } => value,
            _ => unreachable!(),
        };
        println!(
            "{}",
            serde_json::json!({"iterations":iterations,"method":"cpu", "graph":args[1], "seed":seed, "pivots":h, "samples":samples, "initial_sha256":positions_sha256_f64(&p.params.positions), "preprocess_sha256":p.fingerprint, "stress":stress, "repeats":repeats,"iteration_samples_seconds":timings,"iteration_seconds":seconds, "final_sha256":positions_sha256_f64(&positions)})
        );
    }
    Ok(())
}

fn sparse_preprocess_sha256(params: &SgdParams) -> String {
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

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}
