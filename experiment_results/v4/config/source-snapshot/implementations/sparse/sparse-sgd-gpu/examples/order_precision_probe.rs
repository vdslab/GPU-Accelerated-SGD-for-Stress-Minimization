//! Diagnostic only: isolate RR ordering from arithmetic precision on the CPU.
//! Host f32 does not emulate Metal fast math. These are not performance timings.
use anyhow::Result;
use experiment_common::{
    evaluate_f64, positions_sha256_f64, OutputFormat, StressMode, StressResult,
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use sparse_sgd_gpu::{
    gpu::GpuContext,
    graph::{self, Graph, SgdParams},
    schedule::{build_schedule, Schedule},
};
use std::{collections::HashMap, path::Path};
#[path = "../../baseline-sparse-sgd-non-gpu/src/algorithm.rs"]
mod cpu;
fn hash32(mut x: u32) -> u32 {
    x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
    x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
    (x >> 16) ^ x
}
fn stress(x: &[[f64; 2]], edges: &[(usize, usize)], samples: usize) -> f64 {
    match evaluate_f64(StressMode::Sampled, x, edges, samples, 0) {
        StressResult::Sampled { value, .. } => value,
        _ => unreachable!(),
    }
}
fn report(method: &str, x: &[[f64; 2]], edges: &[(usize, usize)], tiny: usize) {
    println!(
        "{}",
        serde_json::json!({"method":method,"stress64":stress(x,edges,64),"stress256":stress(x,edges,256),"final_sha256":positions_sha256_f64(x),"tiny_fallbacks":if tiny==usize::MAX {None} else {Some(tiny)}})
    );
}
// Recover original f64 weights by endpoint identity. Distances are integral.
fn original_weights(p: &SgdParams, s: &Schedule) -> (Vec<f64>, Vec<[f64; 2]>) {
    let h = p.pivots.len();
    let mut slots = vec![usize::MAX; p.positions.len()];
    for (i, &v) in p.pivots.iter().enumerate() {
        slots[v] = i;
    }
    let indices: HashMap<_, _> = s
        .two_sided
        .iter()
        .enumerate()
        .map(|(i, e)| ((e.u as usize, e.v as usize), i))
        .collect();
    let mut one = vec![0.0; s.one_sided.len()];
    let mut two = vec![[0.0; 2]; s.two_sided.len()];
    let (mut ones, mut twos) = (0, 0);
    for e in &p.pairs {
        if e.weight_u > 0.0 && e.weight_v > 0.0 {
            two[indices[&(e.u, e.v)]] = [e.weight_u, e.weight_v];
            twos += 1;
        } else if e.weight_u > 0.0 {
            assert_ne!(slots[e.v], usize::MAX);
            one[e.u * h + slots[e.v]] = e.weight_u;
            ones += 1;
        } else {
            assert_ne!(slots[e.u], usize::MAX);
            one[e.v * h + slots[e.u]] = e.weight_v;
            ones += 1;
        }
    }
    assert_eq!(ones, s.one_sided_count);
    assert_eq!(twos, s.two_sided.len());
    for (a, b) in one.iter().zip(&s.one_sided) {
        assert_eq!(*a as f32, b.weight);
    }
    for (a, b) in two.iter().zip(&s.two_sided) {
        assert_eq!([a[0] as f32, a[1] as f32], [b.weight_u, b.weight_v]);
    }
    (one, two)
}
macro_rules! rr_replay {
    ($name:ident,$float:ty) => {
        fn $name(
            p: &SgdParams,
            s: &Schedule,
            weights: &(Vec<f64>, Vec<[f64; 2]>),
            seed: u64,
        ) -> (Vec<[f64; 2]>, usize) {
            let mut x: Vec<[$float; 2]> = p
                .positions
                .iter()
                .map(|p| [p[0] as $float, p[1] as $float])
                .collect();
            let h = p.pivots.len();
            let mut rng = StdRng::seed_from_u64(seed ^ 0x4750_555f_5350_4152);
            let mut perm: Vec<u32> = (0..h as u32).collect();
            let mut rounds: Vec<usize> = (0..s.round_count()).collect();
            let mut tiny = 0;
            for (iteration, &eta) in p.etas.iter().enumerate() {
                perm.shuffle(&mut rng);
                rounds.shuffle(&mut rng);
                let one_first = rng.random_bool(0.5);
                let eta = eta as $float;
                for phase in 0..2 {
                    if (phase == 0) == one_first {
                        for v in 0..x.len() {
                            let rotation =
                                (hash32(v as u32 ^ seed as u32 ^ iteration as u32) as usize) % h;
                            for k in 0..h {
                                let pi = perm[(k + rotation) % h] as usize;
                                let wi = v * h + pi;
                                let w = weights.0[wi] as $float;
                                if w <= 0.0 {
                                    continue;
                                }
                                let u = p.pivots[pi];
                                let mut dx = x[u][0] - x[v][0];
                                let mut dy = x[u][1] - x[v][1];
                                let mut norm = (dx * dx + dy * dy).sqrt();
                                if norm < 1e-12 {
                                    tiny += 1;
                                    let angle = (hash32(
                                        v as u32 ^ u as u32 ^ seed as u32 ^ iteration as u32,
                                    ) % 6283) as $float
                                        * 0.001;
                                    dx = angle.cos() * 1e-6;
                                    dy = angle.sin() * 1e-6;
                                    norm = (dx * dx + dy * dy).sqrt();
                                }
                                let scale = (norm - s.one_sided[wi].dij as $float) / (2.0 * norm);
                                let mu = (w * eta).min(1.0);
                                x[v][0] += mu * (scale * dx);
                                x[v][1] += mu * (scale * dy);
                            }
                        }
                    } else {
                        for &ri in &rounds {
                            for i in s.round_range(ri) {
                                let e = s.two_sided[i];
                                let (u, v) = (e.u as usize, e.v as usize);
                                let mut dx = x[v][0] - x[u][0];
                                let mut dy = x[v][1] - x[u][1];
                                let mut norm = (dx * dx + dy * dy).sqrt();
                                if norm < 1e-12 {
                                    tiny += 1;
                                    let angle = (hash32(
                                        u as u32 ^ v as u32 ^ seed as u32 ^ iteration as u32,
                                    ) % 6283) as $float
                                        * 0.001;
                                    dx = angle.cos() * 1e-6;
                                    dy = angle.sin() * 1e-6;
                                    norm = (dx * dx + dy * dy).sqrt();
                                }
                                let scale = (norm - e.dij as $float) / (2.0 * norm);
                                let d = [scale * dx, scale * dy];
                                let mu = (weights.1[i][0] as $float * eta).min(1.0);
                                let mv = (weights.1[i][1] as $float * eta).min(1.0);
                                x[u][0] += mu * d[0];
                                x[u][1] += mu * d[1];
                                x[v][0] -= mv * d[0];
                                x[v][1] -= mv * d[1];
                            }
                        }
                    }
                }
            }
            let mut x: Vec<[f64; 2]> = x.into_iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
            graph::center_inplace(&mut x);
            (x, tiny)
        }
    };
}
rr_replay!(rr_f64, f64);
rr_replay!(rr_f32, f32);
fn shuffled_f32(p: &SgdParams, mut rng: StdRng) -> (Vec<[f64; 2]>, usize) {
    let mut x: Vec<[f32; 2]> = p
        .positions
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32])
        .collect();
    let mut order: Vec<usize> = (0..p.pairs.len()).collect();
    let mut tiny = 0;
    for &eta in &p.etas {
        order.shuffle(&mut rng);
        for &i in &order {
            let e = p.pairs[i];
            let (u, v) = (e.u, e.v);
            let mut dx = x[v][0] - x[u][0];
            let mut dy = x[v][1] - x[u][1];
            let mut norm = (dx * dx + dy * dy).sqrt();
            if norm < 1e-12 {
                tiny += 1;
                let angle = rng.random::<f64>() * std::f64::consts::TAU;
                dx = (angle.cos() * 1e-6) as f32;
                dy = (angle.sin() * 1e-6) as f32;
                norm = (dx * dx + dy * dy).sqrt();
            }
            let scale = (norm - e.dij as f32) / (2.0 * norm);
            let d = [scale * dx, scale * dy];
            let mu = (e.weight_u as f32 * eta as f32).min(1.0);
            let mv = (e.weight_v as f32 * eta as f32).min(1.0);
            x[u][0] += mu * d[0];
            x[u][1] += mu * d[1];
            x[v][0] -= mv * d[0];
            x[v][1] -= mv * d[1];
        }
    }
    let mut x: Vec<[f64; 2]> = x.into_iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    graph::center_inplace(&mut x);
    (x, tiny)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 2,
        "usage: order_precision_probe GRAPH [SEED] [PIVOTS] [compare|rounding|rounding-init|rounding-coeff|cpu-rounding-init|gpu-phases]"
    );
    let seed = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let h = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(200);
    let g = Graph::from_mtx(Path::new(&args[1]))?.largest_connected_component()?;
    let mut rng = StdRng::seed_from_u64(seed);
    let p = g.prepare_sgd_params(15, 0.1, h, true, &mut rng)?;
    let s = build_schedule(&g, &p, seed)?;
    let weights = original_weights(&p, &s);
    let edges: Vec<_> = g
        .edge_src
        .iter()
        .copied()
        .zip(g.edge_dst.iter().copied())
        .collect();
    println!(
        "{}",
        serde_json::json!({"graph":args[1],"seed":seed,"pivots":h,"initial_sha256":positions_sha256_f64(&p.positions),"one_sided":s.one_sided_count,"two_sided":s.two_sided.len(),"rounds":s.round_count()})
    );
    let mode = args.get(4).map(String::as_str).unwrap_or("compare");
    anyhow::ensure!(
        [
            "compare",
            "rounding",
            "rounding-init",
            "rounding-coeff",
            "cpu-rounding-init",
            "gpu-phases"
        ]
        .contains(&mode),
        "unknown mode"
    );
    if mode == "cpu-rounding-init" {
        let mut p = p;
        for x in &mut p.positions {
            x[0] = x[0] as f32 as f64;
            x[1] = x[1] as f32 as f64;
        }
        let x = cpu::execute_sgd(p, &mut rng);
        report(
            "global_shuffle_cpu_f64_rounded_initial",
            &x,
            &edges,
            usize::MAX,
        );
        return Ok(());
    }
    if mode.starts_with("rounding") {
        let round_init = mode != "rounding-coeff";
        let round_coeff = mode != "rounding-init";
        let rounded = SgdParams {
            positions: p
                .positions
                .iter()
                .map(|&x| {
                    if round_init {
                        [x[0] as f32 as f64, x[1] as f32 as f64]
                    } else {
                        x
                    }
                })
                .collect(),
            etas: p
                .etas
                .iter()
                .map(|&x| if round_coeff { x as f32 as f64 } else { x })
                .collect(),
            pivots: p.pivots.clone(),
            pairs: vec![],
            center: true,
        };
        let rounded_weights = (
            weights
                .0
                .iter()
                .map(|&x| if round_coeff { x as f32 as f64 } else { x })
                .collect(),
            weights
                .1
                .iter()
                .map(|&x| {
                    if round_coeff {
                        [x[0] as f32 as f64, x[1] as f32 as f64]
                    } else {
                        x
                    }
                })
                .collect(),
        );
        println!(
            "{}",
            serde_json::json!({"mode":mode,"effective_initial_sha256":positions_sha256_f64(&rounded.positions),"max_initial_rounding_error":p.positions.iter().zip(&rounded.positions).flat_map(|(a,b)| [(a[0]-b[0]).abs(),(a[1]-b[1]).abs()]).fold(0.0,f64::max)})
        );
        let (x, t) = rr_f64(&rounded, &s, &rounded_weights, seed);
        report(&format!("rr_order_cpu_f64_{mode}"), &x, &edges, t);
        return Ok(());
    }
    if mode == "gpu-phases" {
        let gpu = GpuContext::new()?;
        for chunks in [1, 4, 16] {
            let gp = SgdParams {
                positions: p.positions.clone(),
                etas: p.etas.clone(),
                pivots: p.pivots.clone(),
                pairs: vec![],
                center: true,
            };
            let run =
                gpu.execute_diagnostic(gp, &s, seed, false, OutputFormat::Json, chunks, false)?;
            report(
                &format!("rr_gpu_chunks_{chunks}"),
                &run.positions,
                &edges,
                usize::MAX,
            );
        }
        return Ok(());
    }
    let (x, t) = rr_f64(&p, &s, &weights, seed);
    report("rr_order_cpu_f64", &x, &edges, t);
    let (x, t) = rr_f32(&p, &s, &weights, seed);
    report("rr_order_cpu_f32", &x, &edges, t);
    let (x, t) = shuffled_f32(&p, rng.clone());
    report("global_shuffle_cpu_f32", &x, &edges, t);
    let gpu = GpuContext::new()?;
    let gp = SgdParams {
        positions: p.positions.clone(),
        etas: p.etas.clone(),
        pivots: p.pivots.clone(),
        pairs: vec![],
        center: true,
    };
    let r = gpu.execute(gp, &s, seed, false, OutputFormat::Json)?;
    report("rr_gpu", &r.positions, &edges, usize::MAX);
    drop(weights);
    drop(s);
    let x = cpu::execute_sgd(p, &mut rng);
    report("global_shuffle_cpu_f64", &x, &edges, usize::MAX);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serial_replay_matches_gpu_for_non_degenerate_updates() {
        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (0, 6),
            (1, 5),
        ];
        let graph = Graph::try_from_edges(7, &edges).unwrap();
        let mut p = graph
            .prepare_sgd_params(15, 0.1, 2, true, &mut StdRng::seed_from_u64(4))
            .unwrap();
        p.etas = vec![0.01];
        let s = build_schedule(&graph, &p, 4).unwrap();
        let w = original_weights(&p, &s);
        let (x, t) = rr_f32(&p, &s, &w, 4);
        assert_eq!(t, 0);
        let (y, t) = rr_f64(&p, &s, &w, 4);
        assert_eq!(t, 0);
        let r = GpuContext::new()
            .unwrap()
            .execute(p, &s, 4, false, OutputFormat::Json)
            .unwrap();
        for ((x, y), g) in x.iter().zip(&y).zip(&r.positions) {
            for k in 0..2 {
                assert!((x[k] - g[k]).abs() < 1e-5);
                assert!((y[k] - g[k]).abs() < 1e-5);
            }
        }
    }
}
