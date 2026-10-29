use anyhow::{ensure, Result};
use graph::{Graph, SgdParams};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde_json::json;
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};
pub use vram_lock_sparse::graph;
#[path = "../../../../implementations/sparse/baseline-sparse-sgd-non-gpu/src/algorithm.rs"]
mod cpu;
use experiment_common::{
    stress::{evaluate_f64, StressMode, StressResult},
    OutputFormat,
};
fn stress(g: &Graph, x: &[[f64; 2]], samples: usize, seed: u64) -> f64 {
    let edges: Vec<_> = g
        .edge_src
        .iter()
        .copied()
        .zip(g.edge_dst.iter().copied())
        .collect();
    match evaluate_f64(StressMode::Sampled, x, &edges, samples, seed) {
        StressResult::Sampled { value, .. } => value,
        _ => unreachable!(),
    }
}
fn read_positions(p: &Path) -> Result<Vec<[f64; 2]>> {
    let mut active = false;
    let mut out = vec![];
    for line in BufReader::new(File::open(p)?).lines() {
        let s = line?;
        if s.starts_with("# Positions") {
            active = true;
            continue;
        }
        if !active || s.trim().is_empty() || s.starts_with('#') {
            continue;
        }
        let v: Vec<f64> = s
            .split_whitespace()
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(v.len() == 2);
        out.push([v[0], v[1]]);
    }
    Ok(out)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let g = Graph::from_mtx(Path::new(&args[2]))?.largest_connected_component()?;
    let mut degrees: Vec<_> = (0..g.node_size).map(|i| g.neighbors(i).len()).collect();
    degrees.sort_unstable();
    let stats = json!({"n":g.node_size,"m":g.edge_size,"degree_min":degrees[0],"degree_median":degrees[degrees.len()/2],"degree_mean":2.0*g.edge_size as f64/g.node_size as f64,"degree_max":degrees[degrees.len()-1]});
    eprintln!("graph {stats}");
    if args[1] == "paired" {
        let files = &args[3..];
        let layouts: Vec<_> = files
            .iter()
            .map(|f| read_positions(Path::new(f)).unwrap())
            .collect();
        for x in &layouts {
            ensure!(x.len() == g.node_size);
        }
        let sources = experiment_common::stress::sample_indices(g.node_size, 4096, 20261027);
        let mut sums = vec![0.0; layouts.len()];
        let mut diff_sum = vec![0.0; layouts.len()];
        let mut diff_sq = vec![0.0; layouts.len()];
        for &u in &sources {
            let ds = g.shortest_path_distances(u);
            let mut row = vec![0.0; layouts.len()];
            for (k, x) in layouts.iter().enumerate() {
                for v in 0..g.node_size {
                    if v == u {
                        continue;
                    }
                    let dx = x[u][0] - x[v][0];
                    let dy = x[u][1] - x[v][1];
                    let e = (dx * dx + dy * dy).sqrt() / ds[v] as f64 - 1.0;
                    row[k] += e * e;
                }
                sums[k] += row[k];
            }
            for k in 0..layouts.len() {
                let d = row[k] - row[0];
                diff_sum[k] += d;
                diff_sq[k] += d * d;
            }
        }
        let k = sources.len() as f64;
        let n = g.node_size as f64;
        for i in 0..layouts.len() {
            let mean = diff_sum[i] / k;
            let var = ((diff_sq[i] - k * mean * mean) / (k - 1.0)).max(0.0);
            let se = n / 2.0 * (var / k * (n - k) / (n - 1.0)).sqrt();
            println!(
                "{}",
                json!({"file":files[i],"baseline":files[0],"samples":sources.len(),"evaluation_seed":20261027,"stress":n/2.0*sums[i]/k,"paired_difference":n/2.0*mean,"paired_difference_95ci":[n/2.0*mean-1.96*se,n/2.0*mean+1.96*se]})
            );
        }
        return Ok(());
    }
    if args[1] == "eval" {
        for file in &args[3..] {
            let x = read_positions(Path::new(file))?;
            ensure!(x.len() == g.node_size);
            let vals: Vec<_> = [0, 1, 2, 3]
                .iter()
                .map(|&s| stress(&g, &x, 1024, s))
                .collect();
            println!(
                "{}",
                json!({"file":file,"graph":stats,"stress_64_seed0":stress(&g,&x,64,0),"stress_1024_eval_seeds_0_1_2_3":vals})
            );
        }
        return Ok(());
    }
    let out = Path::new(&args[3]);
    fs::create_dir_all(out)?;
    let context = vram_lock_sparse::metal::MetalContext::new()?;
    for seed in 0..3 {
        let mut prep_rng = StdRng::seed_from_u64(seed);
        let params = g.prepare_sgd_params(15, 0.1, 200, true, &mut prep_rng)?;
        let mut incident = vec![0usize; g.node_size];
        let mut zero_sides = 0;
        let mut edge_count = 0;
        let mut longest = 0;
        let mut current = 0;
        let mut last = usize::MAX;
        for p in &params.pairs {
            incident[p.u] += 1;
            incident[p.v] += 1;
            if p.weight_u == 0.0 || p.weight_v == 0.0 {
                zero_sides += 1;
            }
            if g.has_edge(p.u, p.v) {
                edge_count += 1;
            }
            if p.u == last {
                current += 1
            } else {
                current = 1;
                last = p.u;
            }
            longest = longest.max(current);
        }
        incident.sort_unstable();
        let ps = json!({"seed":seed,"graph":stats,"constraints":params.pairs.len(),"actual_edge_constraints":edge_count,"one_sided_constraints":zero_sides,"constraint_degree_mean":2.0*params.pairs.len() as f64/g.node_size as f64,"constraint_degree_max":incident[incident.len()-1],"constraint_degree_median":incident[incident.len()/2],"longest_same_u_submission_run":longest,"etas":params.etas});
        fs::write(
            out.join(format!("stats-s{seed}.json")),
            serde_json::to_string_pretty(&ps)?,
        )?;
        for mode in [
            "cpu_shuffle",
            "cpu_fixed",
            "cpu_f32_shuffle",
            "cpu_f32_fixed",
            "gpu_shuffle",
            "gpu_fixed",
        ] {
            let mut rng = prep_rng.clone();
            let mut pairs = params.pairs.clone();
            let mut x = params.positions.clone();
            let start = Instant::now();
            let mut dispatches = 0;
            if mode.starts_with("gpu") {
                let p = SgdParams {
                    pairs,
                    positions: x,
                    etas: params.etas.clone(),
                    pivots: params.pivots.clone(),
                    center: true,
                };
                let r = context.execute_sgd_with_cpu_shuffle(
                    p,
                    &mut rng,
                    false,
                    OutputFormat::Json,
                    mode == "gpu_shuffle",
                )?;
                ensure!(r.completed_updates == r.attempted_updates);
                dispatches = r.dispatches;
                x = r.positions;
            } else if mode.starts_with("cpu_f32") {
                let mut y: Vec<[f32; 2]> = x.iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
                for &eta in &params.etas {
                    if mode.ends_with("shuffle") {
                        pairs.shuffle(&mut rng)
                    }
                    for p in &pairs {
                        let a = y[p.u];
                        let b = y[p.v];
                        let mut dx = b[0] - a[0];
                        let mut dy = b[1] - a[1];
                        let mut norm = (dx * dx + dy * dy).sqrt();
                        if norm < 1e-12 {
                            dx = 1e-6;
                            dy = 1e-6;
                            norm = (dx * dx + dy * dy).sqrt();
                        }
                        let scale = (norm - p.dij as f32) / (2.0 * norm);
                        let rx = scale * dx;
                        let ry = scale * dy;
                        let mu = (eta as f32 * p.weight_u as f32).min(1.0);
                        let mv = (eta as f32 * p.weight_v as f32).min(1.0);
                        y[p.u][0] += mu * rx;
                        y[p.u][1] += mu * ry;
                        y[p.v][0] -= mv * rx;
                        y[p.v][1] -= mv * ry;
                    }
                }
                x = y.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
                graph::center_inplace(&mut x);
            } else {
                for &eta in &params.etas {
                    if mode.ends_with("shuffle") {
                        pairs.shuffle(&mut rng)
                    }
                    for &p in &pairs {
                        cpu::apply_constraint(&mut x, p, eta, &mut rng)
                    }
                }
                graph::center_inplace(&mut x);
            }
            let elapsed = start.elapsed().as_secs_f64();
            let s64 = stress(&g, &x, 64, 0);
            let s1024 = stress(&g, &x, 1024, 0);
            let record = json!({"seed":seed,"mode":mode,"seconds_including_diagnostic_clones_and_setup":elapsed,"stress64":s64,"stress1024":s1024,"dispatches":dispatches});
            fs::write(
                out.join(format!("s{seed}-{mode}.json")),
                serde_json::to_string_pretty(&record)?,
            )?;
            let mut f = BufWriter::new(File::create(
                out.join(format!("s{seed}-{mode}.positions.txt")),
            )?);
            writeln!(f, "# Positions (x y)")?;
            for p in x {
                writeln!(f, "{} {}", p[0], p[1])?;
            }
            f.flush()?;
            println!("{record}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moving_shared_endpoint_tracks_serial_recurrence() {
        let context = vram_lock_sparse::metal::MetalContext::new().unwrap();
        let count = 4096;
        let eta = 0.0001;
        let mut expected = 0.0f32;
        for _ in 0..count {
            let diff = 3.0 - expected;
            let r = ((diff - 1.0) / (2.0 * diff)) * diff;
            expected += eta as f32 * r;
        }
        for seed in 0..20 {
            let mut positions = vec![[3.0, 0.0]; count + 1];
            positions[0] = [0.0, 0.0];
            let pairs = (1..=count)
                .map(|v| graph::EdgeInfo {
                    u: 0,
                    v,
                    dij: 1.0,
                    weight_u: 1.0,
                    weight_v: 0.0,
                })
                .collect();
            let params = SgdParams {
                positions,
                pairs,
                etas: vec![eta],
                pivots: vec![0],
                center: false,
            };
            let r = context
                .execute_sgd_with_cpu_shuffle(
                    params,
                    &mut StdRng::seed_from_u64(seed),
                    false,
                    OutputFormat::Json,
                    seed % 2 == 1,
                )
                .unwrap();
            assert_eq!(r.completed_updates, count as u64);
            assert!(
                (r.positions[0][0] - expected as f64).abs() < 0.0001,
                "{} != {}",
                r.positions[0][0],
                expected
            );
            assert_eq!(r.positions[0][1], 0.0);
            assert!(r.positions[1..].iter().all(|&p| p == [3.0, 0.0]));
        }
    }
    #[test]
    fn preprocessing_matches_bruteforce_paper_definition() {
        for n in [5, 9, 17] {
            let mut edges: Vec<_> = (1..n).map(|i| (i - 1, i)).collect();
            for i in 0..n {
                for j in i + 2..n {
                    if (i * 13 + j * 7) % 5 == 0 {
                        edges.push((i, j));
                    }
                }
            }
            let g = Graph::try_from_edges(n, &edges).unwrap();
            let mut d = vec![vec![100000usize; n]; n];
            for i in 0..n {
                d[i][i] = 0;
            }
            for &(u, v) in &edges {
                d[u][v] = 1;
                d[v][u] = 1;
            }
            for k in 0..n {
                for i in 0..n {
                    for j in 0..n {
                        d[i][j] = d[i][j].min(d[i][k] + d[k][j]);
                    }
                }
            }
            for h in [1, 3, n] {
                for seed in 0..5 {
                    let p = g
                        .prepare_sgd_params(15, 0.1, h, false, &mut StdRng::seed_from_u64(seed))
                        .unwrap();
                    let region: Vec<_> = (0..n)
                        .map(|v| *p.pivots.iter().min_by_key(|&&q| d[v][q]).unwrap())
                        .collect();
                    let actual: std::collections::BTreeMap<_, _> =
                        p.pairs.iter().map(|p| ((p.u, p.v), p)).collect();
                    let mut count = 0;
                    for i in 0..n {
                        for j in i + 1..n {
                            if !g.has_edge(i, j) && !p.pivots.contains(&i) && !p.pivots.contains(&j)
                            {
                                assert!(!actual.contains_key(&(i, j)));
                                continue;
                            }
                            count += 1;
                            let pair = actual[&(i, j)];
                            assert_eq!(pair.dij, d[i][j] as f64);
                            let weight = |v: usize, q: usize| {
                                if g.has_edge(v, q) {
                                    1.0
                                } else if !p.pivots.contains(&q) {
                                    0.0
                                } else {
                                    let s = (0..n)
                                        .filter(|&k| region[k] == q && 2 * d[k][q] <= d[v][q])
                                        .count();
                                    s as f64 / (d[v][q] * d[v][q]) as f64
                                }
                            };
                            assert_eq!(pair.weight_u, weight(i, j));
                            assert_eq!(pair.weight_v, weight(j, i));
                            if h == n {
                                assert_eq!(pair.weight_u, 1.0 / (d[i][j] * d[i][j]) as f64);
                                assert_eq!(pair.weight_v, pair.weight_u);
                            }
                        }
                    }
                    assert_eq!(count, p.pairs.len());
                }
            }
        }
    }
}
