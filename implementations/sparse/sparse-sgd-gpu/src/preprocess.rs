//! Streaming RR preprocessing. The CPU reference stays independent so equivalence
//! of constraints, RNG consumption, learning rates and schedules can be tested.
use crate::graph::{calc_learning_rate, init_positions_random, EdgeInfo, Graph, SgdParams};
use crate::schedule::OneSidedEntry;
use anyhow::{ensure, Context, Result};
use experiment_common::FingerprintBuilder;
use rand::Rng;

pub struct PreparedRun {
    /// Only two-sided constraints are retained here, in canonical pair order.
    pub params: SgdParams,
    pub one_sided: Vec<OneSidedEntry>,
    pub one_sided_count: usize,
    pub weight_residuals: Vec<f32>,
    pub constraint_count: usize,
    pub fingerprint: String,
}

struct Distances {
    pivots: Vec<usize>,
    slot: Vec<usize>,
    distances: Vec<Vec<u32>>,
    prefix: Vec<Vec<usize>>,
}

impl Distances {
    fn new<R: Rng + ?Sized>(graph: &Graph, h: usize, rng: &mut R) -> Result<Self> {
        let n = graph.node_size;
        ensure!(
            n > 1 && n < u32::MAX as usize && h > 0,
            "nontrivial graph and positive pivot count required"
        );
        let h = h.min(n);
        let mut queue = Vec::with_capacity(n);
        let bfs = |source: usize, queue: &mut Vec<usize>| -> Result<Vec<u32>> {
            let mut distances = vec![u32::MAX; n];
            queue.clear();
            queue.push(source);
            distances[source] = 0;
            let mut head = 0;
            while head < queue.len() {
                let u = queue[head];
                head += 1;
                for &v in graph.neighbors(u) {
                    if distances[v] == u32::MAX {
                        distances[v] = distances[u] + 1;
                        queue.push(v);
                    }
                }
            }
            ensure!(queue.len() == n, "connected graph required");
            Ok(distances)
        };
        let first = rng.random_range(0..n);
        let first_distances = bfs(first, &mut queue)?;
        let mut nearest = first_distances.clone();
        let mut regions = vec![0; n];
        let mut pivots = vec![first];
        let mut distances = vec![first_distances];
        while pivots.len() < h {
            let total = nearest.iter().try_fold(0_u64, |sum, &d| {
                sum.checked_add(d as u64).context("distance sum overflow")
            })?;
            ensure!(total > 0, "empty pivot sampling distribution");
            let mut ticket = rng.random_range(0..total);
            let next = nearest
                .iter()
                .position(|&d| {
                    if ticket < d as u64 {
                        true
                    } else {
                        ticket -= d as u64;
                        false
                    }
                })
                .context("pivot sampling failed")?;
            let row = bfs(next, &mut queue)?;
            for v in 0..n {
                // Strict comparison preserves the reference's first-pivot tie rule.
                if row[v] < nearest[v] {
                    nearest[v] = row[v];
                    regions[v] = pivots.len();
                }
            }
            pivots.push(next);
            distances.push(row);
        }
        let mut prefix = Vec::with_capacity(h);
        for (p, row) in distances.iter().enumerate() {
            let mut counts = vec![0; *row.iter().max().unwrap() as usize + 1];
            for (v, &region) in regions.iter().enumerate() {
                if region == p {
                    counts[row[v] as usize] += 1;
                }
            }
            for i in 1..counts.len() {
                counts[i] += counts[i - 1];
            }
            prefix.push(counts);
        }
        let mut slot = vec![usize::MAX; n];
        for (p, &v) in pivots.iter().enumerate() {
            slot[v] = p;
        }
        Ok(Self {
            pivots,
            slot,
            distances,
            prefix,
        })
    }

    /// Merge sorted adjacency and pivot endpoints per row. This is exactly the
    /// BTreeMap reference's lexicographic order, without materializing that map.
    fn visit(&self, graph: &Graph, mut emit: impl FnMut(EdgeInfo)) {
        let n = graph.node_size;
        let mut sorted_pivots = self.pivots.clone();
        sorted_pivots.sort_unstable();
        for u in 0..n {
            let pu = self.slot[u];
            let neighbors = graph.neighbors(u);
            let mut edge_index = neighbors.partition_point(|&v| v <= u);
            let mut pivot_index = sorted_pivots.partition_point(|&v| v <= u);
            let mut next_vertex = u + 1;
            loop {
                let candidate = if pu != usize::MAX {
                    if next_vertex < n {
                        next_vertex
                    } else {
                        usize::MAX
                    }
                } else {
                    sorted_pivots
                        .get(pivot_index)
                        .copied()
                        .unwrap_or(usize::MAX)
                };
                let edge = neighbors.get(edge_index).copied().unwrap_or(usize::MAX);
                let v = candidate.min(edge);
                if v == usize::MAX {
                    break;
                }
                if candidate == v {
                    next_vertex += 1;
                    pivot_index += 1;
                }
                if edge == v {
                    edge_index += 1;
                    emit(EdgeInfo {
                        u,
                        v,
                        dij: 1.0,
                        weight_u: 1.0,
                        weight_v: 1.0,
                    });
                } else {
                    let pv = self.slot[v];
                    let distance = if pu != usize::MAX {
                        self.distances[pu][v]
                    } else {
                        self.distances[pv][u]
                    } as usize;
                    let dij = distance as f64;
                    let weight_u = if pv != usize::MAX {
                        self.prefix[pv][distance / 2] as f64 / (dij * dij)
                    } else {
                        0.0
                    };
                    let weight_v = if pu != usize::MAX {
                        self.prefix[pu][distance / 2] as f64 / (dij * dij)
                    } else {
                        0.0
                    };
                    emit(EdgeInfo {
                        u,
                        v,
                        dij,
                        weight_u,
                        weight_v,
                    });
                }
            }
        }
    }
}

pub fn prepare<R: Rng + ?Sized>(
    graph: &Graph,
    iterations: usize,
    epsilon: f64,
    h: usize,
    center: bool,
    rng: &mut R,
) -> Result<PreparedRun> {
    prepare_with_residuals(graph, iterations, epsilon, h, center, rng, false)
}

/// Preserve coefficient low parts for the f32 extended-precision diagnostic.
#[allow(clippy::too_many_arguments)]
pub fn prepare_with_residuals<R: Rng + ?Sized>(
    graph: &Graph,
    iterations: usize,
    epsilon: f64,
    h: usize,
    center: bool,
    rng: &mut R,
    residuals: bool,
) -> Result<PreparedRun> {
    let distances = Distances::new(graph, h, rng)?;
    // Unweighted graph edges have distance one and override both weights with 1.
    // Traverse each BFS row contiguously; avoid a second canonical pair traversal.
    let mut wmin: f64 = 1.0;
    let mut wmax: f64 = 1.0;
    for (p, row) in distances.distances.iter().enumerate() {
        for &d in row {
            if d <= 1 {
                continue;
            }
            let dij = d as f64;
            let w = distances.prefix[p][d as usize / 2] as f64 / (dij * dij);
            if w > 0.0 {
                wmin = wmin.min(w);
                wmax = wmax.max(w);
            }
        }
    }
    let h = distances.pivots.len();
    let pivot_pairs = graph
        .node_size
        .checked_mul(h)
        .context("constraint count overflow")?
        - h.checked_mul(h + 1).context("pivot count overflow")? / 2;
    let nonpivot_edges = graph
        .edge_src
        .iter()
        .zip(&graph.edge_dst)
        .filter(|(&u, &v)| distances.slot[u] == usize::MAX && distances.slot[v] == usize::MAX)
        .count();
    let constraint_count = pivot_pairs
        .checked_add(nonpivot_edges)
        .context("constraint count overflow")?;
    let etas = calc_learning_rate(iterations, wmin, wmax, epsilon)?;
    let mut fingerprint = FingerprintBuilder::new("sparse-preprocess-v1");
    fingerprint.usize(distances.pivots.len());
    for &p in &distances.pivots {
        fingerprint.usize(p);
    }
    fingerprint.usize(etas.len());
    for &eta in &etas {
        fingerprint.f64(eta);
    }
    fingerprint.usize(constraint_count);
    let h = distances.pivots.len();
    let dense_len = graph
        .node_size
        .checked_mul(h)
        .context("dense table overflow")?;
    let mut one_sided = vec![OneSidedEntry::default(); dense_len];
    let mut weight_residuals = if residuals {
        vec![0.0; dense_len]
    } else {
        vec![]
    };
    let mut one_sided_count = 0;
    let mut pairs = Vec::new();
    distances.visit(graph, |pair| {
        fingerprint
            .usize(pair.u)
            .usize(pair.v)
            .f64(pair.dij)
            .f64(pair.weight_u)
            .f64(pair.weight_v);
        let (vertex, pivot, weight) = if pair.weight_v == 0.0 {
            (pair.u, pair.v, pair.weight_u)
        } else if pair.weight_u == 0.0 {
            (pair.v, pair.u, pair.weight_v)
        } else {
            pairs.push(pair);
            return;
        };
        one_sided[vertex * h + distances.slot[pivot]] = OneSidedEntry {
            dij: pair.dij as f32,
            weight: weight as f32,
        };
        if residuals {
            weight_residuals[vertex * h + distances.slot[pivot]] =
                (weight - f64::from(weight as f32)) as f32;
        }
        one_sided_count += 1;
    });
    ensure!(
        one_sided_count + pairs.len() == constraint_count,
        "streamed constraint count mismatch"
    );
    let params = SgdParams {
        etas,
        positions: init_positions_random(graph.node_size, center, rng),
        pairs,
        pivots: distances.pivots,
        center,
    };
    Ok(PreparedRun {
        params,
        one_sided,
        one_sided_count,
        weight_residuals,
        constraint_count,
        fingerprint: fingerprint.finish(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::{build_prepared_schedule, build_schedule};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn optional_residuals_preserve_f64_one_sided_coefficients() {
        let graph =
            Graph::try_from_edges(64, &(1..64).map(|v| (v - 1, v)).collect::<Vec<_>>()).unwrap();
        let plain = prepare(&graph, 15, 0.1, 8, true, &mut StdRng::seed_from_u64(3)).unwrap();
        let extended = prepare_with_residuals(
            &graph,
            15,
            0.1,
            8,
            true,
            &mut StdRng::seed_from_u64(3),
            true,
        )
        .unwrap();
        assert!(plain.weight_residuals.is_empty());
        assert_eq!(plain.fingerprint, extended.fingerprint);
        assert_eq!(plain.params.positions, extended.params.positions);
        let reference = graph
            .prepare_sgd_params(15, 0.1, 8, true, &mut StdRng::seed_from_u64(3))
            .unwrap();
        for pair in reference.pairs {
            let (vertex, pivot, weight) = if pair.weight_v == 0.0 {
                (pair.u, pair.v, pair.weight_u)
            } else if pair.weight_u == 0.0 {
                (pair.v, pair.u, pair.weight_v)
            } else {
                continue;
            };
            let slot = extended
                .params
                .pivots
                .iter()
                .position(|&v| v == pivot)
                .unwrap();
            let index = vertex * 8 + slot;
            let reconstructed = f64::from(extended.one_sided[index].weight)
                + f64::from(extended.weight_residuals[index]);
            assert!((reconstructed - weight).abs() <= weight.abs() * 1e-14);
        }
    }

    #[test]
    fn streaming_preserves_constraints_on_long_paths_and_stars() {
        for edges in [
            (1..257).map(|v| (v - 1, v)).collect::<Vec<_>>(),
            (1..257).map(|v| (0, v)).collect(),
        ] {
            let graph = Graph::try_from_edges(257, &edges).unwrap();
            for h in [1, 16, 200, 300] {
                let mut rng = StdRng::seed_from_u64(13);
                let reference = graph
                    .prepare_sgd_params(15, 0.1, h, true, &mut rng)
                    .unwrap();
                let mut rng = StdRng::seed_from_u64(13);
                let rows = Distances::new(&graph, h, &mut rng).unwrap();
                let mut constraints = Vec::new();
                rows.visit(&graph, |p| constraints.push(p));
                assert_eq!(constraints, reference.pairs);
                let mut rng = StdRng::seed_from_u64(13);
                let direct = prepare(&graph, 15, 0.1, h, true, &mut rng).unwrap();
                assert_eq!(direct.params.etas, reference.etas);
                assert_eq!(direct.constraint_count, reference.pairs.len());
            }
        }
    }

    #[test]
    fn streaming_matches_independent_cpu_reference_and_schedule_exactly() {
        for n in [2, 7, 32, 73] {
            for seed in [0, 1, 3, 19] {
                let mut edges: Vec<_> = (1..n).map(|v| (v - 1, v)).collect();
                let mut random = StdRng::seed_from_u64(seed);
                for _ in 0..n * 3 {
                    edges.push((random.random_range(0..n), random.random_range(0..n)));
                }
                let graph = Graph::try_from_edges(n, &edges).unwrap();
                for h in [1, 3, 8, n + 1] {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let reference = graph
                        .prepare_sgd_params(15, 0.1, h, true, &mut rng)
                        .unwrap();
                    let mut direct_rng = StdRng::seed_from_u64(seed);
                    let distances = Distances::new(&graph, h, &mut direct_rng).unwrap();
                    let mut actual = Vec::new();
                    distances.visit(&graph, |p| actual.push(p));
                    assert_eq!(actual, reference.pairs, "n={n}, h={h}, seed={seed}");
                    let mut direct_rng = StdRng::seed_from_u64(seed);
                    let direct = prepare(&graph, 15, 0.1, h, true, &mut direct_rng).unwrap();
                    assert_eq!(direct.params.positions, reference.positions);
                    assert_eq!(direct.params.etas, reference.etas);
                    assert_eq!(direct.params.pivots, reference.pivots);
                    assert_eq!(direct.constraint_count, reference.pairs.len());
                    assert_eq!(rng.random::<u64>(), direct_rng.random::<u64>());
                    let old = build_schedule(&graph, &reference, seed).unwrap();
                    let new = build_prepared_schedule(
                        &graph,
                        &direct.params,
                        direct.one_sided,
                        direct.one_sided_count,
                        seed,
                    )
                    .unwrap();
                    assert_eq!(old.one_sided, new.one_sided);
                    assert_eq!(old.two_sided, new.two_sided);
                    assert_eq!(old.round_offsets, new.round_offsets);
                    let mut f = FingerprintBuilder::new("sparse-preprocess-v1");
                    f.usize(reference.pivots.len());
                    for p in reference.pivots {
                        f.usize(p);
                    }
                    f.usize(reference.etas.len());
                    for eta in reference.etas {
                        f.f64(eta);
                    }
                    f.usize(reference.pairs.len());
                    for p in reference.pairs {
                        f.usize(p.u)
                            .usize(p.v)
                            .f64(p.dij)
                            .f64(p.weight_u)
                            .f64(p.weight_v);
                    }
                    assert_eq!(direct.fingerprint, f.finish());
                }
            }
        }
    }
}
