//! Streaming RR preprocessing. The CPU reference stays independent so equivalence
//! of constraints, RNG consumption, learning rates and schedules can be tested.
use crate::graph::{calc_learning_rate, init_positions_random, EdgeInfo, Graph, SgdParams};
use crate::schedule::OneSidedEntry;
use anyhow::{ensure, Context, Result};
use experiment_common::FingerprintBuilder;
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BfsBackend {
    #[default]
    Auto,
    Gpu,
    Cpu,
}

impl std::str::FromStr for BfsBackend {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(BfsBackend::Auto),
            "gpu" => Ok(BfsBackend::Gpu),
            "cpu" => Ok(BfsBackend::Cpu),
            other => anyhow::bail!("Unknown BFS backend '{other}'. Valid options: auto, gpu, cpu"),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn graph_to_csr(graph: &Graph) -> webgpu_bfs_core::CsrGraph {
    let num_vertices = graph.node_size;
    let mut row_offsets = Vec::with_capacity(num_vertices + 1);
    let mut col_indices = Vec::with_capacity(graph.edge_size * 2);
    row_offsets.push(0);
    for u in 0..num_vertices {
        col_indices.extend(graph.neighbors(u).iter().map(|&v| v as u32));
        row_offsets.push(col_indices.len() as u32);
    }
    webgpu_bfs_core::CsrGraph {
        num_vertices,
        num_edges: col_indices.len(),
        row_offsets,
        col_indices,
    }
}

pub struct PreparedRun {
    /// Only two-sided constraints are retained here, in canonical pair order.
    pub params: SgdParams,
    pub one_sided: Vec<OneSidedEntry>,
    pub one_sided_count: usize,
    pub constraint_count: usize,
    pub wmin: f64,
    pub wmax: f64,
    pub fingerprint: String,
}

pub struct Distances {
    pub pivots: Vec<usize>,
    pub slot: Vec<usize>,
    pub flat_distances: Vec<u32>,
    pub prefix: Vec<Vec<usize>>,
}

impl Distances {
    #[allow(dead_code)]
    pub fn new<R: Rng + ?Sized>(graph: &Graph, h: usize, rng: &mut R) -> Result<Self> {
        Self::new_with_backend(graph, h, rng, BfsBackend::Auto)
    }

    pub fn new_with_backend<R: Rng + ?Sized>(
        graph: &Graph,
        h: usize,
        rng: &mut R,
        backend: BfsBackend,
    ) -> Result<Self> {
        let n = graph.node_size;
        ensure!(
            n > 1 && n < u32::MAX as usize && h > 0,
            "nontrivial graph and positive pivot count required"
        );
        let h = h.min(n);

        // Topology-aware method selection
        #[cfg(not(target_arch = "wasm32"))]
        let use_gpu = {
            let max_degree = (0..n).map(|u| graph.neighbors(u).len()).max().unwrap_or(0);
            let is_road_or_narrow = max_degree <= 8 && graph.edge_size <= 4 * n;
            match backend {
                BfsBackend::Auto => is_road_or_narrow && n >= 20_000 && n <= 114_688,
                BfsBackend::Gpu => true,
                BfsBackend::Cpu => false,
            }
        };
        #[cfg(target_arch = "wasm32")]
        let _ = backend;

        let mut flat_distances = vec![u32::MAX; h * n];
        let mut queue = Vec::with_capacity(n);
        let mut nearest = vec![u32::MAX; n];
        let mut regions = vec![0; n];
        let mut pivots = Vec::with_capacity(h);

        #[cfg(not(target_arch = "wasm32"))]
        let (csr, gpu_engine, mut gpu_row) = if use_gpu {
            let csr = graph_to_csr(graph);
            let engine = webgpu_bfs_gpu::engine::GpuBfsEngine::default_with_method(
                webgpu_bfs_gpu::engine::GpuBfsMethod::WorkgroupPersistentBfs,
            )
            .context("Failed to initialize webgpu-bfs GPU engine for pivot BFS")?;
            let gpu_row = vec![u32::MAX; n];
            (Some(csr), Some(engine), gpu_row)
        } else {
            (None, None, Vec::new())
        };

        let mut run_bfs = |source: usize, row_dest: &mut [u32]| -> Result<()> {
            #[cfg(not(target_arch = "wasm32"))]
            if use_gpu {
                let engine = gpu_engine.as_ref().unwrap();
                let sources = [source as u32];
                engine.run_ms_bfs_into(csr.as_ref().unwrap(), &sources, &mut gpu_row)?;
                row_dest.copy_from_slice(&gpu_row[..n]);
                return Ok(());
            }
            {
                row_dest.fill(u32::MAX);
                queue.clear();
                queue.push(source);
                row_dest[source] = 0;
                let mut head = 0;
                while head < queue.len() {
                    let u = queue[head];
                    head += 1;
                    let next_d = row_dest[u] + 1;
                    for &v in graph.neighbors(u) {
                        if row_dest[v] == u32::MAX {
                            row_dest[v] = next_d;
                            queue.push(v);
                        }
                    }
                }
                ensure!(queue.len() == n, "connected graph required");
                Ok(())
            }
        };

        // Pivot 0: uniform random selection
        let first = rng.random_range(0..n);
        pivots.push(first);
        run_bfs(first, &mut flat_distances[0..n])?;
        nearest.copy_from_slice(&flat_distances[0..n]);

        // Pivots 1 .. h-1: distance-proportional sampling with exact tie breaking
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

            let k = pivots.len();
            let row_offset = k * n;
            run_bfs(next, &mut flat_distances[row_offset..row_offset + n])?;

            let row = &flat_distances[row_offset..row_offset + n];
            for v in 0..n {
                // Strict comparison preserves the reference's lowest pivot ID tie rule
                if row[v] < nearest[v] {
                    nearest[v] = row[v];
                    regions[v] = k;
                }
            }
            pivots.push(next);
        }

        // Inverted O(N) Prefix Counts
        let mut max_d_per_pivot = vec![0u32; h];
        for p in 0..h {
            let row = &flat_distances[p * n..(p + 1) * n];
            max_d_per_pivot[p] = *row.iter().max().unwrap_or(&0);
        }

        let mut prefix: Vec<Vec<usize>> = max_d_per_pivot
            .iter()
            .map(|&max_d| vec![0usize; max_d as usize + 1])
            .collect();

        for v in 0..n {
            let reg = regions[v];
            let d = flat_distances[reg * n + v] as usize;
            prefix[reg][d] += 1;
        }

        for p in 0..h {
            for i in 1..prefix[p].len() {
                prefix[p][i] += prefix[p][i - 1];
            }
        }

        let mut slot = vec![usize::MAX; n];
        for (p, &v) in pivots.iter().enumerate() {
            slot[v] = p;
        }

        Ok(Self {
            pivots,
            slot,
            flat_distances,
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
                        self.flat_distances[pu * n + v]
                    } else {
                        self.flat_distances[pv * n + u]
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
    prepare_with_backend(graph, iterations, epsilon, h, center, rng, BfsBackend::Auto)
}

pub fn prepare_with_backend<R: Rng + ?Sized>(
    graph: &Graph,
    iterations: usize,
    epsilon: f64,
    h: usize,
    center: bool,
    rng: &mut R,
    backend: BfsBackend,
) -> Result<PreparedRun> {
    let distances = Distances::new_with_backend(graph, h, rng, backend)?;
    // Unweighted graph edges have distance one and override both weights with 1.
    // Traverse each BFS row contiguously; avoid a second canonical pair traversal.
    let mut wmin: f64 = 1.0;
    let mut wmax: f64 = 1.0;
    let n = graph.node_size;
    for p in 0..distances.pivots.len() {
        let row = &distances.flat_distances[p * n..(p + 1) * n];
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
        constraint_count,
        wmin,
        wmax,
        fingerprint: fingerprint.finish(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::{build_prepared_schedule, build_schedule};
    use rand::{rngs::StdRng, SeedableRng};

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

    #[test]
    fn cpu_and_gpu_pivot_bfs_produce_bit_exact_fingerprints() {
        let edges: Vec<_> = (1..100).map(|v| (v - 1, v)).collect();
        let graph = Graph::try_from_edges(100, &edges).unwrap();
        let mut rng1 = StdRng::seed_from_u64(42);
        let cpu_run =
            prepare_with_backend(&graph, 5, 0.1, 10, true, &mut rng1, BfsBackend::Cpu).unwrap();
        let mut rng2 = StdRng::seed_from_u64(42);
        let gpu_run =
            prepare_with_backend(&graph, 5, 0.1, 10, true, &mut rng2, BfsBackend::Gpu).unwrap();

        assert_eq!(cpu_run.fingerprint, gpu_run.fingerprint);
        assert_eq!(cpu_run.params.positions, gpu_run.params.positions);
        assert_eq!(cpu_run.params.pivots, gpu_run.params.pivots);
        assert_eq!(cpu_run.params.etas, gpu_run.params.etas);
        assert_eq!(cpu_run.constraint_count, gpu_run.constraint_count);
    }
}
