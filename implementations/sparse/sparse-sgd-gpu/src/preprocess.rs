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
    pub weight_table: Vec<Vec<f64>>,
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

        // Sequential single-source pivot BFS (S=1 per pivot) is always faster on CPU
        // than 200 sequential single-source GPU dispatches + second wgpu device init.
        #[cfg(not(target_arch = "wasm32"))]
        let use_gpu = matches!(backend, BfsBackend::Gpu);
        #[cfg(target_arch = "wasm32")]
        let _ = backend;

        let mut flat_distances = vec![u32::MAX; h * n];
        let mut queue = Vec::<u32>::with_capacity(n);
        let mut nearest = vec![u32::MAX; n];
        let mut regions = vec![0usize; n];
        let mut pivots = Vec::with_capacity(h);
        let mut max_d_per_pivot = vec![0u32; h];

        let (csr_offsets, csr_neighbors) = {
            let mut offsets = Vec::with_capacity(n + 1);
            let mut neighbors = Vec::with_capacity(graph.edge_size * 2);
            offsets.push(0u32);
            for u in 0..n {
                neighbors.extend(graph.neighbors(u).iter().map(|&v| v as u32));
                offsets.push(neighbors.len() as u32);
            }
            (offsets, neighbors)
        };

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

        let mut run_bfs = |source: usize, row_dest: &mut [u32]| -> Result<u32> {
            #[cfg(not(target_arch = "wasm32"))]
            if use_gpu {
                let engine = gpu_engine.as_ref().unwrap();
                let sources = [source as u32];
                engine.run_ms_bfs_into(csr.as_ref().unwrap(), &sources, &mut gpu_row)?;
                row_dest.copy_from_slice(&gpu_row[..n]);
                ensure!(
                    row_dest.iter().all(|&d| d != u32::MAX),
                    "connected graph required"
                );
                return Ok(*row_dest.iter().max().unwrap_or(&0));
            }
            {
                queue.clear();
                queue.push(source as u32);
                row_dest[source] = 0;
                let mut head = 0;
                while head < queue.len() {
                    let u = queue[head] as usize;
                    head += 1;
                    let next_d = row_dest[u] + 1;
                    let start_edge = csr_offsets[u] as usize;
                    let end_edge = csr_offsets[u + 1] as usize;
                    for &v_u32 in &csr_neighbors[start_edge..end_edge] {
                        let v = v_u32 as usize;
                        if row_dest[v] == u32::MAX {
                            row_dest[v] = next_d;
                            queue.push(v_u32);
                        }
                    }
                }
                ensure!(queue.len() == n, "connected graph required");
                let max_d = row_dest[*queue.last().unwrap() as usize];
                Ok(max_d)
            }
        };

        // Pivot 0: uniform random selection
        let first = rng.random_range(0..n);
        pivots.push(first);
        max_d_per_pivot[0] = run_bfs(first, &mut flat_distances[0..n])?;
        nearest.copy_from_slice(&flat_distances[0..n]);
        let mut total = nearest.iter().try_fold(0_u64, |sum, &d| {
            sum.checked_add(d as u64).context("distance sum overflow")
        })?;

        // Pivots 1 .. h-1: distance-proportional sampling with exact tie breaking
        while pivots.len() < h {
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
            max_d_per_pivot[k] = run_bfs(next, &mut flat_distances[row_offset..row_offset + n])?;

            let row = &flat_distances[row_offset..row_offset + n];
            let mut next_total = 0_u64;
            for v in 0..n {
                let d = row[v];
                let near = nearest[v];
                // Strict comparison preserves the reference's lowest pivot ID tie rule
                if d < near {
                    nearest[v] = d;
                    regions[v] = k;
                    next_total += d as u64;
                } else {
                    next_total += near as u64;
                }
            }
            total = next_total;
            pivots.push(next);
        }

        // Inverted O(N) Prefix Counts
        let mut prefix: Vec<Vec<usize>> = max_d_per_pivot
            .iter()
            .map(|&max_d| vec![0usize; max_d as usize + 1])
            .collect();

        for v in 0..n {
            let reg = regions[v];
            let d = flat_distances[reg * n + v] as usize;
            prefix[reg][d] += 1;
        }

        let mut weight_table = Vec::with_capacity(h);
        for pref in &mut prefix {
            for i in 1..pref.len() {
                pref[i] += pref[i - 1];
            }
            let max_d = pref.len().saturating_sub(1);
            let mut table = vec![0.0_f64; max_d + 1];
            for (d, slot) in table.iter_mut().enumerate().skip(1) {
                let dij = d as f64;
                *slot = pref[d / 2] as f64 / (dij * dij);
            }
            weight_table.push(table);
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
            weight_table,
        })
    }

    /// Merge sorted adjacency and pivot endpoints per row. This is exactly the
    /// BTreeMap reference's lexicographic order, without materializing that map.
    #[allow(dead_code)]
    fn visit(&self, graph: &Graph, mut emit: impl FnMut(EdgeInfo)) {
        self.visit_with_slots(graph, |pair, _, _| emit(pair));
    }

    fn visit_with_slots(&self, graph: &Graph, mut emit: impl FnMut(EdgeInfo, usize, usize)) {
        let n = graph.node_size;
        let h = self.pivots.len();
        let mut sorted_pivots: Vec<(usize, usize)> = self
            .pivots
            .iter()
            .enumerate()
            .map(|(slot_idx, &v)| (v, slot_idx))
            .collect();
        sorted_pivots.sort_unstable_by_key(|&(v, _)| v);

        const STRIP: usize = 64;
        let mut tile_dist = vec![0u32; STRIP * h];
        let mut first_p = 0usize;

        for u_start in (0..n).step_by(STRIP) {
            let u_end = (u_start + STRIP).min(n);
            let strip_len = u_end - u_start;
            while first_p < h && sorted_pivots[first_p].0 <= u_start {
                first_p += 1;
            }
            for (p_idx, &(_, pv_slot)) in sorted_pivots.iter().enumerate().skip(first_p) {
                let src = &self.flat_distances[pv_slot * n + u_start..pv_slot * n + u_end];
                for i in 0..strip_len {
                    tile_dist[i * h + p_idx] = src[i];
                }
            }

            let mut u_first_p = first_p;
            for u in u_start..u_end {
                while u_first_p < h && sorted_pivots[u_first_p].0 <= u {
                    u_first_p += 1;
                }
                let pu = self.slot[u];
                let neighbors = graph.neighbors(u);
                let edge_start = neighbors.partition_point(|&v| v <= u);
                let rem_edges = &neighbors[edge_start..];
                if pu == usize::MAX {
                    let mut pivot_index = u_first_p;
                    let tile_row = &tile_dist[(u - u_start) * h..(u - u_start + 1) * h];
                    for &edge_v in rem_edges {
                        while pivot_index < h && sorted_pivots[pivot_index].0 < edge_v {
                            let (v, pv) = sorted_pivots[pivot_index];
                            let distance = tile_row[pivot_index] as usize;
                            pivot_index += 1;
                            emit(
                                EdgeInfo {
                                    u,
                                    v,
                                    dij: distance as f64,
                                    weight_u: self.weight_table[pv][distance],
                                    weight_v: 0.0,
                                },
                                usize::MAX,
                                pv,
                            );
                        }
                        let pv = if pivot_index < h && sorted_pivots[pivot_index].0 == edge_v {
                            let pv = sorted_pivots[pivot_index].1;
                            pivot_index += 1;
                            pv
                        } else {
                            usize::MAX
                        };
                        emit(
                            EdgeInfo {
                                u,
                                v: edge_v,
                                dij: 1.0,
                                weight_u: 1.0,
                                weight_v: 1.0,
                            },
                            usize::MAX,
                            pv,
                        );
                    }
                    for &(v, pv) in &sorted_pivots[pivot_index..h] {
                        let distance = tile_row[pivot_index] as usize;
                        pivot_index += 1;
                        emit(
                            EdgeInfo {
                                u,
                                v,
                                dij: distance as f64,
                                weight_u: self.weight_table[pv][distance],
                                weight_v: 0.0,
                            },
                            usize::MAX,
                            pv,
                        );
                    }
                } else {
                    let dist_row = &self.flat_distances[pu * n..(pu + 1) * n];
                    let pu_weights = &self.weight_table[pu];
                    let mut cur_v = u + 1;
                    for &edge_v in rem_edges {
                        for (v, (&dist_u32, &pv)) in (cur_v..edge_v)
                            .zip(dist_row[cur_v..edge_v].iter().zip(&self.slot[cur_v..edge_v]))
                        {
                            let distance = dist_u32 as usize;
                            let weight_u = if pv != usize::MAX {
                                self.weight_table[pv][distance]
                            } else {
                                0.0
                            };
                            emit(
                                EdgeInfo {
                                    u,
                                    v,
                                    dij: distance as f64,
                                    weight_u,
                                    weight_v: pu_weights[distance],
                                },
                                pu,
                                pv,
                            );
                        }
                        emit(
                            EdgeInfo {
                                u,
                                v: edge_v,
                                dij: 1.0,
                                weight_u: 1.0,
                                weight_v: 1.0,
                            },
                            pu,
                            self.slot[edge_v],
                        );
                        cur_v = edge_v + 1;
                    }
                    for (v, (&dist_u32, &pv)) in
                        (cur_v..n).zip(dist_row[cur_v..n].iter().zip(&self.slot[cur_v..n]))
                    {
                        let distance = dist_u32 as usize;
                        let weight_u = if pv != usize::MAX {
                            self.weight_table[pv][distance]
                        } else {
                            0.0
                        };
                        emit(
                            EdgeInfo {
                                u,
                                v,
                                dij: distance as f64,
                                weight_u,
                                weight_v: pu_weights[distance],
                            },
                            pu,
                            pv,
                        );
                    }
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
    // Every integer distance 2..=max_d_per_pivot[p] occurs in connected unweighted BFS.
    let mut wmin: f64 = 1.0;
    let mut wmax: f64 = 1.0;
    for table in &distances.weight_table {
        for &w in table.iter().skip(2) {
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
    let two_sided_capacity = graph.edge_size + h * h.saturating_sub(1) / 2;
    let mut pairs = Vec::with_capacity(two_sided_capacity);
    distances.visit_with_slots(graph, |pair, pu, pv| {
        fingerprint.edge_info(pair.u, pair.v, pair.dij, pair.weight_u, pair.weight_v);
        let (vertex, pivot_slot, weight) = if pair.weight_v == 0.0 {
            (pair.u, pv, pair.weight_u)
        } else if pair.weight_u == 0.0 {
            (pair.v, pu, pair.weight_v)
        } else {
            pairs.push(pair);
            return;
         };
        one_sided[vertex * h + pivot_slot] = OneSidedEntry {
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
