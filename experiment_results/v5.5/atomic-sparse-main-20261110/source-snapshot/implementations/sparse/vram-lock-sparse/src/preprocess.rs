//! Flat CSR CPU BFS Preprocessing for Atomic Sparse SGD (P4).
//! Streams constraints directly into `Vec<GpuPair>` (20 B/pair) while computing
//! `preprocess_sha256` in a single pass, eliminating intermediate `BTreeMap` and `Vec<EdgeInfo>`.

use crate::gpu::GpuPair;
use crate::graph::{calc_learning_rate, init_positions_random, Graph};
use anyhow::{ensure, Context, Result};
use experiment_common::FingerprintBuilder;
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BfsBackend {
    #[default]
    Auto,
    Cpu,
}

impl std::str::FromStr for BfsBackend {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(BfsBackend::Auto),
            "cpu" => Ok(BfsBackend::Cpu),
            other => anyhow::bail!("Unknown BFS backend '{other}'. Valid options: auto, cpu"),
        }
    }
}

pub struct FlatCsr {
    pub num_vertices: usize,
    pub row_offsets: Vec<u32>,
    pub col_indices: Vec<u32>,
}

impl FlatCsr {
    pub fn from_graph(graph: &Graph) -> Self {
        let num_vertices = graph.node_size;
        let mut row_offsets = Vec::with_capacity(num_vertices + 1);
        let mut col_indices = Vec::with_capacity(graph.edge_size * 2);
        row_offsets.push(0);
        for u in 0..num_vertices {
            col_indices.extend(graph.neighbors(u).iter().map(|&v| v as u32));
            row_offsets.push(col_indices.len() as u32);
        }
        Self {
            num_vertices,
            row_offsets,
            col_indices,
        }
    }

    #[inline(always)]
    pub fn neighbors(&self, u: usize) -> &[u32] {
        let start = self.row_offsets[u] as usize;
        let end = self.row_offsets[u + 1] as usize;
        &self.col_indices[start..end]
    }
}

pub struct Distances {
    pub pivots: Vec<usize>,
    pub slot: Vec<usize>,
    pub flat_distances: Vec<u32>,
    pub prefix: Vec<Vec<usize>>,
    pub weight_table: Vec<Vec<f64>>,
}

impl Distances {
    pub fn new<R: Rng + ?Sized>(graph: &Graph, h: usize, rng: &mut R) -> Result<Self> {
        let n = graph.node_size;
        ensure!(
            n > 1 && n < u32::MAX as usize && h > 0,
            "nontrivial graph and positive pivot count required"
        );
        let h = h.min(n);
        let csr = FlatCsr::from_graph(graph);

        let mut flat_distances = vec![u32::MAX; h * n];
        let mut nearest = vec![u32::MAX; n];
        let mut regions = vec![0usize; n];
        let mut pivots = Vec::with_capacity(h);
        let mut max_d_per_pivot = vec![0u32; h];

        let mut queue = Vec::with_capacity(n);
        let mut seen_epoch = vec![0u32; n];
        let mut current_epoch: u32 = 0;

        let mut run_bfs = |source: usize, row_dest: &mut [u32], epoch: &mut u32| -> Result<u32> {
            *epoch += 1;
            let ep = *epoch;
            queue.clear();
            queue.push(source as u32);
            seen_epoch[source] = ep;
            row_dest[source] = 0;
            let mut head = 0;
            while head < queue.len() {
                let u = queue[head] as usize;
                head += 1;
                let next_d = row_dest[u] + 1;
                for &v_u32 in csr.neighbors(u) {
                    let v = v_u32 as usize;
                    if seen_epoch[v] != ep {
                        seen_epoch[v] = ep;
                        row_dest[v] = next_d;
                        queue.push(v_u32);
                    }
                }
            }
            ensure!(queue.len() == n, "connected graph required");
            Ok(row_dest[queue[n - 1] as usize])
        };

        // Pivot 0: uniform random selection
        let first = rng.random_range(0..n);
        pivots.push(first);
        max_d_per_pivot[0] = run_bfs(first, &mut flat_distances[0..n], &mut current_epoch)?;
        nearest.copy_from_slice(&flat_distances[0..n]);
        let mut total: u64 = nearest.iter().map(|&d| d as u64).sum();

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
            max_d_per_pivot[k] = run_bfs(
                next,
                &mut flat_distances[row_offset..row_offset + n],
                &mut current_epoch,
            )?;

            let row = &flat_distances[row_offset..row_offset + n];
            let mut next_total = 0u64;
            for v in 0..n {
                // Strict comparison preserves the reference's lowest pivot ID tie rule
                let d = row[v];
                let cur = nearest[v];
                if d < cur {
                    nearest[v] = d;
                    regions[v] = k;
                    next_total += d as u64;
                } else {
                    next_total += cur as u64;
                }
            }
            total = next_total;
            pivots.push(next);
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

        for pref in &mut prefix {
            for i in 1..pref.len() {
                pref[i] += pref[i - 1];
            }
        }

        let mut weight_table = Vec::with_capacity(h);
        for (&max_d_u32, pref) in max_d_per_pivot.iter().zip(&prefix) {
            let max_d = max_d_u32 as usize;
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

    /// Visit pairs in exact lexicographical (u, v) order, streaming directly into Vec<GpuPair>
    /// and updating SHA-256 fingerprint simultaneously.
    pub fn visit_gpu_pairs(
        &self,
        graph: &Graph,
        fingerprint: &mut FingerprintBuilder,
        pairs: &mut Vec<GpuPair>,
    ) {
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
                            let dij = distance as f64;
                            let weight_u = self.weight_table[pv][distance];
                            fingerprint.edge_info(u, v, dij, weight_u, 0.0);
                            pairs.push(GpuPair {
                                u: u as u32,
                                v: v as u32,
                                dij: dij as f32,
                                weight_u: weight_u as f32,
                                weight_v: 0.0,
                            });
                        }
                        if pivot_index < h && sorted_pivots[pivot_index].0 == edge_v {
                            pivot_index += 1;
                        }
                        fingerprint.edge_info(u, edge_v, 1.0, 1.0, 1.0);
                        pairs.push(GpuPair {
                            u: u as u32,
                            v: edge_v as u32,
                            dij: 1.0,
                            weight_u: 1.0,
                            weight_v: 1.0,
                        });
                    }
                    for &(v, pv) in &sorted_pivots[pivot_index..h] {
                        let distance = tile_row[pivot_index] as usize;
                        pivot_index += 1;
                        let dij = distance as f64;
                        let weight_u = self.weight_table[pv][distance];
                        fingerprint.edge_info(u, v, dij, weight_u, 0.0);
                        pairs.push(GpuPair {
                            u: u as u32,
                            v: v as u32,
                            dij: dij as f32,
                            weight_u: weight_u as f32,
                            weight_v: 0.0,
                        });
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
                            let dij = distance as f64;
                            let weight_u = if pv != usize::MAX {
                                self.weight_table[pv][distance]
                            } else {
                                0.0
                            };
                            let weight_v = pu_weights[distance];
                            fingerprint.edge_info(u, v, dij, weight_u, weight_v);
                            pairs.push(GpuPair {
                                u: u as u32,
                                v: v as u32,
                                dij: dij as f32,
                                weight_u: weight_u as f32,
                                weight_v: weight_v as f32,
                            });
                        }
                        fingerprint.edge_info(u, edge_v, 1.0, 1.0, 1.0);
                        pairs.push(GpuPair {
                            u: u as u32,
                            v: edge_v as u32,
                            dij: 1.0,
                            weight_u: 1.0,
                            weight_v: 1.0,
                        });
                        cur_v = edge_v + 1;
                    }
                    for (v, (&dist_u32, &pv)) in
                        (cur_v..n).zip(dist_row[cur_v..n].iter().zip(&self.slot[cur_v..n]))
                    {
                        let distance = dist_u32 as usize;
                        let dij = distance as f64;
                        let weight_u = if pv != usize::MAX {
                            self.weight_table[pv][distance]
                        } else {
                            0.0
                        };
                        let weight_v = pu_weights[distance];
                        fingerprint.edge_info(u, v, dij, weight_u, weight_v);
                        pairs.push(GpuPair {
                            u: u as u32,
                            v: v as u32,
                            dij: dij as f32,
                            weight_u: weight_u as f32,
                            weight_v: weight_v as f32,
                        });
                    }
                }
            }
        }
    }
}

pub struct PreparedSparse {
    pub etas: Vec<f64>,
    pub initial_positions: Vec<[f64; 2]>,
    pub pairs: Vec<GpuPair>,
    pub pivots: Vec<usize>,
    pub constraint_count: usize,
    pub preprocess_sha256: String,
    pub fingerprint: String,
    pub center: bool,
    pub wmin: f64,
    pub wmax: f64,
}

pub fn prepare<R: Rng + ?Sized>(
    graph: &Graph,
    iterations: usize,
    epsilon: f64,
    h: usize,
    center: bool,
    rng: &mut R,
) -> Result<PreparedSparse> {
    prepare_with_backend(graph, iterations, epsilon, h, center, rng, BfsBackend::Auto)
}

pub fn prepare_with_backend<R: Rng + ?Sized>(
    graph: &Graph,
    iterations: usize,
    epsilon: f64,
    h: usize,
    center: bool,
    rng: &mut R,
    _backend: BfsBackend,
) -> Result<PreparedSparse> {
    let distances = Distances::new(graph, h, rng)?;

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
    let initial_positions = init_positions_random(graph.node_size, center, rng);

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

    let mut pairs = Vec::with_capacity(constraint_count);
    distances.visit_gpu_pairs(graph, &mut fingerprint, &mut pairs);
    ensure!(
        pairs.len() == constraint_count,
        "constraint count mismatch"
    );

    let preprocess_sha256 = fingerprint.finish();
    let fingerprint_str = preprocess_sha256.clone();
    let pivots = distances.pivots.clone();

    // Drop flat_distances immediately before returning
    drop(distances);

    Ok(PreparedSparse {
        etas,
        initial_positions,
        pairs,
        pivots,
        constraint_count,
        preprocess_sha256,
        fingerprint: fingerprint_str,
        center,
        wmin,
        wmax,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn streaming_matches_baseline_reference_sha256_bit_exact() {
        for edges in [
            (1..20).map(|v| (v - 1, v)).collect::<Vec<_>>(),
            (1..20).map(|v| (0, v)).collect(),
        ] {
            let graph = Graph::try_from_edges(20, &edges).unwrap();
            for h in [1, 3, 10, 25] {
                let mut rng1 = StdRng::seed_from_u64(42);
                let ref_params = graph.prepare_sgd_params(5, 0.1, h, true, &mut rng1).unwrap();
                let mut ref_fp = FingerprintBuilder::new("sparse-preprocess-v1");
                ref_fp.usize(ref_params.pivots.len());
                for &p in &ref_params.pivots {
                    ref_fp.usize(p);
                }
                ref_fp.usize(ref_params.etas.len());
                for &eta in &ref_params.etas {
                    ref_fp.f64(eta);
                }
                ref_fp.usize(ref_params.pairs.len());
                for p in &ref_params.pairs {
                    ref_fp.usize(p.u).usize(p.v).f64(p.dij).f64(p.weight_u).f64(p.weight_v);
                }
                let ref_sha = ref_fp.finish();

                let mut rng2 = StdRng::seed_from_u64(42);
                let prepared = prepare(&graph, 5, 0.1, h, true, &mut rng2).unwrap();

                assert_eq!(prepared.preprocess_sha256, ref_sha);
                assert_eq!(prepared.constraint_count, ref_params.pairs.len());
                assert_eq!(prepared.pivots, ref_params.pivots);
                assert_eq!(prepared.etas, ref_params.etas);
            }
        }
    }
}
