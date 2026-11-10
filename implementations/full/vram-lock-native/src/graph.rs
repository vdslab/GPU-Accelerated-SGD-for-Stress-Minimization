use anyhow::{ensure, Result};
use bytemuck::{Pod, Zeroable};
#[cfg(not(target_arch = "wasm32"))]
use sprs::io::read_matrix_market;
#[cfg(not(target_arch = "wasm32"))]
use sprs::num_kinds::Pattern;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[derive(Debug)]
pub struct Graph {
    pub node_size: usize,
    pub edge_size: usize,
    pub edge_src: Vec<usize>,
    pub edge_dst: Vec<usize>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable, Default)]
pub struct EdgeInfo {
    pub u: u32,
    pub v: u32,
    pub dij: f32,
    pub wij: f32,
}

#[allow(dead_code)]
pub type GpuPair = EdgeInfo;
#[allow(dead_code)]
pub type ShortestPathPair = EdgeInfo;

#[allow(dead_code)]
#[derive(Debug)]
pub struct SgdParams {
    pub etas: Vec<f64>,
    pub positions: Vec<[f64; 2]>,
    pub pairs: Vec<EdgeInfo>,
    pub center: bool,
}

pub struct FlatCsr {
    pub offsets: Vec<u32>,
    pub neighbors: Vec<u32>,
}


impl Graph {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_mtx(path: &Path) -> Result<Self> {
        let matrix: sprs::TriMat<Pattern> = read_matrix_market(path)?;
        let node_size: usize = matrix.rows();

        let mut edge_src = Vec::new();
        let mut edge_dst = Vec::new();

        for (row, col) in matrix.row_inds().iter().zip(matrix.col_inds().iter()) {
            if row != col {
                edge_src.push(*row);
                edge_dst.push(*col);
            }
        }

        let edge_size = edge_src.len();
        Ok(Graph {
            node_size,
            edge_size,
            edge_src,
            edge_dst,
        })
    }

    #[allow(dead_code)]
    pub fn from_edges(node_size: usize, edge_src: Vec<usize>, edge_dst: Vec<usize>) -> Self {
        let edge_size = edge_src.len();
        Graph {
            node_size,
            edge_size,
            edge_src,
            edge_dst,
        }
    }

    pub fn to_flat_csr(&self) -> FlatCsr {
        let n = self.node_size;
        let mut degrees = vec![0u32; n];
        for i in 0..self.edge_size {
            degrees[self.edge_src[i]] += 1;
            degrees[self.edge_dst[i]] += 1;
        }

        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        let mut sum = 0u32;
        for &deg in &degrees {
            sum += deg;
            offsets.push(sum);
        }

        let mut cursor = offsets[0..n].to_vec();
        let mut neighbors = vec![0u32; sum as usize];
        for i in 0..self.edge_size {
            let u = self.edge_src[i];
            let v = self.edge_dst[i];

            let u_pos = cursor[u] as usize;
            neighbors[u_pos] = v as u32;
            cursor[u] += 1;

            let v_pos = cursor[v] as usize;
            neighbors[v_pos] = u as u32;
            cursor[v] += 1;
        }

        FlatCsr { offsets, neighbors }
    }

    pub fn calc_edge_info_csr(&self, csr: &FlatCsr) -> Result<(Vec<EdgeInfo>, f64, f64)> {
        let n = self.node_size;
        if n <= 1 {
            return Ok((Vec::new(), f64::INFINITY, 0.0));
        }

        let offsets = &csr.offsets;
        let neighbors = &csr.neighbors;

        // Upfront connectivity check
        if n > 1 {
            let mut visited = vec![false; n];
            let mut queue = Vec::with_capacity(n);
            visited[0] = true;
            queue.push(0u32);
            let mut head = 0;
            while head < queue.len() {
                let curr = queue[head] as usize;
                head += 1;
                let start = offsets[curr] as usize;
                let end = offsets[curr + 1] as usize;
                for &nbr in &neighbors[start..end] {
                    let nbr_idx = nbr as usize;
                    if !visited[nbr_idx] {
                        visited[nbr_idx] = true;
                        queue.push(nbr);
                    }
                }
            }
            ensure!(
                queue.len() == n,
                "Graph must be connected for Full SGD: visited {} of {} nodes",
                queue.len(),
                n
            );
        }

        let total_pairs = n * (n - 1) / 2;
        let mut pairs = vec![EdgeInfo::default(); total_pairs];

        let mut dmin = f64::INFINITY;
        let mut dmax = 0.0f64;

        let mut dist = vec![u32::MAX; n];
        let mut queue = Vec::with_capacity(n);

        for u in 0..n - 1 {
            let slice_offset = u * n - u * (u + 1) / 2;

            dist[u] = 0;
            queue.push(u as u32);
            let mut head = 0;

            while head < queue.len() {
                let curr = queue[head] as usize;
                head += 1;
                let next_d = dist[curr] + 1;
                let start = offsets[curr] as usize;
                let end = offsets[curr + 1] as usize;

                for &nbr in &neighbors[start..end] {
                    let nbr_idx = nbr as usize;
                    if dist[nbr_idx] == u32::MAX {
                        dist[nbr_idx] = next_d;
                        queue.push(nbr);
                    }
                }
            }

            for v in (u + 1)..n {
                let d = dist[v];
                if d != u32::MAX && d > 0 {
                    let dij = d as f32;
                    let wij = 1.0f32 / (dij * dij);
                    let d_f64 = d as f64;
                    if d_f64 < dmin {
                        dmin = d_f64;
                    }
                    if d_f64 > dmax {
                        dmax = d_f64;
                    }
                    pairs[slice_offset + (v - u - 1)] = EdgeInfo {
                        u: u as u32,
                        v: v as u32,
                        dij,
                        wij,
                    };
                }
            }

            for &v in queue.iter() {
                dist[v as usize] = u32::MAX;
            }
            queue.clear();
        }

        let wmin = 1.0 / (dmax * dmax);
        let wmax = 1.0 / (dmin * dmin);

        Ok((pairs, wmin, wmax))
    }

    /// Precompute SGD parameters using single-threaded CSR BFS
    pub fn prepare_sgd_params(
        &self,
        iterations: usize,
        epsilon: f64,
        center: bool,
        seed: u64,
    ) -> Result<SgdParams> {
        let csr = self.to_flat_csr();
        let (pairs, wmin, wmax) = self.calc_edge_info_csr(&csr)?;

        let etas = calc_learning_rate(iterations, wmin, wmax, epsilon);
        let positions = experiment_common::seed::seeded_positions(self.node_size, seed, center);

        Ok(SgdParams {
            etas,
            positions,
            pairs,
            center,
        })
    }
}

pub fn calc_learning_rate(tmax: usize, wmin: f64, wmax: f64, eps: f64) -> Vec<f64> {
    let eta_max = 1.0 / wmin;
    let eta_min = eps / wmax;
    if tmax == 1 {
        return vec![eta_max];
    }
    let lamb = (eta_max / eta_min).ln() / (tmax - 1) as f64;

    (0..tmax)
        .map(|t| eta_max * (-lamb * t as f64).exp())
        .collect()
}

#[allow(dead_code)]
pub fn build_pairs_from_dist_u16(n: usize, dist_u16: &[u16], seed: u64) -> Vec<EdgeInfo> {
    use experiment_common::seed::{rng_for_stream, FULL_UPDATE_STREAM};
    use rand::seq::SliceRandom;

    let total_pairs = n * n.saturating_sub(1) / 2;
    let mut pairs = Vec::with_capacity(total_pairs);
    for u in 0..n.saturating_sub(1) {
        let row_offset = u * n;
        for v in (u + 1)..n {
            let d = dist_u16[row_offset + v];
            if d != u16::MAX && d > 0 {
                let dij = f32::from(d);
                let wij = 1.0f32 / (dij * dij);
                pairs.push(EdgeInfo {
                    u: u as u32,
                    v: v as u32,
                    dij,
                    wij,
                });
            }
        }
    }
    let mut shuffle_rng = rng_for_stream(seed, FULL_UPDATE_STREAM);
    pairs.shuffle(&mut shuffle_rng);
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_slice_indices() {
        let n = 6;
        let mut seen = vec![false; n * (n - 1) / 2];
        for u in 0..n - 1 {
            let offset = u * n - u * (u + 1) / 2;
            for v in (u + 1)..n {
                let idx = offset + (v - u - 1);
                assert!(!seen[idx], "Duplicate write at {idx}");
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "Missing write to some pair");
    }

    #[test]
    fn test_flat_csr_and_bfs_cycle6() {
        let graph = Graph {
            node_size: 6,
            edge_size: 6,
            edge_src: vec![0, 1, 2, 3, 4, 5],
            edge_dst: vec![1, 2, 3, 4, 5, 0],
        };
        let csr = graph.to_flat_csr();
        assert_eq!(csr.offsets.len(), 7);
        assert_eq!(csr.neighbors.len(), 12);

        let (pairs, wmin, wmax) = graph.calc_edge_info_csr(&csr).unwrap();
        assert_eq!(pairs.len(), 15);
        assert!((wmin - 1.0 / 9.0).abs() < 1e-6);
        assert!((wmax - 1.0).abs() < 1e-6);

        // Verify canonical ordering u < v
        for pair in &pairs {
            assert!(pair.u < pair.v);
            assert!(pair.dij > 0.0);
            assert!(pair.wij > 0.0);
        }
    }

    #[test]
    fn test_disconnected_graph_rejected() {
        let graph = Graph {
            node_size: 4,
            edge_size: 2,
            edge_src: vec![0, 2],
            edge_dst: vec![1, 3],
        };
        let csr = graph.to_flat_csr();
        let result = graph.calc_edge_info_csr(&csr);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Graph must be connected for Full SGD: visited 2 of 4 nodes"),
            "unexpected error message: {err}"
        );
    }
}
