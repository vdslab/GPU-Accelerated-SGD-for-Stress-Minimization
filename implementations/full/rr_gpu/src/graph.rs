use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use sprs::io::read_matrix_market;
#[cfg(not(target_arch = "wasm32"))]
use sprs::num_kinds::Pattern;
use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[derive(Debug)]
pub struct Graph {
    pub node_size: usize,
    pub edge_size: usize,
    pub edge_src: Vec<usize>,
    pub edge_dst: Vec<usize>,
}

#[path = "bfs.rs"]
pub mod bfs;

#[derive(Debug, Clone)]
pub struct SgdParams {
    pub etas: Vec<f64>,
    pub positions: Vec<[f64; 2]>,
    pub dist_u16: Vec<u16>,
    pub constraint_count: usize,
    pub pairs: Vec<EdgeInfo>,
    #[allow(dead_code)]
    pub center: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeInfo {
    pub u: usize,
    pub v: usize,
    pub dij: f64,
    #[allow(dead_code)]
    pub wij: f64,
}

impl Graph {
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

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_mtx(path: &Path) -> Result<Self> {
        let matrix: sprs::TriMat<Pattern> = read_matrix_market(path)?;

        let node_size: usize = matrix.rows();

        // Filter out self-loops
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

    pub fn calc_adj_matrix(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.node_size];
        for i in 0..self.edge_size {
            adj[self.edge_src[i]].push(self.edge_dst[i]);
            adj[self.edge_dst[i]].push(self.edge_src[i]);
        }
        adj
    }

    #[allow(dead_code)]
    pub fn calc_dist_matrix(&self) -> Vec<Vec<usize>> {
        let adj = Self::calc_adj_matrix(self);
        let n = adj.len();

        (0..n)
            .map(|i| {
                let mut dist_row = vec![usize::MAX; n];
                let mut deq = VecDeque::new();
                let mut seen = vec![false; n];

                dist_row[i] = 0;
                deq.push_back(i);
                seen[i] = true;

                while let Some(v) = deq.pop_front() {
                    for &u in &adj[v] {
                        if !seen[u] {
                            seen[u] = true;
                            dist_row[u] = dist_row[v] + 1;
                            deq.push_back(u);
                        }
                    }
                }
                dist_row
            })
            .collect()
    }

    #[allow(dead_code)]
    pub fn calc_edge_info(&self, dist: &[Vec<usize>]) -> (Vec<EdgeInfo>, f64, f64) {
        let n = dist.len();

        let row_results: Vec<(Vec<EdgeInfo>, f64, f64)> = (0..n)
            .map(|u| {
                let mut local_pairs = Vec::new();
                let mut dmin = f64::INFINITY;
                let mut dmax = 0.0_f64;

                for v in (u + 1)..n {
                    if dist[u][v] == usize::MAX {
                        continue;
                    }
                    let dij = dist[u][v] as f64;
                    if dij <= 0.0 {
                        continue;
                    }
                    let wij = 1.0 / (dij * dij);
                    local_pairs.push(EdgeInfo { u, v, dij, wij });
                    dmin = dmin.min(dij);
                    dmax = dmax.max(dij);
                }
                (local_pairs, dmin, dmax)
            })
            .collect();

        // 各行の結果を統合
        let mut pairs = Vec::new();
        let mut dmin = f64::INFINITY;
        let mut dmax = 0.0_f64;
        for (local_pairs, local_dmin, local_dmax) in row_results {
            pairs.extend(local_pairs);
            dmin = dmin.min(local_dmin);
            dmax = dmax.max(local_dmax);
        }

        let wmin = 1.0 / (dmax * dmax);
        let wmax = 1.0 / (dmin * dmin);

        (pairs, wmin, wmax)
    }

    /// Precompute SGD parameters
    #[allow(dead_code)]
    pub fn prepare_sgd_params(
        &self,
        iterations: usize,
        epsilon: f64,
        center: bool,
        seed: u64,
    ) -> SgdParams {
        let (params, _timing) = self
            .prepare_sgd_params_with_backend(
                iterations,
                epsilon,
                center,
                seed,
                self::bfs::BfsBackend::Auto,
            )
            .expect("APSP calculation failed");
        params
    }

    pub fn prepare_sgd_params_with_backend(
        &self,
        iterations: usize,
        epsilon: f64,
        center: bool,
        seed: u64,
        backend: self::bfs::BfsBackend,
    ) -> Result<(SgdParams, self::bfs::BfsTiming)> {
        let outcome = self::bfs::compute_apsp(self, backend)?;
        let (wmin, wmax) = if outcome.dmin.is_finite() && outcome.dmax > 0.0 {
            (
                1.0 / (outcome.dmax * outcome.dmax),
                1.0 / (outcome.dmin * outcome.dmin),
            )
        } else {
            (1.0, 1.0)
        };
        let etas = calc_learning_rate(iterations, wmin, wmax, epsilon);
        let positions = experiment_common::seed::seeded_positions(self.node_size, seed, center);
        let params = SgdParams {
            etas,
            positions,
            dist_u16: outcome.dist_u16,
            constraint_count: outcome.constraint_count,
            pairs: Vec::new(),
            center,
        };
        Ok((params, outcome.timing))
    }
}

pub fn calc_learning_rate(tmax: usize, wmin: f64, wmax: f64, eps: f64) -> Vec<f64> {
    let eta_max = 1.0 / wmin;
    let eta_min = eps / wmax;
    if tmax == 1 {
        return vec![eta_max];
    }
    let lamb = (eta_max / eta_min).ln() / (tmax - 1) as f64;

    let etas: Vec<f64> = (0..tmax)
        .map(|t| eta_max * (-lamb * t as f64).exp())
        .collect();

    etas
}
