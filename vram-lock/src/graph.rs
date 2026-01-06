use anyhow::Result;
use rand::Rng;
use sprs::io::read_matrix_market;
use sprs::num_kinds::Pattern;
use std::collections::VecDeque;
use std::path::Path;

#[derive(Debug)]
pub struct Graph {
    pub node_size: usize,
    pub edge_size: usize,
    pub edge_src: Vec<usize>,
    pub edge_dst: Vec<usize>,
}

#[derive(Debug)]
pub struct SgdParams {
    pub etas: Vec<f64>,
    pub positions: Vec<[f64; 2]>,
    pub pairs: Vec<EdgeInfo>,
}

#[derive(Debug)]
pub struct EdgeInfo {
    pub u: usize,
    pub v: usize,
    pub dij: f64,
    pub wij: f64,
}

impl Graph {
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

    fn calc_adj_matrix(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.node_size];
        for i in 0..self.edge_size {
            adj[self.edge_src[i]].push(self.edge_dst[i]);
            adj[self.edge_dst[i]].push(self.edge_src[i]);
        }
        adj
    }

    pub fn calc_dist_matrix(&self) -> Vec<Vec<usize>> {
        let adj = Self::calc_adj_matrix(self);
        let n = adj.len();
        let mut dist_matrix = vec![vec![usize::MAX; n]; n];

        // bfs
        for i in 0..n {
            let mut deq = VecDeque::new();
            let mut seen = vec![false; n];

            deq.push_back(i);
            seen[i] = true;
            dist_matrix[i][i] = 0;

            while let Some(v) = deq.pop_front() {
                for &u in &adj[v] {
                    if seen[u] {
                        continue;
                    }
                    deq.push_back(u);
                    seen[u] = true;
                    dist_matrix[i][u] = dist_matrix[i][v] + 1;
                }
            }
        }
        dist_matrix
    }

    pub fn calc_edge_info(&self, dist: &[Vec<usize>]) -> (Vec<EdgeInfo>, f64, f64) {
        let mut pairs = Vec::new();
        let mut dmin: f64 = f64::INFINITY;
        let mut dmax: f64 = 0.0;

        for u in 0..dist.len() {
            for v in 0..dist[u].len() {
                if u >= v {
                    continue;
                }

                // Skip unreachable nodes (distance == usize::MAX)
                if dist[u][v] == usize::MAX {
                    continue;
                }

                let dij = dist[u][v] as f64;
                if dij <= 0.0 {
                    continue;
                }

                let wij = 1.0 / (dij * dij);
                pairs.push(EdgeInfo { u, v, dij, wij });

                dmin = dmin.min(dij);
                dmax = dmax.max(dij);
            }
        }

        let wmin = 1.0 / (dmax * dmax);
        let wmax = 1.0 / (dmin * dmin);

        (pairs, wmin, wmax)
    }

    /// Precompute SGD parameters
    pub fn prepare_sgd_params(
        &self,
        iterations: usize,
        epsilon: f64,
        center: bool,
    ) -> SgdParams {
        let dist = self.calc_dist_matrix();
        let (pairs, wmin, wmax) = self.calc_edge_info(&dist);

        let etas = calc_learning_rate(iterations, wmin, wmax, epsilon);

        let positions = init_positions_random(self.node_size, center);

        SgdParams {
            etas,
            positions,
            pairs,
        }
    }
}

pub fn calc_learning_rate(tmax: usize, wmin: f64, wmax: f64, eps: f64) -> Vec<f64> {
    let eta_max = 1.0 / wmin;
    let eta_min = eps / wmax;
    let lamb = (eta_max / eta_min).ln() / (tmax - 1) as f64;

    let etas: Vec<f64> = (0..tmax)
        .map(|t| eta_max * (-lamb * t as f64).exp())
        .collect();

    etas
}

pub fn init_positions_random(n_nodes: usize, center: bool) -> Vec<[f64; 2]> {
    let mut rng = rand::rng();

    // Random coordinates in the range [0, 1)
    let mut positions: Vec<[f64; 2]> = (0..n_nodes)
        .map(|_| [rng.random::<f64>(), rng.random::<f64>()])
        .collect();

    // centering if center is true
    if center {
        // Calc the center of the positions
        let sum_x: f64 = positions.iter().map(|p| p[0]).sum();
        let sum_y: f64 = positions.iter().map(|p| p[1]).sum();
        let mean_x = sum_x / n_nodes as f64;
        let mean_y = sum_y / n_nodes as f64;

        // move to the center
        for pos in &mut positions {
            pos[0] -= mean_x;
            pos[1] -= mean_y;
        }
    }

    positions
}
