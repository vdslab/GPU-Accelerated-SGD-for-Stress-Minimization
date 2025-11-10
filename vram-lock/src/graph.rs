use anyhow::Result;
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

impl Graph {
    pub fn from_mtx(path: &Path) -> Result<Self> {
        let matrix: sprs::TriMat<Pattern> = read_matrix_market(path)?;

        let node_size: usize = matrix.rows();
        let edge_size: usize = matrix.nnz();
        let edge_src: Vec<usize> = matrix.row_inds().to_vec();
        let edge_dst: Vec<usize> = matrix.col_inds().to_vec();

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
}
