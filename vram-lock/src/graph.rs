use anyhow::Result;
use sprs::io::read_matrix_market;
use sprs::num_kinds::Pattern;
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

    pub fn calc_adj_matrix(graph: Self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); graph.node_size];
        for i in 0..graph.edge_size {
            adj[graph.edge_src[i]].push(graph.edge_dst[i]);
        }
        adj
    }
}
