use anyhow::{anyhow, Context, Result};
use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Graph {
    pub node_size: usize,
    pub edge_size: usize,
    pub edge_src: Vec<usize>,
    pub edge_dst: Vec<usize>,
    adjacency: Vec<Vec<usize>>,
}

impl Graph {
    pub fn from_mtx(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let header = lines
            .next()
            .transpose()?
            .ok_or_else(|| anyhow!("empty Matrix Market file"))?;
        let header_lower = header.to_ascii_lowercase();
        if !header_lower.starts_with("%%matrixmarket matrix coordinate") {
            return Err(anyhow!(
                "unsupported Matrix Market header: expected coordinate matrix, got {header}"
            ));
        }

        let size_line = loop {
            let line = lines
                .next()
                .transpose()?
                .ok_or_else(|| anyhow!("missing Matrix Market size line"))?;
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('%') {
                break trimmed.to_string();
            }
        };

        let mut size_parts = size_line.split_ascii_whitespace();
        let rows: usize = size_parts
            .next()
            .context("missing Matrix Market row count")?
            .parse()?;
        let cols: usize = size_parts
            .next()
            .context("missing Matrix Market column count")?
            .parse()?;
        let _entries: usize = size_parts
            .next()
            .context("missing Matrix Market entry count")?
            .parse()?;

        if rows != cols {
            return Err(anyhow!(
                "SparseSGD expects a square adjacency matrix, got {rows}x{cols}"
            ));
        }

        let node_size = rows;
        let mut seen_edges = HashSet::new();

        for line in lines {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('%') {
                continue;
            }

            let mut parts = trimmed.split_ascii_whitespace();
            let row_1based: usize = parts
                .next()
                .context("missing Matrix Market row index")?
                .parse()?;
            let col_1based: usize = parts
                .next()
                .context("missing Matrix Market column index")?
                .parse()?;

            if row_1based == 0 || col_1based == 0 {
                return Err(anyhow!("Matrix Market indices must be 1-based"));
            }

            let row = row_1based - 1;
            let col = col_1based - 1;
            if row >= node_size || col >= node_size {
                return Err(anyhow!(
                    "Matrix Market index out of bounds: ({row_1based}, {col_1based}) for {node_size} nodes"
                ));
            }

            if row == col {
                continue;
            }
            let edge = if row < col { (row, col) } else { (col, row) };
            seen_edges.insert(edge);
        }

        let mut edges: Vec<(usize, usize)> = seen_edges.into_iter().collect();
        edges.sort_unstable();

        let mut adjacency = vec![Vec::new(); node_size];
        let mut edge_src = Vec::with_capacity(edges.len());
        let mut edge_dst = Vec::with_capacity(edges.len());

        for (u, v) in edges {
            edge_src.push(u);
            edge_dst.push(v);
            adjacency[u].push(v);
            adjacency[v].push(u);
        }

        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
        }

        Ok(Self {
            node_size,
            edge_size: edge_src.len(),
            edge_src,
            edge_dst,
            adjacency,
        })
    }

    pub fn from_edges(node_size: usize, edges: &[(usize, usize)]) -> Self {
        let mut seen_edges = HashSet::new();
        for &(u, v) in edges {
            if u == v {
                continue;
            }
            let edge = if u < v { (u, v) } else { (v, u) };
            seen_edges.insert(edge);
        }

        let mut edges: Vec<(usize, usize)> = seen_edges.into_iter().collect();
        edges.sort_unstable();

        let mut adjacency = vec![Vec::new(); node_size];
        let mut edge_src = Vec::with_capacity(edges.len());
        let mut edge_dst = Vec::with_capacity(edges.len());

        for (u, v) in edges {
            edge_src.push(u);
            edge_dst.push(v);
            adjacency[u].push(v);
            adjacency[v].push(u);
        }

        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
        }

        Self {
            node_size,
            edge_size: edge_src.len(),
            edge_src,
            edge_dst,
            adjacency,
        }
    }

    pub fn neighbors(&self, node: usize) -> &[usize] {
        &self.adjacency[node]
    }

    pub fn degree(&self, node: usize) -> usize {
        self.adjacency[node].len()
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.adjacency[u].binary_search(&v).is_ok()
    }

    pub fn ensure_connected(&self) -> Result<()> {
        if self.node_size <= 1 {
            return Ok(());
        }

        let mut seen = vec![false; self.node_size];
        let mut queue = VecDeque::new();
        seen[0] = true;
        queue.push_back(0);

        while let Some(u) = queue.pop_front() {
            for &v in &self.adjacency[u] {
                if !seen[v] {
                    seen[v] = true;
                    queue.push_back(v);
                }
            }
        }

        let visited = seen.iter().filter(|&&v| v).count();
        if visited == self.node_size {
            Ok(())
        } else {
            Err(anyhow!(
                "SparseSGD v1 requires a connected graph: visited {visited}/{} nodes",
                self.node_size
            ))
        }
    }

    pub fn update_min_distances_from_pivot(&self, pivot: usize, dist_to_pivot: &mut [usize]) {
        let mut queue = VecDeque::new();
        let mut local_dist = vec![usize::MAX; self.node_size];

        local_dist[pivot] = 0;
        queue.push_back(pivot);

        while let Some(u) = queue.pop_front() {
            let next_dist = local_dist[u] + 1;
            for &v in &self.adjacency[u] {
                if local_dist[v] == usize::MAX {
                    local_dist[v] = next_dist;
                    queue.push_back(v);
                }
            }
        }

        for (best, dist) in dist_to_pivot.iter_mut().zip(local_dist) {
            *best = (*best).min(dist);
        }
    }
}
