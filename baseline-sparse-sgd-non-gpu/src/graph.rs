use anyhow::{anyhow, bail, Context, Result};
use rand::prelude::IndexedRandom;
use rand::Rng;
use std::collections::{BTreeMap, HashSet, VecDeque};
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeInfo {
    pub u: usize,
    pub v: usize,
    pub dij: f64,
    pub wij: f64,
}

#[derive(Debug)]
pub struct SgdParams {
    pub etas: Vec<f64>,
    pub positions: Vec<[f64; 2]>,
    pub pairs: Vec<EdgeInfo>,
    pub pivots: Vec<usize>,
    pub center: bool,
}

impl Graph {
    pub fn from_mtx(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Matrix Market ファイルを開けません: {}", path.display()))?;
        let mut lines = BufReader::new(file).lines();

        let header = lines
            .next()
            .transpose()?
            .ok_or_else(|| anyhow!("Matrix Market ファイルが空です"))?;
        if !header
            .to_ascii_lowercase()
            .starts_with("%%matrixmarket matrix coordinate")
        {
            bail!("coordinate 形式ではない Matrix Market ヘッダーです: {header}");
        }

        let size_line = loop {
            let line = lines
                .next()
                .transpose()?
                .ok_or_else(|| anyhow!("Matrix Market のサイズ行がありません"))?;
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('%') {
                break trimmed.to_owned();
            }
        };

        let mut size_parts = size_line.split_ascii_whitespace();
        let rows: usize = size_parts
            .next()
            .context("行数がありません")?
            .parse()
            .context("行数が不正です")?;
        let cols: usize = size_parts
            .next()
            .context("列数がありません")?
            .parse()
            .context("列数が不正です")?;
        let _entries: usize = size_parts
            .next()
            .context("非ゼロ要素数がありません")?
            .parse()
            .context("非ゼロ要素数が不正です")?;

        if rows != cols {
            bail!("隣接行列は正方行列である必要があります: {rows}x{cols}");
        }

        let mut edges = Vec::new();
        for line in lines {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('%') {
                continue;
            }

            let mut parts = trimmed.split_ascii_whitespace();
            let row: usize = parts
                .next()
                .context("辺の行番号がありません")?
                .parse()
                .context("辺の行番号が不正です")?;
            let col: usize = parts
                .next()
                .context("辺の列番号がありません")?
                .parse()
                .context("辺の列番号が不正です")?;

            if row == 0 || col == 0 {
                bail!("Matrix Market の添字は1始まりである必要があります");
            }
            let u = row - 1;
            let v = col - 1;
            if u >= rows || v >= rows {
                bail!("頂点番号が範囲外です: ({row}, {col}), 頂点数={rows}");
            }
            edges.push((u, v));
        }

        Self::try_from_edges(rows, &edges)
    }

    pub fn try_from_edges(node_size: usize, edges: &[(usize, usize)]) -> Result<Self> {
        let mut unique_edges = HashSet::new();
        for &(u, v) in edges {
            if u >= node_size || v >= node_size {
                bail!("頂点番号が範囲外です: ({u}, {v}), 頂点数={node_size}");
            }
            if u != v {
                unique_edges.insert(canonical_pair(u, v));
            }
        }

        let mut edges: Vec<_> = unique_edges.into_iter().collect();
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

    pub fn neighbors(&self, vertex: usize) -> &[usize] {
        &self.adjacency[vertex]
    }

    pub fn shortest_path_distances(&self, start: usize) -> Vec<usize> {
        assert!(start < self.node_size, "始点が頂点数の範囲外です");
        let mut distances = vec![usize::MAX; self.node_size];
        let mut queue = VecDeque::new();
        distances[start] = 0;
        queue.push_back(start);

        while let Some(u) = queue.pop_front() {
            let next_distance = distances[u] + 1;
            for &v in self.neighbors(u) {
                if distances[v] == usize::MAX {
                    distances[v] = next_distance;
                    queue.push_back(v);
                }
            }
        }
        distances
    }

    pub fn select_pivots<R: Rng + ?Sized>(&self, h: usize, rng: &mut R) -> Vec<usize> {
        let target = h.min(self.node_size);
        if target == 0 {
            return Vec::new();
        }

        let first = rng.random_range(0..self.node_size);
        let mut pivots = vec![first];
        let mut selected = vec![false; self.node_size];
        selected[first] = true;
        let mut nearest_distance = self.shortest_path_distances(first);

        while pivots.len() < target {
            let maximum = (0..self.node_size)
                .filter(|&vertex| !selected[vertex])
                .map(|vertex| nearest_distance[vertex])
                .max()
                .expect("未選択頂点が存在する必要があります");
            let candidates: Vec<_> = (0..self.node_size)
                .filter(|&vertex| !selected[vertex] && nearest_distance[vertex] == maximum)
                .collect();
            let next = *candidates
                .choose(rng)
                .expect("max-min 候補が存在する必要があります");

            selected[next] = true;
            pivots.push(next);
            let distances = self.shortest_path_distances(next);
            for (nearest, distance) in nearest_distance.iter_mut().zip(distances) {
                *nearest = (*nearest).min(distance);
            }
        }
        pivots
    }

    pub fn build_sparse_constraints(&self, pivots: &[usize]) -> Result<Vec<EdgeInfo>> {
        let mut constraints = BTreeMap::new();

        for &pivot in pivots {
            if pivot >= self.node_size {
                bail!("pivot が頂点数の範囲外です: {pivot}");
            }
            for (vertex, distance) in self.shortest_path_distances(pivot).into_iter().enumerate() {
                if vertex == pivot || distance == usize::MAX {
                    continue;
                }
                let dij = distance as f64;
                let (u, v) = canonical_pair(vertex, pivot);
                constraints.insert(
                    (u, v),
                    EdgeInfo {
                        u,
                        v,
                        dij,
                        wij: 1.0 / (dij * dij),
                    },
                );
            }
        }

        for (&u, &v) in self.edge_src.iter().zip(&self.edge_dst) {
            let (u, v) = canonical_pair(u, v);
            constraints.insert(
                (u, v),
                EdgeInfo {
                    u,
                    v,
                    dij: 1.0,
                    wij: 1.0,
                },
            );
        }

        Ok(constraints.into_values().collect())
    }

    pub fn prepare_sgd_params<R: Rng + ?Sized>(
        &self,
        iterations: usize,
        epsilon: f64,
        h: usize,
        center: bool,
        rng: &mut R,
    ) -> Result<SgdParams> {
        let pivots = self.select_pivots(h, rng);
        let pairs = self.build_sparse_constraints(&pivots)?;
        let (wmin, wmax) = positive_weight_range(&pairs)?;
        let etas = calc_learning_rate(iterations, wmin, wmax, epsilon)?;
        let positions = init_positions_random(self.node_size, center, rng);

        Ok(SgdParams {
            etas,
            positions,
            pairs,
            pivots,
            center,
        })
    }
}

fn canonical_pair(u: usize, v: usize) -> (usize, usize) {
    if u < v {
        (u, v)
    } else {
        (v, u)
    }
}

pub fn positive_weight_range(pairs: &[EdgeInfo]) -> Result<(f64, f64)> {
    let mut wmin = f64::INFINITY;
    let mut wmax: f64 = 0.0;
    for pair in pairs {
        if pair.wij.is_finite() && pair.wij > 0.0 {
            wmin = wmin.min(pair.wij);
            wmax = wmax.max(pair.wij);
        }
    }
    if !wmin.is_finite() || wmax == 0.0 {
        bail!("正の有限な Sparse SGD 制約がありません");
    }
    Ok((wmin, wmax))
}

pub fn calc_learning_rate(
    iterations: usize,
    wmin: f64,
    wmax: f64,
    epsilon: f64,
) -> Result<Vec<f64>> {
    if iterations == 0 {
        bail!("反復回数は1以上である必要があります");
    }
    if !(wmin.is_finite() && wmin > 0.0 && wmax.is_finite() && wmax >= wmin) {
        bail!("学習率の重み範囲が不正です: wmin={wmin}, wmax={wmax}");
    }
    if !(epsilon.is_finite() && epsilon > 0.0) {
        bail!("epsilon は正の有限値である必要があります");
    }

    let eta_max = 1.0 / wmin;
    if iterations == 1 {
        return Ok(vec![eta_max]);
    }
    let eta_min = epsilon / wmax;
    let lambda = (eta_max / eta_min).ln() / (iterations - 1) as f64;
    Ok((0..iterations)
        .map(|iteration| eta_max * (-lambda * iteration as f64).exp())
        .collect())
}

pub fn init_positions_random<R: Rng + ?Sized>(
    node_size: usize,
    center: bool,
    rng: &mut R,
) -> Vec<[f64; 2]> {
    let mut positions: Vec<_> = (0..node_size)
        .map(|_| [rng.random::<f64>(), rng.random::<f64>()])
        .collect();
    if center {
        center_inplace(&mut positions);
    }
    positions
}

pub fn center_inplace(positions: &mut [[f64; 2]]) {
    if positions.is_empty() {
        return;
    }
    let count = positions.len() as f64;
    let mean_x = positions.iter().map(|position| position[0]).sum::<f64>() / count;
    let mean_y = positions.iter().map(|position| position[1]).sum::<f64>() / count;
    for position in positions {
        position[0] -= mean_x;
        position[1] -= mean_y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::fs;
    use std::io::Write;

    fn graph(node_size: usize, edges: &[(usize, usize)]) -> Graph {
        Graph::try_from_edges(node_size, edges).unwrap()
    }

    #[test]
    fn matrix_market_loading_filters_loops_and_duplicate_directions() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(file, "3 3 5").unwrap();
        writeln!(file, "1 1").unwrap();
        writeln!(file, "1 2").unwrap();
        writeln!(file, "2 1").unwrap();
        writeln!(file, "2 3").unwrap();
        writeln!(file, "3 2").unwrap();

        let loaded = Graph::from_mtx(file.path()).unwrap();
        assert_eq!(loaded.node_size, 3);
        assert_eq!(loaded.edge_size, 2);
        assert_eq!(loaded.edge_src, vec![0, 1]);
        assert_eq!(loaded.edge_dst, vec![1, 2]);
        assert_eq!(loaded.neighbors(1), &[0, 2]);
    }

    #[test]
    fn matrix_market_loading_rejects_out_of_range_indices() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("invalid.mtx");
        fs::write(
            &path,
            "%%MatrixMarket matrix coordinate pattern symmetric\n2 2 1\n1 3\n",
        )
        .unwrap();
        assert!(Graph::from_mtx(&path).is_err());
    }

    #[test]
    fn bfs_marks_unreachable_vertices() {
        let graph = graph(5, &[(0, 1), (1, 2), (3, 4)]);
        assert_eq!(
            graph.shortest_path_distances(0),
            vec![0, 1, 2, usize::MAX, usize::MAX]
        );
    }

    #[test]
    fn max_min_pivots_are_unique_clamped_and_cover_components() {
        let graph = graph(5, &[(0, 1), (1, 2), (3, 4)]);
        let mut rng = StdRng::seed_from_u64(7);
        let pivots = graph.select_pivots(99, &mut rng);
        let unique: HashSet<_> = pivots.iter().copied().collect();
        assert_eq!(pivots.len(), 5);
        assert_eq!(unique.len(), 5);

        let first_component = pivots[0] <= 2;
        let second_component = pivots[1] <= 2;
        assert_ne!(first_component, second_component);
    }

    #[test]
    fn sparse_constraints_are_deduplicated_and_edges_have_unit_distance() {
        let graph = graph(4, &[(0, 1), (1, 2), (2, 3), (1, 0)]);
        let constraints = graph.build_sparse_constraints(&[0, 3]).unwrap();
        assert_eq!(constraints.len(), 6);
        assert_eq!(
            constraints.iter().find(|pair| (pair.u, pair.v) == (0, 1)),
            Some(&EdgeInfo {
                u: 0,
                v: 1,
                dij: 1.0,
                wij: 1.0
            })
        );
        let long = constraints
            .iter()
            .find(|pair| (pair.u, pair.v) == (0, 3))
            .unwrap();
        assert_eq!(long.dij, 3.0);
        assert!((long.wij - 1.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn unreachable_pivot_relations_are_omitted() {
        let graph = graph(4, &[(0, 1), (2, 3)]);
        let constraints = graph.build_sparse_constraints(&[0]).unwrap();
        assert_eq!(constraints.len(), 2);
        assert!(constraints
            .iter()
            .all(|pair| pair.dij.is_finite() && pair.wij.is_finite()));
    }

    #[test]
    fn learning_rate_has_expected_endpoints() {
        let rates = calc_learning_rate(5, 0.25, 1.0, 0.1).unwrap();
        assert_eq!(rates.len(), 5);
        assert!((rates[0] - 4.0).abs() < 1e-12);
        assert!((rates[4] - 0.1).abs() < 1e-12);
        assert!(rates.windows(2).all(|window| window[0] > window[1]));
    }

    #[test]
    fn random_positions_can_be_centered() {
        let mut rng = StdRng::seed_from_u64(11);
        let positions = init_positions_random(20, true, &mut rng);
        let sum_x: f64 = positions.iter().map(|position| position[0]).sum();
        let sum_y: f64 = positions.iter().map(|position| position[1]).sum();
        assert!(sum_x.abs() < 1e-12);
        assert!(sum_y.abs() < 1e-12);
    }
}
