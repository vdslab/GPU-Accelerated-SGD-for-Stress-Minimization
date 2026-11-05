use anyhow::{anyhow, bail, Context, Result};
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
    pub component_info: ComponentInfo,
    adjacency: Vec<Vec<usize>>,
}

/// Matrix Market入力と、レイアウトに採用した最大連結成分の対応情報。
/// `original_vertex_ids[local_id]`は入力の0始まり頂点番号を表す。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentInfo {
    pub original_node_size: usize,
    pub original_edge_size: usize,
    pub component_count: usize,
    pub original_vertex_ids: Vec<usize>,
}

impl ComponentInfo {
    pub fn retained_vertex_ratio(&self) -> f64 {
        if self.original_node_size == 0 {
            0.0
        } else {
            self.original_vertex_ids.len() as f64 / self.original_node_size as f64
        }
    }

    pub fn retained_edge_ratio(&self, selected_edge_size: usize) -> f64 {
        if self.original_edge_size == 0 {
            0.0
        } else {
            selected_edge_size as f64 / self.original_edge_size as f64
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeInfo {
    pub u: usize,
    pub v: usize,
    pub dij: f64,
    pub weight_u: f64,
    pub weight_v: f64,
}

#[derive(Debug, PartialEq)]
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
        let edge_size = edges.len();

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

        let mut graph = Self {
            node_size,
            edge_size,
            edge_src,
            edge_dst,
            component_info: ComponentInfo {
                original_node_size: node_size,
                original_edge_size: edge_size,
                component_count: 0,
                original_vertex_ids: (0..node_size).collect(),
            },
            adjacency,
        };
        graph.component_info.component_count = graph.connected_components().len();
        Ok(graph)
    }

    pub fn neighbors(&self, vertex: usize) -> &[usize] {
        &self.adjacency[vertex]
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.adjacency[u].binary_search(&v).is_ok()
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

    pub fn ensure_connected(&self) -> Result<()> {
        if self.node_size == 0 {
            bail!("空のグラフでは Sparse SGD を実行できません");
        }
        let distances = self.shortest_path_distances(0);
        let reached = distances
            .iter()
            .filter(|&&distance| distance != usize::MAX)
            .count();
        if reached != self.node_size {
            bail!(
                "Sparse SGD には連結グラフが必要です: 到達頂点数={reached}, 全頂点数={}",
                self.node_size
            );
        }
        Ok(())
    }

    /// 全成分を列挙する。各成分の頂点番号は昇順で、成分列は最小頂点番号順となる。
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.node_size];
        let mut components = Vec::new();
        for start in 0..self.node_size {
            if visited[start] {
                continue;
            }
            let mut queue = VecDeque::new();
            let mut component = Vec::new();
            visited[start] = true;
            queue.push_back(start);
            while let Some(u) = queue.pop_front() {
                component.push(u);
                for &v in self.neighbors(u) {
                    if !visited[v] {
                        visited[v] = true;
                        queue.push_back(v);
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }
        components
    }

    /// 最大連結成分を選び、頂点を0..k-1に再番号付けしたレイアウト用グラフを返す。
    /// 同率なら最小の元頂点番号を含む成分を選ぶ。
    pub fn largest_connected_component(&self) -> Result<Self> {
        let components = self.connected_components();
        let component_count = components.len();
        let mut selected = components
            .into_iter()
            .max_by(|left, right| {
                left.len().cmp(&right.len()).then_with(|| {
                    self.component_info.original_vertex_ids[right[0]]
                        .cmp(&self.component_info.original_vertex_ids[left[0]])
                })
            })
            .ok_or_else(|| anyhow!("空のグラフでは最大連結成分を選択できません"))?;
        selected.sort_by_key(|&vertex| self.component_info.original_vertex_ids[vertex]);

        let mut local_ids = vec![usize::MAX; self.node_size];
        for (local_id, &vertex) in selected.iter().enumerate() {
            local_ids[vertex] = local_id;
        }
        let selected_edges: Vec<_> = self
            .edge_src
            .iter()
            .zip(&self.edge_dst)
            .filter_map(|(&u, &v)| {
                (local_ids[u] != usize::MAX && local_ids[v] != usize::MAX)
                    .then_some((local_ids[u], local_ids[v]))
            })
            .collect();
        if selected.len() < 2 || selected_edges.is_empty() {
            bail!(
                "最大連結成分では Sparse SGD を実行できません: 採用頂点数={}, 採用辺数={}, 全成分数={component_count}",
                selected.len(),
                selected_edges.len(),
            );
        }

        let mut graph = Self::try_from_edges(selected.len(), &selected_edges)?;
        graph.component_info = ComponentInfo {
            original_node_size: self.component_info.original_node_size,
            original_edge_size: self.component_info.original_edge_size,
            component_count,
            original_vertex_ids: selected
                .into_iter()
                .map(|vertex| self.component_info.original_vertex_ids[vertex])
                .collect(),
        };
        Ok(graph)
    }

    fn select_pivots_with_distances<R: Rng + ?Sized>(
        &self,
        h: usize,
        rng: &mut R,
    ) -> Result<(Vec<usize>, Vec<Vec<usize>>)> {
        let target = h.min(self.node_size);
        if target == 0 {
            bail!("pivot 数は1以上である必要があります");
        }

        let first = rng.random_range(0..self.node_size);
        let mut pivots = vec![first];
        let first_distances = self.shortest_path_distances(first);
        let mut nearest_distance = first_distances.clone();
        let mut pivot_distances = vec![first_distances];

        while pivots.len() < target {
            let total = nearest_distance.iter().try_fold(0_u64, |sum, &distance| {
                let distance = u64::try_from(distance)
                    .map_err(|_| anyhow!("最短路距離を u64 に変換できません"))?;
                sum.checked_add(distance)
                    .ok_or_else(|| anyhow!("pivot sampling の距離合計が大きすぎます"))
            })?;
            if total == 0 {
                bail!("未選択 pivot の距離重みがありません");
            }

            let mut ticket = rng.random_range(0..total);
            let mut next = None;
            for (vertex, &distance) in nearest_distance.iter().enumerate() {
                let weight = distance as u64;
                if ticket < weight {
                    next = Some(vertex);
                    break;
                }
                ticket -= weight;
            }
            let next = next.ok_or_else(|| anyhow!("pivot の weighted sampling に失敗しました"))?;

            pivots.push(next);
            let distances = self.shortest_path_distances(next);
            for (nearest, &distance) in nearest_distance.iter_mut().zip(&distances) {
                *nearest = (*nearest).min(distance);
            }
            pivot_distances.push(distances);
        }

        Ok((pivots, pivot_distances))
    }

    fn build_sparse_constraints_from_distances(
        &self,
        pivots: &[usize],
        pivot_distances: &[Vec<usize>],
    ) -> Result<Vec<EdgeInfo>> {
        if pivots.len() != pivot_distances.len() || pivots.is_empty() {
            bail!("pivot と距離配列の数が一致しません");
        }
        if pivot_distances
            .iter()
            .any(|distances| distances.len() != self.node_size)
        {
            bail!("pivot 距離配列の頂点数が一致しません");
        }

        let regions = assign_regions(pivot_distances, self.node_size)?;
        let prefix_counts = region_distance_prefix_counts(pivot_distances, &regions)?;
        let mut constraints = BTreeMap::new();

        for (pivot_index, &pivot) in pivots.iter().enumerate() {
            let distances = &pivot_distances[pivot_index];
            for (vertex, &distance) in distances.iter().enumerate() {
                if vertex == pivot || self.has_edge(vertex, pivot) {
                    continue;
                }
                if distance == usize::MAX || distance == 0 {
                    bail!("連結グラフに有限でない pivot 距離があります");
                }

                let s_ip = prefix_counts[pivot_index][distance / 2];
                let dij = distance as f64;
                let weight = s_ip as f64 / (dij * dij);
                let (u, v) = canonical_pair(vertex, pivot);
                let pair = constraints.entry((u, v)).or_insert(EdgeInfo {
                    u,
                    v,
                    dij,
                    weight_u: 0.0,
                    weight_v: 0.0,
                });
                if (pair.dij - dij).abs() > f64::EPSILON {
                    bail!("同じ頂点対で目標距離が一致しません");
                }
                if vertex == u {
                    pair.weight_u = weight;
                } else {
                    pair.weight_v = weight;
                }
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
                    weight_u: 1.0,
                    weight_v: 1.0,
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
        self.ensure_connected()?;
        let (pivots, pivot_distances) = self.select_pivots_with_distances(h, rng)?;
        let pairs = self.build_sparse_constraints_from_distances(&pivots, &pivot_distances)?;
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

fn assign_regions(pivot_distances: &[Vec<usize>], node_size: usize) -> Result<Vec<usize>> {
    if pivot_distances.is_empty() {
        bail!("領域割当に pivot が必要です");
    }
    let mut regions = vec![0; node_size];
    for vertex in 0..node_size {
        let mut best_distance = usize::MAX;
        let mut best_pivot = 0;
        for (pivot_index, distances) in pivot_distances.iter().enumerate() {
            let distance = distances[vertex];
            if distance < best_distance {
                best_distance = distance;
                best_pivot = pivot_index;
            }
        }
        if best_distance == usize::MAX {
            bail!("どの pivot からも到達できない頂点があります: {vertex}");
        }
        regions[vertex] = best_pivot;
    }
    Ok(regions)
}

fn region_distance_prefix_counts(
    pivot_distances: &[Vec<usize>],
    regions: &[usize],
) -> Result<Vec<Vec<usize>>> {
    let mut prefix_counts = Vec::with_capacity(pivot_distances.len());
    for (pivot_index, distances) in pivot_distances.iter().enumerate() {
        let max_distance = distances
            .iter()
            .copied()
            .filter(|&distance| distance != usize::MAX)
            .max()
            .ok_or_else(|| anyhow!("pivot 距離が空です"))?;
        let mut counts = vec![0_usize; max_distance + 1];
        for (vertex, &region) in regions.iter().enumerate() {
            if region == pivot_index {
                counts[distances[vertex]] += 1;
            }
        }
        for distance in 1..counts.len() {
            counts[distance] += counts[distance - 1];
        }
        prefix_counts.push(counts);
    }
    Ok(prefix_counts)
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
    for weight in pairs.iter().flat_map(|pair| [pair.weight_u, pair.weight_v]) {
        if weight.is_finite() && weight > 0.0 {
            wmin = wmin.min(weight);
            wmax = wmax.max(weight);
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
    fn bfs_marks_unreachable_vertices_and_connectivity_rejects_graph() {
        let graph = graph(5, &[(0, 1), (1, 2), (3, 4)]);
        assert_eq!(
            graph.shortest_path_distances(0),
            vec![0, 1, 2, usize::MAX, usize::MAX]
        );
        assert!(graph.ensure_connected().is_err());
    }

    #[test]
    fn largest_component_reindexes_vertices_and_records_statistics() {
        let graph = graph(8, &[(0, 1), (2, 4), (4, 6)]);
        let largest = graph.largest_connected_component().unwrap();
        assert_eq!(largest.node_size, 3);
        assert_eq!(largest.edge_size, 2);
        assert_eq!(largest.edge_src, vec![0, 1]);
        assert_eq!(largest.edge_dst, vec![1, 2]);
        assert_eq!(largest.component_info.original_node_size, 8);
        assert_eq!(largest.component_info.original_edge_size, 3);
        assert_eq!(largest.component_info.component_count, 5);
        assert_eq!(largest.component_info.original_vertex_ids, vec![2, 4, 6]);
        assert!((largest.component_info.retained_vertex_ratio() - 0.375).abs() < f64::EPSILON);
        assert!((largest.component_info.retained_edge_ratio(2) - 2.0 / 3.0).abs() < f64::EPSILON);
        largest.ensure_connected().unwrap();
    }

    #[test]
    fn equally_sized_components_choose_smallest_original_vertex() {
        let graph = graph(6, &[(0, 2), (3, 5)]);
        let largest = graph.largest_connected_component().unwrap();
        assert_eq!(largest.component_info.original_vertex_ids, vec![0, 2]);
    }

    #[test]
    fn largest_component_rejects_self_loop_only_input_with_statistics() {
        let graph = graph(3, &[(0, 0), (1, 1), (2, 2)]);
        let error = graph.largest_connected_component().unwrap_err().to_string();
        assert!(error.contains("採用頂点数=1"));
        assert!(error.contains("採用辺数=0"));
        assert!(error.contains("全成分数=3"));
    }

    #[test]
    fn connected_graph_largest_component_is_identity_map() {
        let graph = graph(3, &[(0, 1), (1, 2)]);
        let largest = graph.largest_connected_component().unwrap();
        assert_eq!(largest.component_info.component_count, 1);
        assert_eq!(largest.component_info.original_vertex_ids, vec![0, 1, 2]);
    }

    #[test]
    fn distance_weighted_pivots_are_unique_clamped_and_reproducible() {
        let graph = graph(7, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]);
        let mut first_rng = StdRng::seed_from_u64(7);
        let mut second_rng = StdRng::seed_from_u64(7);
        let first = graph
            .select_pivots_with_distances(99, &mut first_rng)
            .unwrap()
            .0;
        let second = graph
            .select_pivots_with_distances(99, &mut second_rng)
            .unwrap()
            .0;
        let unique: HashSet<_> = first.iter().copied().collect();
        assert_eq!(first, second);
        assert_eq!(first.len(), 7);
        assert_eq!(unique.len(), 7);
    }

    #[test]
    fn regions_use_early_pivot_for_ties_and_prefix_counts_match_formula() {
        let graph = graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let distances = vec![
            graph.shortest_path_distances(0),
            graph.shortest_path_distances(4),
        ];
        let regions = assign_regions(&distances, 5).unwrap();
        assert_eq!(regions, vec![0, 0, 0, 1, 1]);
        let prefix = region_distance_prefix_counts(&distances, &regions).unwrap();
        assert_eq!(prefix[0][1], 2);
        assert_eq!(prefix[0][2], 3);
        assert_eq!(prefix[1][1], 2);
    }

    #[test]
    fn sparse_constraints_have_region_corrected_directional_weights() {
        let graph = graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let pivot_distances = vec![
            graph.shortest_path_distances(0),
            graph.shortest_path_distances(4),
        ];
        let constraints = graph
            .build_sparse_constraints_from_distances(&[0, 4], &pivot_distances)
            .unwrap();

        let edge = constraints
            .iter()
            .find(|pair| (pair.u, pair.v) == (1, 2))
            .unwrap();
        assert_eq!((edge.dij, edge.weight_u, edge.weight_v), (1.0, 1.0, 1.0));

        let vertex_pivot = constraints
            .iter()
            .find(|pair| (pair.u, pair.v) == (0, 3))
            .unwrap();
        assert!((vertex_pivot.weight_u - 0.0).abs() < 1e-12);
        assert!((vertex_pivot.weight_v - 2.0 / 9.0).abs() < 1e-12);

        let pivot_pair = constraints
            .iter()
            .find(|pair| (pair.u, pair.v) == (0, 4))
            .unwrap();
        assert!((pivot_pair.weight_u - 2.0 / 16.0).abs() < 1e-12);
        assert!((pivot_pair.weight_v - 3.0 / 16.0).abs() < 1e-12);
    }

    #[test]
    fn learning_rate_has_expected_endpoints() {
        let pairs = [EdgeInfo {
            u: 0,
            v: 1,
            dij: 2.0,
            weight_u: 0.25,
            weight_v: 1.0,
        }];
        let (wmin, wmax) = positive_weight_range(&pairs).unwrap();
        let rates = calc_learning_rate(5, wmin, wmax, 0.1).unwrap();
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

    #[test]
    fn prepared_parameters_are_reproducible_for_same_seed() {
        let graph = graph(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        let mut first_rng = StdRng::seed_from_u64(19);
        let mut second_rng = StdRng::seed_from_u64(19);
        let first = graph
            .prepare_sgd_params(5, 0.1, 3, true, &mut first_rng)
            .unwrap();
        let second = graph
            .prepare_sgd_params(5, 0.1, 3, true, &mut second_rng)
            .unwrap();
        assert_eq!(first, second);
    }
}
