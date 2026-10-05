use crate::graph::{
    calc_learning_rate, init_positions_random, positive_weight_range, Graph, SgdParams,
};
use anyhow::{bail, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;
pub const INF: u32 = u32::MAX;
#[derive(Clone, Debug)]
pub struct Csr {
    pub rows: Vec<u32>,
    pub columns: Vec<u32>,
}
impl Csr {
    pub fn from_graph(graph: &Graph) -> Result<Self> {
        let mut rows = vec![0];
        let mut columns = Vec::new();
        for u in 0..graph.node_size {
            for &v in graph.neighbors(u) {
                columns.push(u32::try_from(v)?);
            }
            rows.push(u32::try_from(columns.len())?);
        }
        let csr = Self { rows, columns };
        csr.validate()?;
        Ok(csr)
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.rows.len().saturating_sub(1);
        if n == 0
            || n >= u32::MAX as usize
            || self.columns.len() > u32::MAX as usize
            || self.rows.first() != Some(&0)
            || self.rows.last().copied().map(|x| x as usize) != Some(self.columns.len())
            || self.rows.windows(2).any(|w| w[0] > w[1])
            || self.columns.iter().any(|&v| v as usize >= n)
        {
            bail!("invalid CSR");
        }
        Ok(())
    }
}
pub trait DistanceBackend {
    fn distances(&mut self, source: usize) -> Result<Vec<u32>>;
    fn transfer_seconds(&self) -> f64 {
        0.0
    }
}
pub struct CpuBfs<'a>(pub &'a Graph);
impl DistanceBackend for CpuBfs<'_> {
    fn distances(&mut self, source: usize) -> Result<Vec<u32>> {
        if source >= self.0.node_size {
            bail!("source out of range");
        }
        Ok(self.0.shortest_path_distances(source))
    }
}
#[derive(Debug, Default, Clone)]
pub struct Timings {
    pub pivot_selection: f64,
    pub distance: f64,
    pub transfer: f64,
    pub constraints: f64,
    pub preparation: f64,
}
pub struct Prepared {
    pub params: SgdParams,
    pub distances: Vec<Vec<u32>>,
    pub timings: Timings,
}
pub fn prepare<R: Rng + ?Sized>(
    graph: &Graph,
    backend: &mut dyn DistanceBackend,
    h: usize,
    iterations: usize,
    epsilon: f64,
    center: bool,
    rng: &mut R,
) -> Result<Prepared> {
    let start = Instant::now();
    graph.ensure_connected()?;
    let connected = start.elapsed().as_secs_f64();
    let transfer_start = backend.transfer_seconds();
    let mut bfs_wall = 0.0;
    let selection_start = Instant::now();
    let (pivots, distances) = graph.select_pivots_using(h, rng, &mut |source| {
        let begin = Instant::now();
        let row = backend.distances(source)?;
        bfs_wall += begin.elapsed().as_secs_f64();
        Ok(row)
    })?;
    let selection_wall = selection_start.elapsed().as_secs_f64();
    let transfer = backend.transfer_seconds() - transfer_start;
    let constraint_start = Instant::now();
    let pairs = graph.build_sparse_constraints_from_distances(&pivots, &distances)?;
    let (wmin, wmax) = positive_weight_range(&pairs)?;
    let etas = calc_learning_rate(iterations, wmin, wmax, epsilon)?;
    let positions = init_positions_random(graph.node_size, center, rng);
    let constraints = constraint_start.elapsed().as_secs_f64();
    let params = SgdParams {
        pairs,
        pivots,
        etas,
        positions,
        center,
    };
    Ok(Prepared {
        params,
        distances,
        timings: Timings {
            pivot_selection: connected + (selection_wall - bfs_wall).max(0.0),
            distance: (bfs_wall - transfer).max(0.0),
            transfer,
            constraints,
            preparation: start.elapsed().as_secs_f64(),
        },
    })
}
// Stable FNV-1a, explicit little-endian fixed-width fields; independent of Rust's DefaultHasher.
pub fn hash_words(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}
pub fn hashes(p: &Prepared) -> [u64; 5] {
    [
        hash_words(p.params.pivots.iter().map(|&x| x as u64)),
        hash_words(p.distances.iter().flatten().map(|&x| u64::from(x))),
        hash_words(p.params.pairs.iter().flat_map(|x| {
            [
                x.u as u64,
                x.v as u64,
                x.dij.to_bits(),
                x.weight_u.to_bits(),
                x.weight_v.to_bits(),
            ]
        })),
        hash_words(p.params.positions.iter().flatten().map(|x| x.to_bits())),
        hash_words(p.params.etas.iter().map(|x| x.to_bits())),
    ]
}
pub fn verify_prepared(cpu: &Prepared, other: &Prepared) -> Result<()> {
    if cpu.params != other.params
        || cpu.distances != other.distances
        || hashes(cpu) != hashes(other)
    {
        bail!(
            "CPU/backend mismatch in pivots, distances, constraints, coordinates or learning rates"
        );
    }
    Ok(())
}
pub fn fixtures() -> Result<Vec<Graph>> {
    let mut result = vec![
        Graph::try_from_edges(3, &[(0, 1), (1, 2)])?,
        Graph::try_from_edges(6, &[(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])?,
        Graph::try_from_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)])?,
        Graph::try_from_edges(6, &[(0, 1), (1, 2), (3, 4)])?,
        Graph::try_from_edges(1, &[])?,
        Graph::try_from_edges(4, &[(0, 0), (0, 1), (1, 0), (0, 1), (2, 3)])?,
    ];
    let mut rng = StdRng::seed_from_u64(0x5eed);
    for n in [2, 7, 31, 65, 257] {
        for probability in [0.03, 0.2, 0.8] {
            let mut edges = Vec::new();
            for u in 0..n {
                for v in u + 1..n {
                    if rng.random::<f64>() < probability {
                        edges.push((u, v));
                    }
                }
            }
            result.push(Graph::try_from_edges(n, &edges)?);
        }
    }
    Ok(result)
}
pub fn verify_all_sources(graph: &Graph, backend: &mut dyn DistanceBackend) -> Result<()> {
    // Full n*n matrices only for small validation fixtures.
    let rows = (0..graph.node_size)
        .map(|u| backend.distances(u))
        .collect::<Result<Vec<_>>>()?;
    for (source, row) in rows.iter().enumerate() {
        if *row != graph.shortest_path_distances(source) {
            bail!("distance mismatch at source {source}");
        }
        if row[source] != 0 {
            bail!("nonzero diagonal");
        }
        for (target, other_row) in rows.iter().enumerate() {
            if row[target] != other_row[source] {
                bail!("asymmetric distances");
            }
        }
    }
    Ok(())
}
// Identical exact full-stress evaluator for both layouts. O(n*(n+m)); NOT part of CLI timing.
pub fn full_stress(graph: &Graph, positions: &[[f64; 2]]) -> f64 {
    let mut stress = 0.0;
    for u in 0..graph.node_size {
        for (v, d) in graph
            .shortest_path_distances(u)
            .into_iter()
            .enumerate()
            .skip(u + 1)
        {
            if d == INF {
                continue;
            }
            let dx = positions[u][0] - positions[v][0];
            let dy = positions[u][1] - positions[v][1];
            let error = dx.hypot(dy) / f64::from(d) - 1.0;
            stress += error * error;
        }
    }
    stress
}
