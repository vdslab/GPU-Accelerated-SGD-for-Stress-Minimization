use std::collections::VecDeque;

pub const AUTO_EXACT_MAX_N: usize = 8_000;
pub const DEFAULT_SAMPLES: usize = 64;
pub const DEFAULT_SEED: u64 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StressMode {
    Auto,
    Exact,
    Sampled,
    Off,
}

impl StressMode {
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "exact" => Ok(Self::Exact),
            "sampled" => Ok(Self::Sampled),
            "off" => Ok(Self::Off),
            _ => anyhow::bail!(
                "不正な --stress モード '{value}' です（auto|exact|sampled|off を指定してください）"
            ),
        }
    }

    pub fn resolve(self, node_count: usize) -> Self {
        match self {
            Self::Auto if node_count <= AUTO_EXACT_MAX_N => Self::Exact,
            Self::Auto => Self::Sampled,
            other => other,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum StressResult {
    Exact(f64),
    Sampled {
        value: f64,
        samples: usize,
        seed: u64,
    },
    Off,
}

pub fn evaluate(
    mode: StressMode,
    positions: &[[f32; 2]],
    edges: &[(usize, usize)],
    samples: usize,
    seed: u64,
) -> StressResult {
    match mode.resolve(positions.len()) {
        StressMode::Exact => StressResult::Exact(exact(positions, edges)),
        StressMode::Sampled => {
            let used = samples.min(positions.len());
            let value = sampled(positions, edges, used, seed);
            StressResult::Sampled {
                value,
                samples: used,
                seed,
            }
        }
        StressMode::Off => StressResult::Off,
        StressMode::Auto => unreachable!("autoは頂点数に応じて解決済み"),
    }
}

/// Σ(i<j) (||x_i-x_j||-d_ij)^2 / d_ij^2。到達不能対は無視する。
pub fn exact(positions: &[[f32; 2]], edges: &[(usize, usize)]) -> f64 {
    let adj = adjacency(positions.len(), edges);
    (0..positions.len())
        .map(|src| {
            let dist = bfs(&adj, src);
            ((src + 1)..positions.len())
                .map(|dst| pair_term(positions, &dist, src, dst))
                .sum::<f64>()
        })
        .sum()
}

/// 一様な始点標本による無向全頂点対ストレスの不偏推定。
pub fn sampled(positions: &[[f32; 2]], edges: &[(usize, usize)], samples: usize, seed: u64) -> f64 {
    let n = positions.len();
    if n == 0 || samples == 0 {
        return 0.0;
    }
    let adj = adjacency(n, edges);
    let sources = sample_indices(n, samples.min(n), seed);
    let directed_sum: f64 = sources
        .iter()
        .map(|&src| {
            let dist = bfs(&adj, src);
            (0..n)
                .filter(|&dst| dst != src)
                .map(|dst| pair_term(positions, &dist, src, dst))
                .sum::<f64>()
        })
        .sum();
    n as f64 * directed_sum / (2.0 * sources.len() as f64)
}

pub fn sample_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut values: Vec<usize> = (0..n).collect();
    let mut state = seed;
    for i in 0..count.min(n) {
        state = splitmix64(state);
        let j = i + (state as usize % (n - i));
        values.swap(i, j);
    }
    values.truncate(count.min(n));
    values
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn adjacency(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for &(u, v) in edges {
        if u < n && v < n {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    adj
}

fn pair_term(positions: &[[f32; 2]], dist: &[u32], src: usize, dst: usize) -> f64 {
    let d = dist[dst];
    if d == u32::MAX || d == 0 {
        return 0.0;
    }
    let d = f64::from(d);
    let dx = f64::from(positions[src][0]) - f64::from(positions[dst][0]);
    let dy = f64::from(positions[src][1]) - f64::from(positions[dst][1]);
    let delta = (dx * dx + dy * dy).sqrt() - d;
    delta * delta / (d * d)
}

fn bfs(adj: &[Vec<usize>], src: usize) -> Vec<u32> {
    let mut dist = vec![u32::MAX; adj.len()];
    dist[src] = 0;
    let mut queue = VecDeque::with_capacity(adj.len());
    queue.push_back(src);
    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if dist[v] == u32::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path4() -> (Vec<[f32; 2]>, Vec<(usize, usize)>) {
        (
            vec![[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]],
            vec![(0, 1), (1, 2), (2, 3)],
        )
    }

    #[test]
    fn exact_regression() {
        let (positions, edges) = path4();
        assert!((exact(&positions, &edges) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn all_sources_matches_exact() {
        let (positions, edges) = path4();
        assert!(
            (sampled(&positions, &edges, positions.len(), 99) - exact(&positions, &edges)).abs()
                < 1e-12
        );
    }

    #[test]
    fn sampling_is_unique_and_repeatable() {
        let a = sample_indices(100, 64, 7);
        let b = sample_indices(100, 64, 7);
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 64);
    }

    #[test]
    fn disconnected_pairs_are_ignored() {
        let positions = vec![[0.0, 0.0], [2.0, 0.0], [100.0, 100.0]];
        let edges = vec![(0, 1)];
        assert!((exact(&positions, &edges) - 1.0).abs() < 1e-12);
        assert!((sampled(&positions, &edges, 3, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn auto_switches_at_threshold() {
        assert_eq!(
            StressMode::Auto.resolve(AUTO_EXACT_MAX_N),
            StressMode::Exact
        );
        assert_eq!(
            StressMode::Auto.resolve(AUTO_EXACT_MAX_N + 1),
            StressMode::Sampled
        );
    }
}
