use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

pub const AUTO_EXACT_MAX_N: usize = 8_000;
pub const DEFAULT_SAMPLES: usize = 64;
pub const DEFAULT_SEED: u64 = 0;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StressMode {
    Auto,
    Exact,
    Sampled,
    Off,
}

impl StressMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "exact" => Ok(Self::Exact),
            "sampled" => Ok(Self::Sampled),
            "off" => Ok(Self::Off),
            _ => anyhow::bail!("不正なstress mode '{value}'です（auto|exact|sampled|off）"),
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StressKind {
    Exact,
    Sampled,
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

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct StressMeasurement {
    pub stress_kind: Option<StressKind>,
    pub stress_value: Option<f64>,
    pub stress_eval_time_ms: Option<f64>,
    pub stress_samples: Option<usize>,
    pub stress_seed: Option<u64>,
}

impl StressMeasurement {
    pub fn from_result(result: StressResult, elapsed_ms: f64) -> Self {
        match result {
            StressResult::Exact(value) => Self {
                stress_kind: Some(StressKind::Exact),
                stress_value: Some(value),
                stress_eval_time_ms: Some(elapsed_ms),
                stress_samples: None,
                stress_seed: None,
            },
            StressResult::Sampled {
                value,
                samples,
                seed,
            } => Self {
                stress_kind: Some(StressKind::Sampled),
                stress_value: Some(value),
                stress_eval_time_ms: Some(elapsed_ms),
                stress_samples: Some(samples),
                stress_seed: Some(seed),
            },
            StressResult::Off => Self::default(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self.stress_kind {
            Some(StressKind::Exact) => {
                ensure!(
                    self.stress_samples.is_none(),
                    "exact stressのsamplesはnullです"
                );
                ensure!(self.stress_seed.is_none(), "exact stressのseedはnullです");
            }
            Some(StressKind::Sampled) => {
                ensure!(
                    self.stress_samples.is_some_and(|value| value > 0),
                    "sampled stressには正のsamplesが必要です"
                );
                ensure!(
                    self.stress_seed.is_some(),
                    "sampled stressにはseedが必要です"
                );
            }
            None => anyhow::bail!("成功recordにはstress_kindが必要です"),
        }
        ensure!(
            self.stress_value
                .is_some_and(|value| value.is_finite() && value >= 0.0),
            "stress_valueは有限な非負値です"
        );
        ensure!(
            self.stress_eval_time_ms
                .is_some_and(|value| value.is_finite() && value >= 0.0),
            "stress_eval_time_msは有限な非負値です"
        );
        Ok(())
    }
}

pub fn measure_auto_f64(positions: &[[f64; 2]], edges: &[(usize, usize)]) -> StressMeasurement {
    let started = Instant::now();
    let result = evaluate_f64(
        StressMode::Auto,
        positions,
        edges,
        DEFAULT_SAMPLES,
        DEFAULT_SEED,
    );
    StressMeasurement::from_result(result, started.elapsed().as_secs_f64() * 1_000.0)
}

pub fn measure_auto_f32(positions: &[[f32; 2]], edges: &[(usize, usize)]) -> StressMeasurement {
    let started = Instant::now();
    let result = evaluate_f32(
        StressMode::Auto,
        positions,
        edges,
        DEFAULT_SAMPLES,
        DEFAULT_SEED,
    );
    StressMeasurement::from_result(result, started.elapsed().as_secs_f64() * 1_000.0)
}

pub fn evaluate_f64(
    mode: StressMode,
    positions: &[[f64; 2]],
    edges: &[(usize, usize)],
    samples: usize,
    seed: u64,
) -> StressResult {
    evaluate_impl(mode, positions.len(), edges, samples, seed, |index| {
        positions[index]
    })
}

pub fn evaluate_f32(
    mode: StressMode,
    positions: &[[f32; 2]],
    edges: &[(usize, usize)],
    samples: usize,
    seed: u64,
) -> StressResult {
    evaluate_impl(mode, positions.len(), edges, samples, seed, |index| {
        [
            f64::from(positions[index][0]),
            f64::from(positions[index][1]),
        ]
    })
}

fn evaluate_impl<F>(
    mode: StressMode,
    node_count: usize,
    edges: &[(usize, usize)],
    samples: usize,
    seed: u64,
    position: F,
) -> StressResult
where
    F: Fn(usize) -> [f64; 2],
{
    let adjacency = adjacency(node_count, edges);
    match mode.resolve(node_count) {
        StressMode::Exact => StressResult::Exact(exact_impl(node_count, &adjacency, &position)),
        StressMode::Sampled => {
            let used = samples.min(node_count);
            let value = sampled_impl(node_count, &adjacency, used, seed, &position);
            StressResult::Sampled {
                value,
                samples: used,
                seed,
            }
        }
        StressMode::Off => StressResult::Off,
        StressMode::Auto => unreachable!("Autoは解決済みです"),
    }
}

fn exact_impl<F>(node_count: usize, adjacency: &[Vec<usize>], position: &F) -> f64
where
    F: Fn(usize) -> [f64; 2],
{
    (0..node_count)
        .map(|source| {
            let distances = bfs(adjacency, source);
            ((source + 1)..node_count)
                .map(|target| pair_term(position, &distances, source, target))
                .sum::<f64>()
        })
        .sum()
}

fn sampled_impl<F>(
    node_count: usize,
    adjacency: &[Vec<usize>],
    samples: usize,
    seed: u64,
    position: &F,
) -> f64
where
    F: Fn(usize) -> [f64; 2],
{
    if node_count == 0 || samples == 0 {
        return 0.0;
    }
    let sources = sample_indices(node_count, samples.min(node_count), seed);
    let directed_sum: f64 = sources
        .iter()
        .map(|&source| {
            let distances = bfs(adjacency, source);
            (0..node_count)
                .filter(|&target| target != source)
                .map(|target| pair_term(position, &distances, source, target))
                .sum::<f64>()
        })
        .sum();
    node_count as f64 * directed_sum / (2.0 * sources.len() as f64)
}

pub fn sample_indices(node_count: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut values: Vec<usize> = (0..node_count).collect();
    let mut state = seed;
    for index in 0..count.min(node_count) {
        state = splitmix64(state);
        let target = index + (state as usize % (node_count - index));
        values.swap(index, target);
    }
    values.truncate(count.min(node_count));
    values
}

fn adjacency(node_count: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for &(source, target) in edges {
        if source < node_count && target < node_count && source != target {
            adjacency[source].push(target);
            adjacency[target].push(source);
        }
    }
    adjacency
}

fn pair_term<F>(position: &F, distances: &[u32], source: usize, target: usize) -> f64
where
    F: Fn(usize) -> [f64; 2],
{
    let distance = distances[target];
    if distance == u32::MAX || distance == 0 {
        return 0.0;
    }
    let distance = f64::from(distance);
    let source_position = position(source);
    let target_position = position(target);
    let dx = source_position[0] - target_position[0];
    let dy = source_position[1] - target_position[1];
    let delta = (dx * dx + dy * dy).sqrt() - distance;
    delta * delta / (distance * distance)
}

fn bfs(adjacency: &[Vec<usize>], source: usize) -> Vec<u32> {
    let mut distances = vec![u32::MAX; adjacency.len()];
    distances[source] = 0;
    let mut queue = VecDeque::with_capacity(adjacency.len());
    queue.push_back(source);
    while let Some(vertex) = queue.pop_front() {
        for &neighbor in &adjacency[vertex] {
            if distances[neighbor] == u32::MAX {
                distances[neighbor] = distances[vertex] + 1;
                queue.push_back(neighbor);
            }
        }
    }
    distances
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path4() -> (Vec<[f64; 2]>, Vec<(usize, usize)>) {
        (
            vec![[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]],
            vec![(0, 1), (1, 2), (2, 3)],
        )
    }

    #[test]
    fn exact_regression() {
        let (positions, edges) = path4();
        assert_eq!(
            evaluate_f64(StressMode::Exact, &positions, &edges, 64, 0),
            StressResult::Exact(6.0)
        );
    }

    #[test]
    fn all_sources_sample_matches_exact() {
        let (positions, edges) = path4();
        let exact = evaluate_f64(StressMode::Exact, &positions, &edges, 64, 0);
        let sampled = evaluate_f64(StressMode::Sampled, &positions, &edges, positions.len(), 99);
        let StressResult::Exact(exact) = exact else {
            unreachable!()
        };
        let StressResult::Sampled { value, .. } = sampled else {
            unreachable!()
        };
        assert!((value - exact).abs() < 1e-12);
    }

    #[test]
    fn sampling_is_unique_and_repeatable() {
        let first = sample_indices(100, 64, 7);
        assert_eq!(first, sample_indices(100, 64, 7));
        let mut unique = first.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 64);
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
        assert_eq!(DEFAULT_SAMPLES, 64);
        assert_eq!(DEFAULT_SEED, 0);
    }

    #[test]
    fn f32_and_f64_paths_match_for_identical_values() {
        let (positions, edges) = path4();
        let positions_f32: Vec<_> = positions
            .iter()
            .map(|position| [position[0] as f32, position[1] as f32])
            .collect();
        assert_eq!(
            evaluate_f64(StressMode::Exact, &positions, &edges, 64, 0),
            evaluate_f32(StressMode::Exact, &positions_f32, &edges, 64, 0)
        );
    }
}
