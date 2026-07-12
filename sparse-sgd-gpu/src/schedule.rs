use crate::graph::{EdgeInfo, Graph, SgdParams};
use anyhow::{bail, ensure, Result};
use bytemuck::{Pod, Zeroable};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::collections::{HashMap, HashSet};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct OneSidedEntry {
    pub pivot: u32,
    pub dij: f32,
    pub weight: f32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuPair {
    pub u: u32,
    pub v: u32,
    pub dij: f32,
    pub weight_u: f32,
    pub weight_v: f32,
    pub _pad: u32,
}

impl From<&EdgeInfo> for GpuPair {
    fn from(e: &EdgeInfo) -> Self {
        Self {
            u: e.u as u32,
            v: e.v as u32,
            dij: e.dij as f32,
            weight_u: e.weight_u as f32,
            weight_v: e.weight_v as f32,
            _pad: 0,
        }
    }
}

#[derive(Debug)]
pub struct Schedule {
    /// Dense CSR rows (n rows, h slots) permit a common shuffled pivot order on GPU.
    pub one_sided: Vec<OneSidedEntry>,
    pub one_sided_count: usize,
    pub two_sided: Vec<GpuPair>,
    pub round_offsets: Vec<u32>,
    pub base_rounds: usize,
    pub spill_rounds: usize,
    pub max_graph_degree: usize,
}

impl Schedule {
    pub fn round_count(&self) -> usize {
        self.round_offsets.len().saturating_sub(1)
    }

    pub fn round_range(&self, round: usize) -> std::ops::Range<usize> {
        self.round_offsets[round] as usize..self.round_offsets[round + 1] as usize
    }
}

fn key(u: usize, v: usize) -> (usize, usize) {
    if u < v {
        (u, v)
    } else {
        (v, u)
    }
}

/// Builds a conflict-free two-phase schedule. Each one-sided invocation owns exactly
/// one writable vertex; every two-sided round is a matching.
pub fn build_schedule(graph: &Graph, prepared: &SgdParams, seed: u64) -> Result<Schedule> {
    let n = graph.node_size;
    let h = prepared.pivots.len();
    ensure!(h > 0, "pivot count must be positive");
    let pivot_set: HashSet<usize> = prepared.pivots.iter().copied().collect();
    let pivot_index: HashMap<usize, usize> = prepared
        .pivots
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();

    // Dense CSR: absent constraints have zero weight. This costs O(nh), but allows
    // every vertex invocation to consume the same seeded pivot permutation.
    let mut one_sided = vec![OneSidedEntry::default(); n * h];
    for v in 0..n {
        for (pi, &p) in prepared.pivots.iter().enumerate() {
            one_sided[v * h + pi].pivot = p as u32;
        }
    }

    let mut one_sided_count = 0;
    let mut pivot_pairs = HashMap::<(usize, usize), EdgeInfo>::new();
    let mut graph_pairs = Vec::<EdgeInfo>::new();
    for e in &prepared.pairs {
        let u_pos = e.weight_u > 0.0;
        let v_pos = e.weight_v > 0.0;
        match (u_pos, v_pos) {
            (true, false) if pivot_set.contains(&e.v) => {
                let pi = pivot_index[&e.v];
                one_sided[e.u * h + pi] = OneSidedEntry {
                    pivot: e.v as u32,
                    dij: e.dij as f32,
                    weight: e.weight_u as f32,
                    _pad: 0,
                };
                one_sided_count += 1;
            }
            (false, true) if pivot_set.contains(&e.u) => {
                let pi = pivot_index[&e.u];
                one_sided[e.v * h + pi] = OneSidedEntry {
                    pivot: e.u as u32,
                    dij: e.dij as f32,
                    weight: e.weight_v as f32,
                    _pad: 0,
                };
                one_sided_count += 1;
            }
            (true, true) => {
                if pivot_set.contains(&e.u) && pivot_set.contains(&e.v) {
                    pivot_pairs.insert(key(e.u, e.v), *e);
                } else {
                    graph_pairs.push(*e);
                }
            }
            _ => bail!("unclassifiable constraint ({}, {})", e.u, e.v),
        }
    }

    let mut rounds = circle_pivot_rounds(&prepared.pivots, &pivot_pairs)?;
    let base_rounds = rounds.len();
    let mut used: Vec<HashSet<usize>> = rounds
        .iter()
        .map(|r| r.iter().flat_map(|e| [e.u, e.v]).collect())
        .collect();

    let mut rng = StdRng::seed_from_u64(seed ^ 0x5350_4152_5345_5347);
    graph_pairs.shuffle(&mut rng);
    for edge in graph_pairs {
        let mut target = None;
        for (ri, vertices) in used.iter().enumerate() {
            if !vertices.contains(&edge.u) && !vertices.contains(&edge.v) {
                target = Some(ri);
                break;
            }
        }
        let ri = if let Some(ri) = target {
            ri
        } else {
            rounds.push(Vec::new());
            used.push(HashSet::new());
            rounds.len() - 1
        };
        used[ri].insert(edge.u);
        used[ri].insert(edge.v);
        rounds[ri].push(edge);
    }

    validate_rounds(&rounds)?;
    let mut two_sided = Vec::new();
    let mut round_offsets = Vec::with_capacity(rounds.len() + 1);
    round_offsets.push(0);
    for round in &rounds {
        two_sided.extend(round.iter().map(GpuPair::from));
        round_offsets.push(two_sided.len() as u32);
    }

    let max_graph_degree = (0..graph.node_size)
        .map(|v| graph.neighbors(v).len())
        .max()
        .unwrap_or(0);
    ensure!(
        rounds.len() <= h + 2 * max_graph_degree,
        "round上限を超えました: rounds={}, h={}, delta={}",
        rounds.len(),
        h,
        max_graph_degree
    );
    Ok(Schedule {
        one_sided,
        one_sided_count,
        two_sided,
        round_offsets,
        base_rounds,
        spill_rounds: rounds.len() - base_rounds,
        max_graph_degree,
    })
}

fn circle_pivot_rounds(
    pivots: &[usize],
    pairs: &HashMap<(usize, usize), EdgeInfo>,
) -> Result<Vec<Vec<EdgeInfo>>> {
    let dummy = usize::MAX;
    let mut players = pivots.to_vec();
    if players.len() % 2 == 1 {
        players.push(dummy);
    }
    let m = players.len();
    let rounds_n = m.saturating_sub(1);
    let mut rounds = Vec::with_capacity(rounds_n);
    for _ in 0..rounds_n {
        let mut round = Vec::new();
        for i in 0..m / 2 {
            let u = players[i];
            let v = players[m - 1 - i];
            if u != dummy && v != dummy {
                let e = pairs
                    .get(&key(u, v))
                    .ok_or_else(|| anyhow::anyhow!("missing pivot pair ({u}, {v})"))?;
                round.push(*e);
            }
        }
        rounds.push(round);
        if m > 2 {
            let last = players.pop().unwrap();
            players.insert(1, last);
        }
    }
    ensure!(
        pairs.len() == pivots.len() * pivots.len().saturating_sub(1) / 2,
        "pivot-pair set is incomplete"
    );
    Ok(rounds)
}

pub fn validate_rounds(rounds: &[Vec<EdgeInfo>]) -> Result<()> {
    let mut all = HashSet::new();
    for (ri, round) in rounds.iter().enumerate() {
        let mut used = HashSet::new();
        for e in round {
            ensure!(used.insert(e.u), "vertex {} repeats in round {ri}", e.u);
            ensure!(used.insert(e.v), "vertex {} repeats in round {ri}", e.v);
            ensure!(
                all.insert(key(e.u, e.v)),
                "duplicate pair ({}, {})",
                e.u,
                e.v
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(u: usize, v: usize) -> EdgeInfo {
        EdgeInfo {
            u,
            v,
            dij: 1.0,
            weight_u: 1.0,
            weight_v: 1.0,
        }
    }

    #[test]
    fn circle_method_even_and_odd_are_complete_matchings() {
        for h in [3usize, 4, 5, 6] {
            let pivots: Vec<_> = (0..h).collect();
            let pairs: HashMap<_, _> = (0..h)
                .flat_map(|u| (u + 1..h).map(move |v| (key(u, v), edge(u, v))))
                .collect();
            let rounds = circle_pivot_rounds(&pivots, &pairs).unwrap();
            validate_rounds(&rounds).unwrap();
            assert_eq!(rounds.len(), if h % 2 == 0 { h - 1 } else { h });
            assert_eq!(rounds.iter().map(Vec::len).sum::<usize>(), h * (h - 1) / 2);
        }
    }

    #[test]
    fn validator_rejects_endpoint_conflict() {
        assert!(validate_rounds(&[vec![edge(0, 1), edge(0, 2)]]).is_err());
    }

    #[test]
    fn prepared_constraints_are_classified_once_and_rounds_are_matchings() {
        use rand::{rngs::StdRng, SeedableRng};
        let graph =
            Graph::try_from_edges(7, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)])
                .unwrap();
        let mut rng = StdRng::seed_from_u64(19);
        let params = graph.prepare_sgd_params(3, 0.1, 3, true, &mut rng).unwrap();
        let schedule = build_schedule(&graph, &params, 19).unwrap();
        assert_eq!(
            schedule.one_sided_count + schedule.two_sided.len(),
            params.pairs.len()
        );
        assert!(schedule.round_count() <= params.pivots.len() + 2 * schedule.max_graph_degree);
        for round in 0..schedule.round_count() {
            let mut vertices = HashSet::new();
            for pair in &schedule.two_sided[schedule.round_range(round)] {
                assert!(vertices.insert(pair.u));
                assert!(vertices.insert(pair.v));
            }
        }
    }
}
