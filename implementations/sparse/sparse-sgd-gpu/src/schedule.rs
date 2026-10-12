use crate::graph::{EdgeInfo, Graph, SgdParams};
use anyhow::{bail, ensure, Context, Result};
use bytemuck::{Pod, Zeroable};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::collections::{HashMap, HashSet};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct OneSidedEntry {
    pub pivot: u32,
    pub dij: f32,
    pub weight: f32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuPair {
    pub u: u32,
    pub v: u32,
    pub dij: f32,
    pub weight_u: f32,
    pub weight_v: f32,
    pub _pad: u32,
}

impl GpuPair {
    fn try_from_edge(edge: &EdgeInfo) -> Result<Self> {
        Ok(Self {
            u: u32::try_from(edge.u).context("two-sided endpoint u exceeds u32")?,
            v: u32::try_from(edge.v).context("two-sided endpoint v exceeds u32")?,
            dij: edge.dij as f32,
            weight_u: edge.weight_u as f32,
            weight_v: edge.weight_v as f32,
            _pad: 0,
        })
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AssignedPair {
    pair_index: usize,
    round: u32,
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
    Ok(build_schedule_compact(graph, prepared, seed)?.0)
}

fn build_schedule_compact(
    graph: &Graph,
    prepared: &SgdParams,
    seed: u64,
) -> Result<(Schedule, usize)> {
    let n = graph.node_size;
    let h = prepared.pivots.len();
    ensure!(h > 0, "pivot count must be positive");
    ensure!(
        n <= u32::MAX as usize,
        "graph has more vertices than the GPU schedule can represent"
    );

    let pivot_set: HashSet<usize> = prepared.pivots.iter().copied().collect();
    let pivot_index: HashMap<usize, usize> = prepared
        .pivots
        .iter()
        .enumerate()
        .map(|(index, &pivot)| (pivot, index))
        .collect();

    // Absent constraints have zero weight. The dense n*h layout is unchanged so the
    // GPU can consume one common seeded pivot permutation for every vertex.
    let dense_len = n
        .checked_mul(h)
        .context("one-sided dense schedule length overflow")?;
    let mut one_sided = vec![OneSidedEntry::default(); dense_len];
    for vertex in 0..n {
        for (pivot_index_in_list, &pivot) in prepared.pivots.iter().enumerate() {
            one_sided[vertex * h + pivot_index_in_list].pivot =
                u32::try_from(pivot).context("pivot endpoint exceeds u32")?;
        }
    }

    let mut one_sided_count = 0_usize;
    let mut pivot_pair_indices = HashMap::<(usize, usize), usize>::new();
    let mut graph_pair_indices = Vec::<usize>::new();
    for (pair_index_in_prepared, edge) in prepared.pairs.iter().enumerate() {
        ensure!(
            edge.u < n && edge.v < n,
            "constraint endpoint exceeds graph size: ({}, {}), n={n}",
            edge.u,
            edge.v
        );
        let u_positive = edge.weight_u > 0.0;
        let v_positive = edge.weight_v > 0.0;
        match (u_positive, v_positive) {
            (true, false) if pivot_set.contains(&edge.v) => {
                let pivot_slot = pivot_index[&edge.v];
                one_sided[edge.u * h + pivot_slot] = OneSidedEntry {
                    pivot: u32::try_from(edge.v).context("pivot endpoint exceeds u32")?,
                    dij: edge.dij as f32,
                    weight: edge.weight_u as f32,
                    _pad: 0,
                };
                one_sided_count = one_sided_count
                    .checked_add(1)
                    .context("one-sided constraint count overflow")?;
            }
            (false, true) if pivot_set.contains(&edge.u) => {
                let pivot_slot = pivot_index[&edge.u];
                one_sided[edge.v * h + pivot_slot] = OneSidedEntry {
                    pivot: u32::try_from(edge.u).context("pivot endpoint exceeds u32")?,
                    dij: edge.dij as f32,
                    weight: edge.weight_v as f32,
                    _pad: 0,
                };
                one_sided_count = one_sided_count
                    .checked_add(1)
                    .context("one-sided constraint count overflow")?;
            }
            (true, true) => {
                if pivot_set.contains(&edge.u) && pivot_set.contains(&edge.v) {
                    ensure!(
                        pivot_pair_indices
                            .insert(key(edge.u, edge.v), pair_index_in_prepared)
                            .is_none(),
                        "duplicate pivot pair ({}, {})",
                        edge.u,
                        edge.v
                    );
                } else {
                    graph_pair_indices.push(pair_index_in_prepared);
                }
            }
            _ => bail!("unclassifiable constraint ({}, {})", edge.u, edge.v),
        }
    }

    let (mut assignments, base_rounds) =
        circle_pivot_assignments(&prepared.pivots, &pivot_pair_indices)?;
    assignments
        .try_reserve(graph_pair_indices.len())
        .context("two-sided assignment allocation failed")?;

    // Each assignment contributes exactly one round ID to each endpoint. Capacities
    // are derived from graph degrees; pivots reserve their complete pivot-pair set.
    let mut vertex_rounds = Vec::with_capacity(n);
    for vertex in 0..n {
        let pivot_capacity = usize::from(pivot_set.contains(&vertex))
            .checked_mul(h.saturating_sub(1))
            .context("pivot round membership capacity overflow")?;
        let capacity = graph
            .neighbors(vertex)
            .len()
            .checked_add(pivot_capacity)
            .context("vertex round membership capacity overflow")?;
        vertex_rounds.push(Vec::<u32>::with_capacity(capacity));
    }
    for assignment in &assignments {
        let edge = &prepared.pairs[assignment.pair_index];
        vertex_rounds[edge.u].push(assignment.round);
        vertex_rounds[edge.v].push(assignment.round);
    }

    let mut rng = StdRng::seed_from_u64(seed ^ 0x5350_4152_5345_5347);
    graph_pair_indices.shuffle(&mut rng);
    let mut round_marks = vec![0_u32; base_rounds];
    let mut epoch = 0_u32;
    for &pair_index_in_prepared in &graph_pair_indices {
        epoch = next_epoch(&mut round_marks, epoch);
        let edge = &prepared.pairs[pair_index_in_prepared];
        for &round in vertex_rounds[edge.u].iter().chain(&vertex_rounds[edge.v]) {
            let round = usize::try_from(round).context("round ID exceeds usize")?;
            ensure!(
                round < round_marks.len(),
                "vertex round membership is out of range"
            );
            round_marks[round] = epoch;
        }

        let round = if let Some(round) = round_marks.iter().position(|&mark| mark != epoch) {
            round
        } else {
            let round = round_marks.len();
            ensure!(
                round < u32::MAX as usize,
                "two-sided round count exceeds u32"
            );
            round_marks.push(0);
            round
        };
        let round = u32::try_from(round).context("round ID exceeds u32")?;
        vertex_rounds[edge.u].push(round);
        vertex_rounds[edge.v].push(round);
        assignments.push(AssignedPair {
            pair_index: pair_index_in_prepared,
            round,
        });
    }

    let round_membership_count = vertex_rounds.iter().try_fold(0_usize, |sum, rounds| {
        sum.checked_add(rounds.len())
            .context("round membership count overflow")
    })?;
    let expected_memberships = assignments
        .len()
        .checked_mul(2)
        .context("round membership expected count overflow")?;
    ensure!(
        round_membership_count == expected_memberships,
        "each two-sided assignment must have two vertex memberships"
    );
    let round_count = round_marks.len();
    drop(vertex_rounds);
    drop(round_marks);
    drop(graph_pair_indices);

    let classified_count = one_sided_count
        .checked_add(assignments.len())
        .context("classified constraint count overflow")?;
    ensure!(
        classified_count == prepared.pairs.len(),
        "classified constraints do not cover the prepared constraint set"
    );

    let (two_sided, round_offsets) =
        flatten_assignments(&prepared.pairs, &assignments, round_count)?;
    drop(assignments);
    validate_flat_rounds(&two_sided, &round_offsets, n)?;

    let max_graph_degree = (0..graph.node_size)
        .map(|vertex| graph.neighbors(vertex).len())
        .max()
        .unwrap_or(0);
    ensure!(
        round_count <= h + 2 * max_graph_degree,
        "round limit exceeded: rounds={round_count}, h={h}, delta={max_graph_degree}"
    );

    Ok((
        Schedule {
            one_sided,
            one_sided_count,
            two_sided,
            round_offsets,
            base_rounds,
            spill_rounds: round_count - base_rounds,
            max_graph_degree,
        },
        round_membership_count,
    ))
}

fn next_epoch(marks: &mut [u32], current: u32) -> u32 {
    if current == u32::MAX {
        marks.fill(0);
        1
    } else {
        current + 1
    }
}

fn circle_pivot_assignments(
    pivots: &[usize],
    pair_indices: &HashMap<(usize, usize), usize>,
) -> Result<(Vec<AssignedPair>, usize)> {
    let expected_pairs = pivots
        .len()
        .checked_mul(pivots.len().saturating_sub(1))
        .context("pivot-pair count overflow")?
        / 2;
    ensure!(
        pair_indices.len() == expected_pairs,
        "pivot-pair set is incomplete"
    );

    let dummy = usize::MAX;
    let mut players = pivots.to_vec();
    if players.len() % 2 == 1 {
        players.push(dummy);
    }
    let player_count = players.len();
    let round_count = player_count.saturating_sub(1);
    ensure!(
        round_count <= u32::MAX as usize,
        "pivot round count exceeds u32"
    );
    let mut assignments = Vec::with_capacity(expected_pairs);
    for round in 0..round_count {
        let round = u32::try_from(round).context("pivot round ID exceeds u32")?;
        for index in 0..player_count / 2 {
            let u = players[index];
            let v = players[player_count - 1 - index];
            if u != dummy && v != dummy {
                let &pair_index = pair_indices
                    .get(&key(u, v))
                    .ok_or_else(|| anyhow::anyhow!("missing pivot pair ({u}, {v})"))?;
                assignments.push(AssignedPair { pair_index, round });
            }
        }
        if player_count > 2 {
            let last = players.pop().expect("player count was checked");
            players.insert(1, last);
        }
    }
    Ok((assignments, round_count))
}

fn flatten_assignments(
    prepared_pairs: &[EdgeInfo],
    assignments: &[AssignedPair],
    round_count: usize,
) -> Result<(Vec<GpuPair>, Vec<u32>)> {
    let mut counts = vec![0_usize; round_count];
    for assignment in assignments {
        let round = usize::try_from(assignment.round).context("round ID exceeds usize")?;
        let count = counts
            .get_mut(round)
            .context("assigned round ID is out of range")?;
        *count = count.checked_add(1).context("round size overflow")?;
    }

    let mut round_offsets = Vec::with_capacity(
        round_count
            .checked_add(1)
            .context("round offset length overflow")?,
    );
    round_offsets.push(0);
    let mut total = 0_usize;
    for count in counts {
        total = total
            .checked_add(count)
            .context("two-sided size overflow")?;
        round_offsets.push(u32::try_from(total).context("two-sided offsets exceed u32")?);
    }
    ensure!(
        total == assignments.len(),
        "round counts do not cover all assignments"
    );

    let mut cursors: Vec<usize> = round_offsets[..round_count]
        .iter()
        .map(|&offset| offset as usize)
        .collect();
    let mut two_sided = vec![GpuPair::default(); assignments.len()];
    for assignment in assignments {
        let round = usize::try_from(assignment.round).context("round ID exceeds usize")?;
        let cursor = cursors
            .get_mut(round)
            .context("assigned round ID is out of range")?;
        let edge = prepared_pairs
            .get(assignment.pair_index)
            .context("assigned pair index is out of range")?;
        two_sided[*cursor] = GpuPair::try_from_edge(edge)?;
        *cursor = cursor.checked_add(1).context("round cursor overflow")?;
    }
    for (round, &cursor) in cursors.iter().enumerate() {
        ensure!(
            cursor == round_offsets[round + 1] as usize,
            "round {round} was not filled exactly"
        );
    }
    Ok((two_sided, round_offsets))
}

pub fn validate_flat_rounds(
    pairs: &[GpuPair],
    round_offsets: &[u32],
    node_count: usize,
) -> Result<()> {
    ensure!(!round_offsets.is_empty(), "round offsets are empty");
    ensure!(round_offsets[0] == 0, "round offsets must start at zero");
    ensure!(
        round_offsets.last().copied().unwrap_or(0) as usize == pairs.len(),
        "round offsets do not cover all pairs"
    );

    let mut seen = vec![0_u32; node_count];
    let mut epoch = 0_u32;
    for (round, offsets) in round_offsets.windows(2).enumerate() {
        ensure!(offsets[0] <= offsets[1], "round offsets are not monotonic");
        epoch = next_epoch(&mut seen, epoch);
        for pair in &pairs[offsets[0] as usize..offsets[1] as usize] {
            let u = pair.u as usize;
            let v = pair.v as usize;
            ensure!(
                u < node_count && v < node_count,
                "pair endpoint exceeds graph size"
            );
            ensure!(seen[u] != epoch, "vertex {u} repeats in round {round}");
            seen[u] = epoch;
            ensure!(seen[v] != epoch, "vertex {v} repeats in round {round}");
            seen[v] = epoch;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn edge(u: usize, v: usize) -> EdgeInfo {
        EdgeInfo {
            u,
            v,
            dij: 1.0,
            weight_u: 1.0,
            weight_v: 1.0,
        }
    }

    fn reference_two_sided(prepared: &SgdParams, seed: u64) -> (Vec<GpuPair>, Vec<u32>, usize) {
        let pivot_set: HashSet<_> = prepared.pivots.iter().copied().collect();
        let mut pivot_pairs = HashMap::<(usize, usize), EdgeInfo>::new();
        let mut graph_pairs = Vec::<EdgeInfo>::new();
        for constraint in &prepared.pairs {
            if constraint.weight_u > 0.0 && constraint.weight_v > 0.0 {
                if pivot_set.contains(&constraint.u) && pivot_set.contains(&constraint.v) {
                    pivot_pairs.insert(key(constraint.u, constraint.v), *constraint);
                } else {
                    graph_pairs.push(*constraint);
                }
            }
        }

        let mut rounds = reference_circle_rounds(&prepared.pivots, &pivot_pairs);
        let base_rounds = rounds.len();
        let mut used: Vec<HashSet<usize>> = rounds
            .iter()
            .map(|round| {
                round
                    .iter()
                    .flat_map(|constraint| [constraint.u, constraint.v])
                    .collect()
            })
            .collect();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5350_4152_5345_5347);
        graph_pairs.shuffle(&mut rng);
        for constraint in graph_pairs {
            let round = used
                .iter()
                .position(|vertices| {
                    !vertices.contains(&constraint.u) && !vertices.contains(&constraint.v)
                })
                .unwrap_or_else(|| {
                    rounds.push(Vec::new());
                    used.push(HashSet::new());
                    rounds.len() - 1
                });
            used[round].insert(constraint.u);
            used[round].insert(constraint.v);
            rounds[round].push(constraint);
        }

        let mut pairs = Vec::new();
        let mut offsets = vec![0];
        for round in rounds {
            pairs.extend(
                round
                    .iter()
                    .map(|constraint| GpuPair::try_from_edge(constraint).unwrap()),
            );
            offsets.push(u32::try_from(pairs.len()).unwrap());
        }
        (pairs, offsets, base_rounds)
    }

    fn reference_circle_rounds(
        pivots: &[usize],
        pairs: &HashMap<(usize, usize), EdgeInfo>,
    ) -> Vec<Vec<EdgeInfo>> {
        let dummy = usize::MAX;
        let mut players = pivots.to_vec();
        if players.len() % 2 == 1 {
            players.push(dummy);
        }
        let player_count = players.len();
        let mut rounds = Vec::with_capacity(player_count.saturating_sub(1));
        for _ in 0..player_count.saturating_sub(1) {
            let mut round = Vec::new();
            for index in 0..player_count / 2 {
                let u = players[index];
                let v = players[player_count - 1 - index];
                if u != dummy && v != dummy {
                    round.push(pairs[&key(u, v)]);
                }
            }
            rounds.push(round);
            if player_count > 2 {
                let last = players.pop().unwrap();
                players.insert(1, last);
            }
        }
        rounds
    }

    #[test]
    fn circle_method_even_and_odd_are_complete_matchings() {
        for h in [3usize, 4, 5, 6] {
            let pivots: Vec<_> = (0..h).collect();
            let edges: Vec<_> = (0..h)
                .flat_map(|u| (u + 1..h).map(move |v| edge(u, v)))
                .collect();
            let indices: HashMap<_, _> = edges
                .iter()
                .enumerate()
                .map(|(index, constraint)| (key(constraint.u, constraint.v), index))
                .collect();
            let (assignments, round_count) = circle_pivot_assignments(&pivots, &indices).unwrap();
            let (pairs, offsets) = flatten_assignments(&edges, &assignments, round_count).unwrap();
            validate_flat_rounds(&pairs, &offsets, h).unwrap();
            assert_eq!(round_count, if h % 2 == 0 { h - 1 } else { h });
            assert_eq!(pairs.len(), h * (h - 1) / 2);
        }
    }

    #[test]
    fn flat_validator_rejects_endpoint_conflict() {
        let pairs = [
            GpuPair::try_from_edge(&edge(0, 1)).unwrap(),
            GpuPair::try_from_edge(&edge(0, 2)).unwrap(),
        ];
        assert!(validate_flat_rounds(&pairs, &[0, 2], 3).is_err());
    }

    #[test]
    fn compact_schedule_matches_reference_first_fit_exactly() {
        let graph = Graph::try_from_edges(
            9,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 0),
            ],
        )
        .unwrap();
        for seed in [0_u64, 1, 19, 77] {
            let mut rng = StdRng::seed_from_u64(seed);
            let params = graph.prepare_sgd_params(3, 0.1, 4, true, &mut rng).unwrap();
            let (schedule, _) = build_schedule_compact(&graph, &params, seed).unwrap();
            let (pairs, offsets, base_rounds) = reference_two_sided(&params, seed);
            assert_eq!(schedule.two_sided, pairs);
            assert_eq!(schedule.round_offsets, offsets);
            assert_eq!(schedule.base_rounds, base_rounds);
            assert_eq!(
                schedule.spill_rounds,
                schedule.round_count() - schedule.base_rounds
            );
        }
    }

    #[test]
    fn prepared_constraints_are_classified_once_and_rounds_are_matchings() {
        let graph =
            Graph::try_from_edges(7, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)])
                .unwrap();
        let mut rng = StdRng::seed_from_u64(19);
        let params = graph.prepare_sgd_params(3, 0.1, 3, true, &mut rng).unwrap();
        let first = build_schedule(&graph, &params, 19).unwrap();
        let second = build_schedule(&graph, &params, 19).unwrap();
        assert_eq!(
            first.one_sided_count + first.two_sided.len(),
            params.pairs.len()
        );
        assert!(first.round_count() <= params.pivots.len() + 2 * first.max_graph_degree);
        assert_eq!(first.one_sided, second.one_sided);
        assert_eq!(first.two_sided, second.two_sided);
        assert_eq!(first.round_offsets, second.round_offsets);
        validate_flat_rounds(&first.two_sided, &first.round_offsets, graph.node_size).unwrap();
    }

    #[test]
    fn high_degree_star_uses_delta_rounds_and_two_memberships_per_pair() {
        let delta = 128;
        let edges: Vec<_> = (1..=delta).map(|leaf| (0, leaf)).collect();
        let graph = Graph::try_from_edges(delta + 1, &edges).unwrap();
        let mut rng = StdRng::seed_from_u64(3);
        let params = graph
            .prepare_sgd_params(1, 0.1, 1, false, &mut rng)
            .unwrap();
        let (schedule, memberships) = build_schedule_compact(&graph, &params, 3).unwrap();
        assert_eq!(schedule.round_count(), delta);
        assert_eq!(schedule.two_sided.len(), delta);
        assert_eq!(memberships, schedule.two_sided.len() * 2);
        validate_flat_rounds(
            &schedule.two_sided,
            &schedule.round_offsets,
            graph.node_size,
        )
        .unwrap();
    }
}
