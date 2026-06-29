use crate::embedding::{Embedding, MIN_DELTA};
use crate::graph::Graph;
use crate::schedule::{Pair, WeightedPair};
use anyhow::{anyhow, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

pub fn sample_pivot_pairs(
    graph: &Graph,
    embedding: &Embedding,
    h: usize,
    seed: u64,
) -> Result<Vec<WeightedPair>> {
    let pivots = select_pivots_max_min_random_sp(graph, h, seed)?;
    let mut weights_by_pair: HashMap<(usize, usize), f32> = HashMap::new();

    for &pivot in &pivots {
        let neighbor_distances: Vec<f32> = graph
            .neighbors(pivot)
            .iter()
            .map(|&neighbor| embedding.distance(pivot, neighbor))
            .collect();

        for i in 0..graph.node_size {
            if i == pivot || graph.has_edge(pivot, i) {
                continue;
            }

            let delta = embedding.distance(pivot, i).max(MIN_DELTA);
            let threshold = delta / 2.0;
            let s = neighbor_distances
                .iter()
                .filter(|&&d| d <= threshold)
                .count() as f32;

            if s <= 0.0 {
                continue;
            }

            let weight = s / (delta * delta);
            let key = if pivot < i { (pivot, i) } else { (i, pivot) };
            *weights_by_pair.entry(key).or_insert(0.0) += weight;
        }
    }

    let mut pairs: Vec<WeightedPair> = weights_by_pair
        .into_iter()
        .map(|((u, v), weight)| WeightedPair {
            pair: Pair {
                u: u as u32,
                v: v as u32,
            },
            weight,
        })
        .collect();
    pairs.sort_by_key(|p| (p.pair.u, p.pair.v));

    if pairs.is_empty() {
        return Err(anyhow!(
            "sampling produced no positive-weight non-adjacent pivot pairs"
        ));
    }

    Ok(pairs)
}

pub fn select_pivots_max_min_random_sp(graph: &Graph, h: usize, seed: u64) -> Result<Vec<usize>> {
    if graph.node_size == 0 {
        return Err(anyhow!("cannot sample pivots from an empty graph"));
    }

    let target = h.min(graph.node_size).max(1);
    let mut rng = StdRng::seed_from_u64(seed);
    let first = rng.random_range(0..graph.node_size);
    let mut pivots = vec![first];
    let mut selected = vec![false; graph.node_size];
    selected[first] = true;

    let mut dist_to_pivot = vec![usize::MAX; graph.node_size];
    graph.update_min_distances_from_pivot(first, &mut dist_to_pivot);

    while pivots.len() < target {
        let max_dist = (0..graph.node_size)
            .filter(|&i| !selected[i])
            .map(|i| dist_to_pivot[i])
            .max()
            .unwrap_or(0);

        let candidates: Vec<usize> = (0..graph.node_size)
            .filter(|&i| !selected[i] && dist_to_pivot[i] == max_dist)
            .collect();

        if candidates.is_empty() {
            break;
        }

        let next = candidates[rng.random_range(0..candidates.len())];
        selected[next] = true;
        pivots.push(next);
        graph.update_min_distances_from_pivot(next, &mut dist_to_pivot);
    }

    Ok(pivots)
}

pub fn positive_weight_range(pairs: &[WeightedPair]) -> Result<(f32, f32)> {
    let mut wmin = f32::INFINITY;
    let mut wmax = 0.0_f32;

    for pair in pairs {
        if pair.weight > 0.0 {
            wmin = wmin.min(pair.weight);
            wmax = wmax.max(pair.weight);
        }
    }

    if wmin.is_finite() && wmax > 0.0 {
        Ok((wmin, wmax))
    } else {
        Err(anyhow!("no positive pair weights"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::compute_spectral_embedding;

    #[test]
    fn sampling_excludes_self_and_adjacent_pairs() {
        let graph = Graph::from_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let embedding = compute_spectral_embedding(&graph, 2, 2).unwrap();
        let pairs = sample_pivot_pairs(&graph, &embedding, 3, 3).unwrap();

        for pair in pairs {
            let u = pair.pair.u as usize;
            let v = pair.pair.v as usize;
            assert_ne!(u, v);
            assert!(!graph.has_edge(u, v));
            assert!(pair.weight > 0.0);
        }
    }
}
