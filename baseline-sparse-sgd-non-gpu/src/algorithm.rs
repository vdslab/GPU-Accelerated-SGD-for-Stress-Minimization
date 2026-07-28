//! CPU 上で動作する論文準拠 Sparse SGD の座標更新。

use crate::graph::{center_inplace, EdgeInfo, SgdParams};
use rand::seq::SliceRandom;
use rand::Rng;

const TINY_NORM: f64 = 1e-12;

fn norm2(vector: [f64; 2]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1]).sqrt()
}

pub fn apply_constraint<R: Rng + ?Sized>(
    positions: &mut [[f64; 2]],
    pair: EdgeInfo,
    eta: f64,
    rng: &mut R,
) {
    let mut difference = [
        positions[pair.v][0] - positions[pair.u][0],
        positions[pair.v][1] - positions[pair.u][1],
    ];
    let mut norm = norm2(difference);

    if norm < TINY_NORM {
        let angle = rng.random::<f64>() * std::f64::consts::TAU;
        difference = [angle.cos() * 1e-6, angle.sin() * 1e-6];
        norm = norm2(difference);
    }

    let scale = (norm - pair.dij) / (2.0 * norm);
    let displacement = [scale * difference[0], scale * difference[1]];
    let mu_u = (pair.weight_u * eta).min(1.0);
    let mu_v = (pair.weight_v * eta).min(1.0);

    positions[pair.u][0] += mu_u * displacement[0];
    positions[pair.u][1] += mu_u * displacement[1];
    positions[pair.v][0] -= mu_v * displacement[0];
    positions[pair.v][1] -= mu_v * displacement[1];
}

pub fn execute_sgd_iterations<R: Rng + ?Sized>(
    sgd_params: SgdParams,
    rng: &mut R,
) -> Vec<[f64; 2]> {
    let mut positions = sgd_params.positions;
    let mut pairs = sgd_params.pairs;

    for eta in sgd_params.etas {
        pairs.shuffle(rng);
        for &pair in &pairs {
            apply_constraint(&mut positions, pair, eta, rng);
        }
    }
    positions
}

pub fn execute_sgd<R: Rng + ?Sized>(sgd_params: SgdParams, rng: &mut R) -> Vec<[f64; 2]> {
    let center = sgd_params.center;
    let mut positions = execute_sgd_iterations(sgd_params, rng);
    if center {
        center_inplace(&mut positions);
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn pair(distance: f64, weight_u: f64, weight_v: f64) -> EdgeInfo {
        EdgeInfo {
            u: 0,
            v: 1,
            dij: distance,
            weight_u,
            weight_v,
        }
    }

    #[test]
    fn symmetric_update_moves_endpoints_equally_in_opposite_directions() {
        let mut positions = vec![[0.0, 0.0], [4.0, 0.0]];
        let mut rng = StdRng::seed_from_u64(1);
        apply_constraint(&mut positions, pair(2.0, 1.0, 1.0), 1.0, &mut rng);
        assert_eq!(positions, vec![[1.0, 0.0], [3.0, 0.0]]);
    }

    #[test]
    fn directional_weights_move_endpoints_by_different_amounts() {
        let mut positions = vec![[0.0, 0.0], [4.0, 0.0]];
        let mut rng = StdRng::seed_from_u64(2);
        apply_constraint(&mut positions, pair(2.0, 1.0, 0.5), 1.0, &mut rng);
        assert_eq!(positions, vec![[1.0, 0.0], [3.5, 0.0]]);
    }

    #[test]
    fn zero_directional_weight_keeps_that_endpoint_fixed() {
        let mut positions = vec![[0.0, 0.0], [4.0, 0.0]];
        let mut rng = StdRng::seed_from_u64(3);
        apply_constraint(&mut positions, pair(2.0, 1.0, 0.0), 1.0, &mut rng);
        assert_eq!(positions, vec![[1.0, 0.0], [4.0, 0.0]]);
    }

    #[test]
    fn each_directional_mu_is_clamped_to_one() {
        let mut positions = vec![[0.0, 0.0], [4.0, 0.0]];
        let mut rng = StdRng::seed_from_u64(4);
        apply_constraint(&mut positions, pair(2.0, 10.0, 10.0), 10.0, &mut rng);
        assert_eq!(positions, vec![[1.0, 0.0], [3.0, 0.0]]);
    }

    #[test]
    fn coincident_positions_remain_finite() {
        let mut positions = vec![[0.0, 0.0], [0.0, 0.0]];
        let mut rng = StdRng::seed_from_u64(5);
        apply_constraint(&mut positions, pair(1.0, 1.0, 0.0), 1.0, &mut rng);
        assert!(positions
            .iter()
            .flatten()
            .all(|coordinate| coordinate.is_finite()));
        assert_ne!(positions[0], positions[1]);
    }

    #[test]
    fn complete_run_centers_the_result() {
        let params = SgdParams {
            etas: vec![1.0, 0.1],
            positions: vec![[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]],
            pairs: vec![
                EdgeInfo {
                    u: 0,
                    v: 1,
                    dij: 1.0,
                    weight_u: 1.0,
                    weight_v: 1.0,
                },
                EdgeInfo {
                    u: 1,
                    v: 2,
                    dij: 1.0,
                    weight_u: 1.0,
                    weight_v: 0.5,
                },
            ],
            pivots: vec![0],
            center: true,
        };
        let mut rng = StdRng::seed_from_u64(6);
        let result = execute_sgd(params, &mut rng);
        let mean_x = result.iter().map(|position| position[0]).sum::<f64>() / 3.0;
        let mean_y = result.iter().map(|position| position[1]).sum::<f64>() / 3.0;
        assert!(mean_x.abs() < 1e-12);
        assert!(mean_y.abs() < 1e-12);
        assert!(result
            .iter()
            .flatten()
            .all(|coordinate| coordinate.is_finite()));
    }

    #[test]
    fn complete_pipeline_is_reproducible_for_same_seed() {
        let graph = Graph::try_from_edges(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]).unwrap();
        let run = |seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let params = graph.prepare_sgd_params(5, 0.1, 3, true, &mut rng).unwrap();
            let pivots = params.pivots.clone();
            let initial = params.positions.clone();
            let final_positions = execute_sgd(params, &mut rng);
            (pivots, initial, final_positions)
        };
        assert_eq!(run(23), run(23));
    }
}
