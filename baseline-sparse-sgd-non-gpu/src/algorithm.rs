//! CPU 上で動作する Sparse SGD の座標更新。

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
    let mu = (pair.wij * eta).min(1.0);

    positions[pair.u][0] += mu * displacement[0];
    positions[pair.u][1] += mu * displacement[1];
    positions[pair.v][0] -= mu * displacement[0];
    positions[pair.v][1] -= mu * displacement[1];
}

pub fn execute_sgd<R: Rng + ?Sized>(sgd_params: SgdParams, rng: &mut R) -> Vec<[f64; 2]> {
    let mut positions = sgd_params.positions;
    let mut pairs = sgd_params.pairs;

    for (iteration, eta) in sgd_params.etas.into_iter().enumerate() {
        pairs.shuffle(rng);
        for &pair in &pairs {
            apply_constraint(&mut positions, pair, eta, rng);
        }
        println!("Iteration: {}", iteration + 1);
    }

    if sgd_params.center {
        center_inplace(&mut positions);
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn pair(distance: f64, weight: f64) -> EdgeInfo {
        EdgeInfo {
            u: 0,
            v: 1,
            dij: distance,
            wij: weight,
        }
    }

    #[test]
    fn update_moves_endpoints_equally_in_opposite_directions() {
        let mut positions = vec![[0.0, 0.0], [4.0, 0.0]];
        let before_center = [
            (positions[0][0] + positions[1][0]) / 2.0,
            (positions[0][1] + positions[1][1]) / 2.0,
        ];
        let mut rng = StdRng::seed_from_u64(1);
        apply_constraint(&mut positions, pair(2.0, 1.0), 1.0, &mut rng);

        assert_eq!(positions, vec![[1.0, 0.0], [3.0, 0.0]]);
        assert_eq!(
            before_center,
            [
                (positions[0][0] + positions[1][0]) / 2.0,
                (positions[0][1] + positions[1][1]) / 2.0,
            ]
        );
    }

    #[test]
    fn mu_is_clamped_to_one() {
        let mut positions = vec![[0.0, 0.0], [4.0, 0.0]];
        let mut rng = StdRng::seed_from_u64(2);
        apply_constraint(&mut positions, pair(2.0, 10.0), 10.0, &mut rng);
        assert_eq!(positions, vec![[1.0, 0.0], [3.0, 0.0]]);
    }

    #[test]
    fn coincident_positions_remain_finite() {
        let mut positions = vec![[0.0, 0.0], [0.0, 0.0]];
        let mut rng = StdRng::seed_from_u64(3);
        apply_constraint(&mut positions, pair(1.0, 1.0), 1.0, &mut rng);
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
                    wij: 1.0,
                },
                EdgeInfo {
                    u: 1,
                    v: 2,
                    dij: 1.0,
                    wij: 1.0,
                },
            ],
            pivots: vec![0],
            center: true,
        };
        let mut rng = StdRng::seed_from_u64(4);
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
}
