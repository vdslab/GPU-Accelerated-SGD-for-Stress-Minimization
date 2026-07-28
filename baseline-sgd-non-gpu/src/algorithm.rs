//! SGD implementation (CPU).
//!
//! This module is intended to be used as a namespace (no stateful struct).

use crate::graph;
use experiment_common::seed::{rng_for_stream, FULL_UPDATE_STREAM};
use experiment_common::OutputFormat;
use rand::seq::SliceRandom;
use rand::Rng;

fn norm2(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

pub fn center_inplace(positions: &mut [[f64; 2]]) {
    if positions.is_empty() {
        return;
    }
    let n = positions.len() as f64;
    let mean_x = positions.iter().map(|p| p[0]).sum::<f64>() / n;
    let mean_y = positions.iter().map(|p| p[1]).sum::<f64>() / n;
    for p in positions {
        p[0] -= mean_x;
        p[1] -= mean_y;
    }
}

/// Execute SGD for stress minimization.
///
/// This follows the same update rule as `sgd_stress_nongpu.py`:
/// - shuffle constraints each iteration
/// - for each pair (u,v):
///   - `r = ((||xv-xu|| - dij)/2) * (diff / ||diff||)`
///   - `mu = min(wij * eta, 1)`
///   - `xu += mu * r`, `xv -= mu * r`
pub fn execute_sgd(
    sgd_params: graph::SgdParams,
    seed: u64,
    verbose: bool,
    output_format: OutputFormat,
) -> Vec<[f64; 2]> {
    let mut rng = rng_for_stream(seed, FULL_UPDATE_STREAM);
    let mut positions = sgd_params.positions.clone();
    let mut pairs = sgd_params.pairs.clone();

    let tiny = 1e-12_f64;

    for (iteration, &eta) in sgd_params.etas.iter().enumerate() {
        pairs.shuffle(&mut rng);

        for pair in &pairs {
            let u = pair.u;
            let v = pair.v;
            let dij = pair.dij;
            let wij = pair.wij;

            let mut diff = sub(positions[v], positions[u]);
            let mut nrm = norm2(diff);

            if nrm < tiny {
                // avoid 0-division; pick a tiny random direction
                let angle = rng.random::<f64>() * std::f64::consts::TAU;
                diff = [angle.cos() * 1e-6, angle.sin() * 1e-6];
                nrm = norm2(diff);
            }

            // NOTE: i から 勾配方向に ずれ*学習率*(1/2) ずつ移動
            let r = [
                ((nrm - dij) / 2.0) * (diff[0] / nrm),
                ((nrm - dij) / 2.0) * (diff[1] / nrm),
            ];
            let mu = (wij * eta).min(1.0);
            positions[u][0] += mu * r[0];
            positions[u][1] += mu * r[1];
            positions[v][0] -= mu * r[0];
            positions[v][1] -= mu * r[1];
        }

        if verbose {
            match output_format {
                OutputFormat::Json => eprintln!("Iteration: {}", iteration + 1),
                OutputFormat::Human => println!("Iteration: {}", iteration + 1),
            }
        }
    }

    positions
}
