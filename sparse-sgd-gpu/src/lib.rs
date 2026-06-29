pub mod embedding;
pub mod gpu;
pub mod graph;
pub mod sampling;
pub mod schedule;

pub fn calc_learning_rate(tmax: usize, wmin: f32, wmax: f32, eps: f32) -> Vec<f32> {
    if tmax == 0 {
        return Vec::new();
    }
    if tmax == 1 {
        return vec![1.0 / wmin];
    }

    let eta_max = 1.0 / wmin;
    let eta_min = eps / wmax;
    let lambda = (eta_max / eta_min).ln() / (tmax - 1) as f32;

    (0..tmax)
        .map(|t| eta_max * (-lambda * t as f32).exp())
        .collect()
}

pub fn center_positions(positions: &mut [[f32; 2]]) {
    if positions.is_empty() {
        return;
    }

    let n = positions.len() as f32;
    let mean_x = positions.iter().map(|p| p[0]).sum::<f32>() / n;
    let mean_y = positions.iter().map(|p| p[1]).sum::<f32>() / n;

    for p in positions {
        p[0] -= mean_x;
        p[1] -= mean_y;
    }
}
