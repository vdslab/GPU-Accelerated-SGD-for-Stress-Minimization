use crate::center_positions;
use crate::graph::Graph;
use anyhow::{anyhow, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const DEFAULT_EMBED_DIM: usize = 10;
pub const MIN_DELTA: f32 = 1.0e-6;

#[derive(Debug, Clone)]
pub struct Embedding {
    pub dim: usize,
    pub values: Vec<f32>,
}

impl Embedding {
    pub fn node(&self, node: usize) -> &[f32] {
        let start = node * self.dim;
        &self.values[start..start + self.dim]
    }

    pub fn distance(&self, u: usize, v: usize) -> f32 {
        if self.dim == 0 {
            return MIN_DELTA;
        }

        let a = self.node(u);
        let b = self.node(v);
        let sum = a
            .iter()
            .zip(b)
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>();

        sum.sqrt().max(MIN_DELTA)
    }

    pub fn initial_positions(&self, n: usize) -> Vec<[f32; 2]> {
        let mut positions = Vec::with_capacity(n);
        for i in 0..n {
            let node = self.node(i);
            let x = node.first().copied().unwrap_or(0.0);
            let y = node.get(1).copied().unwrap_or(0.0);
            positions.push([x, y]);
        }
        center_positions(&mut positions);
        positions
    }
}

pub fn compute_spectral_embedding(
    graph: &Graph,
    requested_dim: usize,
    seed: u64,
) -> Result<Embedding> {
    let n = graph.node_size;
    if n == 0 {
        return Err(anyhow!("cannot embed an empty graph"));
    }

    let dim = requested_dim.min(n.saturating_sub(1));
    if dim == 0 {
        return Ok(Embedding {
            dim,
            values: Vec::new(),
        });
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(dim);
    let mut eigenvalues: Vec<f64> = Vec::with_capacity(dim);

    for _ in 0..dim {
        let mut x: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        orthogonalize(&mut x, &eigenvectors);
        normalize_or_randomize(&mut x, &eigenvectors, &mut rng);

        for _ in 0..50 {
            let mut solved = pcg_solve_shifted_laplacian(graph, &x, 1.0e-3, 1.0e-6, 400);
            orthogonalize(&mut solved, &eigenvectors);
            normalize_or_randomize(&mut solved, &eigenvectors, &mut rng);
            x = solved;
        }

        let lambda = rayleigh_quotient(graph, &x).max(1.0e-8);
        eigenvalues.push(lambda);
        eigenvectors.push(x);
    }

    let mut values = vec![0.0_f32; n * dim];
    for (k, (lambda, eigenvector)) in eigenvalues.iter().zip(&eigenvectors).enumerate() {
        let scale = 1.0 / lambda.sqrt();
        for i in 0..n {
            values[i * dim + k] = (eigenvector[i] * scale) as f32;
        }
    }

    Ok(Embedding { dim, values })
}

fn laplacian_matvec(graph: &Graph, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; graph.node_size];
    for i in 0..graph.node_size {
        let mut value = graph.degree(i) as f64 * x[i];
        for &j in graph.neighbors(i) {
            value -= x[j];
        }
        y[i] = value;
    }
    y
}

fn shifted_laplacian_matvec(graph: &Graph, x: &[f64], shift: f64) -> Vec<f64> {
    let mut y = laplacian_matvec(graph, x);
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += shift * xi;
    }
    y
}

fn pcg_solve_shifted_laplacian(
    graph: &Graph,
    b: &[f64],
    shift: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = graph.node_size;
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut z = apply_jacobi_preconditioner(graph, &r, shift);
    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);
    let b_norm = dot(b, b).sqrt().max(1.0e-12);

    for _ in 0..max_iter {
        let ap = shifted_laplacian_matvec(graph, &p, shift);
        let denom = dot(&p, &ap);
        if denom.abs() < 1.0e-20 {
            break;
        }

        let alpha = rz_old / denom;
        axpy(alpha, &p, &mut x);
        axpy(-alpha, &ap, &mut r);

        if dot(&r, &r).sqrt() / b_norm < tol {
            break;
        }

        z = apply_jacobi_preconditioner(graph, &r, shift);
        let rz_new = dot(&r, &z);
        if rz_old.abs() < 1.0e-20 {
            break;
        }

        let beta = rz_new / rz_old;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }

    x
}

fn apply_jacobi_preconditioner(graph: &Graph, r: &[f64], shift: f64) -> Vec<f64> {
    (0..graph.node_size)
        .map(|i| r[i] / (graph.degree(i) as f64 + shift))
        .collect()
}

fn rayleigh_quotient(graph: &Graph, x: &[f64]) -> f64 {
    let lx = laplacian_matvec(graph, x);
    dot(x, &lx) / dot(x, x).max(1.0e-20)
}

fn orthogonalize(x: &mut [f64], basis: &[Vec<f64>]) {
    if x.is_empty() {
        return;
    }

    let mean = x.iter().sum::<f64>() / x.len() as f64;
    for xi in x.iter_mut() {
        *xi -= mean;
    }

    for b in basis {
        let projection = dot(x, b);
        for (xi, bi) in x.iter_mut().zip(b) {
            *xi -= projection * bi;
        }
    }
}

fn normalize_or_randomize(x: &mut [f64], basis: &[Vec<f64>], rng: &mut StdRng) {
    let mut norm = dot(x, x).sqrt();
    if norm < 1.0e-12 {
        for xi in x.iter_mut() {
            *xi = rng.random_range(-1.0..1.0);
        }
        orthogonalize(x, basis);
        norm = dot(x, x).sqrt();
    }

    if norm < 1.0e-12 {
        return;
    }

    for xi in x {
        *xi /= norm;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_distances_are_positive_and_symmetric() {
        let graph = Graph::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let embedding = compute_spectral_embedding(&graph, 2, 1).unwrap();

        assert_eq!(embedding.dim, 2);
        let d01 = embedding.distance(0, 1);
        let d10 = embedding.distance(1, 0);
        assert!(d01 >= MIN_DELTA);
        assert!((d01 - d10).abs() < 1.0e-6);
    }
}
