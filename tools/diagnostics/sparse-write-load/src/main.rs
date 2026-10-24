//! Offline CPU preprocessing only: no SGD, Metal or wgpu execution.
#[path = "../../../../implementations/sparse/baseline-sparse-sgd-non-gpu/src/graph.rs"]
mod graph;

use anyhow::{ensure, Context, Result};
use experiment_common::{positions_sha256_f64, sha256_file, FingerprintBuilder};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{collections::{BTreeMap, HashSet}, fs, path::PathBuf};

#[derive(Deserialize)]
struct Request {
    input_path: PathBuf,
    seed: u64,
    pivots: usize,
    iterations: usize,
    epsilon: f64,
    expected_input_sha256: String,
    expected_graph_source_sha256: String,
    expected_preprocess_sha256: String,
    expected_initial_positions_sha256: String,
    expected_nodes: usize,
    expected_constraints: usize,
}

#[derive(Serialize)]
struct Distribution {
    vertices: usize,
    total_per_iteration: u64,
    mean: Option<f64>,
    max: Option<u64>,
    coefficient_of_variation: Option<f64>,
    p50: Option<u64>,
    p90: Option<u64>,
    p99: Option<u64>,
    top_1pct_vertex_count: usize,
    top_1pct_total: u64,
    top_1pct_share: Option<f64>,
    share_of_all_vertex_total: Option<f64>,
    histogram_count_to_vertices: BTreeMap<u64, usize>,
}

fn distribution(counts: &[u64], all_total: u64) -> Distribution {
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let total: u64 = sorted.iter().sum();
    let mean = (n > 0).then(|| total as f64 / n as f64);
    let cv = mean.filter(|&v| v > 0.0).map(|mean| {
        let variance = sorted.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        variance.sqrt() / mean
    });
    let quantile = |q: f64| {
        (n > 0).then(|| sorted[((q * n as f64).ceil() as usize).saturating_sub(1).min(n - 1)])
    };
    let top_n = n.div_ceil(100);
    let top_total: u64 = sorted.iter().rev().take(top_n).sum();
    let mut histogram = BTreeMap::new();
    for &value in &sorted { *histogram.entry(value).or_default() += 1; }
    Distribution {
        vertices: n, total_per_iteration: total, mean, max: sorted.last().copied(),
        coefficient_of_variation: cv, p50: quantile(0.5), p90: quantile(0.9), p99: quantile(0.99),
        top_1pct_vertex_count: top_n, top_1pct_total: top_total,
        top_1pct_share: (total > 0).then(|| top_total as f64 / total as f64),
        share_of_all_vertex_total: (all_total > 0).then(|| total as f64 / all_total as f64),
        histogram_count_to_vertices: histogram,
    }
}

fn grouped(counts: &[u64], pivots: &HashSet<usize>) -> serde_json::Value {
    let total = counts.iter().sum();
    let pivot_counts: Vec<_> = counts.iter().enumerate().filter_map(|(i, &v)| pivots.contains(&i).then_some(v)).collect();
    let nonpivot_counts: Vec<_> = counts.iter().enumerate().filter_map(|(i, &v)| (!pivots.contains(&i)).then_some(v)).collect();
    serde_json::json!({
        "all": distribution(counts, total),
        "pivot": distribution(&pivot_counts, total),
        "nonpivot": distribution(&nonpivot_counts, total),
    })
}

// Identical domain and field order to each measured Sparse main.rs.
fn preprocess_sha256(params: &graph::SgdParams) -> String {
    let mut fingerprint = FingerprintBuilder::new("sparse-preprocess-v1");
    fingerprint.usize(params.pivots.len());
    for &pivot in &params.pivots { fingerprint.usize(pivot); }
    fingerprint.usize(params.etas.len());
    for &eta in &params.etas { fingerprint.f64(eta); }
    fingerprint.usize(params.pairs.len());
    for pair in &params.pairs {
        fingerprint.usize(pair.u).usize(pair.v).f64(pair.dij).f64(pair.weight_u).f64(pair.weight_v);
    }
    fingerprint.finish()
}

fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    ensure!(arguments.len() == 4 && arguments[0] == "--request" && arguments[2] == "--output",
            "Usage: sparse-write-load --request REQUEST.json --output OUTPUT.json");
    let request: Request = serde_json::from_slice(&fs::read(&arguments[1])?)?;
    let output = PathBuf::from(&arguments[3]);
    ensure!(!output.exists(), "Refusing to overwrite existing diagnostic output");
    ensure!(sha256_file(&request.input_path)? == request.expected_input_sha256, "Input SHA mismatch");
    let graph_source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../implementations/sparse/baseline-sparse-sgd-non-gpu/src/graph.rs");
    let graph_source_sha256 = sha256_file(&graph_source)?;
    ensure!(graph_source_sha256 == request.expected_graph_source_sha256, "Shared graph source SHA mismatch");
    let graph = graph::Graph::from_mtx(&request.input_path)?.largest_connected_component()?;
    let mut rng = StdRng::seed_from_u64(request.seed);
    let params = graph.prepare_sgd_params(request.iterations, request.epsilon, request.pivots, true, &mut rng)?;
    let preprocess_sha256 = preprocess_sha256(&params);
    let initial_positions_sha256 = positions_sha256_f64(&params.positions);
    ensure!(preprocess_sha256 == request.expected_preprocess_sha256, "Preprocess SHA mismatch");
    ensure!(initial_positions_sha256 == request.expected_initial_positions_sha256, "Initial-position SHA mismatch");
    ensure!(graph.node_size == request.expected_nodes && params.pairs.len() == request.expected_constraints,
            "Prepared graph/constraint count mismatch");
    let mut updates = vec![0_u64; graph.node_size];
    let mut lock_targets = vec![0_u64; graph.node_size];
    for pair in &params.pairs {
        if pair.weight_u > 0.0 { updates[pair.u] += 1; }
        if pair.weight_v > 0.0 { updates[pair.v] += 1; }
        lock_targets[pair.u] += 1;
        lock_targets[pair.v] += 1;
    }
    let pivot_set: HashSet<_> = params.pivots.iter().copied().collect();
    let result = serde_json::json!({
        "status": "success", "diagnostic_only": true, "gpu_executed": false, "sgd_executed": false,
        "seed": request.seed, "pivots": params.pivots.len(), "iterations": request.iterations,
        "epsilon": request.epsilon, "nodes": graph.node_size, "constraints": params.pairs.len(),
        "input_sha256": request.expected_input_sha256, "graph_source_sha256": graph_source_sha256,
        "preprocess_sha256": preprocess_sha256, "initial_positions_sha256": initial_positions_sha256,
        "hashes_match_measured_record": true, "rng_version": "rand 0.9.5 StdRng",
        "positive_weight_endpoint_updates": grouped(&updates, &pivot_set),
        "atomic_lock_target_touches": grouped(&lock_targets, &pivot_set),
        "definitions": {
            "positive_weight_endpoint_updates": "Per iteration, add one at u/v only when the corresponding f64 weight is positive. Semantic eligibility for an update, not a hardware memory-store count.",
            "atomic_lock_target_touches": "Per iteration, add one at both endpoints of every constraint, including zero-weight endpoints. Intended pair-level lock targets, not actual attempts/retries/waits/collisions.",
            "top_1pct": "Largest ceil(group vertex count / 100) loads; share uses that group's total.",
            "cv": "Population standard deviation divided by mean; null for empty or all-zero groups.",
            "whole_run": "Per-iteration counts can be multiplied by iterations; no SGD trajectory or timing is simulated.",
        },
    });
    if let Some(parent) = output.parent() { fs::create_dir_all(parent)?; }
    let temporary = output.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&result)?).context("write diagnostic output")?;
    fs::rename(temporary, output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planned_load_statistics_include_zeros_and_use_population_cv() {
        let statistics = distribution(&[0, 1, 2, 7], 10);
        assert_eq!(statistics.vertices, 4);
        assert_eq!(statistics.total_per_iteration, 10);
        assert_eq!(statistics.mean, Some(2.5));
        assert_eq!(statistics.max, Some(7));
        assert_eq!(statistics.p50, Some(1));
        assert_eq!(statistics.p99, Some(7));
        assert_eq!(statistics.top_1pct_vertex_count, 1);
        assert_eq!(statistics.top_1pct_share, Some(0.7));
        assert!((statistics.coefficient_of_variation.unwrap() - 7.25_f64.sqrt() / 2.5).abs() < 1e-12);
        assert_eq!(statistics.histogram_count_to_vertices.get(&0), Some(&1));
    }

    #[test]
    fn empty_and_all_zero_groups_have_no_defined_cv() {
        assert!(distribution(&[], 0).coefficient_of_variation.is_none());
        assert!(distribution(&[0, 0], 0).coefficient_of_variation.is_none());
        assert!(distribution(&[], 0).mean.is_none());
        assert!(distribution(&[0, 0], 0).top_1pct_share.is_none());
    }

    #[test]
    fn pivot_and_nonpivot_groups_keep_their_share_of_total() {
        let groups = grouped(&[2, 3, 5], &[1_usize].into_iter().collect());
        assert_eq!(groups["pivot"]["total_per_iteration"], 3);
        assert_eq!(groups["pivot"]["share_of_all_vertex_total"], 0.3);
        assert_eq!(groups["nonpivot"]["vertices"], 2);
    }
}
