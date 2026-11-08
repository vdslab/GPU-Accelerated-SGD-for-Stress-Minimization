use anyhow::{ensure, Context, Result};
use experiment_common::stress::{evaluate_f64, StressMode, StressResult};
use std::io::{self, BufRead};
use std::time::Instant;

fn main() -> Result<()> {
    let precision = std::env::args().nth(1).context("expected f32 or f64")?;
    ensure!(precision == "f32" || precision == "f64", "invalid precision");
    let mut positions = Vec::new();
    let mut edges = Vec::new();
    let mut section = "";
    let mut expected_nodes: Option<usize> = None;
    let mut expected_edges: Option<usize> = None;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let s = line.trim();
        if let Some(n) = s.strip_prefix("# Node count:") { expected_nodes = Some(n.trim().parse()?); }
        if let Some(n) = s.strip_prefix("# Edge count:") { expected_edges = Some(n.trim().parse()?); }
        if s.starts_with("# Edges") { section = "edges"; continue; }
        if s.starts_with("# Positions") { section = "positions"; continue; }
        if s.is_empty() || s.starts_with('#') { continue; }
        let pair: Vec<_> = s.split_whitespace().collect();
        ensure!(pair.len() == 2, "invalid data row");
        if section == "edges" {
            edges.push((pair[0].parse::<usize>()?, pair[1].parse::<usize>()?));
        } else if section == "positions" {
            let parse = |s: &str| -> Result<f64> {
                Ok(if precision == "f32" { f64::from(s.parse::<f32>()?) } else { s.parse::<f64>()? })
            };
            positions.push([parse(pair[0])?, parse(pair[1])?]);
        }
    }
    ensure!(!positions.is_empty(), "no positions");
    if let Some(n) = expected_nodes { ensure!(positions.len() == n, "node count mismatch"); }
    if let Some(n) = expected_edges { ensure!(edges.len() == n, "edge count mismatch"); }
    ensure!(positions.iter().flatten().all(|x| x.is_finite()), "nonfinite coordinates");
    ensure!(edges.iter().all(|&(u,v)| u < positions.len() && v < positions.len()), "invalid endpoint");
    let started = Instant::now();
    let StressResult::Exact(value) = evaluate_f64(StressMode::Exact, &positions, &edges, 0, 0) else { unreachable!() };
    println!("{}", serde_json::json!({"stress_kind":"exact","stress_value":value,"nodes":positions.len(),"edges":edges.len(),"coordinate_precision":precision,"evaluation_ms":started.elapsed().as_secs_f64()*1000.0}));
    Ok(())
}
