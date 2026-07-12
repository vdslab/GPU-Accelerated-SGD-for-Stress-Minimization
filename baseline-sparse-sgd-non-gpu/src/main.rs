mod algorithm;
mod graph;

use anyhow::{Context, Result};
use chrono::Local;
use graph::Graph;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug)]
struct Config {
    input: PathBuf,
    iterations: usize,
    pivot_count: usize,
    epsilon: f64,
    center: bool,
}

impl Config {
    fn from_args() -> Self {
        Self {
            input: std::env::args()
                .nth(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("../data/USpowerGrid.mtx")),
            iterations: 15,
            pivot_count: 200,
            epsilon: 0.1,
            center: true,
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let config = Config::from_args();
    let graph = Graph::from_mtx(&config.input)
        .with_context(|| format!("グラフを読み込めません: {}", config.input.display()))?;
    println!(
        "Graph: nodes={}, edges={}",
        graph.node_size, graph.edge_size
    );

    let mut rng = rand::rng();
    let sgd_params = graph.prepare_sgd_params(
        config.iterations,
        config.epsilon,
        config.pivot_count,
        config.center,
        &mut rng,
    )?;
    let initial_positions = sgd_params.positions.clone();
    let pivots = sgd_params.pivots.clone();
    let constraint_count = sgd_params.pairs.len();

    println!(
        "Sparse constraints: {}, pivots: {}",
        constraint_count,
        pivots.len()
    );
    let start = Instant::now();
    let result = algorithm::execute_sgd(sgd_params, &mut rng);
    println!("Time taken: {:?}", start.elapsed());

    create_dir_all("../output")?;
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let data_name = config
        .input
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let initial_path =
        format!("../output/baseline-sparse-sgd-non-gpu-{data_name}-{timestamp}-0.txt");
    save_result(
        Path::new(&initial_path),
        "Initial (Randomized)",
        &graph,
        &initial_positions,
        &config,
        &pivots,
        constraint_count,
    )?;
    println!("Initial result saved to {initial_path}");

    let processed_path =
        format!("../output/baseline-sparse-sgd-non-gpu-{data_name}-{timestamp}-1.txt");
    save_result(
        Path::new(&processed_path),
        "Processed",
        &graph,
        &result,
        &config,
        &pivots,
        constraint_count,
    )?;
    println!("Processed result saved to {processed_path}");

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn save_result(
    path: &Path,
    stage: &str,
    graph: &Graph,
    positions: &[[f64; 2]],
    config: &Config,
    pivots: &[usize],
    constraint_count: usize,
) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "# Rust CPU Result (baseline-sparse-sgd-non-gpu) - {stage}"
    )?;
    writeln!(
        file,
        "# Timestamp: {}",
        Local::now().format("%Y-%m-%d %H:%M:%S")
    )?;
    writeln!(file, "# Dataset: {}", config.input.display())?;
    writeln!(file, "# Node count: {}", graph.node_size)?;
    writeln!(file, "# Edge count: {}", graph.edge_size)?;
    writeln!(file, "# Iterations: {}", config.iterations)?;
    writeln!(file, "# Epsilon: {}", config.epsilon)?;
    writeln!(file, "# Centered: {}", config.center)?;
    writeln!(file, "# Pivot count: {}", pivots.len())?;
    writeln!(
        file,
        "# Pivots: {}",
        pivots
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    )?;
    writeln!(file, "# Constraint count: {constraint_count}")?;
    writeln!(file)?;
    writeln!(file, "# Edges (source target)")?;
    for (&source, &target) in graph.edge_src.iter().zip(&graph.edge_dst) {
        writeln!(file, "{source} {target}")?;
    }
    writeln!(file)?;
    writeln!(file, "# Positions (x y)")?;
    for position in positions {
        writeln!(file, "{} {}", position[0], position[1])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_file_contains_metadata_edges_and_positions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("result.txt");
        let graph = Graph::try_from_edges(2, &[(0, 1)]).unwrap();
        let config = Config {
            input: PathBuf::from("graph.mtx"),
            iterations: 3,
            pivot_count: 1,
            epsilon: 0.1,
            center: true,
        };
        save_result(
            &path,
            "Processed",
            &graph,
            &[[0.0, 0.0], [1.0, 1.0]],
            &config,
            &[0],
            1,
        )
        .unwrap();

        let contents = std::fs::read_to_string(path).unwrap();
        assert!(contents.contains("# Pivot count: 1"));
        assert!(contents.contains("# Edges (source target)\n0 1"));
        assert!(contents.contains("# Positions (x y)\n0 0\n1 1"));
    }
}
