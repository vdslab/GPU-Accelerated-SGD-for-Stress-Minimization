mod algorithm;
mod graph;

use anyhow::{bail, Context, Result};
use chrono::Local;
use graph::Graph;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const PIVOT_SELECTION: &str = "max-min-random-sp-distance-proportional";
const WEIGHT_MODEL: &str = "ortmann-region-directed-weight";

#[derive(Debug, Clone, PartialEq)]
struct Config {
    input: PathBuf,
    iterations: usize,
    pivot_count: usize,
    epsilon: f64,
    seed: u64,
    center: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input: PathBuf::from("../data/bcspwr10.mtx"),
            iterations: 15,
            pivot_count: 200,
            epsilon: 0.1,
            seed: 0,
            center: true,
        }
    }
}

impl Config {
    fn from_args() -> Result<Self> {
        Self::from_iter(std::env::args().skip(1))
    }

    fn from_iter<I>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = String>,
    {
        let mut config = Self::default();
        let mut args = args.into_iter().peekable();
        let mut positional_input_seen = false;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                "--input" => {
                    config.input = PathBuf::from(next_value(&mut args, "--input")?);
                    positional_input_seen = true;
                }
                "--pivots" => {
                    config.pivot_count = next_value(&mut args, "--pivots")?.parse()?;
                }
                "--iterations" => {
                    config.iterations = next_value(&mut args, "--iterations")?.parse()?;
                }
                "--epsilon" => {
                    config.epsilon = next_value(&mut args, "--epsilon")?.parse()?;
                }
                "--seed" => {
                    config.seed = next_value(&mut args, "--seed")?.parse()?;
                }
                "--no-center" => config.center = false,
                value if !value.starts_with('-') && !positional_input_seen => {
                    config.input = PathBuf::from(value);
                    positional_input_seen = true;
                }
                _ => bail!("不明な引数です: {arg}"),
            }
        }

        if config.iterations == 0 {
            bail!("--iterations は1以上である必要があります");
        }
        if config.pivot_count == 0 {
            bail!("--pivots は1以上である必要があります");
        }
        if !(config.epsilon.is_finite() && config.epsilon > 0.0) {
            bail!("--epsilon は正の有限値である必要があります");
        }
        Ok(config)
    }
}

fn next_value<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .with_context(|| format!("{flag} には値が必要です"))
}

fn print_help() {
    println!(
        "Usage: baseline-sparse-sgd-non-gpu [INPUT] [--input PATH] [--pivots N] [--iterations N] [--epsilon F] [--seed N] [--no-center]"
    );
}

fn main() -> Result<()> {
    env_logger::init();
    let config = Config::from_args()?;
    let graph = Graph::from_mtx(&config.input)
        .with_context(|| format!("グラフを読み込めません: {}", config.input.display()))?;
    graph.ensure_connected()?;
    println!(
        "Graph: nodes={}, edges={}, seed={}",
        graph.node_size, graph.edge_size, config.seed
    );

    let mut rng = StdRng::seed_from_u64(config.seed);
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
    let timestamp = Local::now().format("%Y%m%d_%H%M%S_%3f");
    let data_name = config
        .input
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let prefix = format!(
        "../output/baseline-sparse-sgd-non-gpu-{data_name}-seed{}-{timestamp}",
        config.seed
    );
    let initial_path = format!("{prefix}-0.txt");
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

    let processed_path = format!("{prefix}-1.txt");
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
    writeln!(file, "# Seed: {}", config.seed)?;
    writeln!(file, "# Centered: {}", config.center)?;
    writeln!(file, "# Pivot selection: {PIVOT_SELECTION}")?;
    writeln!(file, "# Weight model: {WEIGHT_MODEL}")?;
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
    fn command_line_parameters_are_parsed() {
        let config = Config::from_iter(
            [
                "--input",
                "graph.mtx",
                "--pivots",
                "12",
                "--iterations",
                "9",
                "--epsilon",
                "0.05",
                "--seed",
                "42",
                "--no-center",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(
            config,
            Config {
                input: PathBuf::from("graph.mtx"),
                iterations: 9,
                pivot_count: 12,
                epsilon: 0.05,
                seed: 42,
                center: false,
            }
        );
    }

    #[test]
    fn invalid_command_line_parameters_are_rejected() {
        assert!(Config::from_iter(["--pivots", "0"].into_iter().map(str::to_owned)).is_err());
        assert!(Config::from_iter(["--iterations", "0"].into_iter().map(str::to_owned)).is_err());
        assert!(Config::from_iter(["--epsilon", "0"].into_iter().map(str::to_owned)).is_err());
    }

    #[test]
    fn result_file_contains_reproduction_metadata_edges_and_positions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("result.txt");
        let graph = Graph::try_from_edges(2, &[(0, 1)]).unwrap();
        let config = Config {
            input: PathBuf::from("graph.mtx"),
            iterations: 3,
            pivot_count: 1,
            epsilon: 0.1,
            seed: 17,
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
        assert!(contents.contains("# Seed: 17"));
        assert!(contents.contains(&format!("# Pivot selection: {PIVOT_SELECTION}")));
        assert!(contents.contains(&format!("# Weight model: {WEIGHT_MODEL}")));
        assert!(contents.contains("# Pivot count: 1"));
        assert!(contents.contains("# Edges (source target)\n0 1"));
        assert!(contents.contains("# Positions (x y)\n0 0\n1 1"));
    }
}
