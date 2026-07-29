mod algorithm;
mod graph;

use anyhow::{bail, Context, Result};
use chrono::Local;
use experiment_common::{
    current_binary, dataset_name, environment_metadata, git_metadata, measure_auto_f64,
    positions_sha256_f64, sha256_file, Artifacts, CommonExperimentArgs, ExperimentRecord,
    GraphMetrics, Method, MethodStats, OutputFormat, Parameters, RecordIdentity, RunMode,
    TimingBreakdown,
};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
struct Config {
    common: CommonExperimentArgs,
    input: PathBuf,
    iterations: usize,
    epsilon: f64,
    seed: u64,
    center: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            common: CommonExperimentArgs::human("../output"),
            input: PathBuf::from("../data/USpowerGrid.mtx"),
            iterations: 15,
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

    fn from_iter<I: IntoIterator<Item = String>>(args: I) -> Result<Self> {
        let mut config = Self::default();
        let mut args = args.into_iter().peekable();
        let mut positional = false;
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                "--run-id" => config.common.run_id = next(&mut args, "--run-id")?,
                "--input" => {
                    config.input = next(&mut args, "--input")?.into();
                    positional = true;
                }
                "--iterations" => config.iterations = next(&mut args, "--iterations")?.parse()?,
                "--epsilon" => config.epsilon = next(&mut args, "--epsilon")?.parse()?,
                "--seed" => config.seed = next(&mut args, "--seed")?.parse()?,
                "--output-format" => {
                    config.common.output_format = next(&mut args, "--output-format")?.parse()?
                }
                "--output-dir" => {
                    config.common.output_dir = next(&mut args, "--output-dir")?.into()
                }
                "--run-mode" => {
                    config.common.run_mode = next(&mut args, "--run-mode")?.parse::<RunMode>()?
                }
                "--verbose" => config.common.verbose = true,
                "--no-center" => config.center = false,
                value if !value.starts_with('-') && !positional => {
                    config.input = value.into();
                    positional = true;
                }
                _ => bail!("不明な引数です: {argument}"),
            }
        }
        if config.iterations == 0 {
            bail!("--iterations は1以上である必要があります");
        }
        if !(config.epsilon.is_finite() && config.epsilon > 0.0) {
            bail!("--epsilon は正の有限値である必要があります");
        }
        config.common.validate()?;
        Ok(config)
    }

    fn log(&self, message: impl AsRef<str>) {
        match self.common.output_format {
            OutputFormat::Json => self.common.log(message),
            OutputFormat::Human => println!("{}", message.as_ref()),
        }
    }
}

fn next<I: Iterator<Item = String>>(
    args: &mut std::iter::Peekable<I>,
    flag: &str,
) -> Result<String> {
    args.next()
        .with_context(|| format!("{flag} には値が必要です"))
}

fn print_help() {
    println!(
        "Usage: baseline-sgd-non-gpu [INPUT] [--run-id ID] [--input PATH] \
         [--iterations N] [--epsilon F] [--seed N] [--output-format human|json] \
         [--output-dir PATH] [--run-mode benchmark|diagnostic] [--verbose] [--no-center]"
    );
}

fn main() -> Result<()> {
    env_logger::init();
    let config = Config::from_args()?;

    let input_started = Instant::now();
    let graph = graph::Graph::from_mtx(&config.input)
        .with_context(|| format!("グラフを読み込めません: {}", config.input.display()))?;
    let input_time = input_started.elapsed();
    config.log(format!(
        "Graph: nodes={}, edges={}, seed={}",
        graph.node_size, graph.edge_size, config.seed
    ));

    let preprocess_started = Instant::now();
    let params = graph.prepare_sgd_params(
        config.iterations,
        config.epsilon,
        config.center,
        config.seed,
    );
    let initial_positions = params.positions.clone();
    let constraint_count = params.pairs.len();
    let center = params.center;
    let common_preprocess_time = preprocess_started.elapsed();

    let iteration_started = Instant::now();
    let mut final_positions = algorithm::execute_sgd(
        params,
        config.seed,
        config.common.verbose,
        config.common.output_format,
    );
    let iteration_time = iteration_started.elapsed();

    let postprocess_started = Instant::now();
    if center {
        algorithm::center_inplace(&mut final_positions);
    }
    let postprocess_time = postprocess_started.elapsed();

    let edges: Vec<_> = graph
        .edge_src
        .iter()
        .copied()
        .zip(graph.edge_dst.iter().copied())
        .collect();
    let stress = measure_auto_f64(&final_positions, &edges);

    create_dir_all(&config.common.output_dir)?;
    let prefix = output_prefix(&config);
    let initial_path = prefix.with_extension("initial.txt");
    let final_path = prefix.with_extension("final.txt");
    save_result(
        &initial_path,
        "Initial",
        &graph,
        &initial_positions,
        &config,
    )?;
    save_result(&final_path, "Processed", &graph, &final_positions, &config)?;

    let record = ExperimentRecord::success(
        RecordIdentity {
            run_id: config.common.run_id.clone(),
            run_mode: config.common.run_mode,
            method: Method::Sgd,
            binary: current_binary(),
            dataset: dataset_name(&config.input),
            input_path: config.input.display().to_string(),
            input_sha256: sha256_file(&config.input)?,
            seed: config.seed,
            initial_positions_sha256: positions_sha256_f64(&initial_positions),
            preprocess_sha256: None,
        },
        git_metadata(Path::new(env!("CARGO_MANIFEST_DIR")))?,
        GraphMetrics {
            nodes: graph.node_size,
            edges: graph.edge_size,
            constraints: Some(constraint_count),
        },
        Parameters {
            pivots: None,
            iterations: config.iterations,
            epsilon: config.epsilon,
        },
        environment_metadata(None, None),
        TimingBreakdown {
            input_time_ms: Some(ms(input_time)),
            common_preprocess_time_ms: Some(ms(common_preprocess_time)),
            method_setup_time_ms: None,
            runtime_init_time_ms: None,
            upload_time_ms: None,
            iteration_time_ms: Some(ms(iteration_time)),
            gpu_device_time_ms: None,
            readback_time_ms: None,
            postprocess_time_ms: Some(ms(postprocess_time)),
            ..TimingBreakdown::default()
        },
        stress,
        MethodStats::default(),
        Artifacts {
            final_positions_path: Some(final_path.display().to_string()),
            vertex_map_path: None,
        },
    )?;

    match config.common.output_format {
        OutputFormat::Json => println!("{}", record.to_json_line()?),
        OutputFormat::Human => {
            println!(
                "Iteration: {:.3} ms, Algorithm: {:.3} ms, CLI total: {:.3} ms",
                record.timings.iteration_time_ms.unwrap(),
                record.timings.algorithm_time_cold_ms.unwrap(),
                record.timings.cli_total_time_cold_ms.unwrap()
            );
            println!("Final result saved to {}", final_path.display());
        }
    }
    Ok(())
}

fn output_prefix(config: &Config) -> PathBuf {
    let suffix = if config.common.run_id == "manual" {
        format!("manual-{}", Local::now().format("%Y%m%d_%H%M%S_%3f"))
    } else {
        config.common.run_id.clone()
    };
    config
        .common
        .output_dir
        .join(format!("sgd-{}-{suffix}", dataset_name(&config.input)))
}

fn save_result(
    path: &Path,
    stage: &str,
    graph: &graph::Graph,
    positions: &[[f64; 2]],
    config: &Config,
) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "# Rust CPU Result (baseline-sgd-non-gpu) - {stage}")?;
    writeln!(file, "# Dataset: {}", config.input.display())?;
    writeln!(file, "# Node count: {}", graph.node_size)?;
    writeln!(file, "# Edge count: {}", graph.edge_size)?;
    writeln!(file, "# Iterations: {}", config.iterations)?;
    writeln!(file, "# Epsilon: {}", config.epsilon)?;
    writeln!(file, "# Seed: {}", config.seed)?;
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

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_arguments_are_parsed() {
        let config = Config::from_iter(
            [
                "--run-id",
                "test-1",
                "--input",
                "graph.mtx",
                "--iterations",
                "3",
                "--epsilon",
                "0.2",
                "--seed",
                "9",
                "--output-format",
                "json",
                "--output-dir",
                "out",
                "--run-mode",
                "diagnostic",
                "--verbose",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.common.run_id, "test-1");
        assert_eq!(config.input, PathBuf::from("graph.mtx"));
        assert_eq!(config.iterations, 3);
        assert_eq!(config.seed, 9);
        assert_eq!(config.common.output_format, OutputFormat::Json);
        assert_eq!(config.common.run_mode, RunMode::Diagnostic);
        assert!(config.common.verbose);
    }
}
