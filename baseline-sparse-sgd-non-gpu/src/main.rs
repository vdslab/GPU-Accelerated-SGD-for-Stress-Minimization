mod algorithm;
mod graph;

use anyhow::{bail, Context, Result};
use chrono::Local;
use experiment_common::{
    current_binary, dataset_name, environment_metadata, git_metadata, measure_auto_f64,
    positions_sha256_f64, sha256_file, Artifacts, CommonExperimentArgs, ExperimentRecord,
    FingerprintBuilder, GraphMetrics, Method, MethodStats, OutputFormat, Parameters,
    RecordIdentity, RunMode, TimingBreakdown,
};
use graph::{Graph, SgdParams};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const PIVOT_SELECTION: &str = "max-min-random-sp-distance-proportional";
const WEIGHT_MODEL: &str = "ortmann-region-directed-weight";

#[derive(Debug, Clone, PartialEq)]
struct Config {
    common: CommonExperimentArgs,
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
            common: CommonExperimentArgs::human("../output"),
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
                "--run-id" => config.common.run_id = next_value(&mut args, "--run-id")?,
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
                "--output-format" => {
                    config.common.output_format =
                        next_value(&mut args, "--output-format")?.parse()?
                }
                "--output-dir" => {
                    config.common.output_dir = PathBuf::from(next_value(&mut args, "--output-dir")?)
                }
                "--run-mode" => {
                    config.common.run_mode =
                        next_value(&mut args, "--run-mode")?.parse::<RunMode>()?
                }
                "--verbose" => config.common.verbose = true,
                "--no-center" => config.center = false,
                "--largest-component" => {}
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

fn next_value<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .with_context(|| format!("{flag} には値が必要です"))
}

fn print_help() {
    println!(
        "Usage: baseline-sparse-sgd-non-gpu [INPUT] [--run-id ID] [--input PATH] \
         [--pivots N] [--iterations N] [--epsilon F] [--seed N] \
         [--output-format human|json] [--output-dir PATH] \
         [--run-mode benchmark|diagnostic] [--verbose] [--no-center]"
    );
}

fn main() -> Result<()> {
    env_logger::init();
    let config = Config::from_args()?;

    let input_started = Instant::now();
    let graph = Graph::from_mtx(&config.input)
        .with_context(|| format!("グラフを読み込めません: {}", config.input.display()))?
        .largest_connected_component()?;
    let input_time = input_started.elapsed();
    config.log(format!(
        "Graph: nodes={}, edges={}, components={}, seed={}",
        graph.node_size, graph.edge_size, graph.component_info.component_count, config.seed
    ));

    let preprocess_started = Instant::now();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let params = graph.prepare_sgd_params(
        config.iterations,
        config.epsilon,
        config.pivot_count,
        config.center,
        &mut rng,
    )?;
    let common_preprocess_time = preprocess_started.elapsed();
    let initial_positions = params.positions.clone();
    let pivots = params.pivots.clone();
    let constraint_count = params.pairs.len();
    let preprocess_sha256 = sparse_preprocess_sha256(&params);
    let center = params.center;

    let iteration_started = Instant::now();
    let mut final_positions = algorithm::execute_sgd_iterations(params, &mut rng);
    let iteration_time = iteration_started.elapsed();

    let postprocess_started = Instant::now();
    if center {
        graph::center_inplace(&mut final_positions);
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
    let vertex_map_path = prefix.with_extension("vertex-map.txt");
    let initial_path = prefix.with_extension("initial.txt");
    let final_path = prefix.with_extension("final.txt");
    save_vertex_map(&vertex_map_path, &graph)?;
    save_result(
        &initial_path,
        "Initial",
        &graph,
        &initial_positions,
        &config,
        &pivots,
        constraint_count,
        &vertex_map_path,
    )?;
    save_result(
        &final_path,
        "Processed",
        &graph,
        &final_positions,
        &config,
        &pivots,
        constraint_count,
        &vertex_map_path,
    )?;

    let record = ExperimentRecord::success(
        RecordIdentity {
            run_id: config.common.run_id.clone(),
            run_mode: config.common.run_mode,
            method: Method::SparseSgd,
            binary: current_binary(),
            dataset: dataset_name(&config.input),
            input_path: config.input.display().to_string(),
            input_sha256: sha256_file(&config.input)?,
            seed: config.seed,
            initial_positions_sha256: positions_sha256_f64(&initial_positions),
            preprocess_sha256: Some(preprocess_sha256),
        },
        git_metadata(Path::new(env!("CARGO_MANIFEST_DIR")))?,
        GraphMetrics {
            nodes: graph.node_size,
            edges: graph.edge_size,
            constraints: Some(constraint_count),
        },
        Parameters {
            pivots: Some(pivots.len()),
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
            vertex_map_path: Some(vertex_map_path.display().to_string()),
        },
    )?;

    match config.common.output_format {
        OutputFormat::Json => println!("{}", record.to_json_line()?),
        OutputFormat::Human => {
            println!(
                "Constraints={}, pivots={}, Iteration={:.3} ms, Algorithm={:.3} ms, CLI total={:.3} ms",
                constraint_count,
                pivots.len(),
                record.timings.iteration_time_ms.unwrap(),
                record.timings.algorithm_time_cold_ms.unwrap(),
                record.timings.cli_total_time_cold_ms.unwrap()
            );
            println!("Final result saved to {}", final_path.display());
        }
    }
    Ok(())
}

fn sparse_preprocess_sha256(params: &SgdParams) -> String {
    let mut fingerprint = FingerprintBuilder::new("sparse-preprocess-v1");
    fingerprint.usize(params.pivots.len());
    for &pivot in &params.pivots {
        fingerprint.usize(pivot);
    }
    fingerprint.usize(params.etas.len());
    for &eta in &params.etas {
        fingerprint.f64(eta);
    }
    fingerprint.usize(params.pairs.len());
    for pair in &params.pairs {
        fingerprint
            .usize(pair.u)
            .usize(pair.v)
            .f64(pair.dij)
            .f64(pair.weight_u)
            .f64(pair.weight_v);
    }
    fingerprint.finish()
}

fn output_prefix(config: &Config) -> PathBuf {
    let suffix = if config.common.run_id == "manual" {
        format!("manual-{}", Local::now().format("%Y%m%d_%H%M%S_%3f"))
    } else {
        config.common.run_id.clone()
    };
    config.common.output_dir.join(format!(
        "sparse-sgd-{}-{suffix}",
        dataset_name(&config.input)
    ))
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
    vertex_map_path: &Path,
) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "# Rust CPU Result (baseline-sparse-sgd-non-gpu) - {stage}"
    )?;
    writeln!(file, "# Dataset: {}", config.input.display())?;
    writeln!(file, "# Node count: {}", graph.node_size)?;
    writeln!(file, "# Edge count: {}", graph.edge_size)?;
    writeln!(
        file,
        "# Original node count: {}",
        graph.component_info.original_node_size
    )?;
    writeln!(
        file,
        "# Original edge count: {}",
        graph.component_info.original_edge_size
    )?;
    writeln!(
        file,
        "# Component count: {}",
        graph.component_info.component_count
    )?;
    writeln!(file, "# Vertex map file: {}", vertex_map_path.display())?;
    writeln!(file, "# Iterations: {}", config.iterations)?;
    writeln!(file, "# Epsilon: {}", config.epsilon)?;
    writeln!(file, "# Seed: {}", config.seed)?;
    writeln!(file, "# Pivot selection: {PIVOT_SELECTION}")?;
    writeln!(file, "# Weight model: {WEIGHT_MODEL}")?;
    writeln!(file, "# Pivot count: {}", pivots.len())?;
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

fn save_vertex_map(path: &Path, graph: &Graph) -> Result<()> {
    let mut file = File::create(path)?;
    for (local_id, &original_id) in graph.component_info.original_vertex_ids.iter().enumerate() {
        writeln!(file, "{local_id} {original_id}")?;
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
    fn command_line_parameters_are_parsed() {
        let config = Config::from_iter(
            [
                "--run-id",
                "sparse-test",
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
                "--output-format",
                "json",
                "--output-dir",
                "out",
                "--run-mode",
                "diagnostic",
                "--no-center",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.common.run_id, "sparse-test");
        assert_eq!(config.input, PathBuf::from("graph.mtx"));
        assert_eq!(config.iterations, 9);
        assert_eq!(config.pivot_count, 12);
        assert_eq!(config.epsilon, 0.05);
        assert_eq!(config.seed, 42);
        assert!(!config.center);
        assert_eq!(config.common.output_format, OutputFormat::Json);
        assert_eq!(config.common.run_mode, RunMode::Diagnostic);
    }

    #[test]
    fn invalid_command_line_parameters_are_rejected() {
        assert!(Config::from_iter(["--pivots", "0"].into_iter().map(str::to_owned)).is_err());
        assert!(Config::from_iter(["--iterations", "0"].into_iter().map(str::to_owned)).is_err());
        assert!(Config::from_iter(["--epsilon", "0"].into_iter().map(str::to_owned)).is_err());
    }

    #[test]
    fn historical_largest_component_flag_remains_accepted() {
        assert!(Config::from_iter(["--largest-component"].into_iter().map(str::to_owned)).is_ok());
    }
}
