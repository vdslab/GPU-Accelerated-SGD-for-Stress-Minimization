use anyhow::{bail, Context, Result};
use chrono::Local;
use rand::{rngs::StdRng, SeedableRng};
use sparse_sgd_gpu::{gpu::GpuContext, graph::Graph, schedule::build_schedule};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const PIVOT_SELECTION: &str = "max-min-random-sp-distance-proportional";
const WEIGHT_MODEL: &str = "ortmann-region-directed-weight";

#[derive(Debug, Clone, PartialEq)]
struct Config {
    input: PathBuf,
    output_dir: PathBuf,
    iterations: usize,
    pivot_count: usize,
    epsilon: f64,
    seed: u64,
    center: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input: PathBuf::from("../data/bcsstk29.mtx"),
            output_dir: PathBuf::from("../output"),
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

    fn from_iter<I: IntoIterator<Item = String>>(args: I) -> Result<Self> {
        let mut config = Self::default();
        let mut args = args.into_iter().peekable();
        let mut positional = false;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                "--input" => {
                    config.input = next(&mut args, "--input")?.into();
                    positional = true;
                }
                "--output-dir" => config.output_dir = next(&mut args, "--output-dir")?.into(),
                "--pivots" => config.pivot_count = next(&mut args, "--pivots")?.parse()?,
                "--iterations" => config.iterations = next(&mut args, "--iterations")?.parse()?,
                "--epsilon" => config.epsilon = next(&mut args, "--epsilon")?.parse()?,
                "--seed" => config.seed = next(&mut args, "--seed")?.parse()?,
                "--no-center" => config.center = false,
                value if !value.starts_with('-') && !positional => {
                    config.input = value.into();
                    positional = true;
                }
                _ => bail!("不明な引数です: {arg}"),
            }
        }
        if config.iterations == 0 || config.pivot_count == 0 {
            bail!("--iterations と --pivots は1以上である必要があります");
        }
        if !(config.epsilon.is_finite() && config.epsilon > 0.0) {
            bail!("--epsilon は正の有限値である必要があります");
        }
        Ok(config)
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
    println!("Usage: sparse-sgd-gpu [INPUT] [--input PATH] [--output-dir PATH] [--pivots N] [--iterations N] [--epsilon F] [--seed N] [--no-center]");
}

fn main() -> Result<()> {
    env_logger::init();
    let total_start = Instant::now();
    let config = Config::from_args()?;
    let graph = Graph::from_mtx(&config.input)
        .with_context(|| format!("グラフを読み込めません: {}", config.input.display()))?;
    graph.ensure_connected()?;
    println!(
        "Graph: nodes={}, edges={}, seed={}",
        graph.node_size, graph.edge_size, config.seed
    );

    let preprocessing_start = Instant::now();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let params = graph.prepare_sgd_params(
        config.iterations,
        config.epsilon,
        config.pivot_count,
        config.center,
        &mut rng,
    )?;
    let preprocessing_time = preprocessing_start.elapsed();
    let scheduling_start = Instant::now();
    let schedule = build_schedule(&graph, &params, config.seed)?;
    let scheduling_time = scheduling_start.elapsed();
    let pivots = params.pivots.clone();
    let constraint_count = params.pairs.len();
    println!(
        "Schedule: one-sided={}, two-sided={}, base_rounds={}, spill_rounds={}, rounds={}, dispatches/iteration={}",
        schedule.one_sided_count,
        schedule.two_sided.len(),
        schedule.base_rounds,
        schedule.spill_rounds,
        schedule.round_count(),
        schedule.round_count() + 1,
    );

    let context = GpuContext::new()?;
    println!("GPU: {}", context.adapter_name);
    let run = context.execute(params, &schedule, config.seed)?;
    println!(
        "Timing: preprocessing={:?}, scheduling={:?}, upload={:?}, compute={:?} ({:?}/iteration), readback={:?}, total={:?}",
        preprocessing_time,
        scheduling_time,
        run.upload_time,
        run.compute_time,
        run.compute_time / config.iterations as u32,
        run.readback_time,
        total_start.elapsed()
    );

    create_dir_all(&config.output_dir)?;
    let timestamp = Local::now().format("%Y%m%d_%H%M%S_%3f");
    let data_name = config
        .input
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let prefix = config.output_dir.join(format!(
        "sparse-sgd-gpu-{data_name}-seed{}-{timestamp}",
        config.seed
    ));
    let initial_path = prefix.with_file_name(format!(
        "{}-0.txt",
        prefix.file_name().unwrap().to_string_lossy()
    ));
    save_result(
        &initial_path,
        "Initial (Randomized)",
        &graph,
        &run.initial_positions,
        &config,
        &pivots,
        constraint_count,
        &schedule,
        &context.adapter_name,
        preprocessing_time,
        scheduling_time,
        run.upload_time,
        run.compute_time,
        run.readback_time,
    )?;
    let processed_path = prefix.with_file_name(format!(
        "{}-1.txt",
        prefix.file_name().unwrap().to_string_lossy()
    ));
    save_result(
        &processed_path,
        "Processed",
        &graph,
        &run.positions,
        &config,
        &pivots,
        constraint_count,
        &schedule,
        &context.adapter_name,
        preprocessing_time,
        scheduling_time,
        run.upload_time,
        run.compute_time,
        run.readback_time,
    )?;
    println!("Initial result saved to {}", initial_path.display());
    println!("Processed result saved to {}", processed_path.display());
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
    schedule: &sparse_sgd_gpu::schedule::Schedule,
    adapter: &str,
    preprocessing: Duration,
    scheduling: Duration,
    upload: Duration,
    compute: Duration,
    readback: Duration,
) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "# Rust GPU Result (sparse-sgd-gpu) - {stage}")?;
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
    writeln!(file, "# GPU adapter: {adapter}")?;
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
    writeln!(
        file,
        "# One-sided constraint count: {}",
        schedule.one_sided_count
    )?;
    writeln!(
        file,
        "# Two-sided constraint count: {}",
        schedule.two_sided.len()
    )?;
    writeln!(file, "# Base rounds: {}", schedule.base_rounds)?;
    writeln!(file, "# Spill rounds: {}", schedule.spill_rounds)?;
    writeln!(file, "# Round count: {}", schedule.round_count())?;
    writeln!(
        file,
        "# Dispatches per iteration: {}",
        schedule.round_count() + 1
    )?;
    writeln!(file, "# Max graph degree: {}", schedule.max_graph_degree)?;
    writeln!(
        file,
        "# Preprocessing ms: {:.3}",
        preprocessing.as_secs_f64() * 1000.0
    )?;
    writeln!(
        file,
        "# Scheduling ms: {:.3}",
        scheduling.as_secs_f64() * 1000.0
    )?;
    writeln!(file, "# Upload ms: {:.3}", upload.as_secs_f64() * 1000.0)?;
    writeln!(file, "# Compute ms: {:.3}", compute.as_secs_f64() * 1000.0)?;
    writeln!(
        file,
        "# Readback ms: {:.3}",
        readback.as_secs_f64() * 1000.0
    )?;
    writeln!(
        file,
        "# Compute ms per iteration: {:.3}",
        compute.as_secs_f64() * 1000.0 / config.iterations as f64
    )?;
    writeln!(file, "\n# Edges (source target)")?;
    for (&u, &v) in graph.edge_src.iter().zip(&graph.edge_dst) {
        writeln!(file, "{u} {v}")?;
    }
    writeln!(file, "\n# Positions (x y)")?;
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
                "graph.mtx",
                "--pivots",
                "16",
                "--iterations",
                "3",
                "--epsilon",
                "0.2",
                "--seed",
                "9",
                "--no-center",
                "--output-dir",
                "out",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.input, PathBuf::from("graph.mtx"));
        assert_eq!(config.pivot_count, 16);
        assert_eq!(config.iterations, 3);
        assert_eq!(config.epsilon, 0.2);
        assert_eq!(config.seed, 9);
        assert!(!config.center);
        assert_eq!(config.output_dir, PathBuf::from("out"));
    }
}
