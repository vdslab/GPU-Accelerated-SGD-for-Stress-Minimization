use anyhow::{bail, Context, Result};
use chrono::Local;
use sparse_sgd_gpu::calc_learning_rate;
use sparse_sgd_gpu::embedding::{compute_spectral_embedding, DEFAULT_EMBED_DIM};
use sparse_sgd_gpu::gpu::{GpuContext, GpuSgdParams};
use sparse_sgd_gpu::graph::Graph;
use sparse_sgd_gpu::sampling::{positive_weight_range, sample_pivot_pairs};
use sparse_sgd_gpu::schedule::{classify_and_pack, validate_dispatch_ranges_conflict_free};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
struct Config {
    input: PathBuf,
    iterations: usize,
    h: usize,
    embed_dim: usize,
    seed: u64,
    epsilon: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input: PathBuf::from("../data/bcspwr10.mtx"),
            iterations: 15,
            h: 50,
            embed_dim: DEFAULT_EMBED_DIM,
            seed: 0,
            epsilon: 0.1,
        }
    }
}

impl Config {
    fn from_args() -> Result<Self> {
        let mut config = Self::default();
        let mut args = std::env::args().skip(1).peekable();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                "--input" => {
                    config.input = PathBuf::from(next_value(&mut args, "--input")?);
                }
                "--iterations" => {
                    config.iterations = next_value(&mut args, "--iterations")?.parse()?;
                }
                "--h" | "--pivots" => {
                    config.h = next_value(&mut args, "--h")?.parse()?;
                }
                "--embed-dim" => {
                    config.embed_dim = next_value(&mut args, "--embed-dim")?.parse()?;
                }
                "--seed" => {
                    config.seed = next_value(&mut args, "--seed")?.parse()?;
                }
                "--epsilon" => {
                    config.epsilon = next_value(&mut args, "--epsilon")?.parse()?;
                }
                value if !value.starts_with('-') => {
                    config.input = PathBuf::from(value);
                }
                _ => bail!("unknown argument: {arg}"),
            }
        }

        if config.iterations == 0 {
            bail!("--iterations must be greater than 0");
        }
        if config.embed_dim == 0 {
            bail!("--embed-dim must be greater than 0");
        }
        if config.h == 0 {
            bail!("--h must be greater than 0");
        }
        if config.epsilon <= 0.0 {
            bail!("--epsilon must be greater than 0");
        }

        Ok(config)
    }
}

fn next_value<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .with_context(|| format!("{flag} requires a value"))
}

fn print_help() {
    println!(
        "Usage: sparse-sgd-gpu [--input PATH] [--iterations N] [--h N] [--embed-dim N] [--seed N] [--epsilon F]"
    );
}

fn main() -> Result<()> {
    env_logger::init();
    let config = Config::from_args()?;

    println!("config: {:?}", config);
    let total_start = Instant::now();

    let graph = Graph::from_mtx(&config.input)
        .with_context(|| format!("failed to load MTX file: {}", config.input.display()))?;
    graph.ensure_connected()?;

    println!(
        "graph loaded: nodes={}, edges={}",
        graph.node_size, graph.edge_size
    );

    let embedding_start = Instant::now();
    let embedding = compute_spectral_embedding(&graph, config.embed_dim, config.seed)
        .context("failed to compute spectral embedding")?;
    println!(
        "embedding complete: dim={}, {:.3}s",
        embedding.dim,
        embedding_start.elapsed().as_secs_f64()
    );

    let sampling_start = Instant::now();
    let sampled_pairs = sample_pivot_pairs(&graph, &embedding, config.h, config.seed)
        .context("failed to sample SparseSGD pivot pairs")?;
    let (wmin, wmax) = positive_weight_range(&sampled_pairs)?;
    println!(
        "sampling complete: pairs={}, wmin={:.6}, wmax={:.6}, {:.3}s",
        sampled_pairs.len(),
        wmin,
        wmax,
        sampling_start.elapsed().as_secs_f64()
    );

    let schedule_start = Instant::now();
    let schedule = classify_and_pack(&sampled_pairs, graph.node_size);
    validate_dispatch_ranges_conflict_free(&schedule).map_err(anyhow::Error::msg)?;
    println!(
        "schedule complete: ranges={}, packed_pairs={}, {:.3}s",
        schedule.dispatch_ranges.len(),
        schedule.packed_pairs.len(),
        schedule_start.elapsed().as_secs_f64()
    );

    let etas = calc_learning_rate(config.iterations, wmin, wmax, config.epsilon);
    let positions = embedding.initial_positions(graph.node_size);

    let ctx = GpuContext::new()?;
    let sgd_start = Instant::now();
    let (initial_positions, final_positions) = ctx.execute_sgd(GpuSgdParams {
        positions,
        embedding: embedding.values,
        embed_dim: embedding.dim,
        schedule,
        etas,
        seed: config.seed,
    })?;
    let sgd_duration = sgd_start.elapsed();

    println!(
        "GPU SparseSGD complete: {:.3}s, total {:.3}s",
        sgd_duration.as_secs_f64(),
        total_start.elapsed().as_secs_f64()
    );

    create_dir_all("../output")?;
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let data_name = config
        .input
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();

    let path_init = format!("../output/sparse-sgd-gpu-{}-{}-0.txt", data_name, timestamp);
    save_result(
        &path_init,
        "sparse-sgd-gpu - Initial (Spectral)",
        &graph,
        &initial_positions,
    )?;
    println!("initial positions saved: {}", path_init);

    let path_final = format!("../output/sparse-sgd-gpu-{}-{}-1.txt", data_name, timestamp);
    save_result(
        &path_final,
        "sparse-sgd-gpu - Processed",
        &graph,
        &final_positions,
    )?;
    println!("final positions saved: {}", path_final);

    Ok(())
}

fn save_result(path: &str, label: &str, graph: &Graph, positions: &[[f32; 2]]) -> Result<()> {
    let mut file = File::create(Path::new(path))?;

    writeln!(file, "# Rust GPU Result ({label})")?;
    writeln!(
        file,
        "# Timestamp: {}",
        Local::now().format("%Y-%m-%d %H:%M:%S")
    )?;
    writeln!(file, "# Node count: {}", graph.node_size)?;
    writeln!(file, "# Edge count: {}", graph.edge_size)?;
    writeln!(file)?;
    writeln!(file, "# Edges (source target)")?;
    for i in 0..graph.edge_size {
        writeln!(file, "{} {}", graph.edge_src[i], graph.edge_dst[i])?;
    }
    writeln!(file)?;
    writeln!(file, "# Positions (x y)")?;
    for pos in positions {
        writeln!(file, "{} {}", pos[0], pos[1])?;
    }

    Ok(())
}
