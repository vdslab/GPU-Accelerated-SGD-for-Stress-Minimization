mod reader;
mod renderer;
mod stress;

use anyhow::{Context, Result};
use image::{ImageBuffer, Rgba};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const DEFAULT_WIDTH: u32 = 2048;
const DEFAULT_HEIGHT: u32 = 2048;

#[derive(Debug)]
struct Config {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    width: u32,
    height: u32,
    stress_mode: stress::StressMode,
    stress_samples: usize,
    stress_seed: u64,
    node_radius: Option<f32>,
    max_edges: Option<usize>,
}

#[derive(Default)]
struct RunTimings {
    read: Duration,
    stress: Duration,
    gpu_init: Duration,
    gpu_prepare: Duration,
    gpu_execute_readback: Duration,
    png: Duration,
}

fn main() -> Result<()> {
    let total_start = Instant::now();
    let mut config = parse_args(std::env::args_os().skip(1))?;
    let mut timings = RunTimings::default();
    let input_path = match config.input.take() {
        Some(path) => resolve_input(&path.to_string_lossy()),
        None => prompt_input()?,
    };
    let output_path = config
        .output
        .take()
        .unwrap_or_else(|| default_output(&input_path));

    println!("Reading: {}", input_path.display());
    let started = Instant::now();
    let graph = reader::read_result_file(&input_path)?;
    timings.read = started.elapsed();
    println!(
        "  nodes={}, edges={} ({:.1}ms)",
        graph.node_count,
        graph.edges.len(),
        ms(timings.read)
    );

    let resolved_mode = config.stress_mode.resolve(graph.node_count);
    if config.stress_mode == stress::StressMode::Exact
        && graph.node_count > stress::AUTO_EXACT_MAX_N
    {
        eprintln!("Warning: {}頂点の厳密ストレスは非常に高コストです。中止する場合はCtrl-C、通常は --stress sampled を使用してください。", graph.node_count);
    }
    let started = Instant::now();
    let stress_result = stress::evaluate(
        config.stress_mode,
        &graph.positions,
        &graph.edges,
        config.stress_samples,
        config.stress_seed,
    );
    timings.stress = started.elapsed();
    print_stress(&stress_result, timings.stress, resolved_mode);

    let node_radius = config.node_radius.unwrap_or_else(|| {
        renderer::auto_node_radius(graph.node_count, config.width, config.height)
    });
    let selected_edges = renderer::select_edges(&graph.edges, config.max_edges, config.stress_seed);
    if selected_edges.len() < graph.edges.len() {
        println!(
            "Edges: {}/{} (explicit --max-edges sampling, seed={})",
            selected_edges.len(),
            graph.edges.len(),
            config.stress_seed
        );
    } else {
        println!(
            "Edges: {}/{} (all)",
            selected_edges.len(),
            graph.edges.len()
        );
    }

    println!("Initialising GPU renderer ...");
    let started = Instant::now();
    let gpu = renderer::GpuRenderer::new()?;
    timings.gpu_init = started.elapsed();
    println!(
        "Rendering {}×{} px (node_r={:.2}px) ...",
        config.width, config.height, node_radius
    );
    let rendered = gpu.render(
        &graph.positions,
        &selected_edges,
        config.width,
        config.height,
        node_radius,
    )?;
    timings.gpu_prepare = rendered.prepare_time;
    timings.gpu_execute_readback = rendered.execute_readback_time;

    let started = Instant::now();
    let image =
        ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(config.width, config.height, rendered.pixels)
            .context("Image buffer size mismatch")?;
    image.save(&output_path)?;
    timings.png = started.elapsed();

    println!("Saved: {}", output_path.display());
    println!("Timing:");
    println!("  read:                 {:>8.1}ms", ms(timings.read));
    println!("  stress:               {:>8.1}ms", ms(timings.stress));
    println!("  GPU init:              {:>8.1}ms", ms(timings.gpu_init));
    println!(
        "  GPU prepare/upload:    {:>8.1}ms",
        ms(timings.gpu_prepare)
    );
    println!(
        "  GPU render/readback:   {:>8.1}ms",
        ms(timings.gpu_execute_readback)
    );
    println!("  PNG encode/save:       {:>8.1}ms", ms(timings.png));
    println!("Total: {:.1}ms", ms(total_start.elapsed()));
    Ok(())
}

fn parse_args<I>(args: I) -> Result<Config>
where
    I: IntoIterator<Item = OsString>,
{
    let args: Vec<String> = args
        .into_iter()
        .map(|s| s.to_string_lossy().into_owned())
        .collect();
    let mut config = Config {
        input: None,
        output: None,
        width: DEFAULT_WIDTH,
        height: DEFAULT_HEIGHT,
        stress_mode: stress::StressMode::Auto,
        stress_samples: stress::DEFAULT_SAMPLES,
        stress_seed: stress::DEFAULT_SEED,
        node_radius: None,
        max_edges: None,
    };
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let option = args[i].as_str();
        if option == "--help" || option == "-h" {
            print_help();
            std::process::exit(0);
        }
        if !option.starts_with('-') {
            positional.push(option.to_owned());
            i += 1;
            continue;
        }
        let value = args
            .get(i + 1)
            .with_context(|| format!("{option} に値が必要です"))?;
        match option {
            "--size" => {
                let (w, h) = value
                    .split_once('x')
                    .context("--size は WIDTHxHEIGHT 形式で指定してください")?;
                config.width = w.parse().context("--size の幅が不正です")?;
                config.height = h.parse().context("--size の高さが不正です")?;
                anyhow::ensure!(
                    config.width > 0 && config.height > 0,
                    "--size は0より大きくしてください"
                );
            }
            "--stress" => config.stress_mode = stress::StressMode::parse(value)?,
            "--stress-samples" => {
                config.stress_samples = value
                    .parse()
                    .context("--stress-samples は正整数で指定してください")?;
                anyhow::ensure!(
                    config.stress_samples > 0,
                    "--stress-samples は1以上にしてください"
                );
            }
            "--stress-seed" => {
                config.stress_seed = value
                    .parse()
                    .context("--stress-seed は整数で指定してください")?
            }
            "--node-radius" => {
                let radius: f32 = value
                    .parse()
                    .context("--node-radius は数値で指定してください")?;
                anyhow::ensure!(
                    radius.is_finite() && radius > 0.0,
                    "--node-radius は有限な正数にしてください"
                );
                config.node_radius = Some(radius);
            }
            "--max-edges" => {
                let max: usize = value
                    .parse()
                    .context("--max-edges は正整数で指定してください")?;
                anyhow::ensure!(max > 0, "--max-edges は1以上にしてください");
                config.max_edges = Some(max);
            }
            _ => anyhow::bail!("不明なオプションです: {option}"),
        }
        i += 2;
    }
    anyhow::ensure!(
        positional.len() <= 2,
        "位置引数は INPUT [OUTPUT] の2個までです"
    );
    config.input = positional.first().map(PathBuf::from);
    config.output = positional.get(1).map(PathBuf::from);
    Ok(config)
}

fn print_help() {
    println!("gpu-visualizer [INPUT] [OUTPUT] [options]\n\n  --size WxH\n  --stress auto|exact|sampled|off\n  --stress-samples K\n  --stress-seed SEED\n  --node-radius PX\n  --max-edges N");
}

fn print_stress(result: &stress::StressResult, elapsed: Duration, resolved: stress::StressMode) {
    println!("{}", stress_summary(result, elapsed));
    debug_assert_ne!(resolved, stress::StressMode::Auto);
}

fn stress_summary(result: &stress::StressResult, elapsed: Duration) -> String {
    match result {
        stress::StressResult::Exact(value) => {
            format!("Stress: {value:.2} (exact, {:.1}ms)", ms(elapsed))
        }
        stress::StressResult::Sampled {
            value,
            samples,
            seed,
        } => format!(
            "Approx stress: {value:.2} (samples={samples}, seed={seed}, {:.1}ms)",
            ms(elapsed)
        ),
        stress::StressResult::Off => {
            format!("Stress: skipped (--stress off, {:.1}ms)", ms(elapsed))
        }
    }
}

fn prompt_input() -> Result<PathBuf> {
    print!("ファイル名 (例: rr-gpu-bcsstm36-20260316_123139-1): ");
    std::io::stdout().flush()?;
    let mut value = String::new();
    std::io::stdin().read_line(&mut value)?;
    Ok(resolve_input(value.trim()))
}

fn resolve_input(value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.extension().is_some() || path.components().count() > 1 {
        path
    } else {
        PathBuf::from(format!("../output/{value}.txt"))
    }
}

fn default_output(input: &Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    input
        .parent()
        .unwrap_or(Path::new("."))
        .join(format!("{stem}.png"))
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(values: &[&str]) -> Result<Config> {
        parse_args(values.iter().map(OsString::from))
    }

    #[test]
    fn parses_all_options_and_positionals() {
        let c = parse(&[
            "in.txt",
            "out.png",
            "--stress",
            "sampled",
            "--stress-samples",
            "10",
            "--stress-seed",
            "3",
            "--node-radius",
            "1.5",
            "--max-edges",
            "20",
        ])
        .unwrap();
        assert_eq!(c.input.unwrap(), PathBuf::from("in.txt"));
        assert_eq!(c.output.unwrap(), PathBuf::from("out.png"));
        assert_eq!(c.stress_mode, stress::StressMode::Sampled);
        assert_eq!(c.stress_samples, 10);
        assert_eq!(c.node_radius, Some(1.5));
        assert_eq!(c.max_edges, Some(20));
    }

    #[test]
    fn rejects_invalid_values() {
        for args in [
            vec!["--stress", "bad"],
            vec!["--stress-samples", "0"],
            vec!["--node-radius", "0"],
            vec!["--node-radius", "NaN"],
            vec!["--max-edges", "0"],
        ] {
            assert!(parse(&args).is_err(), "accepted {args:?}");
        }
    }

    #[test]
    fn exact_and_sampled_labels_cannot_be_confused() {
        let exact = stress_summary(&stress::StressResult::Exact(12.0), Duration::ZERO);
        let sampled = stress_summary(
            &stress::StressResult::Sampled {
                value: 12.0,
                samples: 64,
                seed: 3,
            },
            Duration::ZERO,
        );
        assert!(exact.starts_with("Stress:"));
        assert!(sampled.starts_with("Approx stress:"));
        assert!(sampled.contains("samples=64, seed=3"));
    }
}
