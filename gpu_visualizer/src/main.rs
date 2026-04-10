mod reader;
mod renderer;
mod stress;

use anyhow::Result;
use image::{ImageBuffer, Rgba};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let input_path: PathBuf = if args.len() > 1 {
        resolve_input(&args[1])
    } else {
        print!("ファイル名 (例: rr-gpu-bcsstm36-20260316_123139-1): ");
        std::io::stdout().flush()?;
        let mut s = String::new();
        std::io::stdin().read_line(&mut s)?;
        resolve_input(s.trim())
    };

    let output_path: PathBuf = if args.len() > 2 {
        PathBuf::from(&args[2])
    } else {
        let stem   = input_path.file_stem().unwrap_or_default().to_string_lossy();
        let parent = input_path.parent().unwrap_or(Path::new("."));
        parent.join(format!("{stem}.png"))
    };

    let (width, height)    = parse_size(&args).unwrap_or((2048, 2048));
    let node_radius_px     = parse_node_radius(&args)
        .unwrap_or(renderer::DEFAULT_NODE_RADIUS_PX);

    // ── Load ─────────────────────────────────────────────────────────────────
    println!("Reading: {}", input_path.display());
    let t0    = Instant::now();
    let graph = reader::read_result_file(&input_path)?;
    println!(
        "  nodes={}, edges={} ({:.1}ms)",
        graph.node_count,
        graph.edges.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );

    if graph.positions.is_empty() {
        anyhow::bail!("No positions found in file");
    }

    // ── Stress ───────────────────────────────────────────────────────────────
    println!("Computing stress ...");
    let ts = Instant::now();
    let stress_str = match stress::calc_stress(&graph.positions, &graph.edges) {
        stress::StressResult::Value(v) => {
            println!("  stress = {v:.2}  ({:.1}ms)", ts.elapsed().as_secs_f64() * 1000.0);
            format!("{v:.2}")
        }
        stress::StressResult::TooLarge => {
            println!("  stress: skipped (n={} > 8000)", graph.node_count);
            "N/A".to_string()
        }
        stress::StressResult::Disconnected => {
            println!("  stress: graph is disconnected");
            "disconnected".to_string()
        }
    };

    // ── GPU render ────────────────────────────────────────────────────────────
    println!("Initialising GPU renderer ...");
    let renderer = renderer::GpuRenderer::new()?;

    println!("Rendering {}×{} px (node_r={}px) ...", width, height, node_radius_px);
    let t1     = Instant::now();
    let pixels = renderer.render(
        &graph.positions,
        &graph.edges,
        width,
        height,
        node_radius_px,
    )?;
    println!("  GPU render: {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);

    // ── Save PNG ──────────────────────────────────────────────────────────────
    let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow::anyhow!("Image buffer size mismatch"))?;
    img.save(&output_path)?;

    println!("Saved:  {}", output_path.display());
    println!("Stress: {stress_str}");
    println!("Total:  {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Expand a bare basename to `../output/<name>.txt`.
fn resolve_input(s: &str) -> PathBuf {
    let p = PathBuf::from(s);
    if p.extension().is_some() || p.components().count() > 1 {
        p
    } else {
        PathBuf::from(format!("../output/{s}.txt"))
    }
}

/// Parse `--size WxH`
fn parse_size(args: &[String]) -> Option<(u32, u32)> {
    let idx = args.iter().position(|a| a == "--size")?;
    let val = args.get(idx + 1)?;
    let (w, h) = val.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

/// Parse `--node-radius N`
fn parse_node_radius(args: &[String]) -> Option<f32> {
    let idx = args.iter().position(|a| a == "--node-radius")?;
    args.get(idx + 1)?.parse().ok()
}
