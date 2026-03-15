mod reader;
mod renderer;

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
        print!("ファイル名 (例: rr-gpu-20260316_042332-1): ");
        std::io::stdout().flush()?;
        let mut s = String::new();
        std::io::stdin().read_line(&mut s)?;
        resolve_input(s.trim())
    };

    let output_path: PathBuf = if args.len() > 2 {
        PathBuf::from(&args[2])
    } else {
        let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();
        let parent = input_path.parent().unwrap_or(Path::new("."));
        parent.join(format!("{stem}.png"))
    };

    // Optional resolution override: --size WxH or default 2048x2048
    let (width, height) = parse_size(&args).unwrap_or((2048, 2048));

    // ---- Load graph data ----
    println!("Reading: {}", input_path.display());
    let t0 = Instant::now();
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

    // ---- GPU render ----
    println!("Initialising GPU renderer ...");
    let renderer = renderer::GpuRenderer::new()?;

    println!("Rendering {}×{} px ...", width, height);
    let t1 = Instant::now();
    let pixels = renderer.render(&graph.positions, &graph.edges, width, height)?;
    println!("  GPU render: {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);

    // ---- Save PNG ----
    let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow::anyhow!("Image buffer size mismatch"))?;
    img.save(&output_path)?;
    println!("Saved: {}", output_path.display());
    println!("Total: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}

/// ベース名だけ入力した場合に ../output/<name>.txt へ補完する。
/// すでにパスや拡張子がついていればそのまま使う。
fn resolve_input(s: &str) -> PathBuf {
    let p = PathBuf::from(s);
    if p.extension().is_some() || p.components().count() > 1 {
        p
    } else {
        PathBuf::from(format!("../output/{s}.txt"))
    }
}

/// Parse --size WxH from args (e.g. --size 4096x4096)
fn parse_size(args: &[String]) -> Option<(u32, u32)> {
    let idx = args.iter().position(|a| a == "--size")?;
    let val = args.get(idx + 1)?;
    let (w, h) = val.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}
