use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const DEFAULT_NODES: usize = 250_000;
const DEFAULT_EDGES: usize = 1_941_926;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let output = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/gpu-visualizer-250k.txt"));
    let nodes = args
        .get(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_NODES);
    let edges = args
        .get(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_EDGES);
    assert!(nodes >= 2);
    assert!(edges >= nodes - 1, "fixture must be connected");

    let mut writer = BufWriter::new(File::create(&output)?);
    writeln!(writer, "# GPU visualizer deterministic large fixture")?;
    writeln!(writer, "# Node count: {nodes}")?;
    writeln!(writer, "# Edge count: {edges}")?;
    writeln!(writer, "# Edges (source target)")?;
    for i in 0..edges {
        let (u, v) = if i < nodes - 1 {
            (i, i + 1)
        } else {
            let u = i % nodes;
            (u, (u + 1 + (i / nodes) * 7_919) % nodes)
        };
        writeln!(writer, "{u} {v}")?;
    }
    writeln!(writer, "# Positions (x y)")?;
    for i in 0..nodes {
        let angle = std::f64::consts::TAU * i as f64 / nodes as f64;
        let radius = 1.0 + (i % 997) as f64 / 4_000.0;
        writeln!(writer, "{} {}", radius * angle.cos(), radius * angle.sin())?;
    }
    writer.flush()?;
    println!(
        "Generated: {} (nodes={nodes}, edges={edges})",
        output.display()
    );
    Ok(())
}
