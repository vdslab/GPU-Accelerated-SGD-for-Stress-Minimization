use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct GraphData {
    pub node_count: usize,
    pub edges: Vec<(usize, usize)>,
    pub positions: Vec<[f32; 2]>,
}

pub fn read_result_file(path: &Path) -> Result<GraphData> {
    let file = File::open(path)
        .with_context(|| format!("Cannot open: {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut node_count = 0usize;
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut positions: Vec<[f32; 2]> = Vec::new();
    let mut mode = "";

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        if line.starts_with("# Node count:") {
            node_count = line
                .split(':')
                .nth(1)
                .unwrap_or("0")
                .trim()
                .parse()
                .context("Invalid node count")?;
        } else if line.starts_with("# Edges") {
            mode = "edges";
        } else if line.starts_with("# Positions") {
            mode = "positions";
        } else if line.starts_with('#') {
            continue;
        } else if mode == "edges" {
            let mut parts = line.split_ascii_whitespace();
            if let (Some(u), Some(v)) = (parts.next(), parts.next()) {
                let u: usize = u.parse().context("Invalid edge src")?;
                let v: usize = v.parse().context("Invalid edge dst")?;
                edges.push((u, v));
            }
        } else if mode == "positions" {
            let mut parts = line.split_ascii_whitespace();
            if let (Some(x), Some(y)) = (parts.next(), parts.next()) {
                let x: f32 = x.parse().context("Invalid position x")?;
                let y: f32 = y.parse().context("Invalid position y")?;
                positions.push([x, y]);
            }
        }
    }

    Ok(GraphData { node_count, edges, positions })
}
