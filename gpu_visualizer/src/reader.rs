use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
pub struct GraphData {
    pub node_count: usize,
    pub edges: Vec<(usize, usize)>,
    pub positions: Vec<[f32; 2]>,
}

pub fn read_result_file(path: &Path) -> Result<GraphData> {
    let file = File::open(path).with_context(|| format!("Cannot open: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut node_count = 0usize;
    let mut expected_edges = None;
    let mut edges = Vec::new();
    let mut positions = Vec::new();
    let mut mode = "";

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(value) = metadata_value(line, "# Node count:") {
            node_count = value.parse().context("Invalid node count")?;
            positions.reserve(node_count);
        } else if let Some(value) = metadata_value(line, "# Edge count:") {
            let count: usize = value.parse().context("Invalid edge count")?;
            expected_edges = Some(count);
            edges.reserve(count);
        } else if line.starts_with("# Edges") {
            mode = "edges";
        } else if line.starts_with("# Positions") {
            mode = "positions";
        } else if line.starts_with('#') {
            continue;
        } else if mode == "edges" {
            let mut parts = line.split_ascii_whitespace();
            let u = parts
                .next()
                .context("Missing edge src")?
                .parse()
                .with_context(|| format!("Invalid edge src at line {}", line_no + 1))?;
            let v = parts
                .next()
                .context("Missing edge dst")?
                .parse()
                .with_context(|| format!("Invalid edge dst at line {}", line_no + 1))?;
            edges.push((u, v));
        } else if mode == "positions" {
            let mut parts = line.split_ascii_whitespace();
            let x = parts
                .next()
                .context("Missing position x")?
                .parse()
                .with_context(|| format!("Invalid position x at line {}", line_no + 1))?;
            let y = parts
                .next()
                .context("Missing position y")?
                .parse()
                .with_context(|| format!("Invalid position y at line {}", line_no + 1))?;
            positions.push([x, y]);
        }
    }
    if node_count == 0 {
        node_count = positions.len();
    }
    if let Some(expected) = expected_edges {
        anyhow::ensure!(
            expected == edges.len(),
            "Edge count mismatch: metadata={expected}, actual={}",
            edges.len()
        );
    }
    let graph = GraphData {
        node_count,
        edges,
        positions,
    };
    validate_graph(&graph)?;
    Ok(graph)
}

pub fn validate_graph(graph: &GraphData) -> Result<()> {
    anyhow::ensure!(!graph.positions.is_empty(), "No positions found in file");
    anyhow::ensure!(
        graph.node_count == graph.positions.len(),
        "Node count mismatch: metadata={}, positions={}",
        graph.node_count,
        graph.positions.len()
    );
    anyhow::ensure!(
        graph.node_count <= u32::MAX as usize,
        "頂点数 {} はu32上限 {} を超えています",
        graph.node_count,
        u32::MAX
    );
    for (i, &[x, y]) in graph.positions.iter().enumerate() {
        anyhow::ensure!(
            x.is_finite() && y.is_finite(),
            "頂点 {i} の座標が有限値ではありません: ({x}, {y})"
        );
    }
    for (i, &(u, v)) in graph.edges.iter().enumerate() {
        anyhow::ensure!(
            u < graph.node_count && v < graph.node_count,
            "辺 {i} の端点が範囲外です: ({u}, {v}), 頂点数={}",
            graph.node_count
        );
    }
    Ok(())
}

fn metadata_value<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    line.strip_prefix(prefix).map(str::trim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_finite_position() {
        let graph = GraphData {
            node_count: 1,
            edges: vec![],
            positions: vec![[f32::NAN, 0.0]],
        };
        assert!(validate_graph(&graph)
            .unwrap_err()
            .to_string()
            .contains("頂点 0"));
    }

    #[test]
    fn rejects_bad_edge_with_index() {
        let graph = GraphData {
            node_count: 1,
            edges: vec![(0, 3)],
            positions: vec![[0.0, 0.0]],
        };
        let message = validate_graph(&graph).unwrap_err().to_string();
        assert!(message.contains("辺 0"));
        assert!(message.contains("(0, 3)"));
    }
}
