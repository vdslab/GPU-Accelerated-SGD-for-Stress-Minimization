mod gpu;
mod graph;

use anyhow::Result;
use sprs::vec;
use std::path::Path;
use std::fs::File;
use std::io::Write;
use chrono::Local;

fn main() -> Result<()> {
    env_logger::init();

    // let mtx_path = Path::new("../data/lesmis_pattern.mtx");
    // let graph = graph::Graph::from_mtx(mtx_path).expect("Failed to load matrix");

    let graph = {
        graph::Graph {
            node_size: 10,
            edge_size: 10,
            edge_src: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            edge_dst: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        }
    };

    // LOG: Print graph information
    println!("{:?}",graph);

    // GPU setup
    let gpu_context = gpu::GpuContext::new()?;

    // Create GPU pipeline
    let pipeline = graph::Graph::create_gpu_pipeline(&graph, &gpu_context, 15, 0.1, true)?;

    // LOG: Print pipeline
    // println!("Pipeline: {:?}", pipeline);

    let result = gpu::GpuContext::execute_compute_pipeline(&gpu_context, pipeline)?;

    // LOG: Print result
    println!("Result: {:?}", result);
    
    // Save result to file with timestamp (including graph structure)
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let filename = format!("../output/vram-lock-{}.txt", timestamp);
    let mut file = File::create(&filename)?;
    writeln!(file, "# Rust GPU Result (vram-lock)")?;
    writeln!(file, "# Timestamp: {}", Local::now().format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(file, "# Node count: {}", graph.node_size)?;
    writeln!(file, "# Edge count: {}", graph.edge_size)?;
    writeln!(file, "")?;
    writeln!(file, "# Edges (source target)")?;
    for i in 0..graph.edge_size {
        writeln!(file, "{} {}", graph.edge_src[i], graph.edge_dst[i])?;
    }
    writeln!(file, "")?;
    writeln!(file, "# Positions (x y)")?;
    for pos in &result {
        writeln!(file, "{} {}", pos[0], pos[1])?;
    }
    println!("Result saved to {}", filename);

    Ok(())
}
