mod gpu;
mod graph;

use std::path::Path;
use std::time::Instant;
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use chrono::Local;

fn main() -> Result<()> {
    env_logger::init();

    let mtx_path = Path::new("../data/bcspwr10.mtx");
    let graph = graph::Graph::from_mtx(mtx_path).expect("Failed to load matrix");

    // let graph = {
    //     graph::Graph {
    //         node_size: 10,
    //         edge_size: 10,
    //         edge_src: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    //         edge_dst: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    //     }
    // };

    // LOG: Print graph information
    // println!("{:?}",graph);

    // GPU setup
    let gpu_context = gpu::GpuContext::new()?;

    // CPU precompute
    let sgd_params = graph.prepare_sgd_params(15, 0.1, true);

    // GPU: convert + create pipeline
    let (pipeline, initial_positions) = gpu_context.create_pipeline_from_cpu_params(sgd_params)?;

    // LOG: Print pipeline
    // println!("Pipeline: {:?}", pipeline);

    let start = Instant::now();
    let result = gpu::GpuContext::execute_compute_pipeline(&gpu_context, pipeline)?;
    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);

    // LOG: Print result
    // println!("Result: {:?}", result);
    
    // Save initial positions (after randomization) to file with timestamp
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let filename_init = format!("../output/vram-lock-{}-0.txt", timestamp);
    let mut file = File::create(&filename_init)?;
    writeln!(file, "# Rust GPU Result (vram-lock) - Initial (Randomized)")?;
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
    for pos in &initial_positions {
        writeln!(file, "{} {}", pos[0], pos[1])?;
    }
    println!("Initial result saved to {}", filename_init);
    
    // Save processed result to file with timestamp
    let filename_processed = format!("../output/vram-lock-{}-1.txt", timestamp);
    let mut file = File::create(&filename_processed)?;
    writeln!(file, "# Rust GPU Result (vram-lock) - Processed")?;
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
    println!("Processed result saved to {}", filename_processed);

    Ok(())
}
