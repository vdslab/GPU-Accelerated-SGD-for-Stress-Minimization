mod gpu;
mod graph;

use anyhow::Result;
use std::path::Path;

fn main() -> Result<()> {
    env_logger::init();

    let mtx_path = Path::new("../data/lesmis_pattern.mtx");
    let graph = graph::Graph::from_mtx(mtx_path).expect("Failed to load matrix");

    // LOG: Print graph information
    // println!("Node size: {:?}", graph.node_size);
    // println!("Edge size: {:?}", graph.edge_size);
    // println!("Edge src: {:?}", graph.edge_src);
    // println!("Edge dst: {:?}", graph.edge_dst);

    let dist = graph::Graph::calc_dist_matrix(&graph);

    // LOG: Print distance matrix
    // println!("Dist matrix: {:?}", dist);

    let (pairs, wmin, wmax) = graph::Graph::calc_edge_info(&graph, &dist);

    let etas = graph::calc_learning_rate(15, wmin, wmax, 0.1);

    // LOG: Print pairs and etas
    // println!("pairs: {:?}", pairs);
    // println!("Etas: {:?}", etas);

    let positions = graph::init_positions_random(graph.node_size, true);

    // LOG: Print positions
    // println!("positions: {:?}", positions);

    // GPU setup
    let gpu_context = gpu::GpuContext::new()?;

    // LOG: Print GPU context
    // println!("GPU context: {:?}", gpu_context);

    // Test computation
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let pipeline = gpu::GpuContext::setup_compute_pipeline(&gpu_context, gpu::GpuParams { data })?;

    // LOG: Print pipeline
    println!("Pipeline: {:?}", pipeline);

    Ok(())
}
