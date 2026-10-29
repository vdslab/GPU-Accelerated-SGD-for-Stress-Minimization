use anyhow::Result;
use rand::{rngs::StdRng, SeedableRng};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
pub use vram_lock_sparse::graph;
#[path = "../probe_metal.rs"]
mod probe_metal;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let g = graph::Graph::from_mtx(Path::new(&a[1]))?.largest_connected_component()?;
    let mut rng = StdRng::seed_from_u64(0);
    let p = g.prepare_sgd_params(15, 0.1, 200, true, &mut rng)?;
    fs::write(
        format!("{}.pivots.json", a[2]),
        serde_json::to_string(&p.pivots)?,
    )?;
    let mut degrees: Vec<_> = (0..g.node_size).map(|i| g.neighbors(i).len()).collect();
    degrees.sort_unstable();
    let mut incidence = vec![0usize; g.node_size];
    let mut one_sided = 0;
    let mut edges = 0;
    for c in &p.pairs {
        incidence[c.u] += 1;
        incidence[c.v] += 1;
        one_sided += (c.weight_u == 0.0 || c.weight_v == 0.0) as usize;
        edges += (c.dij == 1.0) as usize;
    }
    incidence.sort_unstable();
    fs::write(
        format!("{}.stats.json", a[2]),
        serde_json::to_string_pretty(
            &serde_json::json!({"n":g.node_size,"m":g.edge_size,"degree_mean":2.0*g.edge_size as f64/g.node_size as f64,"degree_median":degrees[g.node_size/2],"degree_max":degrees[g.node_size-1],"constraints":p.pairs.len(),"one_sided":one_sided,"edge_constraints":edges,"constraint_degree_max":incidence[g.node_size-1],"constraint_degree_median":incidence[g.node_size/2],"etas":p.etas}),
        )?,
    )?;
    let result = probe_metal::MetalContext::new()?.execute_sgd_with_cpu_shuffle(
        p,
        &mut rng,
        false,
        experiment_common::OutputFormat::Json,
        true,
    )?;
    let mut f = BufWriter::new(File::create(format!("{}.positions.txt", a[2]))?);
    writeln!(f, "# Positions (x y)")?;
    for p in &result.positions {
        writeln!(f, "{} {}", p[0], p[1])?;
    }
    f.flush()?;
    println!(
        "{}",
        serde_json::json!({"completed":result.completed_updates,"attempted":result.attempted_updates,"seconds":result.iteration_time.as_secs_f64(),"dispatches":result.dispatches})
    );
    Ok(())
}
