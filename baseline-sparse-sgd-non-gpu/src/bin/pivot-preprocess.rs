use anyhow::{bail, Context, Result};
use baseline_sparse_sgd_non_gpu::{algorithm, graph::Graph, preprocessing::*};
use rand::{rngs::StdRng, SeedableRng};
use std::{path::PathBuf, time::Instant};
#[derive(Debug)]
struct Config {
    input: Option<PathBuf>,
    mode: String,
    backend: String,
    pivots: usize,
    iterations: usize,
    epsilon: f64,
    seed: u64,
}
fn config() -> Result<Config> {
    let mut c = Config {
        input: None,
        mode: "run".into(),
        backend: "cpu".into(),
        pivots: 200,
        iterations: 15,
        epsilon: 0.1,
        seed: 0,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!("pivot-preprocess --mode validate|verify|run --backend cpu|cuda --input GRAPH.mtx [--pivots 200 --iterations 15 --epsilon 0.1 --seed 0]");
            std::process::exit(0);
        }
        let value = args
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => c.input = Some(value.into()),
            "--mode" => c.mode = value,
            "--backend" => c.backend = value,
            "--pivots" => c.pivots = value.parse()?,
            "--iterations" => c.iterations = value.parse()?,
            "--epsilon" => c.epsilon = value.parse()?,
            "--seed" => c.seed = value.parse()?,
            _ => bail!("unknown flag {flag}"),
        }
    }
    if !["validate", "verify", "run"].contains(&c.mode.as_str())
        || !["cpu", "cuda"].contains(&c.backend.as_str())
        || c.pivots == 0
        || c.iterations == 0
        || !c.epsilon.is_finite()
        || c.epsilon <= 0.0
    {
        bail!("invalid configuration");
    }
    Ok(c)
}
fn backend<'a>(
    name: &str,
    graph: &'a Graph,
    csr: Option<&Csr>,
) -> Result<Box<dyn DistanceBackend + 'a>> {
    if name == "cpu" {
        return Ok(Box::new(CpuBfs(graph)));
    }
    #[cfg(feature = "cuda")]
    {
        return Ok(Box::new(baseline_sparse_sgd_non_gpu::cuda::CudaBfs::new(
            csr.context("missing CSR")?,
        )?));
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = csr;
        bail!("CUDA requested: rebuild with --features cuda (no CPU fallback)");
    }
}
fn validate(name: &str) -> Result<()> {
    let graphs = fixtures()?;
    for graph in &graphs {
        let csr = Csr::from_graph(graph)?;
        verify_all_sources(graph, backend(name, graph, Some(&csr))?.as_mut())?;
    }
    println!(
        "validated {} small graphs: all distances, diagonal, symmetry and INF agree",
        graphs.len()
    );
    Ok(())
}
fn main() -> Result<()> {
    let c = config()?;
    if c.mode == "validate" {
        return validate(&c.backend);
    }
    // Verify mode gates actual-input comparison on small CUDA fixtures.
    if c.mode == "verify" {
        validate("cuda")?;
    }
    let total = Instant::now();
    let read_start = Instant::now();
    let graph = Graph::from_mtx(c.input.as_ref().context("--input required")?)?
        .largest_connected_component()?;
    let csr = if c.backend == "cuda" || c.mode == "verify" {
        Some(Csr::from_graph(&graph)?)
    } else {
        None
    };
    let load = read_start.elapsed().as_secs_f64();
    if c.mode == "verify" {
        let mut cpu_rng = StdRng::seed_from_u64(c.seed);
        let mut gpu_rng = cpu_rng.clone();
        let cpu = prepare(
            &graph,
            &mut CpuBfs(&graph),
            c.pivots,
            c.iterations,
            c.epsilon,
            true,
            &mut cpu_rng,
        )?;
        let gpu = prepare(
            &graph,
            backend("cuda", &graph, csr.as_ref())?.as_mut(),
            c.pivots,
            c.iterations,
            c.epsilon,
            true,
            &mut gpu_rng,
        )?;
        verify_prepared(&cpu, &gpu)?;
        let a = algorithm::execute_sgd(cpu.params, &mut cpu_rng);
        let b = algorithm::execute_sgd(gpu.params, &mut gpu_rng);
        if a != b {
            bail!("final coordinates differ");
        }
        let stress_cpu = full_stress(&graph, &a);
        let stress_cuda = full_stress(&graph, &b);
        if !stress_cpu.is_finite() || stress_cpu != stress_cuda {
            bail!("stress differs or is nonfinite");
        }
        println!("verified h*n distances, full constraints, initial coordinates, learning rates, final coordinates and stress: {stress_cpu:.17}");
        return Ok(());
    }
    let setup = Instant::now();
    let mut bfs = backend(&c.backend, &graph, csr.as_ref())?;
    let setup_wall = setup.elapsed().as_secs_f64();
    let setup_transfer = bfs.transfer_seconds();
    // Warm this exact context without consuming the pivot/SGD RNG.
    let warmup = Instant::now();
    bfs.distances(0)?;
    let warmup_wall = warmup.elapsed().as_secs_f64();
    let mut rng = StdRng::seed_from_u64(c.seed);
    let prepared = prepare(
        &graph,
        bfs.as_mut(),
        c.pivots,
        c.iterations,
        c.epsilon,
        true,
        &mut rng,
    )?;
    let hashes = hashes(&prepared);
    let t = prepared.timings.clone();
    let count = prepared.params.pairs.len();
    let h = prepared.params.pivots.len();
    let sgd = Instant::now();
    let result = algorithm::execute_sgd(prepared.params, &mut rng);
    let sgd_wall = sgd.elapsed().as_secs_f64();
    let pipeline = (total.elapsed().as_secs_f64() - warmup_wall).max(0.0);
    let final_hash = hash_words(result.iter().flatten().map(|x| x.to_bits()));
    // CSV header/row; external benchmark script measures true whole-process CLI wall time.
    println!("backend,n,h,seed,iterations,epsilon,load_csr_s,setup_nontransfer_s,pivot_selection_s,pivot_distance_s,transfer_s,constraints_s,sgd_s,preprocess_s,pipeline_s,warmup_s,constraint_count,pivot_hash,distance_hash,constraint_hash,initial_hash,eta_hash,final_hash");
    println!("{},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{:016x},{:016x},{:016x},{:016x},{:016x},{:016x}",
        c.backend,graph.node_size,h,c.seed,c.iterations,c.epsilon,load,(setup_wall-setup_transfer).max(0.0),
        t.pivot_selection,t.distance,t.transfer+setup_transfer,t.constraints,sgd_wall,
        load+setup_wall+t.preparation,pipeline,warmup_wall,count,hashes[0],hashes[1],hashes[2],hashes[3],hashes[4],final_hash);
    Ok(())
}
