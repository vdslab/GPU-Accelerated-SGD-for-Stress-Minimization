use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::time::Instant;
use webgpu_bfs_core::CsrGraph;
use webgpu_bfs_gpu::engine::{GpuBfsEngine, GpuBfsMethod};

/// Maximum vertex count supported by Full SGD's 4-bank 512 MiB distance storage (N <= 16,384).
pub const FULL_SGD_MAX_NODES: usize = 16_384;

/// Threshold at or above which GPU APSP BFS is selected under `Auto`.
///
/// On WSL2 Mesa Gallium D3D12, initializing a standalone `GpuBfsEngine` and compiling
/// WGSL compute pipelines to DXIL incurs ~2.2 - 2.7 seconds of latency. Meanwhile,
/// unweighted CPU FIFO BFS is extremely fast (0.18 ms at N=128, 116 ms at N=2,000,
/// and 2.4 s at N=8,000). Consequently, CPU BFS strictly outperforms cold GPU BFS for all
/// graphs that fit within Full SGD's multi-bank VRAM saturation boundary (N <= 16,384).
///
/// Setting `GPU_APSP_THRESHOLD = FULL_SGD_MAX_NODES + 1` ensures that `--bfs-engine auto`
/// selects CPU BFS for all valid Full SGD inputs (preventing up to 15,000x regression), while
/// explicit `--bfs-engine gpu` remains available for benchmarking and profiling.
pub const GPU_APSP_THRESHOLD: usize = FULL_SGD_MAX_NODES + 1;

/// Timing breakdown for APSP BFS execution.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BfsTiming {
    /// WebGPU device & pipeline initialization time in milliseconds (0.0 for CPU BFS).
    pub engine_init_time_ms: f64,
    /// Pure BFS kernel dispatch / traversal execution time in milliseconds.
    pub bfs_kernel_time_ms: f64,
    /// Distance matrix compaction time into lossless u16 buffer.
    pub compaction_time_ms: f64,
}

/// Outcome of APSP preprocessing ready for RoundRobin Full SGD.
#[derive(Debug, Clone)]
pub struct FullBfsOutcome {
    pub dist_u16: Vec<u16>,
    pub dmin: f64,
    pub dmax: f64,
    pub constraint_count: usize,
    pub timing: BfsTiming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BfsBackend {
    #[default]
    Auto,
    Gpu,
    Cpu,
}

impl std::str::FromStr for BfsBackend {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(BfsBackend::Auto),
            "gpu" => Ok(BfsBackend::Gpu),
            "cpu" => Ok(BfsBackend::Cpu),
            other => anyhow::bail!("Unknown BFS backend '{other}'. Valid options: auto, gpu, cpu"),
        }
    }
}

/// Computes APSP distance matrix and prepares lossless u16 multi-bank buffer.
pub fn compute_apsp(
    graph: &super::Graph,
    backend: BfsBackend,
) -> Result<FullBfsOutcome> {
    let n = graph.node_size;
    if n == 0 {
        return Ok(FullBfsOutcome {
            dist_u16: Vec::new(),
            dmin: f64::INFINITY,
            dmax: 0.0,
            constraint_count: 0,
            timing: BfsTiming::default(),
        });
    }

    let use_gpu = match backend {
        BfsBackend::Auto => n >= GPU_APSP_THRESHOLD,
        BfsBackend::Gpu => true,
        BfsBackend::Cpu => false,
    };

    if use_gpu {
        compute_apsp_gpu(graph)
    } else {
        compute_apsp_cpu(graph)
    }
}

pub fn compute_apsp_gpu(graph: &super::Graph) -> Result<FullBfsOutcome> {
    let n = graph.node_size;

    // 1. Build symmetric CSR graph
    let mut edges = Vec::with_capacity(graph.edge_size * 2);
    for i in 0..graph.edge_size {
        let u = graph.edge_src[i] as u32;
        let v = graph.edge_dst[i] as u32;
        edges.push((u, v));
        edges.push((v, u));
    }
    let csr = CsrGraph::from_edge_list(n, &edges);

    // 2. Isolate engine initialization (wgpu device creation + shader compilation)
    let init_start = Instant::now();
    let engine = GpuBfsEngine::default_with_method(GpuBfsMethod::Auto)
        .context("Failed to initialize webgpu-bfs GPU engine")?;
    let engine_init_time_ms = init_start.elapsed().as_secs_f64() * 1000.0;

    // 3. Execute pure GPU Multi-Source BFS across all N sources
    // Auto selects MultiWordBitParallel for large batches on general graphs
    // or WorkgroupPersistentBfs on road networks.
    let kernel_start = Instant::now();
    let sources: Vec<u32> = (0..n as u32).collect();
    let mut dist_flat_u32 = vec![u32::MAX; n * n];
    engine
        .run_ms_bfs_into(&csr, &sources, &mut dist_flat_u32)
        .context("GPU BFS execution failed")?;
    let bfs_kernel_time_ms = kernel_start.elapsed().as_secs_f64() * 1000.0;

    // 4. Compact into lossless u16 multi-bank buffer & calculate dmin/dmax/constraints
    // Directly without allocating gigabytes of EdgeInfo or Vec<Vec<usize>>
    let compaction_start = Instant::now();
    let total_distances = n * n;
    let total_words = total_distances.div_ceil(2);
    let mut dist_u16 = vec![0u16; total_words * 2];
    let mut dmin = f64::INFINITY;
    let mut dmax = 0.0_f64;
    let mut constraint_count: usize = 0;

    for u in 0..n {
        let row_offset = u * n;
        for v in (u + 1)..n {
            let d = dist_flat_u32[row_offset + v];
            if d != u32::MAX && d > 0 {
                let d_f64 = d as f64;
                dmin = dmin.min(d_f64);
                dmax = dmax.max(d_f64);
                constraint_count += 1;
                let d_u16 = d as u16;
                dist_u16[row_offset + v] = d_u16;
                dist_u16[v * n + u] = d_u16;
            }
        }
    }
    let compaction_time_ms = compaction_start.elapsed().as_secs_f64() * 1000.0;

    Ok(FullBfsOutcome {
        dist_u16,
        dmin,
        dmax,
        constraint_count,
        timing: BfsTiming {
            engine_init_time_ms,
            bfs_kernel_time_ms,
            compaction_time_ms,
        },
    })
}

pub fn compute_apsp_cpu(graph: &super::Graph) -> Result<FullBfsOutcome> {
    let kernel_start = Instant::now();
    let n = graph.node_size;
    let total_distances = n * n;
    let total_words = total_distances.div_ceil(2);
    let mut dist_u16 = vec![0u16; total_words * 2];
    let mut dmin = f64::INFINITY;
    let mut dmax = 0.0_f64;
    let mut constraint_count: usize = 0;

    let adj = graph.calc_adj_matrix();
    let mut queue = VecDeque::with_capacity(n);
    let mut dist_row = vec![u32::MAX; n];

    for u in 0..n {
        dist_row.fill(u32::MAX);
        queue.clear();
        dist_row[u] = 0;
        queue.push_back(u);

        while let Some(v) = queue.pop_front() {
            let next_d = dist_row[v] + 1;
            for &w in &adj[v] {
                if dist_row[w] == u32::MAX {
                    dist_row[w] = next_d;
                    queue.push_back(w);
                }
            }
        }

        let row_offset = u * n;
        for v in (u + 1)..n {
            let d = dist_row[v];
            if d != u32::MAX && d > 0 {
                let d_f64 = d as f64;
                dmin = dmin.min(d_f64);
                dmax = dmax.max(d_f64);
                constraint_count += 1;
                let d_u16 = d as u16;
                dist_u16[row_offset + v] = d_u16;
                dist_u16[v * n + u] = d_u16;
            }
        }
    }
    let bfs_kernel_time_ms = kernel_start.elapsed().as_secs_f64() * 1000.0;

    Ok(FullBfsOutcome {
        dist_u16,
        dmin,
        dmax,
        constraint_count,
        timing: BfsTiming {
            engine_init_time_ms: 0.0,
            bfs_kernel_time_ms,
            compaction_time_ms: 0.0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::Graph;

    #[test]
    fn test_cpu_and_gpu_apsp_parity_on_small_cycle() {
        // Simple cycle graph on 6 vertices
        let graph = Graph {
            node_size: 6,
            edge_size: 6,
            edge_src: vec![0, 1, 2, 3, 4, 5],
            edge_dst: vec![1, 2, 3, 4, 5, 0],
        };

        let cpu_outcome = compute_apsp_cpu(&graph).unwrap();
        let gpu_outcome = compute_apsp_gpu(&graph).unwrap();

        assert_eq!(cpu_outcome.constraint_count, gpu_outcome.constraint_count);
        assert_eq!(cpu_outcome.dmin, gpu_outcome.dmin);
        assert_eq!(cpu_outcome.dmax, gpu_outcome.dmax);
        assert_eq!(cpu_outcome.dist_u16, gpu_outcome.dist_u16);
    }

    #[test]
    fn test_auto_routing_protects_against_cold_gpu_regression() {
        // Under Auto, graphs with N <= FULL_SGD_MAX_NODES (16,384) use CPU BFS
        assert!(GPU_APSP_THRESHOLD > FULL_SGD_MAX_NODES);
        assert_eq!(FULL_SGD_MAX_NODES, 16_384);
    }
}
