use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::time::Instant;
use webgpu_bfs_core::CsrGraph;
use webgpu_bfs_gpu::engine::{GpuBfsEngine, GpuBfsMethod};

/// Outcome of APSP preprocessing ready for RoundRobin Full SGD.
#[derive(Debug, Clone)]
pub struct FullBfsOutcome {
    pub dist_u16: Vec<u16>,
    pub dmin: f64,
    pub dmax: f64,
    pub constraint_count: usize,
    pub bfs_time_ms: f64,
    pub compaction_time_ms: f64,
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

/// Threshold below which CPU BFS is faster due to GPU kernel dispatch & staging overhead.
pub const GPU_APSP_THRESHOLD: usize = 128;

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
            bfs_time_ms: 0.0,
            compaction_time_ms: 0.0,
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
    let bfs_start = Instant::now();

    // 1. Build symmetric CSR graph
    let mut edges = Vec::with_capacity(graph.edge_size * 2);
    for i in 0..graph.edge_size {
        let u = graph.edge_src[i] as u32;
        let v = graph.edge_dst[i] as u32;
        edges.push((u, v));
        edges.push((v, u));
    }
    let csr = CsrGraph::from_edge_list(n, &edges);

    // 2. Execute GPU Multi-Source BFS across all N sources
    // Auto selects MultiWordBitParallel for large batches on general graphs
    // or WorkgroupPersistentBfs on road networks.
    let engine = GpuBfsEngine::default_with_method(GpuBfsMethod::Auto)
        .context("Failed to initialize webgpu-bfs GPU engine")?;
    let sources: Vec<u32> = (0..n as u32).collect();
    let mut dist_flat_u32 = vec![u32::MAX; n * n];
    engine
        .run_ms_bfs_into(&csr, &sources, &mut dist_flat_u32)
        .context("GPU BFS execution failed")?;
    let bfs_time_ms = bfs_start.elapsed().as_secs_f64() * 1000.0;

    // 3. Compact into lossless u16 multi-bank buffer & calculate dmin/dmax/constraints
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
        bfs_time_ms,
        compaction_time_ms,
    })
}

pub fn compute_apsp_cpu(graph: &super::Graph) -> Result<FullBfsOutcome> {
    let bfs_start = Instant::now();
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
    let bfs_time_ms = bfs_start.elapsed().as_secs_f64() * 1000.0;

    Ok(FullBfsOutcome {
        dist_u16,
        dmin,
        dmax,
        constraint_count,
        bfs_time_ms,
        compaction_time_ms: 0.0,
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
}
