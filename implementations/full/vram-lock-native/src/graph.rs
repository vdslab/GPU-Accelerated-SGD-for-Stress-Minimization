use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use sprs::io::read_matrix_market;
use sprs::num_kinds::Pattern;
use std::cell::RefCell;
use std::path::Path;
use std::sync::OnceLock;

#[derive(Debug)]
pub struct Graph {
    pub node_size: usize,
    pub edge_size: usize,
    pub edge_src: Vec<usize>,
    pub edge_dst: Vec<usize>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable, Default)]
pub struct EdgeInfo {
    pub u: u32,
    pub v: u32,
    pub dij: f32,
    pub wij: f32,
}

#[allow(dead_code)]
pub type GpuPair = EdgeInfo;
#[allow(dead_code)]
pub type ShortestPathPair = EdgeInfo;

#[allow(dead_code)]
#[derive(Debug)]
pub struct SgdParams {
    pub etas: Vec<f64>,
    pub positions: Vec<[f64; 2]>,
    pub pairs: Vec<EdgeInfo>,
    pub center: bool,
}

pub struct FlatCsr {
    pub offsets: Vec<u32>,
    pub neighbors: Vec<u32>,
}

#[derive(Clone, Copy)]
struct SendPtr(*mut EdgeInfo);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    unsafe fn add(self, count: usize) -> *mut EdgeInfo {
        self.0.add(count)
    }
}

thread_local! {
    static BFS_BUFFERS: RefCell<(Vec<u32>, Vec<u32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

fn get_rayon_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let max_threads = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(8)
            })
            .clamp(1, 12);
        rayon::ThreadPoolBuilder::new()
            .num_threads(max_threads)
            .thread_name(|i| format!("full-bfs-{}", i))
            .build()
            .expect("Failed to build BFS Rayon thread pool")
    })
}

impl Graph {
    pub fn from_mtx(path: &Path) -> Result<Self> {
        let matrix: sprs::TriMat<Pattern> = read_matrix_market(path)?;
        let node_size: usize = matrix.rows();

        let mut edge_src = Vec::new();
        let mut edge_dst = Vec::new();

        for (row, col) in matrix.row_inds().iter().zip(matrix.col_inds().iter()) {
            if row != col {
                edge_src.push(*row);
                edge_dst.push(*col);
            }
        }

        let edge_size = edge_src.len();
        Ok(Graph {
            node_size,
            edge_size,
            edge_src,
            edge_dst,
        })
    }

    pub fn to_flat_csr(&self) -> FlatCsr {
        let n = self.node_size;
        let mut degrees = vec![0u32; n];
        for i in 0..self.edge_size {
            degrees[self.edge_src[i]] += 1;
            degrees[self.edge_dst[i]] += 1;
        }

        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        let mut sum = 0u32;
        for &deg in &degrees {
            sum += deg;
            offsets.push(sum);
        }

        let mut cursor = offsets[0..n].to_vec();
        let mut neighbors = vec![0u32; sum as usize];
        for i in 0..self.edge_size {
            let u = self.edge_src[i];
            let v = self.edge_dst[i];

            let u_pos = cursor[u] as usize;
            neighbors[u_pos] = v as u32;
            cursor[u] += 1;

            let v_pos = cursor[v] as usize;
            neighbors[v_pos] = u as u32;
            cursor[v] += 1;
        }

        FlatCsr { offsets, neighbors }
    }

    pub fn calc_edge_info_csr(&self, csr: &FlatCsr) -> (Vec<EdgeInfo>, f64, f64) {
        let n = self.node_size;
        if n <= 1 {
            return (Vec::new(), f64::INFINITY, 0.0);
        }

        let total_pairs = n * (n - 1) / 2;
        let mut pairs = vec![EdgeInfo::default(); total_pairs];
        let pairs_ptr = SendPtr(pairs.as_mut_ptr());

        let offsets = &csr.offsets;
        let neighbors = &csr.neighbors;

        let pool = get_rayon_pool();
        let (dmin, dmax) = pool.install(|| {
            (0..n - 1)
                .into_par_iter()
                .map(move |u| {
                    let slice_offset = u * n - u * (u + 1) / 2;
                    let mut local_dmin = f64::INFINITY;
                    let mut local_dmax = 0.0f64;

                    BFS_BUFFERS.with_borrow_mut(|(dist, queue)| {
                        if dist.len() != n {
                            dist.resize(n, u32::MAX);
                        }
                        if queue.capacity() < n {
                            queue.reserve(n - queue.capacity());
                        }

                        dist[u] = 0;
                        queue.push(u as u32);
                        let mut head = 0;

                        while head < queue.len() {
                            let curr = queue[head] as usize;
                            head += 1;
                            let next_d = dist[curr] + 1;
                            let start = offsets[curr] as usize;
                            let end = offsets[curr + 1] as usize;

                            for &nbr in &neighbors[start..end] {
                                let nbr_idx = nbr as usize;
                                if dist[nbr_idx] == u32::MAX {
                                    dist[nbr_idx] = next_d;
                                    queue.push(nbr);
                                }
                            }
                        }

                        let out_ptr = unsafe { pairs_ptr.add(slice_offset) };
                        for v in (u + 1)..n {
                            let d = dist[v];
                            if d != u32::MAX && d > 0 {
                                let dij = d as f32;
                                let wij = 1.0f32 / (dij * dij);
                                let d_f64 = d as f64;
                                if d_f64 < local_dmin {
                                    local_dmin = d_f64;
                                }
                                if d_f64 > local_dmax {
                                    local_dmax = d_f64;
                                }
                                unsafe {
                                    *out_ptr.add(v - u - 1) = EdgeInfo {
                                        u: u as u32,
                                        v: v as u32,
                                        dij,
                                        wij,
                                    };
                                }
                            }
                        }

                        for &v in queue.iter() {
                            dist[v as usize] = u32::MAX;
                        }
                        queue.clear();
                    });

                    (local_dmin, local_dmax)
                })
                .reduce(
                    || (f64::INFINITY, 0.0f64),
                    |a, b| (a.0.min(b.0), a.1.max(b.1)),
                )
        });

        let wmin = 1.0 / (dmax * dmax);
        let wmax = 1.0 / (dmin * dmin);

        (pairs, wmin, wmax)
    }

    /// Precompute SGD parameters using parallel CSR BFS
    pub fn prepare_sgd_params(
        &self,
        iterations: usize,
        epsilon: f64,
        center: bool,
        seed: u64,
    ) -> SgdParams {
        let csr = self.to_flat_csr();
        let (pairs, wmin, wmax) = self.calc_edge_info_csr(&csr);

        let etas = calc_learning_rate(iterations, wmin, wmax, epsilon);
        let positions = experiment_common::seed::seeded_positions(self.node_size, seed, center);

        SgdParams {
            etas,
            positions,
            pairs,
            center,
        }
    }
}

pub fn calc_learning_rate(tmax: usize, wmin: f64, wmax: f64, eps: f64) -> Vec<f64> {
    let eta_max = 1.0 / wmin;
    let eta_min = eps / wmax;
    if tmax == 1 {
        return vec![eta_max];
    }
    let lamb = (eta_max / eta_min).ln() / (tmax - 1) as f64;

    (0..tmax)
        .map(|t| eta_max * (-lamb * t as f64).exp())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_slice_indices() {
        let n = 6;
        let mut seen = vec![false; n * (n - 1) / 2];
        for u in 0..n - 1 {
            let offset = u * n - u * (u + 1) / 2;
            for v in (u + 1)..n {
                let idx = offset + (v - u - 1);
                assert!(!seen[idx], "Duplicate write at {idx}");
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "Missing write to some pair");
    }

    #[test]
    fn test_flat_csr_and_bfs_cycle6() {
        let graph = Graph {
            node_size: 6,
            edge_size: 6,
            edge_src: vec![0, 1, 2, 3, 4, 5],
            edge_dst: vec![1, 2, 3, 4, 5, 0],
        };
        let csr = graph.to_flat_csr();
        assert_eq!(csr.offsets.len(), 7);
        assert_eq!(csr.neighbors.len(), 12);

        let (pairs, wmin, wmax) = graph.calc_edge_info_csr(&csr);
        assert_eq!(pairs.len(), 15);
        assert!((wmin - 1.0 / 9.0).abs() < 1e-6);
        assert!((wmax - 1.0).abs() < 1e-6);

        // Verify canonical ordering u < v
        for pair in &pairs {
            assert!(pair.u < pair.v);
            assert!(pair.dij > 0.0);
            assert!(pair.wij > 0.0);
        }
    }
}
