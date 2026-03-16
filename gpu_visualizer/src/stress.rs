use std::collections::VecDeque;

/// Threshold above which stress computation is skipped (O(n²) memory/time).
const MAX_N_FOR_STRESS: usize = 8_000;

pub enum StressResult {
    Value(f64),
    TooLarge,
    Disconnected,
}

/// Compute graph-layout stress:
///   Σ_{i<j} w_ij (||p_i - p_j|| - d_ij)²   where w_ij = 1/d_ij²
///
pub fn calc_stress(
    positions: &[[f32; 2]],
    edges: &[(usize, usize)],
) -> StressResult {
    let n = positions.len();
    if n > MAX_N_FOR_STRESS {
        return StressResult::TooLarge;
    }

    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        if u < n && v < n {
            adj[u].push(v);
            adj[v].push(u);
        }
    }

    let stress: f64 = (0..n)
        .map(|src| {
            let dist = bfs(&adj, n, src);
            let mut s = 0.0f64;
            for dst in (src + 1)..n {
                let d = dist[dst];
                if d == u32::MAX {
                    continue; // unreachable pair – skip
                }
                let d = d as f64;
                let dx = positions[src][0] as f64 - positions[dst][0] as f64;
                let dy = positions[src][1] as f64 - positions[dst][1] as f64;
                let euc = (dx * dx + dy * dy).sqrt();
                let w = 1.0 / (d * d);
                s += w * (euc - d) * (euc - d);
            }
            s
        })
        .sum();

    // If every off-diagonal BFS entry was MAX the graph is fully disconnected
    // (stress would be 0.0 — report Disconnected instead)
    if stress == 0.0 && n > 1 && edges.is_empty() {
        return StressResult::Disconnected;
    }

    StressResult::Value(stress)
}

fn bfs(adj: &[Vec<usize>], n: usize, src: usize) -> Vec<u32> {
    let mut dist = vec![u32::MAX; n];
    dist[src] = 0;
    let mut queue = VecDeque::new();
    queue.push_back(src);
    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if dist[v] == u32::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }
    dist
}
