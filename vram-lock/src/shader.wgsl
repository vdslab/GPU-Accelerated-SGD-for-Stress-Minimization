struct EdgeInfo {
    u: u32,
    v: u32,
    dij: f32,
    wij: f32,
}

@group(0) @binding(0)
var<storage, read> etas: array<f32>;

@group(0) @binding(1)
var<storage, read_write> positions: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read> pairs: array<EdgeInfo>;

@group(0) @binding(3)
var<uniform> iteration: u32;

@group(0) @binding(4)
var<storage, read_write> locks: array<atomic<u32>>;

// Atomic lock helper functions (based on WebGPU best practices)
fn try_lock(node: u32) -> bool {
    // Try to swap 0 -> 1. If old value was 0, we got the lock
    let old_value = atomicExchange(&locks[node], 1u);
    return old_value == 0u;
}

fn unlock(node: u32) {
    atomicExchange(&locks[node], 0u);
}

fn acquire_locks(node1: u32, node2: u32) -> bool {
    // Always lock in order: smaller index first (deadlock prevention)
    let first = min(node1, node2);
    let second = max(node1, node2);
    
    // Spin until both locks are acquired (with timeout for safety)
    let max_retries = 1000000u;
    for (var retry = 0u; retry < max_retries; retry++) {
        if (try_lock(first)) {
            if (try_lock(second)) {
                return true;  // Both locks acquired successfully
            } else {
                unlock(first);  // Release first lock and retry
            }
        }
        // Small yield to allow other workgroups to progress
    }
    
    // Timeout (should be extremely rare with proper GPU parallelism)
    return false;
}

fn release_locks(node1: u32, node2: u32) {
    unlock(node1);
    unlock(node2);
}

@compute @workgroup_size(32, 1, 1)
fn sgd(@builtin(local_invocation_id) local_id: vec3<u32>,@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let node_size = arrayLength(&positions);
    
    // 1 workgroup = 32 threads (= 1 warp)
    // each workgroup handles 1 pair
    // use 2D dispatch to handle more pairs: pair_idx = y * 65535 + x
    // only local_id.x == 0 does the work
    let total_pairs = arrayLength(&pairs);
    let pair_idx = workgroup_id.y * 65535u + workgroup_id.x;
    
    // only thread 0 in the workgroup does the work
    if (local_id.x != 0u) {
        return;
    }
    
    if (pair_idx >= total_pairs) {
        return;
    }
    
    let pair = pairs[pair_idx];
    let i = pair.u;
    let j = pair.v;
    
    // Only process upper triangular matrix (i < j)
    if (i >= j) {
        return;
    }
    
    let dij = pair.dij;
    let wij = pair.wij;
    
    // Get learning rate for this iteration
    let eta = etas[iteration];
    
    // Acquire locks for both nodes (deadlock-free with retry limit)
    if (!acquire_locks(i, j)) {
        // Failed to acquire locks after max retries
        // Skip this pair in this iteration
        return;
    }
    
    // SGD update (matching Python implementation)
    let tiny = 1e-12;
    var diff = positions[j] - positions[i];
    var dist = length(diff);
    
    // Handle zero/tiny distance case
    if (dist < tiny) {
        diff = vec2<f32>(1e-6, 1e-6);
        dist = length(diff);
    }
    
    let r = ((dist - dij) / 2.0) * (diff / dist);
    let mu = min(wij * eta, 1.0);
    
    positions[i] += mu * r;
    positions[j] -= mu * r;
    
    // Release locks
    release_locks(i, j);
}