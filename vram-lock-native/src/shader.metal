#include <metal_stdlib>
using namespace metal;

struct EdgeInfo {
    uint u;
    uint v;
    float dij;
    float wij;
};

// Helper: atomic float add using native Metal 3.1+ atomic<float>
// This is O(1) instead of O(k) for the CAS-loop approach
inline void atomic_add_float(device atomic<float>* addr, float value) {
    atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}

// Atomic lock helper functions (matching WGSL implementation)
inline bool try_lock(device atomic_uint* locks, uint node) {
    // Try to swap 0 -> 1. If old value was 0, we got the lock
    uint expected = 0;
    return atomic_compare_exchange_weak_explicit(
        &locks[node], &expected, 1u,
        memory_order_relaxed, memory_order_relaxed);
}

inline void unlock(device atomic_uint* locks, uint node) {
    atomic_store_explicit(&locks[node], 0u, memory_order_relaxed);
}

inline bool acquire_locks(device atomic_uint* locks, uint node1, uint node2) {
    // Always lock in order: smaller index first (deadlock prevention)
    uint first = min(node1, node2);
    uint second = max(node1, node2);
    
    // Spin until both locks are acquired (with timeout for safety)
    const uint max_retries = 1000000;
    for (uint retry = 0; retry < max_retries; retry++) {
        if (try_lock(locks, first)) {
            if (try_lock(locks, second)) {
                return true;  // Both locks acquired successfully
            } else {
                unlock(locks, first);  // Release first lock and retry
            }
        }
        // Small yield to allow other workgroups to progress
    }
    
    // Timeout (should be extremely rare with proper GPU parallelism)
    return false;
}

inline void release_locks(device atomic_uint* locks, uint node1, uint node2) {
    unlock(locks, node1);
    unlock(locks, node2);
}

kernel void sgd(
    constant float* etas [[buffer(0)]],
    device atomic<float>* positions [[buffer(1)]],  // Native atomic<float> for Metal 3.1+
    constant EdgeInfo* pairs [[buffer(2)]],
    constant uint& iteration [[buffer(3)]],
    device atomic_uint* locks [[buffer(4)]],
    device uint* updated_pairs [[buffer(5)]],
    device atomic_uint& updated_count [[buffer(6)]],
    device float4* positions_before [[buffer(7)]],
    constant uint& num_pairs [[buffer(8)]],  // Add num_pairs parameter
    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 workgroup_id [[threadgroup_position_in_grid]]
) {
    // 1 workgroup = 32 threads (= 1 warp)
    // each workgroup handles 1 pair
    // use 2D dispatch to handle more pairs: pair_idx = y * 65535 + x
    // only local_id.x == 0 does the work
    uint pair_idx = workgroup_id.y * 65535u + workgroup_id.x;
    
    // only thread 0 in the workgroup does the work
    if (local_id.x != 0u) {
        return;
    }
    
    // Check bounds - CRITICAL for performance!
    if (pair_idx >= num_pairs) {
        return;
    }
    
    EdgeInfo pair = pairs[pair_idx];
    uint i = pair.u;
    uint j = pair.v;
    
    // Only process upper triangular matrix (i < j)
    if (i >= j) {
        return;
    }
    
    float dij = pair.dij;
    float wij = pair.wij;
    float eta = etas[iteration];
    
    // Acquire locks for both nodes (deadlock-free with retry limit)
    if (!acquire_locks(locks, i, j)) {
        // Failed to acquire locks after max retries
        // Skip this pair in this iteration
        return;
    }
    
    // Record the pair index
    uint record_idx = atomic_fetch_add_explicit(&updated_count, 1u, memory_order_relaxed);
    updated_pairs[record_idx] = pair_idx;
    
    // Calculate position indices
    uint i_x = i * 2;
    uint i_y = i_x + 1;
    uint j_x = j * 2;
    uint j_y = j_x + 1;
    
    // Read positions AFTER acquiring locks (need atomic_load for memory visibility)
    float2 pos_i = float2(
        atomic_load_explicit(&positions[i_x], memory_order_relaxed),
        atomic_load_explicit(&positions[i_y], memory_order_relaxed)
    );
    float2 pos_j = float2(
        atomic_load_explicit(&positions[j_x], memory_order_relaxed),
        atomic_load_explicit(&positions[j_y], memory_order_relaxed)
    );
    
    positions_before[record_idx] = float4(pos_i.x, pos_i.y, pos_j.x, pos_j.y);
    
    // SGD update (matching WGSL implementation)
    const float tiny = 1e-12;
    float2 diff = pos_j - pos_i;
    float dist = length(diff);
    
    // Handle zero/tiny distance case
    if (dist < tiny) {
        diff = float2(1e-6, 1e-6);
        dist = length(diff);
    }
    
    float2 r = ((dist - dij) / 2.0) * (diff / dist);
    float mu = min(wij * eta, 1.0);
    
    float2 delta = mu * r;
    
    // Atomic update of positions (need atomic for memory visibility across workgroups)
    atomic_add_float(&positions[i_x], delta.x);
    atomic_add_float(&positions[i_y], delta.y);
    atomic_add_float(&positions[j_x], -delta.x);
    atomic_add_float(&positions[j_y], -delta.y);
    
    // Release locks
    release_locks(locks, i, j);
}

