#include <metal_stdlib>
using namespace metal;

struct SparsePair {
    uint u;
    uint v;
    float dij;
    float weight_u;
    float weight_v;
};

inline bool try_lock(device atomic_uint* locks, uint node) {
    uint expected = 0u;
    return atomic_compare_exchange_weak_explicit(
        &locks[node], &expected, 1u, memory_order_relaxed, memory_order_relaxed);
}
inline void unlock(device atomic_uint* locks, uint node) {
    atomic_store_explicit(&locks[node], 0u, memory_order_relaxed);
}

kernel void sparse_sgd(
    device atomic<float>* positions [[buffer(0)]],
    device const SparsePair* pairs [[buffer(1)]],
    device atomic_uint* locks [[buffer(2)]],
    device const uint* pending [[buffer(3)]],
    device uint* completed [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant float& eta [[buffer(6)]],
    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    // Keep native atomic SGD's one active lane per 32-thread group.
    uint slot = group_id.y * 65535u + group_id.x;
    if (local_id.x != 0u || slot >= count) return;
    SparsePair pair = pairs[pending[slot]];
    uint first = min(pair.u, pair.v);
    uint second = max(pair.u, pair.v);
    bool acquired = false;
    // Bound spinning. The host resubmits only unfinished pairs after GPU completion.
    for (uint retry = 0u; retry < 128u; ++retry) {
        if (try_lock(locks, first)) {
            if (try_lock(locks, second)) {
                acquired = true;
                break;
            }
            unlock(locks, first);
        }
    }
    if (!acquired) return;

    float2 a = float2(
        atomic_load_explicit(&positions[pair.u * 2u], memory_order_relaxed),
        atomic_load_explicit(&positions[pair.u * 2u + 1u], memory_order_relaxed));
    float2 b = float2(
        atomic_load_explicit(&positions[pair.v * 2u], memory_order_relaxed),
        atomic_load_explicit(&positions[pair.v * 2u + 1u], memory_order_relaxed));
    float2 diff = b - a;
    float norm = length(diff);
    if (norm < 1e-12f) {
        // Use native atomic SGD's deterministic fallback; CPU uses a random angle.
        diff = float2(1e-6f, 1e-6f);
        norm = length(diff);
    }
    float2 r = ((norm - pair.dij) / (2.0f * norm)) * diff;
    float2 du = min(eta * pair.weight_u, 1.0f) * r;
    float2 dv = min(eta * pair.weight_v, 1.0f) * r;
    atomic_fetch_add_explicit(&positions[pair.u * 2u], du.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&positions[pair.u * 2u + 1u], du.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&positions[pair.v * 2u], -dv.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&positions[pair.v * 2u + 1u], -dv.y, memory_order_relaxed);
    unlock(locks, second);
    unlock(locks, first);
    completed[slot] = 1u;
}
