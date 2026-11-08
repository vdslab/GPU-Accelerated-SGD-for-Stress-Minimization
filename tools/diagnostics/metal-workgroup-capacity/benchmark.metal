#include <metal_stdlib>
using namespace metal;

struct Params {
    uint iterations;
    uint tile;
    uint rounds;
    uint stride;
    float a;
    float b;
    float eta;
    float d;
};

// One dependent recurrence per lane; and four independent recurrences per lane.
// Runtime arguments prevent folding the recurrence away. Explicit FMA and
// disabled fast math keep the arithmetic independent of dispatch geometry.
kernel void fma1(device const float4 *initial [[buffer(0)]],
                 device float4 *output [[buffer(1)]],
                 constant Params &p [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    float x = initial[gid].x;
    for (uint i = 0; i < p.iterations; ++i) x = fma(x, p.a, p.b);
    output[gid] = float4(x, 0, 0, 0);
}

kernel void fma4(device const float4 *initial [[buffer(0)]],
                 device float4 *output [[buffer(1)]],
                 constant Params &p [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    float4 x = initial[gid];
    for (uint i = 0; i < p.iterations; ++i) x = fma(x, float4(p.a), float4(p.b));
    output[gid] = x;
}

// Cross-tile matching, the same two-endpoint update formula and a threadgroup
// barrier after each matching. Groups have independent writable coordinates.
// All groups deliberately reuse the same read-only inputs. This is an on-chip
// scheduling/synchronization probe, not a substitute for whole-graph timing.
template<bool matrix_distance>
void rr_body(device const float4 *initial, device float4 *output,
             constant Params &p, device const float *distances,
             threadgroup float2 *coordinates,
             uint group, uint tx, uint width) {
    for (uint local = tx; local < p.tile; local += width) {
        coordinates[local] = initial[local].xy;
        coordinates[p.tile + local] = initial[local].zw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint round = 0; round < p.rounds; ++round) {
        for (uint s = 0; s < p.tile; ++s) {
            uint offset = (s * 17u + 3u) % p.tile;
            for (uint local = tx; local < p.tile; local += width) {
                uint other = (local + offset) % p.tile;
                float d = matrix_distance ? distances[local * p.stride + p.tile + other] : p.d;
                float2 a = coordinates[local];
                float2 b = coordinates[p.tile + other];
                float2 diff = b - a;
                float current = max(length(diff), 1e-12f);
                float2 r = ((current - d) / 2.0f) * (diff / current);
                float mu = min(p.eta / (d * d), 1.0f);
                float2 delta = mu * r;
                coordinates[local] = a + delta;
                coordinates[p.tile + other] = b - delta;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint local = tx; local < p.tile; local += width) {
        output[group * p.tile + local] = float4(coordinates[local], coordinates[p.tile + local]);
    }
}

kernel void rr_uniform(device const float4 *initial [[buffer(0)]],
                       device float4 *output [[buffer(1)]],
                       constant Params &p [[buffer(2)]],
                       device const float *distances [[buffer(3)]],
                       threadgroup float2 *coordinates [[threadgroup(0)]],
                       uint group [[threadgroup_position_in_grid]],
                       uint tx [[thread_index_in_threadgroup]],
                       uint width [[threads_per_threadgroup]]) {
    rr_body<false>(initial, output, p, distances, coordinates, group, tx, width);
}

kernel void rr_matrix(device const float4 *initial [[buffer(0)]],
                      device float4 *output [[buffer(1)]],
                      constant Params &p [[buffer(2)]],
                      device const float *distances [[buffer(3)]],
                      threadgroup float2 *coordinates [[threadgroup(0)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint tx [[thread_index_in_threadgroup]],
                      uint width [[threads_per_threadgroup]]) {
    rr_body<true>(initial, output, p, distances, coordinates, group, tx, width);
}
