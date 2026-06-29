const WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    n: u32,
    embed_dim: u32,
    pair_start: u32,
    pair_len: u32,
    eta: f32,
    min_delta: f32,
    pad0: u32,
    pad1: u32,
}

struct Pair {
    u: u32,
    v: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> pairs: array<Pair>;
@group(0) @binding(3) var<storage, read> pair_weights: array<f32>;
@group(0) @binding(4) var<storage, read> embedding: array<f32>;

fn ideal_distance(u: u32, v: u32) -> f32 {
    var sum = 0.0;
    for (var k = 0u; k < uniforms.embed_dim; k = k + 1u) {
        let du = embedding[u * uniforms.embed_dim + k] - embedding[v * uniforms.embed_dim + k];
        sum = sum + du * du;
    }
    return max(sqrt(sum), uniforms.min_delta);
}

fn fallback_direction(u: u32, v: u32) -> vec2<f32> {
    let hash = ((u % 65535u) * 251u + (v % 65535u) * 997u) % 65535u;
    let angle = (f32(hash) / 65535.0) * 6.28318530718;
    return vec2<f32>(cos(angle), sin(angle));
}

@compute @workgroup_size(256, 1, 1)
fn sparse_sgd(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let local_idx = wg.x * WORKGROUP_SIZE + lid.x;
    if (local_idx >= uniforms.pair_len) {
        return;
    }

    let pair_idx = uniforms.pair_start + local_idx;
    let pair = pairs[pair_idx];
    let weight = pair_weights[pair_idx];
    if (weight <= 0.0) {
        return;
    }

    var diff = positions[pair.v] - positions[pair.u];
    var current = length(diff);
    if (current < 1.0e-12) {
        diff = fallback_direction(pair.u, pair.v) * 1.0e-6;
        current = 1.0e-6;
    }

    let delta = ideal_distance(pair.u, pair.v);
    let r = ((current - delta) / 2.0) * (diff / current);
    let mu = min(weight * uniforms.eta, 1.0);

    positions[pair.u] = positions[pair.u] + mu * r;
    positions[pair.v] = positions[pair.v] - mu * r;
}
