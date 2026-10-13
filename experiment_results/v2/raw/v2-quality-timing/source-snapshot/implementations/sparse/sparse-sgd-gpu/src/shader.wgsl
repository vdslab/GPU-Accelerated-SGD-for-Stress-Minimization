struct Uniforms {
    n: u32,
    start: u32,
    count: u32,
    iteration: u32,
    eta: f32,
    seed: u32,
    pivot_count: u32,
    pad: u32,
}

struct OneSidedEntry {
    pivot: u32,
    dij: f32,
    weight: f32,
    pad: u32,
}

struct Pair {
    u: u32,
    v: u32,
    dij: f32,
    weight_u: f32,
    weight_v: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> one_sided: array<OneSidedEntry>;
@group(0) @binding(3) var<storage, read> pivot_permutation: array<u32>;
@group(0) @binding(4) var<storage, read> pairs: array<Pair>;

fn hash32(value: u32) -> u32 {
    var x = value;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    return (x >> 16u) ^ x;
}

fn safe_difference(pos_u: vec2<f32>, pos_v: vec2<f32>, salt: u32) -> vec2<f32> {
    let difference = pos_v - pos_u;
    if length(difference) >= 1e-12 {
        return difference;
    }
    let angle = f32(hash32(salt) % 6283u) * 0.001;
    return vec2<f32>(cos(angle), sin(angle)) * 1e-6;
}

fn displacement(pos_u: vec2<f32>, pos_v: vec2<f32>, dij: f32, salt: u32) -> vec2<f32> {
    let difference = safe_difference(pos_u, pos_v, salt);
    let norm = length(difference);
    return ((norm - dij) / (2.0 * norm)) * difference;
}

// One invocation owns one non-pivot vertex. It is the only writer of that vertex;
// pivots are read-only throughout this dispatch.
@compute @workgroup_size(256)
fn one_sided_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vertex = gid.x;
    if vertex >= uniforms.n {
        return;
    }
    var local_position = positions[vertex];
    let rotation = hash32(vertex ^ uniforms.seed ^ uniforms.iteration) % uniforms.pivot_count;
    for (var k = 0u; k < uniforms.pivot_count; k++) {
        let pivot_index = pivot_permutation[(k + rotation) % uniforms.pivot_count];
        let entry = one_sided[vertex * uniforms.pivot_count + pivot_index];
        if entry.weight > 0.0 {
            let pivot_position = positions[entry.pivot];
            let delta = displacement(
                local_position,
                pivot_position,
                entry.dij,
                vertex ^ entry.pivot ^ uniforms.seed ^ uniforms.iteration,
            );
            let mu = min(entry.weight * uniforms.eta, 1.0);
            local_position += mu * delta;
        }
    }
    positions[vertex] = local_position;
}

// The host guarantees that each dispatched slice is a matching, so both endpoint
// writes are race-free without atomics or locks.
@compute @workgroup_size(256)
fn two_sided_round(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= uniforms.count {
        return;
    }
    let pair_index = uniforms.start + gid.x;
    let pair = pairs[pair_index];
    let pos_u = positions[pair.u];
    let pos_v = positions[pair.v];
    let delta = displacement(
        pos_u,
        pos_v,
        pair.dij,
        pair.u ^ pair.v ^ uniforms.seed ^ uniforms.iteration,
    );
    let mu_u = min(pair.weight_u * uniforms.eta, 1.0);
    let mu_v = min(pair.weight_v * uniforms.eta, 1.0);
    positions[pair.u] = pos_u + mu_u * delta;
    positions[pair.v] = pos_v - mu_v * delta;
}

