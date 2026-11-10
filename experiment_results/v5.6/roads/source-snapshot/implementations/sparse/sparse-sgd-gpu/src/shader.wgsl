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
    dij: f32,
    weight: f32,
}

struct Pair {
    u: u32,
    v: u32,
    dij: f32,
    weight_u: f32,
    weight_v: f32,
}

const WORKGROUP_SIZE: u32 = 128u;
const BANK_SHIFT: u32 = 25u;
const BANK_MASK: u32 = 0x01ffffffu;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> one_sided_bank0: array<OneSidedEntry>;
@group(0) @binding(3) var<storage, read> pivot_permutation: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> pairs: array<Pair>;
@group(0) @binding(5) var<storage, read> one_sided_bank1: array<OneSidedEntry>;
@group(0) @binding(6) var<storage, read> one_sided_bank2: array<OneSidedEntry>;
@group(0) @binding(7) var<storage, read> one_sided_bank3: array<OneSidedEntry>;

fn get_one_sided_entry(entry_idx: u32) -> OneSidedEntry {
    let bank_idx = entry_idx >> BANK_SHIFT;
    let local_idx = entry_idx & BANK_MASK;
    switch bank_idx {
        case 0u: { return one_sided_bank0[local_idx]; }
        case 1u: { return one_sided_bank1[local_idx]; }
        case 2u: { return one_sided_bank2[local_idx]; }
        default: { return one_sided_bank3[local_idx]; }
    }
}

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
@compute @workgroup_size(128)
fn one_sided_phase(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
    let vertex = gid.x + gid.y * grid.x * WORKGROUP_SIZE;
    if vertex >= uniforms.n {
        return;
    }
    var local_position = positions[vertex];
    var changed = false;
    let rotation = hash32(vertex ^ uniforms.seed ^ uniforms.iteration) % uniforms.pivot_count;
    var stride = 1u;
    if (uniforms.pad & 1u) != 0u && uniforms.pivot_count > 1u {
        stride = 1u + hash32(vertex ^ uniforms.seed ^ (uniforms.iteration * 0x9e3779b9u)) % (uniforms.pivot_count - 1u);
        loop {
            var a = stride;
            var b = uniforms.pivot_count;
            loop {
                if b == 0u { break; }
                let t = a % b;
                a = b;
                b = t;
            }
            if a == 1u { break; }
            stride = stride % (uniforms.pivot_count - 1u) + 1u;
        }
    }
    let perm_base = (uniforms.pad >> 1u) * uniforms.pivot_count;
    let row_base = vertex * uniforms.pivot_count;
    let salt_base = vertex ^ uniforms.seed ^ uniforms.iteration;
    let end_k = uniforms.start + uniforms.count;
    for (var k = uniforms.start; k < end_k; k++) {
        let step = (k * stride + rotation) % uniforms.pivot_count;
        let p_info = pivot_permutation[perm_base + step];
        let entry = get_one_sided_entry(row_base + p_info.x);
        if entry.weight > 0.0 {
            let pivot = p_info.y;
            let pivot_position = positions[pivot];
            changed = true;
            let delta = displacement(
                local_position,
                pivot_position,
                entry.dij,
                salt_base ^ pivot,
            );
            let mu = min(entry.weight * uniforms.eta, 1.0);
            local_position += mu * delta;
        }
    }
    // Pivot rows have no positive one-sided entries: never write their coordinates.
    if changed {
        positions[vertex] = local_position;
    }
}

@compute @workgroup_size(128)
fn one_sided_phase_single(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
    let vertex = gid.x + gid.y * grid.x * WORKGROUP_SIZE;
    if vertex >= uniforms.n {
        return;
    }
    var local_position = positions[vertex];
    var changed = false;
    let rotation = hash32(vertex ^ uniforms.seed ^ uniforms.iteration) % uniforms.pivot_count;
    var stride = 1u;
    if (uniforms.pad & 1u) != 0u && uniforms.pivot_count > 1u {
        stride = 1u + hash32(vertex ^ uniforms.seed ^ (uniforms.iteration * 0x9e3779b9u)) % (uniforms.pivot_count - 1u);
        loop {
            var a = stride;
            var b = uniforms.pivot_count;
            loop {
                if b == 0u { break; }
                let t = a % b;
                a = b;
                b = t;
            }
            if a == 1u { break; }
            stride = stride % (uniforms.pivot_count - 1u) + 1u;
        }
    }
    let perm_base = (uniforms.pad >> 1u) * uniforms.pivot_count;
    let row_base = vertex * uniforms.pivot_count;
    let salt_base = vertex ^ uniforms.seed ^ uniforms.iteration;
    let end_k = uniforms.start + uniforms.count;
    for (var k = uniforms.start; k < end_k; k++) {
        let step = (k * stride + rotation) % uniforms.pivot_count;
        let p_info = pivot_permutation[perm_base + step];
        let entry = one_sided_bank0[row_base + p_info.x];
        if entry.weight > 0.0 {
            let pivot = p_info.y;
            let pivot_position = positions[pivot];
            changed = true;
            let delta = displacement(
                local_position,
                pivot_position,
                entry.dij,
                salt_base ^ pivot,
            );
            let mu = min(entry.weight * uniforms.eta, 1.0);
            local_position += mu * delta;
        }
    }
    // Pivot rows have no positive one-sided entries: never write their coordinates.
    if changed {
        positions[vertex] = local_position;
    }
}

// The host guarantees that each dispatched slice is a matching, so both endpoint
// writes are race-free without atomics or locks.
@compute @workgroup_size(128)
fn two_sided_round(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
    let flat_id = gid.x + gid.y * grid.x * WORKGROUP_SIZE;
    if flat_id >= uniforms.count {
        return;
    }
    let pair_index = uniforms.start + flat_id;
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

