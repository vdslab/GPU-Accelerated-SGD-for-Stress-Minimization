// Port of implementations/sparse/sparse-sgd-gpu/src/shader*.wgsl
// (Phase 1B: shared-memory pivot caching, 20-byte Pair, 4-bank one_sided storage).
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
const BANK_SHIFT: u32 = 24u;
const BANK_MASK: u32 = 0x00ffffffu;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> one_sided_bank0: array<OneSidedEntry>;
@group(0) @binding(3) var<storage, read> pivot_permutation: array<u32>;
@group(0) @binding(4) var<storage, read> pairs: array<Pair>;
@group(0) @binding(5) var<storage, read> one_sided_bank1: array<OneSidedEntry>;
@group(0) @binding(6) var<storage, read> one_sided_bank2: array<OneSidedEntry>;
@group(0) @binding(7) var<storage, read> one_sided_bank3: array<OneSidedEntry>;

// Workgroup shared memory caching for pivot data (4.0 KiB total <= 16.0 KiB limit)
var<workgroup> wg_pivot_perm: array<u32, 256>;
var<workgroup> wg_pivot_id: array<u32, 256>;
var<workgroup> wg_pivot_pos: array<vec2<f32>, 256>;

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

@compute @workgroup_size(128)
fn one_sided_phase(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    for (var i = lid.x; i < uniforms.pivot_count; i += WORKGROUP_SIZE) {
        wg_pivot_perm[i] = pivot_permutation[i];
        let pid = pivot_permutation[uniforms.pivot_count + i];
        wg_pivot_id[i] = pid;
        wg_pivot_pos[i] = positions[pid];
    }
    workgroupBarrier();

    let vertex = gid.x;
    if vertex >= uniforms.n {
        return;
    }
    var local_position = positions[vertex];
    var changed = false;
    for (var k = uniforms.start; k < uniforms.start + uniforms.count; k++) {
        let pivot_index = wg_pivot_perm[shuffle_pivot(k, vertex ^ uniforms.seed ^ (uniforms.iteration * 0x9e3779b9u), uniforms.pivot_count)];
        let entry = get_one_sided_entry(vertex * uniforms.pivot_count + pivot_index);
        if entry.weight > 0.0 {
            let pivot = wg_pivot_id[pivot_index];
            let pivot_position = wg_pivot_pos[pivot_index];
            changed = true;
            let delta = displacement(
                local_position,
                pivot_position,
                entry.dij,
                vertex ^ pivot ^ uniforms.seed ^ uniforms.iteration,
            );
            let mu = min(entry.weight * uniforms.eta, 1.0);
            local_position += mu * delta;
        }
    }
    if changed {
        positions[vertex] = local_position;
    }
}

@compute @workgroup_size(128)
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

fn shuffle_pivot(index: u32, key: u32, count: u32) -> u32 {
    if count <= 1u { return 0u; }
    var remaining = count - 1u;
    var bits = 0u;
    loop {
        if remaining == 0u { break; }
        bits += 1u;
        remaining >>= 2u;
    }
    let mask = (1u << bits) - 1u;
    var value = index;
    loop {
        var left = value >> bits;
        var right = value & mask;
        for (var round = 0u; round < 4u; round++) {
            let next = left ^ (hash32(right ^ key ^ ((round + 1u) * 0x85ebca6bu)) & mask);
            left = right;
            right = next;
        }
        value = (left << bits) | right;
        if value < count { break; }
    }
    return value;
}
