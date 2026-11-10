// Two-level Round-robin Full SGD (Phase 1A: lossless u16 4-bank distance storage).
// A workgroup owns one tile (two blocks of at most 128 vertices).
// Inner matchings are disjoint, and a barrier separates successive matchings.
const BLOCK_T: u32 = 128u;
const WORKGROUP_SIZE: u32 = 64u;
const BANK_SHIFT: u32 = 25u;
const BANK_MASK: u32 = 0x01ffffffu;

struct Uniforms {
    n: u32,
    eta: f32,
    full_self_perm_offset: u32,
    partial_self_perm_offset: u32,
    chunk: u32,
    chunks: u32,
    tiles_offset: u32,
    perm_offset: u32,
    subround_start: u32,
    subround_end: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> dist_bank0: array<u32>;
@group(0) @binding(3) var<storage, read> block_lens: array<u32>;
@group(0) @binding(4) var<storage, read> tiles: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> inner_perm: array<u32>;
@group(0) @binding(6) var<storage, read> dist_bank1: array<u32>;
@group(0) @binding(7) var<storage, read> dist_bank2: array<u32>;
@group(0) @binding(8) var<storage, read> dist_bank3: array<u32>;

var<workgroup> wg_pos_i: array<vec2<f32>, 128>;
var<workgroup> wg_pos_j: array<vec2<f32>, 128>;
var<workgroup> wg_tile: vec2<u32>;

fn get_dist(u: u32, v: u32) -> f32 {
    let flat_idx = u * uniforms.n + v;
    let word_idx = flat_idx >> 1u;
    let bank_idx = word_idx >> BANK_SHIFT;
    let local_idx = word_idx & BANK_MASK;
    var word = 0u;
    switch bank_idx {
        case 0u: { word = dist_bank0[local_idx]; }
        case 1u: { word = dist_bank1[local_idx]; }
        case 2u: { word = dist_bank2[local_idx]; }
        default: { word = dist_bank3[local_idx]; }
    }
    let shift = (flat_idx & 1u) * 16u;
    return f32((word >> shift) & 0xffffu);
}

fn displacement(left: vec2<f32>, right: vec2<f32>, desired_distance: f32) -> vec2<f32> {
    let difference = right - left;
    let current = max(length(difference), 1e-12);
    let step = min(uniforms.eta / (desired_distance * desired_distance), 1.0);
    return step * ((current - desired_distance) * 0.5) * (difference / current);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if lane == 0u {
        wg_tile = tiles[uniforms.tiles_offset + workgroup.x];
    }
    let tile = workgroupUniformLoad(&wg_tile);
    let i = tile.x;
    let j = tile.y;
    let li = block_lens[i];
    let lj = block_lens[j];

    for (var local = lane; local < BLOCK_T; local += WORKGROUP_SIZE) {
        if local < li {
            wg_pos_i[local] = positions[i * BLOCK_T + local];
        }
        if local < lj {
            wg_pos_j[local] = positions[j * BLOCK_T + local];
        }
    }
    workgroupBarrier();

    if i != j {
        for (var round = uniforms.subround_start; round < uniforms.subround_end; round++) {
            let offset = inner_perm[uniforms.perm_offset + round];
            for (var left = lane; left < li; left += WORKGROUP_SIZE) {
                let right = (left + offset) % BLOCK_T;
                if right < lj {
                    let u = i * BLOCK_T + left;
                    let v = j * BLOCK_T + right;
                    let d = get_dist(u, v);
                    if d >= 0.5 {
                        let delta = displacement(wg_pos_i[left], wg_pos_j[right], d);
                        wg_pos_i[left] += delta;
                        wg_pos_j[right] -= delta;
                    }
                }
            }
            workgroupBarrier();
        }
    } else {
        let effective_length = li + li % 2u;
        let round_count = effective_length - 1u;
        let offset = uniforms.perm_offset + select(uniforms.full_self_perm_offset, uniforms.partial_self_perm_offset, li < BLOCK_T);
        for (var step = uniforms.chunk * round_count / uniforms.chunks; step < (uniforms.chunk + 1u) * round_count / uniforms.chunks; step++) {
            let round = inner_perm[offset + step];
            for (var slot = lane; slot < effective_length / 2u; slot += WORKGROUP_SIZE) {
                var left = effective_length - 1u;
                var right = round;
                if slot > 0u {
                    left = (round + slot) % round_count;
                    right = (round + round_count - slot) % round_count;
                }
                if left < li && right < li {
                    let u = i * BLOCK_T + left;
                    let v = i * BLOCK_T + right;
                    let d = get_dist(u, v);
                    if d >= 0.5 {
                        let delta = displacement(wg_pos_i[left], wg_pos_i[right], d);
                        wg_pos_i[left] += delta;
                        wg_pos_i[right] -= delta;
                    }
                }
            }
            workgroupBarrier();
        }
    }

    for (var local = lane; local < li; local += WORKGROUP_SIZE) {
        positions[i * BLOCK_T + local] = wg_pos_i[local];
    }
    if i != j {
        for (var local = lane; local < lj; local += WORKGROUP_SIZE) {
            positions[j * BLOCK_T + local] = wg_pos_j[local];
        }
    }
}
