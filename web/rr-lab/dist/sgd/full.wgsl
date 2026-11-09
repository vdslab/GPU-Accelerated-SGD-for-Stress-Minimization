// Two-level Round-robin Full SGD.
// A workgroup owns one tile (two blocks of at most 128 vertices).
// Inner matchings are disjoint, and a barrier separates successive matchings.
const TILE_SIZE: u32 = 128u;
const WORKGROUP_SIZE: u32 = 64u;
const FULL_SELF_OFFSET: u32 = 128u;

struct Parameters {
    vertex_count: u32,
    eta: f32,
    partial_self_offset: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> parameters: Parameters;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> distances: array<f32>;
@group(0) @binding(3) var<storage, read> tiles: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> permutation: array<u32>;

var<workgroup> left_positions: array<vec2<f32>, 128>;
var<workgroup> right_positions: array<vec2<f32>, 128>;
var<workgroup> tile_index: vec2<u32>;

fn displacement(left: vec2<f32>, right: vec2<f32>, desired_distance: f32) -> vec2<f32> {
    let difference = right - left;
    let current = max(length(difference), 1e-12);
    let step = min(parameters.eta / (desired_distance * desired_distance), 1.0);
    return step * ((current - desired_distance) * 0.5) * (difference / current);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if lane == 0u {
        tile_index = tiles[workgroup.x];
    }
    // Broadcast a uniform tile identity so all lanes reach the same barriers.
    let tile = workgroupUniformLoad(&tile_index);
    let left_block = tile.x;
    let right_block = tile.y;
    let left_length = min(TILE_SIZE, parameters.vertex_count - left_block * TILE_SIZE);
    let right_length = min(TILE_SIZE, parameters.vertex_count - right_block * TILE_SIZE);

    for (var local = lane; local < TILE_SIZE; local += WORKGROUP_SIZE) {
        if local < left_length {
            left_positions[local] = positions[left_block * TILE_SIZE + local];
        }
        if local < right_length {
            right_positions[local] = positions[right_block * TILE_SIZE + local];
        }
    }
    workgroupBarrier();

    if left_block != right_block {
        // Cyclic offsets cover every cross-block pair once, skipping padded vertices.
        for (var round = 0u; round < TILE_SIZE; round++) {
            let offset = permutation[round];
            for (var left = lane; left < left_length; left += WORKGROUP_SIZE) {
                let right = (left + offset) % TILE_SIZE;
                if right < right_length {
                    let u = left_block * TILE_SIZE + left;
                    let v = right_block * TILE_SIZE + right;
                    let desired_distance = distances[u * parameters.vertex_count + v];
                    let delta = displacement(left_positions[left], right_positions[right], desired_distance);
                    left_positions[left] += delta;
                    right_positions[right] -= delta;
                }
            }
            workgroupBarrier();
        }
    } else {
        // Circle method for the actual block length, with one dummy when it is odd.
        let effective_length = left_length + left_length % 2u;
        let round_count = effective_length - 1u;
        let offset = select(FULL_SELF_OFFSET, parameters.partial_self_offset, left_length < TILE_SIZE);
        for (var step = 0u; step < round_count; step++) {
            let round = permutation[offset + step];
            for (var slot = lane; slot < effective_length / 2u; slot += WORKGROUP_SIZE) {
                var left = effective_length - 1u;
                var right = round;
                if slot > 0u {
                    left = (round + slot) % round_count;
                    right = (round + round_count - slot) % round_count;
                }
                if left < left_length && right < left_length {
                    let u = left_block * TILE_SIZE + left;
                    let v = left_block * TILE_SIZE + right;
                    let desired_distance = distances[u * parameters.vertex_count + v];
                    let delta = displacement(left_positions[left], left_positions[right], desired_distance);
                    left_positions[left] += delta;
                    left_positions[right] -= delta;
                }
            }
            workgroupBarrier();
        }
    }

    for (var local = lane; local < left_length; local += WORKGROUP_SIZE) {
        positions[left_block * TILE_SIZE + local] = left_positions[local];
    }
    if left_block != right_block {
        for (var local = lane; local < right_length; local += WORKGROUP_SIZE) {
            positions[right_block * TILE_SIZE + local] = right_positions[local];
        }
    }
}
