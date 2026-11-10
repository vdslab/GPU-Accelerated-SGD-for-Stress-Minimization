// Full Atomic SGD compute kernel for WebGPU / D3D12 (RTX 3050)
// Features: atomicExchange spinlocks, rollback on 2nd lock failure,
// storageBarrier visibility, @workgroup_size(1, 1, 1), 2D grid dispatch.

struct EdgeInfo {
    u: u32,
    v: u32,
    dij: f32,
    wij: f32,
}

struct Uniforms {
    iteration: u32,
    num_pairs: u32,
    chunk_offset: u32,
    chunk_size: u32,
}

@group(0) @binding(0)
var<storage, read> etas: array<f32>;

@group(0) @binding(1)
var<storage, read_write> positions: array<atomic<u32>>;

@group(0) @binding(2)
var<storage, read> pairs: array<EdgeInfo>;

@group(0) @binding(3)
var<storage, read_write> locks: array<atomic<u32>>;

@group(0) @binding(4)
var<storage, read_write> updated_count: atomic<u32>;

@group(0) @binding(5)
var<uniform> uniforms: Uniforms;

fn try_lock(node: u32) -> bool {
    let acquired = atomicExchange(&locks[node], 1u) == 0u;
    if (acquired) {
        storageBarrier();
    }
    return acquired;
}

fn unlock(node: u32) {
    storageBarrier();
    atomicExchange(&locks[node], 0u);
}

fn acquire_locks(node1: u32, node2: u32) {
    let first = min(node1, node2);
    let second = max(node1, node2);
    loop {
        if (try_lock(first)) {
            if (try_lock(second)) {
                return;
            }
            unlock(first);
        }
    }
}

fn release_locks(node1: u32, node2: u32) {
    let first = min(node1, node2);
    let second = max(node1, node2);
    unlock(second);
    unlock(first);
}

@compute @workgroup_size(1, 1, 1)
fn sgd(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let local_pair_idx = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    if (local_pair_idx >= uniforms.chunk_size) {
        return;
    }
    let pair_idx = uniforms.chunk_offset + local_pair_idx;
    if (pair_idx >= uniforms.num_pairs) {
        return;
    }

    let pair = pairs[pair_idx];
    let i = pair.u;
    let j = pair.v;

    if (i >= j) {
        return;
    }

    let dij = pair.dij;
    let wij = pair.wij;
    let eta = etas[uniforms.iteration];

    acquire_locks(i, j);

    // Read positions atomically via atomicAdd(0u) ensuring L2 cache coherence across workgroups
    let i_x = atomicAdd(&positions[2u * i], 0u);
    let i_y = atomicAdd(&positions[2u * i + 1u], 0u);
    let pos_i = vec2<f32>(bitcast<f32>(i_x), bitcast<f32>(i_y));

    let j_x = atomicAdd(&positions[2u * j], 0u);
    let j_y = atomicAdd(&positions[2u * j + 1u], 0u);
    let pos_j = vec2<f32>(bitcast<f32>(j_x), bitcast<f32>(j_y));

    let tiny = 1e-12f;
    var diff = pos_j - pos_i;
    var dist = length(diff);

    if (dist < tiny) {
        diff = vec2<f32>(1e-6f, 1e-6f);
        dist = length(diff);
    }

    let r = ((dist - dij) * 0.5f) * (diff / dist);
    let mu = min(wij * eta, 1.0f);
    let delta = mu * r;

    let new_pos_i = pos_i + delta;
    let new_pos_j = pos_j - delta;

    // Write back updated coordinates atomically ensuring L2 cache coherence across workgroups
    atomicExchange(&positions[2u * i], bitcast<u32>(new_pos_i.x));
    atomicExchange(&positions[2u * i + 1u], bitcast<u32>(new_pos_i.y));
    atomicExchange(&positions[2u * j], bitcast<u32>(new_pos_j.x));
    atomicExchange(&positions[2u * j + 1u], bitcast<u32>(new_pos_j.y));

    release_locks(i, j);

    atomicAdd(&updated_count, 1u);
}
