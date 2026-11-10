// Sparse Atomic SGD compute kernel for WebGPU / D3D12 (RTX 3050)
// Features: atomicExchange spinlocks, rollback on 2nd lock failure,
// storageBarrier visibility, @workgroup_size(1, 1, 1), 2D grid dispatch.

struct Uniforms {
    chunk_start: u32,
    chunk_count: u32,
    epoch: u32,
    eta: f32,
    pair_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Pair {
    u: u32,
    v: u32,
    dij: f32,
    weight_u: f32,
    weight_v: f32,
}

@group(0) @binding(0) var<storage, read_write> positions: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> locks: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> pending: array<u32>;
@group(0) @binding(4) var<storage, read_write> completed: array<u32>;
@group(0) @binding(5) var<uniform> uniforms: Uniforms;
@group(0) @binding(6) var<storage, read_write> incomplete_counts: array<atomic<u32>>;

fn try_lock(node: u32) -> bool {
    let old = atomicExchange(&locks[node], 1u);
    if (old == 0u) {
        storageBarrier();
        return true;
    }
    return false;
}

fn unlock(node: u32) {
    storageBarrier();
    atomicExchange(&locks[node], 0u);
}

fn acquire_locks(first: u32, second: u32) {
    // Retry loop until both locks are acquired.
    // Prohibits MAX_LOCK_ATTEMPTS capping to prevent Selective Deferral Bias.
    loop {
        if (try_lock(first)) {
            if (try_lock(second)) {
                return; // Both locks acquired successfully
            }
            // Immediate rollback: release first lock before retrying
            unlock(first);
        }
    }
}

@compute @workgroup_size(1, 1, 1)
fn sparse_sgd(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let slot_in_chunk = group_id.y * num_workgroups.x + group_id.x;
    if (slot_in_chunk >= uniforms.chunk_count) {
        return;
    }

    let slot = uniforms.chunk_start + slot_in_chunk;
    let pair_idx = pending[slot];
    let pair = pairs[pair_idx];

    let u = pair.u;
    let v = pair.v;
    if (u == v) {
        completed[slot] = uniforms.epoch;
        return;
    }

    let first = min(u, v);
    let second = max(u, v);

    // Acquire locks in canonical index order
    acquire_locks(first, second);

    // Read positions atomically via atomicAdd(0u) ensuring L2 cache coherence across workgroups
    let u_x = atomicAdd(&positions[2u * u], 0u);
    let u_y = atomicAdd(&positions[2u * u + 1u], 0u);
    let pos_u = vec2<f32>(bitcast<f32>(u_x), bitcast<f32>(u_y));

    let v_x = atomicAdd(&positions[2u * v], 0u);
    let v_y = atomicAdd(&positions[2u * v + 1u], 0u);
    let pos_v = vec2<f32>(bitcast<f32>(v_x), bitcast<f32>(v_y));

    var diff = pos_v - pos_u;
    var norm = length(diff);

    if (norm < 1e-12) {
        // Deterministic tiny fallback matching native Atomic SGD
        diff = vec2<f32>(1e-6, 1e-6);
        norm = length(diff);
    }

    let r = ((norm - pair.dij) / (2.0 * norm)) * diff;
    let du = min(uniforms.eta * pair.weight_u, 1.0) * r;
    let dv = min(uniforms.eta * pair.weight_v, 1.0) * r;

    let new_pos_u = pos_u + du;
    let new_pos_v = pos_v - dv;

    // Write back updated coordinates atomically ensuring L2 cache coherence across workgroups
    atomicExchange(&positions[2u * u], bitcast<u32>(new_pos_u.x));
    atomicExchange(&positions[2u * u + 1u], bitcast<u32>(new_pos_u.y));
    atomicExchange(&positions[2u * v], bitcast<u32>(new_pos_v.x));
    atomicExchange(&positions[2u * v + 1u], bitcast<u32>(new_pos_v.y));

    // Release locks in reverse order
    unlock(second);
    unlock(first);

    // Record completion epoch
    completed[slot] = uniforms.epoch;
}

@compute @workgroup_size(256)
fn verify_epoch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.pair_count) {
        return;
    }
    if (completed[idx] != uniforms.epoch) {
        atomicAdd(&incomplete_counts[uniforms.epoch - 1u], 1u);
    }
}
