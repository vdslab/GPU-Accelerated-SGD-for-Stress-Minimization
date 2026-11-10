// Sparse Atomic SGD compute kernel for WebGPU (vram-lock-sparse)
// Features: ordered per-vertex atomicExchange spinlocks, zero-barrier rollback,
// atomic RMW position reads/writes, @workgroup_size(1, 1, 1), 2D grid dispatch.

struct Pair {
    u: u32,
    v: u32,
    dij: f32,
    weight_u: f32,
    weight_v: f32,
};

struct Uniforms {
    chunk_start: u32,
    chunk_count: u32,
    epoch: u32,
    eta: f32,
    pair_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> locks: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read_write> updated_count: atomic<u32>;

fn acquire_pair_locks(u: u32, v: u32) -> bool {
    let first = min(u, v);
    let second = max(u, v);
    if (atomicExchange(&locks[first], 1u) != 0u) {
        return false;
    }
    if (atomicExchange(&locks[second], 1u) == 0u) {
        return true;
    }
    atomicExchange(&locks[first], 0u);
    return false;
}

fn release_pair_locks(u: u32, v: u32) {
    let first = min(u, v);
    let second = max(u, v);
    atomicExchange(&locks[second], 0u);
    atomicExchange(&locks[first], 0u);
}

fn update_pair(p: Pair) {
    let u = p.u;
    let v = p.v;

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

    let r = ((norm - p.dij) / (2.0 * norm)) * diff;
    let du = min(uniforms.eta * p.weight_u, 1.0) * r;
    let dv = min(uniforms.eta * p.weight_v, 1.0) * r;

    let new_pos_u = pos_u + du;
    let new_pos_v = pos_v - dv;

    // Write back updated coordinates atomically ensuring L2 cache coherence across workgroups
    atomicExchange(&positions[2u * u], bitcast<u32>(new_pos_u.x));
    atomicExchange(&positions[2u * u + 1u], bitcast<u32>(new_pos_u.y));
    atomicExchange(&positions[2u * v], bitcast<u32>(new_pos_v.x));
    atomicExchange(&positions[2u * v + 1u], bitcast<u32>(new_pos_v.y));
}

@compute @workgroup_size(1, 1, 1)
fn sparse_sgd(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>,
) {
    let linear_idx = group_id.y * num_wgs.x + group_id.x;
    if (linear_idx >= uniforms.chunk_count) {
        return;
    }
    let slot = uniforms.chunk_start + linear_idx;
    if (slot >= uniforms.pair_count) {
        return;
    }

    let p = pairs[slot];
    if (p.u == p.v) {
        atomicAdd(&updated_count, 1u);
        return;
    }

    loop {
        if (acquire_pair_locks(p.u, p.v)) {
            update_pair(p);
            release_pair_locks(p.u, p.v);
            atomicAdd(&updated_count, 1u);
            break;
        }
    }
}
