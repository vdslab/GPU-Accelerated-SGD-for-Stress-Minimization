// Two-level round robin: one workgroup owns a tile, one matching at a time.
// Host substitutes T, W and the shared/global coordinate helpers.
const BLOCK_T: u32 = __BLOCK_T__u;
const WORKGROUP_SIZE: u32 = __WORKGROUP_SIZE__u;

struct Uniforms {
    n: u32,
    eta: f32,
    full_self_perm_offset: u32,
    partial_self_perm_offset: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> dist_flat: array<f32>;
@group(0) @binding(3) var<storage, read> block_lens: array<u32>;
@group(0) @binding(4) var<storage, read> tiles: array<vec2<u32>>;
// Packed exact permutations: cross T offsets, full self rounds, partial self rounds.
@group(0) @binding(5) var<storage, read> inner_perm: array<u32>;

__MEMORY_HELPERS__

fn sgd_delta(pos_u: vec2<f32>, pos_v: vec2<f32>, d: f32, eta: f32) -> vec2<f32> {
    let diff = pos_v - pos_u;
    let dist_cur = max(length(diff), 1e-12);
    let r = ((dist_cur - d) / 2.0) * (diff / dist_cur);
    let w = 1.0 / (d * d);
    let mu = min(w * eta, 1.0);
    return mu * r;
}

@compute @workgroup_size(__WORKGROUP_SIZE__, 1, 1)
fn sgd_rr(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tile = tiles[wg.x];
    let i = tile.x;
    let j = tile.y;
    let li = block_lens[i];
    let lj = block_lens[j];
    let tx = lid.x;

    if i != j {
        __INIT_CROSS__
        // Both sides are padded to T. The cyclic offset is a bijection on T,
        // so every subround is a disjoint matching even for a partial last tile.
        // Skipping dummies still visits every real cross pair exactly once.
        for (var s = 0u; s < BLOCK_T; s++) {
            let offset = inner_perm[s];
            for (var local = tx; local < li; local += WORKGROUP_SIZE) {
                let other = (local + offset) % BLOCK_T;
                if other < lj {
                    let gu = i * BLOCK_T + local;
                    let gv = j * BLOCK_T + other;
                    let d = dist_flat[gu * uniforms.n + gv];
                    if d >= 0.5 {
                        let pos_u = read_i(i, local);
                        let pos_v = read_j(j, other);
                        let delta = sgd_delta(pos_u, pos_v, d, uniforms.eta);
                        write_i(i, local, pos_u + delta);
                        write_j(j, other, pos_v - delta);
                    }
                }
            }
            __MEMORY_BARRIER__
        }
        __FLUSH_CROSS__
        return;
    }

    __INIT_SELF__
    // Circle method for each actual block length, adding one dummy if odd.
    // Its own exact permutation avoids the old prefix-modulo repeats/omissions.
    if li > 1u {
        let effective_len = li + li % 2u;
        let rounds = effective_len - 1u;
        let perm_offset = select(uniforms.full_self_perm_offset,
                                 uniforms.partial_self_perm_offset, li < BLOCK_T);
        for (var s = 0u; s < rounds; s++) {
            let r = inner_perm[perm_offset + s];
            for (var slot = tx; slot < effective_len / 2u; slot += WORKGROUP_SIZE) {
                var lu = effective_len - 1u;
                var lv = r;
                if slot > 0u {
                    lu = (r + slot) % rounds;
                    lv = (r + rounds - slot) % rounds;
                }
                if lu < li && lv < li {
                    let gu = i * BLOCK_T + lu;
                    let gv = i * BLOCK_T + lv;
                    let d = dist_flat[gu * uniforms.n + gv];
                    if d >= 0.5 {
                        let pos_u = read_i(i, lu);
                        let pos_v = read_i(i, lv);
                        let delta = sgd_delta(pos_u, pos_v, d, uniforms.eta);
                        write_i(i, lu, pos_u + delta);
                        write_i(i, lv, pos_v - delta);
                    }
                }
            }
            __MEMORY_BARRIER__
        }
    }
    __FLUSH_SELF__
}
