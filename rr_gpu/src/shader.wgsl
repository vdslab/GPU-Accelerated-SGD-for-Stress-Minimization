// Bidirectional Round-Robin GPU SGD カーネル
//
// スケジューリング変更点:
//   旧: cyclic shift j=(g+r_outer)%B → 各ラウンドで各ブロックが2回登場 → 一方向更新のみ可能
//   新: Berger 1-factorization      → 各ラウンドで各ブロックが高々1回 → 両側更新が安全
//
// ケースA (i != j): 異なるブロック間
//   wg_pos_i に i側、wg_pos_j に j側を並列ロード
//   内側ラウンド r_inner (0..lj-1) を逐次実行（サイクルアルゴリズム）
//     thread tx: pair (wg_pos_i[tx], wg_pos_j[(tx+r_inner)%lj])
//     wg_pos_i[tx] += delta  （i側更新）
//     wg_pos_j[b]  -= delta  （j側更新: b=(tx+r_inner)%lj はラウンド内で tx の全単射）
//     → j スロット b を同時に書くスレッドは tx だけ → 競合なし
//   最後に両バッファを positions に書き戻し
//
// ケースB (i == j): 同一ブロック内（変更なし）
//   Berger テーブルによる対称更新

const BLOCK_T: u32 = 1024u;

struct Uniforms {
    n   : u32,
    eta : f32,
    // r_outer / big_b は廃止。tiles バッファから (i,j) を受け取る。
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform>            uniforms   : Uniforms;
@group(0) @binding(1) var<storage, read_write> positions  : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       dist_flat  : array<f32>;
@group(0) @binding(3) var<storage, read>       block_lens : array<u32>;
// binding 4: ラウンドごとのタイル割り当て。tiles[g] = vec2<u32>(i, j)
@group(0) @binding(4) var<storage, read>       tiles      : array<vec2<u32>>;
// binding 5: ケースA内側ループのオフセット順列（長さ T、ラウンドごとにシャッフル）
@group(0) @binding(5) var<storage, read>       inner_perm : array<u32>;

// ワークグループキャッシュ（各 1024×8 = 8KB, 合計 16KB < Metal 上限 32KB）
var<workgroup> wg_pos_i : array<vec2<f32>, 1024>; // i側
var<workgroup> wg_pos_j : array<vec2<f32>, 1024>; // j側

fn sgd_delta(pos_u: vec2<f32>, pos_v: vec2<f32>, d: f32, eta: f32) -> vec2<f32> {
    let diff     = pos_v - pos_u;
    let dist_cur = max(length(diff), 1e-12);
    let r        = ((dist_cur - d) / 2.0) * (diff / dist_cur);
    let w        = 1.0 / (d * d);
    let mu       = min(w * eta, 1.0);
    return mu * r;
}

@compute @workgroup_size(1024, 1, 1)
fn sgd_rr(
    @builtin(workgroup_id)        wg  : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let g  = wg.x;
    // タイル割り当てを tiles バッファから読む
    let tile = tiles[g];
    let i  = tile.x;
    let j  = tile.y;
    let li = block_lens[i];
    let lj = block_lens[j];

    let tx = lid.x; // 0..1023

    // =========================================================
    // ケースA: 異なるブロック間（両側更新）
    // =========================================================
    if i != j {
        // 両側を並列ロード
        if tx < li { wg_pos_i[tx] = positions[i * BLOCK_T + tx]; }
        if tx < lj { wg_pos_j[tx] = positions[j * BLOCK_T + tx]; }
        workgroupBarrier();

        // 内側サイクルアルゴリズム:
        //   r_inner ごとに thread tx は j スロット b=(tx+offset)%lj を担当。
        //   offset = inner_perm[r_inner]（ラウンドごとにシャッフルされた順列）
        //   tx ↦ b は lj 上の全単射 → 同一 r_inner 内に b への書き込みは1スレッドのみ。
        for (var r_inner = 0u; r_inner < lj; r_inner++) {
            if tx < li {
                let offset = inner_perm[r_inner];
                let b   = (tx + offset) % lj;
                let gu  = i * BLOCK_T + tx;
                let gv  = j * BLOCK_T + b;
                let d   = dist_flat[gu * uniforms.n + gv];
                if d >= 0.5 {
                    let delta = sgd_delta(wg_pos_i[tx], wg_pos_j[b], d, uniforms.eta);
                    wg_pos_i[tx] += delta; // i側更新
                    wg_pos_j[b]  -= delta; // j側更新（競合なし）
                }
            }
            workgroupBarrier();
        }

        // 両側書き戻し
        if tx < li { positions[i * BLOCK_T + tx] = wg_pos_i[tx]; }
        if tx < lj { positions[j * BLOCK_T + tx] = wg_pos_j[tx]; }

        return;
    }

    // =========================================================
    // ケースB: 同一ブロック内（Berger テーブル・対称更新、変更なし）
    // =========================================================

    if tx < li {
        wg_pos_i[tx] = positions[i * BLOCK_T + tx];
    }
    workgroupBarrier();

    let L     = li;
    let L_eff : u32 = L + (L % 2u); // 奇数のときダミーを追加して偶数化
    let rounds: u32 = L_eff - 1u;

    for (var r = 0u; r < rounds; r++) {
        var lu: u32 = 0u;
        var lv: u32 = 0u;
        var do_work: bool = false;

        if tx == 0u {
            lu = L_eff - 1u;
            lv = r;
            do_work = true;
        } else if tx < L_eff / 2u {
            lu = (r + tx) % (L_eff - 1u);
            lv = (r + L_eff - 1u - tx) % (L_eff - 1u);
            do_work = true;
        }

        if do_work && lu < L && lv < L {
            let gu    = i * BLOCK_T + lu;
            let gv    = i * BLOCK_T + lv;
            let d     = dist_flat[gu * uniforms.n + gv];
            if d >= 0.5 {
                let delta = sgd_delta(wg_pos_i[lu], wg_pos_i[lv], d, uniforms.eta);
                wg_pos_i[lu] += delta;
                wg_pos_i[lv] -= delta;
            }
        }
        workgroupBarrier();
    }

    if tx < li {
        positions[i * BLOCK_T + tx] = wg_pos_i[tx];
    }
}
