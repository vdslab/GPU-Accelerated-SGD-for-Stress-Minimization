// ラウンドロビン GPU SGD カーネル（T=1024 最適化版）
//
// ブロックサイズ T=1024 により WG の 1024 スレッドを全て活用する。
// B = ceil(n/1024) となり dispatch 数が T=32 版の約 1/32 に削減される。
//
// ケースA (i != j): 異なるブロック間
//   全1024スレッドで i側ポジションを wg_pos に並列ロード
//   内側ラウンド r_inner (0..lj-1) を順番に処理（サイクルアルゴリズム）
//     ラウンド r_inner: thread tx → pair (wg_pos[tx], positions[j*T+(tx+r_inner)%lj])
//     各ラウンド後に workgroupBarrier()
//   最後に wg_pos を positions に並列書き戻し
//
// ケースB (i == j): 同一ブロック内
//   全スレッドで positions を wg_pos に並列ロード
//   Berger テーブルのサイクルアルゴリズムで対称更新（最大1023ラウンド）
//   ラウンド間は workgroupBarrier() で同期
//   最後に wg_pos を positions に並列書き戻し

const BLOCK_T: u32 = 1024u;

struct Uniforms {
    r_outer : u32,
    n       : u32,
    big_b   : u32,
    eta     : f32,
}

@group(0) @binding(0) var<uniform>            uniforms   : Uniforms;
@group(0) @binding(1) var<storage, read_write> positions  : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       dist_flat  : array<f32>;
@group(0) @binding(3) var<storage, read>       block_lens : array<u32>;

// ケースA・ケースBで排他的に共有する座標キャッシュ（8192 bytes = 8KB）
var<workgroup> wg_pos : array<vec2<f32>, 1024>;

fn sgd_delta(pos_u: vec2<f32>, pos_v: vec2<f32>, d: f32, eta: f32) -> vec2<f32> {
    let diff = pos_v - pos_u;
    let dist_cur = max(length(diff), 1e-12);
    let r = ((dist_cur - d) / 2.0) * (diff / dist_cur);
    let w = 1.0 / (d * d);
    let mu = min(w * eta, 1.0);
    return mu * r;
}

@compute @workgroup_size(1024, 1, 1)
fn sgd_rr(
    @builtin(workgroup_id)        wg  : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let g  = wg.x;
    let i  = g;
    let j  = (g + uniforms.r_outer) % uniforms.big_b;
    let li = block_lens[i];
    let lj = block_lens[j];

    let tx = lid.x;  // 0..1023（全スレッド活動）

    // =========================================================
    // ケースA: 異なるブロック間（一方向更新）
    // =========================================================
    if i != j {
        // --- i側を全1024スレッドで並列ロード ---
        if tx < li {
            wg_pos[tx] = positions[i * BLOCK_T + tx];
        }
        workgroupBarrier();

        // --- 内側ラウンドを順番に実行（サイクルアルゴリズム）---
        // ラウンド r_inner: thread tx → pair (wg_pos[tx], positions[j*T+(tx+r_inner)%lj])
        // 各ラウンド内で node u=i*T+tx を担当するのは thread tx のみ → 競合なし
        // ラウンドをまたいで wg_pos[tx] が更新されるため逐次 SGD と等価
        for (var r_inner = 0u; r_inner < lj; r_inner++) {
            if tx < li {
                let b = (tx + r_inner) % lj;
                let v = j * BLOCK_T + b;
                let d = dist_flat[(i * BLOCK_T + tx) * uniforms.n + v];
                if d >= 0.5 {
                    let delta = sgd_delta(wg_pos[tx], positions[v], d, uniforms.eta);
                    wg_pos[tx] += delta;  // i側のみ更新（一方向）
                }
            }
            workgroupBarrier();  // 各ラウンド後に全スレッドで同期
        }

        // --- 書き戻し: wg_pos → positions ---
        if tx < li {
            positions[i * BLOCK_T + tx] = wg_pos[tx];
        }

        return;
    }

    // =========================================================
    // ケースB: 同一ブロック内（サイクルアルゴリズム・対称更新）
    // =========================================================

    // --- 全スレッドで positions を wg_pos に並列ロード ---
    if tx < li {
        wg_pos[tx] = positions[i * BLOCK_T + tx];
    }
    workgroupBarrier();

    let L     = li;
    // L が奇数のときダミーノードを追加して偶数化
    let L_eff : u32 = L + (L % 2u);
    let rounds: u32 = L_eff - 1u;

    // Berger テーブル方式（最大1023ラウンド）:
    //   固定ノード: L_eff-1
    //   ラウンド r (0..rounds-1):
    //     tx=0:   (L_eff-1, r)
    //     tx=k>0: ((r+k)%(L_eff-1), (r+L_eff-1-k)%(L_eff-1))
    //   1ラウンドあたり最大512スレッドが並列に担当ペアを処理
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

        // lu, lv が有効ノード（ダミーではない）かチェック
        if do_work && lu < L && lv < L {
            let gu = i * BLOCK_T + lu;
            let gv = i * BLOCK_T + lv;
            let d  = dist_flat[gu * uniforms.n + gv];
            if d >= 0.5 {
                let delta = sgd_delta(wg_pos[lu], wg_pos[lv], d, uniforms.eta);
                wg_pos[lu] += delta;
                wg_pos[lv] -= delta;
            }
        }
        workgroupBarrier();
    }

    // --- 全スレッドで wg_pos → positions に並列書き戻し ---
    if tx < li {
        positions[i * BLOCK_T + tx] = wg_pos[tx];
    }
}
