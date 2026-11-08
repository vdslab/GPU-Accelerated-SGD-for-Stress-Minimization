"""Summarize controlled Full-RR tile experiments; excludes warmups/profiling."""
import collections
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiment_results/validation/rr-tile-cause-20261107"
if len(sys.argv) > 1:
    OUT = Path(sys.argv[1]).resolve()


def read(name):
    return [json.loads(line) for line in (OUT / name).read_text().splitlines()]


def summarize(rows, keys):
    groups = collections.defaultdict(list)
    for row in rows:
        if not row["warmup"]:
            groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, runs in sorted(groups.items()):
        row = dict(zip(keys, key))
        row["runs"] = len(runs)
        for metric in ["wall_ms", "gpu_ms"]:
            values = [r[metric] for r in runs if r[metric] is not None]
            if values:
                row[metric] = statistics.median(values)
                row[metric + "_min"] = min(values)
                row[metric + "_max"] = max(values)
        if "updates" in runs[0]:
            row["updates"] = runs[0]["updates"]
            row["ns_per_update"] = row["gpu_ms"] * 1e6 / row["updates"]
        result.append(row)
    return result


real = {}
for graph in ["3elt", "fe_4elt2"]:
    rows = read(graph + ".jsonl")
    assert len(rows) == 100
    hashes = collections.defaultdict(set)
    for row in rows:
        hashes[row["tile"]].add(row["position_sha"])
    assert all(len(values) == 1 for values in hashes.values())
    real[graph] = summarize(rows, ["tile", "workgroup", "capacity", "memory"])
micro_rows = read("micro-controls.jsonl")
assert len(micro_rows) == 360
hashes = collections.defaultdict(set)
for row in micro_rows:
    hashes[row["tile"]].add(row["position_sha"])
assert all(len(values) == 1 for values in hashes.values())
micro = summarize(micro_rows, ["tile", "replicas", "distance_mode"])
no_timestamp = summarize(read("3elt-no-timestamps.jsonl"),
                         ["tile", "workgroup", "capacity", "memory"])
summary = {"real": real, "micro": micro, "no_timestamp": no_timestamp}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def baseline(rows):
    return [r for r in rows if r["workgroup"] == 1024
            and r["tile"] == r["capacity"] and r["memory"] == "shared"]


lines = ["# RR Full：128〜256付近が速い原因の切り分け", "",
"2026-11-07。Apple M4 Pro、GPU 20 cores（system_profilerで確認）、Metal、release。標準の更新コードは変更せず、専用exampleで比較した。", "",
"## 結論", "",
"大きいTで利用可能なworkgroupが減ることによるスループット低下を、独立レプリカの追加で確認した。さらに、内側でも共有メモリに載っていない距離行列の読み出し配置が大きく効く。単に共有メモリ使用量が増えるという説明だけでは結果を説明できない。", "",
"ユーザーの『総更新数と外側×内側のステップ数は同じで、T増加は座標の再利用を増やす』という理解は正しい。ただし同じ総仕事量でも、コアへの分配と距離読み出しの処理効率は同じにはならない。", "",
"## 実グラフでの再現", "",
"各設定はseed10、15反復、内外shuffleあり、phase_chunks=1。25設定×（warmup1回＋測定3回）。毎周設定順を固定乱数でシャッフル。時間は15反復合計の中央値。GPU時間はcompute passのtimestamp合計、wallは標準実装のSGD区間。前処理・pipeline構築はwall区間外。", "",
"| graph | T | cross WG/round | GPU ms | wall ms |", "|---|---:|---:|---:|---:|"]
for graph, rows in real.items():
    n = read(graph + ".jsonl")[0]["nodes"]
    for r in baseline(rows):
        lines.append(f"| {graph} (n={n}) | {r['tile']} | {(n+r['tile']-1)//r['tile']//2} | {r['gpu_ms']:.3f} | {r['wall_ms']:.3f} |")
lines += ["", "両入力でT1024の悪化はGPU時間にも現れた。CPUのsubmit費用だけによる曲線ではない。今回の再測定は元のv5測定とは別集計で、同じ最速T・倍率の完全再現を保証しない。", "",
"### Wと共有メモリの比較", "",
"T固定のW32/128/256/1024比較ではWを減らすだけではT1024は回復せず、むしろ悪化。W<Tでは各スレッドの担当ペア数も増えるため、これは純粋なoccupancyのみの介入ではない。", "",
"| graph | T | memory | declared shared capacity | GPU ms |", "|---|---:|---|---:|---:|"]
for graph, rows in real.items():
    for r in rows:
        if r["tile"] in [128, 256, 1024] and r["workgroup"] == 1024:
            lines.append(f"| {graph} | {r['tile']} | {r['memory']} | {r['capacity'] if r['memory']=='shared' else 0} | {r['gpu_ms']:.3f} |")
lines += ["", "T128/256の配列宣言容量だけ1024に増やすと小幅なGPU時間増加があったが、T1024の全悪化には届かない。宣言容量は実際の物理メモリ常駐量を直接測った値ではない。global版はworkgroup配列を使わないがstorageBarrierに変わるため、共有メモリだけを除く完全な単変量比較ではない。", "",
"## 独立レプリカによる因果介入", "",
"n8192、crossタイルだけの1ラウンドを8回dispatch。W1024。全距離d=1、同じ初期座標、固定シャッフル。各レプリカの書込先座標は独立、距離行列は読み取り専用で共用。各設定warmup1回＋測定5回。レプリカの追加は1タイル内の処理・同期・共有メモリ容量を変えず、配れる仕事量を増やす。", "",
"| T | distance source | replicas | WG/dispatch | updates | GPU ms | ns/update |", "|---:|---|---:|---:|---:|---:|---:|"]
for r in micro:
    if r["tile"] == 1024 and r["distance_mode"] in ["matrix", "uniform", "packed"]:
        lines.append(f"| {r['tile']} | {r['distance_mode']} | {r['replicas']} | {8192//r['tile']//2*r['replicas']} | {r['updates']:,} | {r['gpu_ms']:.3f} | {r['ns_per_update']:.4f} |")
lines += ["", "uniform（実行時uniformからd=1を読む）では、T1024で1→4レプリカにすると更新数は4倍でもGPU時間はほぼ一定。距離行列のレプリカ間キャッシュ再利用では説明できず、この条件に未使用の処理能力があったことを支持する。ただしコアごとの稼働率を測った値ではない。", "",
"## 距離行列へのアクセス", "",
"共有メモリに置くのは座標2T個で、距離dは内側の更新ごとにdist_flat[gu*n+gv]から読み出している。隣接localでguもgvも変わるので、通常のcrossタイルでは隣接スレッドの読み出しは概ねn+1要素離れる（循環する境界を除く）。これは連続アクセスではない。", "",
"matrixは標準の行列アドレス。packedはcrossタイル・offset・local順に並べた配置に相当し、隣接localが連続した距離を読む。どちらも実行時のstorage bufferから同じd=1を同数読み、更新式・出力座標は同じ。uniformは距離のstorage読み出しを除く対照。constantも測ったが定数畳み込みで演算まで減るため、主な根拠には使わない。", "",
"T1024・1レプリカでmatrix→packedだけでも大きく改善。したがってworkgroup数と共有メモリ容量だけを見た説明では不十分。packed/uniformでもレプリカ増加の効果が残る。分配不足とメモリアクセスの影響は相互作用するので、速度比を足して寄与率にしない。", "",
"## timestamp計測の影響", "",
"3eltの標準設定だけをtimestamp無効でwarmup1回＋測定5回再測定した。profiling中の値は混ぜていない。", "",
"| T | wall ms (timestamp off) | min–max ms |", "|---:|---:|---:|"]
for r in no_timestamp:
    lines.append(f"| {r['tile']} | {r['wall_ms']:.3f} | {r['wall_ms_min']:.3f}–{r['wall_ms_max']:.3f} |")
lines += ["", "## 正しさと限界", "",
"- 実グラフ：同じTの全W・memory・capacity・反復で最終座標SHA256が一致。T間は更新順が異なるので一致を要求しない。", "- micro：同じTの全distance source・レプリカ数・反復で先頭レプリカの最終座標SHA256が一致し、全レプリカの座標も完全一致。", "- microは合成距離・crossのみ・同じラウンドの反復で、実グラフのFull全反復や品質評価の代用ではない。", "- 実グラフは2入力・1seed。温度・クロック・他アプリ負荷は固定していない。設定順のランダム化とwarmupで一部を緩和した。", "- 全体の遅さを何%ずつ各要因へ分解したわけではない。共有メモリの銀行競合・バリア待ち・レジスタ圧の個別寄与も未確定。", "- T128/256という正確な最適値はGPU・n・入力・W・投入費用の釣り合いで決まり、普遍的なハードウェア定数ではない。", "",
"## 再現と根拠", "",
"専用example：`implementations/full/rr_gpu/examples/tile_cause.rs`。標準GPUコードをincludeし、専用経路のみでshader・configを切り替える。実行手順は同ディレクトリのREADME.md。全設定の中央値・最小最大はsummary.json。", "",
"Appleの一次資料：", "",
"- https://developer.apple.com/videos/play/wwdc2022/10159/ ：workgroupのコアへの分配、workgroup不足による非飽和。", "- https://developer.apple.com/videos/play/tech-talks/111373/ ：Apple Family 9以降のキャッシュ階層では、deviceとthreadgroupのアクセス性能が近くなる条件がある。今回のM4 Proでshared/globalが近いという観測と整合するが、それ自体がキャッシュhit率の実測ではない。", ""]
(OUT / "RESULTS.md").write_text("\n".join(lines))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
for graph, rows in real.items():
    data = baseline(rows)
    axes[0].plot([r['tile'] for r in data], [r['gpu_ms'] / 15 for r in data], 'o-', label=graph)
axes[0].set(xscale='log', yscale='log', xlabel='Tile size T', ylabel='GPU ms / iteration', title='Actual Full SGD, W=1024')
axes[0].legend()
for mode in ['matrix', 'packed', 'uniform']:
    data = [r for r in micro if r['tile']==1024 and r['distance_mode']==mode]
    axes[1].plot([r['replicas'] for r in data], [r['gpu_ms'] for r in data], 'o-', label=mode)
    axes[2].plot([r['replicas'] for r in data], [r['ns_per_update'] for r in data], 'o-', label=mode)
axes[1].set(xscale='log', xlabel='Independent replicas', ylabel='GPU ms / 8 dispatches', title='T1024: more work, similar time initially')
axes[2].set(xscale='log', yscale='log', xlabel='Independent replicas', ylabel='GPU ns / update', title='T1024: throughput improves')
for ax in axes[1:]: ax.legend()
for ax in axes: ax.grid(alpha=.2)
fig.savefig(OUT / 'diagnosis.png', dpi=180)
fig.savefig(OUT / 'diagnosis.pdf')
print(OUT / 'RESULTS.md')
