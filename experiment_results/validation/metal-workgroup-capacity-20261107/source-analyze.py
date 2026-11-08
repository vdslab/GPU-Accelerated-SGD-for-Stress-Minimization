"""Aggregate native Metal capacity probes and produce scientific figures."""
import collections
import csv
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiment_results/validation/metal-workgroup-capacity-20261107"


def summarize(name, expected):
    rows = [json.loads(s) for s in (OUT / (name + ".jsonl")).read_text().splitlines()]
    metadata = rows.pop(0)
    assert metadata["kind"] == "metadata" and len(rows) == expected
    groups = collections.defaultdict(list)
    for row in rows:
        if not row["warmup"]:
            groups[row["kernel"], row["tile"], row["width"], row["groups"]].append(row)
    result = []
    for key, runs in sorted(groups.items()):
        assert len(runs) == 5
        assert len({r["output_sha256"] for r in runs}) == 1
        values = [r["gpu_ms"] / r["dispatches"] * 1000 for r in runs]
        result.append(dict(zip(["kernel", "tile", "width", "groups"], key)) | {
            "gpu_us": statistics.median(values), "gpu_us_min": min(values), "gpu_us_max": max(values),
            "repetitions": len(runs), "work_per_dispatch": runs[0]["work_items"] // runs[0]["dispatches"],
            "output_sha256": runs[0]["output_sha256"],
        })
    return metadata, result


sweep_meta, sweep = summarize("sweep", 2856)
boundary_meta, boundary = summarize("boundary", 1440)
s = {(r['kernel'], r['tile'], r['width'], r['groups']): r for r in sweep}
b = {(r['kernel'], r['tile'], r['width'], r['groups']): r for r in boundary}
(OUT / "summary.json").write_text(json.dumps({"sweep_metadata":sweep_meta,"sweep":sweep,
    "boundary_metadata":boundary_meta,"boundary":boundary},indent=2)+"\n")
for name, rows in [("sweep",sweep),("boundary",boundary)]:
    with (OUT / (name + "-summary.csv")).open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

widths = [32,64,128,256,512,1024]
lines = ["# M4 Pro：workgroupの規模と個数の実測", "", "2026-11-07。Apple M4 Pro GPU 20 cores、macOS 15.7.2。標準RR実装を変更せず、native Metal専用ベンチマークを作成して測定した。", "",
"## 結論の読み方", "",
"同時常駐（実行途中のSIMDグループを保持）と、同一サイクルの命令発行と、仕事を増やしても経過時間が増えない範囲は別である。時間の平坦部を『ハードウェアの最大常駐workgroup数』や『すべてが同じクロックで演算した証明』とは呼ばない。", "",
"今回、常駐に関するCompute SIMD Groups Inflightカウンタと、条件別の経過時間の両方を取得した。最大スレッド数1024というAPI上限だけでは、同じ仕事量を最短時間で処理できる分割は決まらない。", "",
"## 測定条件", "",
"- 主掃引476設定×（warmup1回＋測定5回）=2,856試行。境界確認240設定×6回=1,440試行。合計4,296試行。各周は固定乱数で順序をランダム化し、GPU測定は直列。", "- fma1/fma4：各スレッドが依存するFMA漸化式を1系列／独立4系列実行。主掃引4,096反復、境界確認16,384反復。GPU出力をCPUのfused FMAと照合し、同じ総スレッド数の分け方間でも出力SHA256が一致。", "- rr_uniform/rr_matrix：crossタイルの更新式、循環matching、各matching後のthreadgroup barrier、2T個の共有座標。1 dispatchで同じタイルの全matchingを4巡。W=TとW1024を比較。同じTで全WG数・W・距離供給元・反復の出力が一致。", "- RR型は全距離d=1の合成診断。matrixはstride8192のstorage読み、uniformは実行時uniform読み。全WGが同じ読取専用初期値・距離を再利用し、書込先は独立。実グラフの全対SGDの代用ではない。", "- 4回のserial dispatchを同じcommand bufferへ記録。表の時間はgpuEndTime−gpuStartTimeを4で割った値（1 dispatch平均）。同一サイクルの命令発行数は測らない。", "- fast math無効。温度・クロック・他アプリ負荷は固定しない。profile実行はカウンタ取得専用で、非profileの時間集計へ混ぜない。", "",
"## 1 workgroup内の規模を変える", "", "fma4、16,384反復。W個の有効スレッドが同じ処理を担当する。Wを増やすと総仕事量も増えるため、これは固定仕事量の比較ではなく、1 WG内で仕事を増やしたときの処理時間の比較。", "",
"| W | SIMD group数（幅32） | GPU µs/dispatch | min–max µs |", "|---:|---:|---:|---:|"]
for w in widths:
    r=b['fma4',w,w,1]
    lines.append(f"| {w} | {w//32} | {r['gpu_us']:.2f} | {r['gpu_us_min']:.2f}–{r['gpu_us_max']:.2f} |")
lines += ["", "このFMA処理ではW32〜256の時間は近い一方、W512/1024では増える。W1024をW128と同じ時間で処理できるという仮定は、この実測では成立しない。ただし平坦部から『同時に発行できるSIMDグループは8個』とは推定しない。依存遅延、ループ制御、パイプライン処理能力が含まれるため。", "",
"## 同じ仕事量を分割する", "", "fma4、総スレッド数8,192、各スレッド4,096反復。同じ入力・同じ総FMA数・同じ出力。", "",
"| W | WG数 | GPU µs/dispatch | min–max µs |", "|---:|---:|---:|---:|"]
for w in widths:
    r=s['fma4',w,w,8192//w]
    lines.append(f"| {w} | {8192//w} | {r['gpu_us']:.2f} | {r['gpu_us_min']:.2f}–{r['gpu_us_max']:.2f} |")
lines += ["", "大きいworkgroupへ集めるより、小さいworkgroupへ分けた方が速い実例を、計算結果を変えずに確認した。これは独立演算の結果であり、RRのTを変えたときの更新順やdispatch回数まで同じにした比較ではない。", "",
"## 大きいworkgroup数の境界", "", "時間が平坦な範囲と、その次の点を直接比較する。", "",
"| 処理 | T | W | G20 µs | G21 µs | G40 µs | G41 µs |", "|---|---:|---:|---:|---:|---:|---:|"]
for kernel,t,w in [('fma4',256,256),('fma4',1024,1024),('rr_uniform',128,1024),('rr_uniform',1024,1024)]:
    vals=[b[kernel,t,w,g]['gpu_us'] for g in [20,21,40,41]]
    lines.append(f"| {kernel} | {t} | {w} | "+" | ".join(f"{v:.2f}" for v in vals)+" |")
lines += ["", "大きいWではG20→21に明確な時間増加が出る。20コアと整合する実効的な処理の段差だが、最大常駐数そのものではない。小さいWでは20を超えても時間が平坦な範囲がある。", "",
"| fma4のW | 実測した平坦部の例：WG数 | 時間 µs（同順） |", "|---:|---|---|"]
for w,counts in [(32,[1,20,80,160,161]),(64,[1,20,40,80,81]),(128,[1,20,40,41]),(256,[1,20,21]),(1024,[1,20,21])]:
    lines.append(f"| {w} | "+", ".join(map(str,counts))+" | "+", ".join(f"{b['fma4',w,w,g]['gpu_us']:.2f}" for g in counts)+" |")
lines += ["", "この表のWG数を普遍的な上限値にしない。RR型ではbarrier・共有メモリがあるため、同じWでもFMAとは曲線が異なる。全条件の値はsummary.jsonとCSVを参照。", "",
"## カウンタによる同時常駐の観測", "",
"通常のMTLDevice.counterSetsではtimestampしか公開されなかったが、Instruments UIでCounter SetをPerformance Limiters、Enable Shader Timelineを有効化したテンプレートを保存し、48種類のカウンタを取得できた。", "",
"Compute SIMD Groups Inflightはshader core当たりの平均実行途中SIMDグループ数。命令発行数ではない。ラベル付きGPU実行区間の中央80%に完全に入るカウンタサンプルを、サンプル持続時間で重み付けして集計する。warmupを除外。カウンタはGPU全体の観測で、他アプリの影響を完全には除いていない。", ""]
for prefix in ['', 'profile2-', 'resident-']:
    path=OUT/(prefix+'counter-summary.json')
    if not path.exists(): continue
    rows=json.loads(path.read_text())
    lines += [f"### {prefix or 'profile1'} カウンタ", "", "| 処理 | T | W | G | 平均inflight SIMD/core | 最大サンプル | サンプル数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r['counter']=='Compute SIMD Groups Inflight':
            lines.append(f"| {r['kernel']} | {r['tile']} | {r['width']} | {r['groups']} | {r['weighted_mean']:.3f} | {r['max_sample']:.3f} | {r['samples']} |")
    lines.append('')
resident_path=OUT/'resident-counter-summary.json'
if resident_path.exists():
    resident=json.loads(resident_path.read_text())
    counter_lookup={(r['kernel'],r['width'],r['counter']):r for r in resident}
    lines += ['### 十分な仕事を投入したときの最大観測値', '',
        '各Wに対して総スレッド数131,072を投入した。fma4とRR型（T=W）の計測。最大サンプルを物理的な普遍上限と断定せず、今回のカーネルでの最大観測値として報告する。', '',
        'WG相当数 = inflight SIMD/core × 20 cores ÷ (W/32)。これは同じサイズのWGへ換算したSIMDグループ数であり、実行途中で一部SIMDグループが終了したWGを個別に数えた値ではない。G1/W1024のRR型で約1.6 SIMD/coreが観測され、32 SIMD÷20 coresという正規化と整合した。', '',
        '| W=T | 投入WG数 | FMA最大SIMD/core | RR最大SIMD/core | RRのGPU全体WG相当 |',
        '|---:|---:|---:|---:|---:|']
    for w in widths:
        f=counter_lookup['fma4',w,'Compute SIMD Groups Inflight']
        r=counter_lookup['rr_uniform',w,'Compute SIMD Groups Inflight']
        lines.append(f"| {w} | {131072//w} | {f['max_sample']:.2f} | {r['max_sample']:.2f} | {r['max_sample']*20/(w/32):.1f} |")
    lines += ['', 'W1024では最大約96 SIMD/core、すなわち約3 WG/core・GPU全体で約60 WG相当が観測された。一方、RR型の経過時間はG20→21ですでに増えた。同時常駐能力と、仕事を増やしても同じ時間で処理できる能力は一致しない。', '']
    fig_r,axes_r=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    for kernel,label in [('fma4','Independent FMA'),('rr_uniform','RR-like, T=W')]:
        rows=[counter_lookup[kernel,w,'Compute SIMD Groups Inflight'] for w in widths]
        axes_r[0].plot(widths,[r['max_sample'] for r in rows],'o-',label=label)
        axes_r[1].plot(widths,[r['max_sample']*20/(w/32) for r,w in zip(rows,widths)],'o-',label=label)
    axes_r[0].set(xscale='log',xlabel='Threads per WG W',ylabel='Peak sampled SIMD groups / core',title='Observed inflight groups, not issued instructions')
    axes_r[0].set_ylim(0,110)
    axes_r[1].set(xscale='log',yscale='log',xlabel='Threads per WG W',ylabel='GPU-wide workgroup equivalents',title='Peak sampled residency, converted by WG size')
    for a in axes_r:a.set_xticks(widths,labels=widths);a.legend();a.grid(alpha=.2)
    fig_r.savefig(OUT/'residency.png',dpi=180);fig_r.savefig(OUT/'residency.pdf')
lines += ["RR型T1024/W1024/G20では約32 SIMD groups/coreが観測された。したがって『workgroup内に32組を置けない』という説明はこの観測と合わない。全組が実行途中として存在することと、全組が同一サイクルに命令を発行することは別である。", "",
"## 今回の実装へどう使うか", "",
"1. W1024が最大という理由だけでT1024を性能最適としない。独立FMAではW32〜256に時間が近い範囲があり、W1024へ増やすと1 WG自体の処理時間が増えた。", "2. 大きいWでのG20前後は実測した処理時間の境界として使える。ただし常駐上限20という意味ではなく、カーネルと資源量が変われば結果も変わる。", "3. 実グラフのT/Wは実際のSGD時間と最終Stressで選ぶ。独立演算の最適Wや『全部同じ瞬間に処理する』という基準だけでは選ばない。", "",
"## 未確認のこと", "",
"- 同一サイクルに発行されるSIMD命令の実数。inflight/occupancyカウンタからは取得できない。", "- 全カーネル共通の最大常駐workgroup数。動的キャッシュ、レジスタ、共有メモリなどに依存する。", "- 1 workgroupが特定の物理コアのどこへ割り当てられたか。今回のinflightはコア当たり平均であり、コア別の個数そのものではない。", "- 実グラフRRの遅さに対する各要因の厳密な寄与率。", "",
"## 参照", "",
"- [Apple: Finding your Metal app’s GPU occupancy](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy) — occupancyと資源制約。", "- [Apple: GPU counters and counter sample buffers](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers) — 公開counter API。", "- 計測カウンタの定義はprofile-counter-info.xmlとprofile2-counter-info.xmlに保存。", "- ソースと再現手順：tools/diagnostics/metal-workgroup-capacity/README.md。", ""]
(OUT/'RESULTS.md').write_text('\n'.join(lines))

fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
data=[b['fma4',w,w,1] for w in widths]
ax[0,0].plot(widths,[r['gpu_us'] for r in data],'o-')
ax[0,0].set(xscale='log',xlabel='Threads per workgroup W',ylabel='GPU µs / dispatch',title='One WG: 4 FMA chains, 16,384 iterations')
ax[0,0].set_xticks(widths,labels=widths)
ax[0,0].set_ylim(bottom=0)
for w in [64,256,1024]:
    data=[r for r in boundary if r['kernel']=='fma4' and r['width']==w]
    ax[0,1].plot([r['groups'] for r in data],[r['gpu_us'] for r in data],'o-',label=f'W{w}')
ax[0,1].set(xscale='log',yscale='log',xlabel='Workgroups G',ylabel='GPU µs / dispatch',title='Same per-thread work, more WGs')
ax[0,1].axvline(20,color='gray',alpha=.4,linestyle='--');ax[0,1].legend()
data=[s['fma4',w,w,8192//w] for w in widths]
ax[1,0].plot(widths,[r['gpu_us'] for r in data],'o-')
ax[1,0].set(xscale='log',xlabel='Threads per workgroup W',ylabel='GPU µs / dispatch',title='Identical work: 8,192 threads × 4,096 iterations')
ax[1,0].set_xticks(widths,labels=widths)
ax[1,0].set_ylim(bottom=0)
for t in [128,1024]:
    data=[r for r in boundary if r['kernel']=='rr_uniform' and r['tile']==t and r['width']==1024]
    ref=b['rr_uniform',t,1024,20]['gpu_us']
    ax[1,1].plot([r['groups'] for r in data],[r['gpu_us']/ref for r in data],'o-',label=f'T{t}, W1024')
ax[1,1].set(xscale='log',yscale='log',xlabel='Workgroups G',ylabel='GPU time / time at G20',title='RR-like kernel: jump between G20 and G21')
ax[1,1].axvline(20,color='gray',alpha=.4,linestyle='--');ax[1,1].legend()
for a in ax.flat:a.grid(alpha=.2)
fig.savefig(OUT/'capacity.png',dpi=180)
fig.savefig(OUT/'capacity.pdf')
print(OUT/'RESULTS.md')
