import csv,statistics
from pathlib import Path
out=Path(__file__).resolve().parent
rows=list(csv.DictReader((out/'summary.csv').open()))+list(csv.DictReader((out/'summary-extra.csv').open()))
assert len(rows)==30
with (out/'summary-all.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
methods=['cpu','before_shuffle','before_fixed','fenced_shuffle','fenced_fixed']
labels={'cpu':'CPU Sparse','before_shuffle':'修正前GPU・shuffleあり','before_fixed':'修正前GPU・shuffleなし','fenced_shuffle':'修正版GPU・shuffleあり','fenced_fixed':'修正版GPU・shuffleなし'}
text='''---
tags: [研究, SGD, GPU, 実験, 正当性]
created: 2026-10-27
updated: 2026-10-27
---

# Atomic Sparseのメモリ同期修正と再実験

関連：[[Atomic Sparseの正当性とstress評価の監査]]、[[Atomic SparseのCPUシャッフル除去比較]]。

## 修正内容

`implementations/sparse/vram-lock-sparse/src/shader.metal` の両端ロック取得後・座標読取り前と、全座標更新後・両端ロック解放前に、それぞれ `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)` を追加した。

Metal 3.2で使用可能なseq_cst fenceのacquire/releaseの働きを使い、relaxed atomicのロック引継ぎを介して、前の保持者の座標更新を次の保持者へ公開する。ロック変数のatomic性だけに依存していた構成を変更した。Metal 3.2未満ではshaderの明示的なコンパイルエラーとし、同期なしへのフォールバックはしない。macOS 15.7.2 / Apple M4 Proでコンパイルと動作を確認。

標準はCPUシャッフルなし、`--cpu-shuffle` でありを選択する従来の切替を維持。Sparseの数式・前処理・学習率・128回のロック試行上限・未完了制約の再投入は変更していない。

共有hubを更新する競合回帰テストを本番クレートへ追加。4096制約の既知の逐次漸化式との比較を20実行（あり／なし各10回）で確認した。本番30テスト、監査9テストが通過。全制約更新、片側固定、方向別重み、2D dispatch、無効入力等も検証している。実グラフの全更新列を記録した逐次再生まで行ったものではない。

## 実験条件

- 15反復、200 pivots、epsilon=0.1、最大連結成分、centeringあり、release build。
- delaunay_n17（131,072頂点）：seed 0,1,2。smallworld（100,000頂点）：seed 0,1,2。smallworldはseed=0の品質差が大きかったため、条件を変えずにseed 1,2を追加した。
- CPU、修正前GPUのあり／なし、修正版GPUのあり／なし＝計30実行。修正前バイナリは再ビルド前に保存した。
- グループごとに方式の実行順をローテーション。ベンチマークと再評価を順に実行し、並行ジョブによる計測干渉を避けた。専用warm-up・同一seed複数回測定は行っていない。
- 同一グラフ・seed内の入力・初期座標・前処理・制約数の一致を確認。全30実行で全制約更新完了。
- 本体JSONが既存形式で出す64始点stressは生データとして残すが、以下の品質比較には使用しない。全レイアウトを4096個の共通始点（評価seed=20261027）で再評価した。
- CPU f64とGPU f32の既存差は残る。修正前後GPUの比較は同じf32処理で、主な実装差はfenceの追加。

## 時間と品質

時間は15反復の合計、stressは全頂点対stressの4096始点による推定。低いほど良い。両グラフとも時間中央値／stress平均（3 seed）。

'''
for d in ['delaunay_n17','smallworld']:
 text+=f'### {d}\n\n| 方式 | 反復 秒 | 前処理・GPU初期化込み 秒 | stress | dispatch数中央値 |\n|---|---:|---:|---:|---:|\n'
 for m in methods:
  rs=[r for r in rows if r['dataset']==d and r['method']==m]
  it=statistics.median(float(r['iteration_time_ms']) for r in rs)/1000
  total=statistics.median(float(r['algorithm_time_cold_ms']) for r in rs)/1000
  stress=statistics.mean(float(r['stress4096']) for r in rs)
  ds=statistics.median(float(r['dispatches'] or 0) for r in rs)
  text+=f'| {labels[m]} | {it:.3f} | {total:.3f} | {stress:,.3f} | {ds:g} |\n'
 text+='\n'
text+='''「前処理・GPU初期化込み」は既存のalgorithm_time_cold指標で、入力ファイル読込み、stress評価、座標ファイル保存を含まない。反復時間にはCPUシャッフル（有効時）、反復内転送、dispatch、待機、未完了抽出・再投入を含む。

## seedごとの修正版GPUの品質差

各行は同じ4096始点の対応比較。区間は最適化seedのばらつきではなく、当該固定レイアウトに対する始点標本の不確実性を示す。単回や3 seedの結果から一般的な有意差を断定しない。

| グラフ | seed | GPUあり stress | GPUなし stress | なし／あり差 | stress差95%区間 |
|---|---:|---:|---:|---:|---:|
'''
for d,s in [('delaunay_n17','0'),('delaunay_n17','1'),('delaunay_n17','2'),('smallworld','0'),('smallworld','1'),('smallworld','2')]:
 a=next(r for r in rows if r['dataset']==d and r['seed']==s and r['method']=='fenced_shuffle')
 b=next(r for r in rows if r['dataset']==d and r['seed']==s and r['method']=='fenced_fixed')
 x,y=float(a['stress4096']),float(b['stress4096'])
 text+=f"| {d} | {s} | {x:,.3f} | {y:,.3f} | {(y/x-1)*100:+.2f}% | [{float(b['delta_ci_low']):,.0f}, {float(b['delta_ci_high']):,.0f}] |\n"
text+='''
## 解釈上の限界

fenceは必要な同期保証を追加する変更であり、stressを必ず改善させるための変更ではない。修正前後でGPU更新順も変わり得るため、stressの差だけから「修正前に可視性違反が発生していた」とは判断できない。GPUの固定投入順は一様ランダムな順列ではないが、CPUの固定逐次順とも同一ではない。

## 保存先と再現

リポジトリ：`/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization`

実験ディレクトリ：`experiment_results/v2/validation/atomic-sparse-fenced-20261027/`。

- `run.py`：最初の20実行と各グループの4096始点再評価。`run_extra.py`：smallworldのseed 1,2の10実行。再実行は結果を上書きするので、保存先を複製してから実行する。
- `summary-all.csv`（`summary.csv`＋`summary-extra.csv`）：4096始点stress・対応差区間・反復時間・前処理込み時間。
- 各結果JSON：生の64始点評価、実行コマンド、更新回数、ハッシュ、時間。
- `*-evaluation.jsonl`：4096始点評価。基準は修正版GPU shuffleあり。
- `metadata.json` と `metadata-extra.json`：条件と各バイナリのSHA256。`source.diff`：実行時の本番変更差分。
- `binaries/vram-lock-sparse-before`：修正前バイナリ。`artifacts/`：全初期・最終座標。

commit・pushはしていない。
'''
vault=Path('/Users/kose/Library/Mobile Documents/iCloud~md~obsidian/Documents/sakamoti55/11_大学院研究')
p=vault/'01_研究フェーズ/実験/Atomic Sparseのメモリ同期修正と再実験.md';p.write_text(text)
audit=vault/'01_研究フェーズ/実験/Atomic Sparseの正当性とstress評価の監査.md';s=audit.read_text();mark='# Atomic Sparseの正当性とstress評価の監査\n';note='\n> 2026-10-27追記：この監査で指摘したAtomicの同期保証不足に対し、ロック前後のdevice-scope fenceを追加して再実験した。以下は修正前の監査記録として保持する。現在のコードと測定結果は [[Atomic Sparseのメモリ同期修正と再実験]] を参照。\n';
if note not in s:audit.write_text(s.replace(mark,mark+note,1))
print(p)
print(text[text.index('## 時間と品質'):text.index('## 解釈上の限界')])
