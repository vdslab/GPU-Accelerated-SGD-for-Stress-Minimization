from pathlib import Path
import json,statistics,csv,hashlib,subprocess
out=Path(__file__).resolve().parent
root=out.parents[3]
rows=[json.loads(l) for l in (out/'runs.jsonl').read_text().splitlines()]
assert len(rows)==18
paired=[json.loads(l) for l in (out/'paired4096.jsonl').read_text().splitlines()]
old=root/'experiment_results/v2/validation/atomic-sparse-no-shuffle-xlarge-20261027'
base={m:json.loads((old/f'delaunay_n17-s0-{m}.json').read_text()) for m in ['cpu','gpu_shuffle','gpu_no_shuffle']}
assert next(r for r in rows if r['seed']==0 and r['mode']=='cpu_shuffle')['stress64']==base['cpu']['stress_value']
with (out/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['mode','seed','stress64','stress1024','dispatches']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
metadata={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'production_changes_this_audit':False,'audit_seeds':[0,1,2],'iterations':15,'pivots':200,'epsilon':0.1,'reevaluation_sources':[64,1024,4096],'paired_eval_seed':20261027,'test_result':'9 tests passed; includes 45 brute-force preprocessing cases and 20 moving-hub GPU cases','binary_sha256':hashlib.sha256((root/'tools/diagnostics/sparse-audit/target/release/sparse-audit').read_bytes()).hexdigest()}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
text='''---
tags: [研究, SGD, GPU, 実験, 正当性]
created: 2026-10-27
updated: 2026-10-27
---

# Atomic Sparseの正当性とstress評価の監査

関連：[[Atomic SparseのCPUシャッフル除去比較]]、[[05_GPU Atomic Sparse SGD]]。

## 結論

delaunay_n17の「シャッフルなしでstressが6.82%良くなった」という先の速報は、64始点の標本に強く依存していた。**同一の保存済み座標を4096始点で再評価すると、なしはありより1.34%高い推定stressとなり、差の95%信頼区間はゼロを含んだ。品質改善を示す根拠として先の6.82%を使わない。** 再実行して都合のよい結果を選んだものではなく、座標を固定して評価だけを変更した比較である。

CPU Sparseの制約・領域重み・更新式・学習率に、今回の監査範囲で論文からの明確な実装誤りは見つからなかった。独立総当たり検証も通過。ただし、Atomic側にはrelaxed atomicだけでロック間の座標可視性を保証していないという重要な正当性上の問題が残る。これは今回の数値異常として再現したとは言えず、先のstress改善の原因と断定しない。

シャッフルの一様性や混合の程度を今回定量計測してはいない。「ランダム性が落ちたから必ず最終stressが悪化する」という単調関係は前提にできない。有限反復の非凸最適化、Sparse近似、更新順、精度差を区別する必要がある。

## 1. 同じ座標の評価だけを変更

M4 Proで先に保存したseed=0のCPU・GPUあり・GPUなしの最終座標を使用。全方式で同じ始点を共有し、4096始点は評価seed=20261027で重複なく選択した。独立実装のBFSと各始点の全頂点への寄与から全stressを推定した。

| 方式 | 旧64始点 stress | 新4096始点 stress |
|---|---:|---:|
'''
for m,i in [('cpu',2),('gpu_shuffle',0),('gpu_no_shuffle',1)]:text+=f"| {m} | {base[m]['stress_value']:,.3f} | {paired[i]['stress']:,.3f} |\n"
text+='''
- 旧64始点：GPUなし／あり差 −6.82%。
- 新4096始点：GPUなし／あり差 +1.34%。
- 始点ごとの対応差による全stress差の95%区間：約 −163万〜+2043万（ゼロを含む）。有限母集団補正を用いた正規近似であり、最適化seedの不確実性を表すものではない。
- 1024始点×評価seed 0,1,2,3でも順位が入れ替わった。これら4組は重なり得るため、4096個の独立した始点と呼ばない。

64始点×約13万対象の大量のペアを見ていても、評価の標本単位は64個の始点である。同じ始点に属するペアは独立標本ではない。誤差式 `n/(2k) × 始点寄与の和` は確認できたが、標本数が小さいと局所的な歪みの分布によって順位を誤認する。現行の共通評価は8,000頂点を超えると64始点・評価seed=0へ固定され、誤差幅を記録していない。これはCPUにもGPUにも共通の評価上の弱点。

## 2. 次数とSparse制約の違い

入力は無向・非重みDelaunayグラフ。公式データは131,072頂点、対称行列の非ゼロ数786,352であり、無向辺数はその半分の393,176。ローカル読込み値と一致。

| 指標 | 実測（seed=0、200 pivots） |
|---|---:|
| 元グラフの平均次数 | 5.9994 |
| 元グラフの次数 中央値／最小／最大 | 6／3／17 |
| Sparse制約数 | 26,586,282 |
| 実辺の制約 | 393,176（約1.48%） |
| 片側重みが0の制約 | 26,173,206（約98.45%） |
| 頂点当たり制約数 平均／中央値 | 405.67／206 |
| 頂点当たり制約数 最大 | 131,071 |
| 固定投入順で同じuが連続する最大件数 | 129,806 |

**元グラフは低次数だが、制約グラフではpivotが大きなハブになる。** 「元グラフの最大次数が17だからロック競合も小さい」とは言えない。片側が動かない制約でも、その頂点は別制約で動くので現在の両端ロックを単純に省略してよいわけではない。

制約をBTreeMapの辞書順で取り出すため、固定投入では同じ端点の制約が連続する。ただし実際のGPU更新順は、ロック獲得と未完了制約の次dispatchへの繰越で変わる。CPUの辞書順逐次更新と同じではなく、一様ランダムな順列とも保証されない。今回、実更新列の記録・一様性検定は未実施。

## 3. 3 seed × 6方式の切り分け

同一seed内で初期座標、pivot、全制約、学習率、前処理後のRNG状態を共有。CPU f32版は監査用にGPUと同じ方向の微小変位を使って実装したが、Metalのfast math・FMA・sqrtまでビット一致させるものではない。CPU f64 shuffledのseed=0は、元の保存済み結果と64始点stressが完全一致した。

全て15反復・200 pivots・epsilon=0.1。下表は1024始点・評価seed=0の推定stress（単位：百万）。3 seedの小規模監査であり、有意差や一般的な優劣を確定するものではない。

| 方式 | seed 0 | seed 1 | seed 2 | 3 seed平均 |
|---|---:|---:|---:|---:|
'''
for mode in dict.fromkeys(r['mode'] for r in rows):
 vals=[r['stress1024']/1e6 for r in rows if r['mode']==mode];text+=f"| {mode} | "+' | '.join(f'{v:.3f}' for v in vals)+f" | {statistics.mean(vals):.3f} |\n"
text+='''
GPUなし／ありの差はseed 0:+0.58%、seed 1:−0.60%、seed 2:+3.01%。GPUでシャッフルを外すと常に改善する／常に悪化する、いずれも今回の結果とは合わない。

CPU f64固定順はseed 0でシャッフルありより約14.15%高stressだった。GPU固定投入順はCPU固定順と同一ではないことと整合する。CPU shuffledをf64→f32にしただけでもseed 2は約4.26%低stressになった。精度・丸め差も非線形な反復の軌道を変えるので、CPU f64対GPU f32の品質差を全て順序に帰属させない。

監査用の時間値は速度比較に使わない。コピーやGPUセットアップの範囲が従来のiteration_timeとは異なり、保存済み座標のCPU再評価やコンパイルを一部並行した。前の速度表をこの時間で更新していない。

## 4. 問題・留意点の分類

### A. 確認済み：共通stress評価の標本数不足

`shared/experiment-common/src/stress.rs:6–8` は大規模で64始点へ自動切替し、評価seedも固定。評価式の明確な誤りは見つからないが、今回の6.82%改善を頑健に評価できていない。品質差を論文で主張する際は、共通の十分な始点数、複数評価標本、対応差の誤差幅を保存する。小規模ではexact評価を併用する。

### B. 重要：Atomicロックのメモリ順序保証不足

`implementations/sparse/vram-lock-sparse/src/shader.metal:12–18,51–72` のロック取得・解放・座標読取り・加算はすべてrelaxed。ロック変数1個のatomic性は、別アドレスの4座標の読取り・更新をロックの前後に順序付ける保証にならない。

Apple公式仕様6.16では、relaxed操作は同期操作ではなく、atomic性と各atomicオブジェクトの変更順だけを保証するとされる。原子的な座標加算も4座標をまとめたSGD更新を1個のトランザクションにするものではない。したがって「ロックがあるから逐次SGDのある順序と等価」とは現状断言できない。

Metal 3.2以降のdevice-scope fence、または対応するMetal版のacquire/releaseを使い、取得後・解放前の可視性を明示して検証することが必要。具体的な修正には対象OS・言語版・対応GPUの確認が必要で、今回本番カーネルは変更していない。同系統のFull Atomicにも同じrelaxed構成があるので、共通の確認対象となる。

**これは仕様上の正当性の問題であり、今回実機で不正な座標読取りが起きたと確認したものではない。** 既存の高競合テストは共有pivotを固定しており、共有座標の更新可視性を十分には検査していなかった。新たに「動く共有hub＋4096固定端点」の既知の逐次漸化式を20実行（シャッフルあり／なし各10回）で比較し、全て許容誤差内、全更新完了だった。テスト通過は今回の実機条件での観測であり、メモリモデルの保証の代わりにはならない。

### C. CPU Sparse本体：主要数式は一致、完全保証ではない

ZhengらAlgorithm 2と著者公開C++実装に対し、次を照合した。

- 最近pivotまでの距離に比例するpivot選択。
- 最近pivotの領域割当、同距離時は先に選んだpivotを保持。
- `s_ip = #{j in R(p) : 2*d(j,p) <= d(i,p)}`、方向別の `s_ip/d(i,p)^2`。
- 実辺は両方向重み1、pivot同士は方向ごとの重みを保持。
- 更新量の1/2、方向ごとのmu上限1、学習率の両端 `1/wmin` と `epsilon/wmax`。

別途、Floyd–Warshallで距離を作り直し、領域内の全頂点を数えて全制約を再構成した。3グラフサイズ×3 pivot数×5 seed＝45条件が一致。全頂点をpivotにした場合にFullの `1/d²` 重みへ戻ることも確認。今回、明確な重みの取り違え・制約欠落・重複・更新方向の誤りは見つからなかった。

著者実装をバイト単位で再現しているわけではない。現CPU版は最初のpivotを乱数選択し、著者公開コードは頂点0を指定する。RNGの種類・ストリームの使い方、ゼロ距離の保護も異なる。これらは現CPU対GPUでは共有条件または明記した差だが、「原著実装と完全一致」とは記述しない。

### D. 元のSparseに残る設計・比較上の制約

SparseはFull stressの近似であり、200 pivots固定の近似品質はグラフに依存する。近似を用いた更新と評価しているFull stressは同一の最適化対象ではない。15反復の固定スケジュールには収束判定がなく、今回も収束済みとは確認していない。これらは実装バグとは区別する。

現読込みはMatrix Marketの値を距離重みとして使わず、辺を無向・非重み化する。delaunay_n17はpattern symmetricなので整合するが、重み付き・有向データに適用する場合はこの変換を実験条件に明記する。

前処理では全pivot距離を保持し、BTreeMapからVecを作るため、大規模時のメモリ負荷も大きい。これはCPUとAtomicで共有する実装上の改善余地であり、今回のstress順位の原因とは確認していない。

## 5. 推奨する対応順

1. 前の品質改善の主張を訂正し、64始点の単回評価では優劣を確定しない。
2. Atomicのメモリ可視性を明示的に保証し、実更新順と座標スナップショットを記録して逐次再生との一致を検証する。
3. CPU f32対GPU f32、複数最適化seed、より十分な共通stress評価で比較する。更新順の混合性能を主張するなら順序自体も計測する。
4. 反復数・pivot数の感度を別実験として確認する。望む結果になるまで条件を選別せず、比較条件を先に固定する。

## 根拠と再現

- [Zhengらの論文（Algorithm 2、Section 4.3.1）](https://arxiv.org/pdf/1710.04626)
- [著者のSparse実装](https://github.com/jxz12/s_gd2/blob/master/cpp/s_gd2/sparse.cpp)
- [Apple Metal Shading Language Specification（2026-06-04、6.16、pp.298–302）](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [SuiteSparse delaunay_n17](https://sparse.tamu.edu/DIMACS10/delaunay_n17)

監査コード：`tools/diagnostics/sparse-audit/`。実行手順は同README。9テスト通過（うち独立前処理45条件、動くhub20実行）。

生データ：`experiment_results/v2/validation/atomic-sparse-audit-20261027/`。
`reevaluation.jsonl` は元座標の1024始点×4組、`paired4096.jsonl` は対応4096始点評価、`runs.jsonl` と `summary.csv` は3 seed×6方式、`delaunay_n17/` は座標・制約統計。`sources/` は照合した論文・仕様・著者コード。

本番実装にはこの監査で変更を加えていない。前ターンのシャッフル切替変更は保持し、commit・pushはしていない。
'''
vault=Path('/Users/kose/Library/Mobile Documents/iCloud~md~obsidian/Documents/sakamoti55/11_大学院研究')
report=vault/'01_研究フェーズ/実験/Atomic Sparseの正当性とstress評価の監査.md';report.write_text(text)
oldnote=vault/'01_研究フェーズ/実験/Atomic SparseのCPUシャッフル除去比較.md'
s=oldnote.read_text();title='# Atomic SparseのCPUシャッフル除去比較\n'
warning='\n> 2026-10-27 監査後の訂正：以下の64始点stressによる順位は速報であり、特にdelaunay_n17の「6.82%改善」は4096始点で再評価すると再現しなかった。同じ保存済み座標ではGPUなしがありより1.34%高い推定値となり、差の95%区間はゼロを含む。速度の測定値は保持するが、品質改善の根拠としてこの単回評価を使わない。詳細：[[Atomic Sparseの正当性とstress評価の監査]]。\n'
if warning not in s:s=s.replace(title,title+warning,1)
oldnote.write_text(s)
print(report)
