## 背景

`baseline-sparse-sgd-non-gpu` は、方向別重みを持つ論文準拠 Sparse SGD を実装し、USPowerGrid・200 pivots・15 iterations・25 runsで平均full stress `726,046.63` を達成しています。GPU版はこの前処理と品質を維持しつつ、約100万制約/iterationを並列化する必要があります。

`rr_gpu` はWGPU 27の初期化、storage buffer、WGSL compute、ラウンド間dispatch、結果download、出力形式の参考になります。しかし、全点対制約を頂点blockへ分ける方式をそのままSparse SGDへ適用すると、pivotは約 `n-1` 制約へ参加するため、競合なしedge coloringの下限が `n-1` roundsになります。これは「round数をpivot数 `h` 程度にする」目標と両立しません。

論文の方向別重みでは、非pivot頂点 `i` とpivot `p` の制約は通常 `w'_ip>0, w'_pi=0` です。この非対称性を利用すると、`i` だけを書き、`p` を読む処理として全pivot制約を1 invocation内へまとめられます。両端を書き込むのはpivot–pivot制約と元グラフの辺だけです。

## 目標と対象外

**目標:**

- `sparse-sgd-gpu` のWGPU実行基盤を作ります。
- CPU版と同一の論文準拠前処理を使用します。
- atomicやspin lockなしで、全roundの書き込み競合を構造的に排除します。
- pivot–pivotと隣接辺の両端更新roundを概ね `h+O(Δ_E)` に抑えます。
- USPowerGrid・200 pivotsで両端更新roundを220以下にします。
- GPU版の25-run平均full stressをCPU版の5%以内にします。
- seedによるスケジュールと結果の再現性、`gpu_visualizer`互換出力を提供します。

**対象外:**

- pivot選択や `s_ip` 前処理自体のGPU化。
- 非連結グラフおよび重み付き最短路。
- CPU版と座標をbit単位で一致させること。
- 1 workgroup内へ全頂点を収める設計。
- lock競合を許容して失敗制約を再試行する方式。

## 設計上の判断

### CPU前処理をGPU版の正本として再利用する

初期実装では `baseline-sparse-sgd-non-gpu` のgraph preprocessingを `sparse-sgd-gpu` へ移植し、同一seedに対するpivot、距離、領域、方向別重み、学習率、初期座標を一致させます。共有crate化は両実装が安定した後の別変更とし、この変更では比較可能性を優先します。

代替案としてGPUでBFSと領域計算を行う方法がありますが、スケジューリングと座標更新の検証に別の変数を持ち込むため対象外とします。

### 制約を一方向phaseと両端更新phaseへ分割する

```text
CPU Sparse constraints
        │
        ├── one-sided: non-pivot i → pivot p
        │      w'_ip > 0, w'_pi = 0
        │      CSR grouped by writable vertex i
        │
        └── two-sided
               ├── pivot ↔ pivot
               └── original graph edge
```

一方向phaseは、非pivot頂点ごとのCSR offsetと、`(pivot, dij, weight_i)` の連続配列をGPUへ渡します。1 invocationは担当頂点の座標をprivate変数へ読み、seed付きpivot順で全CSR entryを逐次適用し、最後に担当頂点だけを書き戻します。pivot座標はphase中に変更しないため、全invocationが安全に読み取れます。

これにより約 `nh` 制約を1 dispatchへまとめます。pivotごとに1 dispatchする案は `h` dispatchで順序を細かく混ぜられますが、dispatch overheadが増えるため、まず1 dispatch方式を採用します。品質がCPU基準を外れる場合は、pivot列を複数chunkへ分けて両端roundと交互に実行できるようデータ構造を保ちます。

### pivot–pivotをcircle methodでh程度のroundへ分解する

全pivot対は元グラフで隣接している場合も含めて1つの制約を持つため、pivot集合上では完全グラフ `K_h` とみなせます。circle methodにより、偶数 `h` は `h-1`、奇数 `h` はdummyを追加して `h` matchingへ分解できます。

各roundではpivotが最大1回だけ現れるため、pivot–pivot制約をworkgroup単位またはinvocation単位で安全に並列実行できます。

### 隣接辺をpivot roundの空きへ詰める

元グラフ辺を次の順で扱います。

1. 両端がpivotの辺は既存pivot–pivot slotの重みを対称な辺重みへ置き換え、重複追加しません。
2. その他の辺は、seedでシャッフルした順に、両端が未使用の既存pivot roundへfirst-fitで追加します。
3. 入らなかった辺は、同じfirst-fit edge coloringで追加matching roundへ分解します。

非pivot–非pivot辺はpivot roundの空きへ入りやすく、pivot–非pivot辺はpivotが全roundで使用中の場合にspillします。単純greedy edge coloringは残余最大次数 `Δ` に対して最大 `2Δ-1` 色なので、全体を保守的に `h+2Δ_E` 以下とします。USPowerGridは `h=200, Δ_E=19` で、実測目標を220以下とします。

スケジュール生成後にCPU検証器が以下を確認します。

- 元のtwo-sided制約が各1回だけ存在する。
- 各roundで各頂点の出現回数が0または1である。
- round offsetとflattened constraint範囲が整合する。
- one-sidedとtwo-sidedの集合和が元制約集合と一致する。

### 1 iterationをh+1前後のdispatch列として実行する

基本形は次の通りです。

```text
iteration η_t
   │
   ├── one-sided vertex→pivots dispatch   (1)
   │
   └── shuffled two-sided matching rounds (≈ h)
          round 0 dispatch
          round 1 dispatch
          ...
```

iterationごとにone-sided/two-sidedのphase順をseedで選択し、two-sided round順をシャッフルします。一方向phaseでは共通pivot permutationにvertex別のdeterministic rotationを加え、全頂点が同じpivot順を処理する偏りを減らします。

roundごとのoffset/count/eta/iteration seedは256-byte aligned dynamic uniform bufferへ事前格納します。1 iteration分を1 command encoderへ順序通りに記録し、roundごとにdynamic offsetを切り替えます。WGPUのdispatch境界でstorage buffer更新を次dispatchから可視にし、`rr_gpu` のようなroundごとのCPU waitを避けます。

代替案として全roundを1 dispatch内ループで処理する方法がありますが、workgroup間barrierが存在せずround境界を保証できないため採用しません。

### GPU bufferとshaderをphase別にする

共通buffer:

- `positions: array<vec2<f32>>` read_write
- `pivots: array<u32>` read
- iteration/round uniforms

one-sided pipeline:

- vertex CSR offsets
- `(pivot, dij, weight)` entry buffer
- pivot permutation buffer
- 1 invocation / vertex、workgroup sizeはadapter limit内の256を初期値とする

two-sided pipeline:

- round順にflattenした `(u, v, dij, weight_u, weight_v)` buffer
- round offsets/counts
- 1 invocation / constraint、workgroup size256

WGSLはCPU版と同じ半変位と方向別clampを使用します。座標一致時は `(vertex ids, iteration, seed)` のhashから決定的な微小方向を生成し、GPU間の再現性を可能な範囲で維持します。

### 品質はstressで判定する

並列phaseはCPUの完全なrandom reshufflingと更新順が異なるため、座標一致ではなくfull stressを品質契約とします。

- 小規模連結グラフ: GPU stressをCPU stressの10%以内
- USPowerGrid、200 pivots、15 iterations、epsilon 0.1、seeds 0〜24:
  - GPU平均をCPU基準 `726,046.63` の5%以内
  - 論文許容範囲 `576,000`〜`864,000`
  - 代表runを `gpu_visualizer` で2048×2048描画

1 dispatchのone-sided phaseで品質を満たせない場合、pivot constraintをchunk化し、two-sided round群と交互にdispatchする方式を採用します。競合不変条件は変わりません。

### rr_gpu互換のCLIと結果形式を使う

`rr_gpu` のWGPU 27初期化と結果保存を基礎に、CPU版と同じ `--input`、`--pivots`、`--iterations`、`--epsilon`、`--seed` を提供します。出力にはadapter、backend、one-sided件数、two-sided件数、round数、dispatch数、前処理時間、GPU時間を追加します。

## リスクとトレードオフ

- [one-sidedをまとめるとCPUのglobal random orderと異なる] → phase順、pivot順、vertex rotationをランダム化し、stress基準を外れた場合はchunk化します。
- [round数は入力グラフ次数に依存する] → `h+2Δ_E` の上限を検証し、実測round数を必ず出力します。
- [f64 CPUからf32 GPUへの変換で差が出る] → 前処理はf64のまま一致させ、GPU upload時だけf32化し、stress許容差で評価します。
- [大きいCSRでstorage buffer limitを超える] → adapter limitsを事前確認し、必要sizeとlimitを含むエラーを返します。
- [dispatch間の同期解釈を誤る] → 同一roundのmatching検証に加え、小規模グラフでCPU/GPU結果と競合検出用canaryをテストします。
- [異なるGPU backendで浮動小数点結果が変わる] → 同一adapterでの再現性を要件とし、cross-adapterはstress品質で比較します。

## 導入手順

1. `sparse-sgd-gpu` crateとCPU前処理・CLI・出力の土台を作ります。
2. 制約分類とCPUスケジューラ、検証器を実装します。
3. one-sided WGSL pipelineを実装します。
4. two-sided round WGSL pipelineとdynamic uniform dispatchを実装します。
5. 小規模グラフで競合なし・完全性・再現性・stressを検証します。
6. USPowerGrid 25 runsでround数とstressを検証し、代表PNGを生成します。
7. 品質未達時のみone-sided chunk interleaveを導入します。

## 未解決事項

- 1 dispatch one-sided方式がCPU版5%以内を満たすかは実測で決めます。データ構造はchunk化へ拡張可能にします。
- USPowerGridでfirst-fit packingが220 rounds以内になるかは実装時に測定し、超える場合は残余pivot–nonpivot辺専用のbipartite edge coloringを検討します。
