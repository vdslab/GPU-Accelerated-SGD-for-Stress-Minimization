# GPU Sparse SGD

`baseline-sparse-sgd-non-gpu`と同じ論文準拠Sparse SGD前処理を使い、競合のない
round-robinスケジュールでWGPU上の座標更新を実行します。

リポジトリ直下から実行します。

```sh
cargo run --release --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml -- data/USpowerGrid.mtx \
  --pivots 200 --iterations 15 --epsilon 0.1 --seed 0
```

出力先の既定値はリポジトリ直下の`output/`です。`--output-dir PATH`で変更できます。生成される
`sparse-sgd-gpu-...-1.txt`は`gpu_visualizer`でそのまま読み込み、full stressの計算と
PNGの描画ができます。

スケジュールは次の2相から成ります。

- one-sided相: 1 invocationが1非pivot頂点だけを書き、pivotは読み取り専用にする
- two-sided相: pivot完全グラフをcircle methodで分解し、実辺を空きへ詰めたmatchingを順に実行する

各反復で相順、matching round順、共通pivot順をseedから再現可能にシャッフルします。
CLIの標準実行では、さらに頂点ごとに独立したpivot順を作り、片側更新とmatching roundを
4分割して交互に処理します（pivotが4個未満ならその個数）。各制約の処理は1反復につき1回です。

頂点別の順序には4段Feistel置換とcycle walkingを使います。pivot数が2の累乗でなくても
重複・欠落がなく、追加の頂点×pivot順序表は不要です。グラフの種類・次数で切り替えません。
座標・GPU演算は単一のf32で、重み・学習率・移動係数の上限1・RR matchingを維持します。
atomicsやlockは使用しません。

JSONには`rr_pivot_order`、`rr_phase_chunks`、`rr_step_cap`、`rr_arithmetic`を記録します。


## 共通の省メモリ・高速化

全グラフで同じ経路を使用します。グラフ名や次数による手法の切り替えはありません。

- `preprocess.rs`はu32のBFS距離を保持し、制約をCPU参照版と同じ辞書順で直接生成します。
  巨大なBTreeMapと全EdgeInfo配列を作らず、片側制約をGPU用表へ格納します。
  pivot選択・領域の同距離tie・f64重み・学習率・乱数消費・前処理SHAは参照版と一致します。
- 片側表は距離と重みの8 bytes/要素。pivot IDは共通配列から参照します。
- 片側相では正重みの更新がある頂点だけを書き戻します。pivotはこの相で書きません。
- 両側制約の不要なcloneを削除し、GPU転送前に不要なCPU制約を解放します。
- 各command bufferに最大256 dispatchを1つのcompute passとして記録します。
  dispatchの更新順序・依存関係を維持し、反復末尾で完了を待ちます。
  次反復のuniform/permutation書き換えは完了待ちの後です。
- 出力形式を維持したままBufWriterと明示flushを使用します。

前処理の同値性と反復高速化を保ち、標準の更新順序を上記の方式へ変更しています。
反復数・epsilon・pivot数は指定値のままです。CPUと同じ15反復で比較し、追加反復は行いません。
[現在のf32品質検証・CPU比較](../../../experiment_results/v2/validation/rr-f32-compensated-20261030/README.md)には
8グラフ×4seed、選択後の未使用seed、Stanfordの4,096始点評価、時間比較を記録しています。
CPUと完全に同じstressや、未検証の全グラフ・seedでの改善を保証するものではありません。

[v3との照合・旧順序の品質課題・100万ノード実測](../../../experiment_results/v2/validation/rr-common-optimization-20261030/README.md)
は旧実装の記録です。

## 品質の診断

CLI標準の対照実験には`quality_search`の`shuffle`と`4@0`を指定します。
`4@0`は最初から4分割する指定です。`standard`と`1@0`は旧RR順序を再生します。
`cpu`を付けると独立したCPU前処理とのSHA一致も確認します。

```sh
cargo run --release --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml --example quality_search -- data/web-NotreDame.mtx 3 '4@0' 256 200 shuffle cpu
```

評価は固定256始点（上記コマンドの場合）のsampled full stressです。小規模グラフでもsampledなので、
CLIのauto exact stressやv3の小規模exact stressと直接比べないでください。
`quality_probe`の循環shift/strideは旧方式の診断用に残しています。
`compensated`・`extended`・`coefficients`・`recenter`・`cap=VALUE`も診断専用で、CLI標準では使用しません。

[追加の反復高速化・Full/Sparseの大規模CPU比較](../../../experiment_results/v2/validation/rr-iteration-20261030/README.md)
では、compute passをまとめた前後の最終座標一致も検証しています。
