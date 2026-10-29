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

各反復で相順、matching round順、pivot順をseedから再現可能にシャッフルします。
atomicsやlockは使用しません。


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

標準の更新順序、15反復、epsilon、pivot数は変更していません。
従来RRとの品質一致はCPU版と同じ品質であることを意味しません。
[v3との照合・品質の未解決点・100万ノード実測](../../../experiment_results/v2/validation/rr-common-optimization-20261030/README.md)を参照してください。

## 品質の診断

`quality_probe`は同じ初期配置と固定256始点のsampled full stressを使います。
小規模グラフでもsampledなので、v3の小規模exact stressと直接比べないでください。

```sh
cargo run --release --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml --example quality_probe -- data/web-NotreDame.mtx 200 3 cpu 1
```

最後の`1`を`1,4,16`とすると、片側相を分割して両側roundと交互に実行する対照実験になります。
さらに`stride`を付けると頂点別の互いに素な巡回strideを試します。いずれも**診断専用**で、
CLIの標準実行では無効です。制約数は変えませんが更新順序と品質が変わります。
別グラフで大きく悪化する例があるため、改善済みの標準手法として扱わないでください。

[追加の反復高速化・Full/Sparseの大規模CPU比較](../../../experiment_results/v2/validation/rr-iteration-20261030/README.md)
では、compute passをまとめた前後の最終座標一致も検証しています。
