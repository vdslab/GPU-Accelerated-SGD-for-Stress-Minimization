# GPU Sparse SGD

`baseline-sparse-sgd-non-gpu`と同じ論文準拠Sparse SGD前処理を使い、競合のない
round-robinスケジュールでWGPU上の座標更新を実行します。

```sh
cargo run --release -- ../data/USpowerGrid.mtx \
  --pivots 200 --iterations 15 --epsilon 0.1 --seed 0
```

出力先の既定値は`../output`です。`--output-dir PATH`で変更できます。生成される
`sparse-sgd-gpu-...-1.txt`は`gpu_visualizer`でそのまま読み込み、full stressの計算と
2048×2048 PNGの描画ができます。

スケジュールは次の2相から成ります。

- one-sided相: 1 invocationが1非pivot頂点だけを書き、pivotは読み取り専用にする
- two-sided相: pivot完全グラフをcircle methodで分解し、実辺を空きへ詰めたmatchingを順に実行する

各反復で相順、matching round順、pivot順をseedから再現可能にシャッフルします。
atomicsやlockは使用しません。

