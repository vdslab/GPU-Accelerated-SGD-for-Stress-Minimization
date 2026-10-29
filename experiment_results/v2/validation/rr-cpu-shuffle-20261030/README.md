# CPUのシャッフルとRR Sparseのstress差の調査

## 結論

CPUは各反復で全制約配列に`pairs.shuffle(rng)`を適用する。
ロックされたrand 0.9.5のSliceRandomはDurstenfeld/Fisher–Yates系のシャッフルで、
循環シフトや制約の一部だけのシャッフルではない。今回の調査ではシャッフル不良は見つからなかった。
これは順序を無作為に選ぶことであり、最終stressが最小になる順序を選ぶ操作ではない。
CPU・GPUは同じ初期配置・制約・15反復でも順序と演算精度が異なるため、到達する配置は異なる。
f64は更新計算の精度を上げるが、配置の大域的な最適性を保証しない。

## 評価の始点数だけで順位が逆転するケース

Stanford seed 0の同じ保存済み配置を評価すると、64始点ではCPU 5,092,620,939.590、
GPU 4,924,176,872.803（GPUが約3.31%低い）。4,096始点ではCPU 5,043,851,029.634、
GPU 5,210,893,656.347（GPUが約3.31%高い）。4,096始点も厳密値ではない。
記録は`../rr-f32-compensated-20261030/stanford-selected-paired4096.jsonl`。

同じ始点を両方式に使っても、各配置の誤差が大きい頂点が異なれば、抽出した始点によって
推定差・順位が変わる。評価関数が同じという条件だけでは推定誤差はなくならない。

EVA seed 1でも、以前の256始点ではGPU/CPU=0.987076（GPUが1.29%低い）だったが、
今回の全頂点対の厳密評価では1.009603（GPUが0.96%高い）に逆転した。
GPU最終座標SHAは以前の実験と一致しており、配置を変えずに評価の違いを確認している。
従来の「40条件・中央値+0.154%」は256始点での推定値であり、厳密stressの結論ではない。

## 厳密評価でもGPUが良いケースと、CPUの順序だけを変えた実験

EVA（4,475頂点）の準備seed 1と11を評価した。両方とも15反復・200 pivots。
各準備seed内で初期座標・pivot・制約・重み・学習率を固定したまま、
CPUの更新用乱数seedだけを0～19に変えて20回実行した。
通常CPUも別に、前処理後の本来の乱数状態から再生した。
更新用乱数はシャッフルのほか、距離が1e-12未満の場合の退避方向にも使われるため、
厳密には更新用乱数全体を変えた比較である（その分岐の発生回数は未計測）。
全配置を既存共通評価器の`StressMode::Exact`、f64で評価した。
GPUは採用済みの方式をそのまま1回実行し、候補選択や方式変更はしていない。

| 準備seed | 通常CPU exact stress | GPU exact stress | GPU/CPU差 | CPU順序20通りの最小～最大 | CPU順序20通りの中央値 | GPUより良いCPU順序 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,111,264.662 | 1,121,936.437 | +0.96% | 1,080,356.280～1,360,289.257 | 1,131,644.181 | 8/20 |
| 11 | 1,163,137.560 | 1,074,915.399 | -7.58% | 1,091,337.558～1,216,967.215 | 1,132,202.125 | 0/20 |

seed 11ではGPUのstressが厳密評価でも7.58%低い。全てをサンプリング誤差では説明できない。
一方、CPUだけで更新乱数を変えてもstressに幅があることから、正規のシャッフルでも
15反復後のstressが一定・最小になるわけではないことが確認できる。
GPUの順序は制約の一様ランダムな全順列ではなく、競合回避・頂点ごとの処理・相分割という
構造を持つ。その構造が特定の配置で有利にも不利にもなりうる。

この実験は「GPUのシャッフルの方がランダム性に優れる」「全グラフで優れる」ことを示さない。
GPUとCPUの差にはf32/f64と更新順序の両方が含まれる。
新しいRR順序のf64逐次再生や各変更単独の比較は今回行っておらず、
seed 11の改善を特定の1要因へ帰属することはできない。
また、Sparse SGDが更新する制約集合・重みは全頂点対のfull stressそのものと同一ではない。

## コードと再現

- CPU更新: `implementations/sparse/baseline-sparse-sgd-non-gpu/src/algorithm.rs:42`
- Sparse制約: `implementations/sparse/baseline-sparse-sgd-non-gpu/src/graph.rs`
- 厳密/抽出評価: `shared/experiment-common/src/stress.rs:205`
- 実験: `implementations/sparse/sparse-sgd-gpu/examples/cpu_shuffle_probe.rs`
- rand実装: Cargo.lockのrand 0.9.5、`rand-0.9.5/src/seq/slice.rs`のSliceRandom実装

```sh
cargo build --release --locked --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml --example cpu_shuffle_probe
implementations/sparse/sparse-sgd-gpu/target/release/examples/cpu_shuffle_probe data/EVA.mtx 1 20
implementations/sparse/sparse-sgd-gpu/target/release/examples/cpu_shuffle_probe data/EVA.mtx 11 20
```

診断exampleのみ追加。標準手法・CPUのシャッフル・ブラウザの評価設定は変更していない。
release build、対象exampleのclippy `-D warnings`、44配置の厳密評価が完了。
