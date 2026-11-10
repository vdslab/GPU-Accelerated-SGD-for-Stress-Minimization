# WGPU Round-robin Full SGD

正式比較のmethod IDは`rr_sgd`。リポジトリ直下から実行する。

```sh
cargo build --release --locked --manifest-path implementations/full/rr_gpu/Cargo.toml
```

| 引数 | 意味 | 既定値 |
|---|---|---|
| `--phase-chunks C` | 内側のshuffle済み列を連続区間に分割し、区間ごとに全外側ラウンドを巡回。実効値はmin(C,T)、C>=1 | 1 |
| `--tile-size T` | 1ブロックの頂点数。2ブロック間のペア集合がタイル。1–1024 | 1024 |
| `--workgroup-size W` | 1 workgroupのスレッド数。1–1024、GPUの制限内 | 1024 |
| `--coordinate-memory MODE` | `shared`で座標を再利用するか、`global`で更新するか | shared |
| `--no-outer-shuffle` | 外側RRのラウンド順のランダム化を止める | ランダム化あり |
| `--no-inner-shuffle` | タイル内RRの更新ラウンド順のランダム化を止める | ランダム化あり |

TとWは独立に指定できる。sharedを使う場合、1 workgroupが2ブロックの座標を保持し、容量は16×T bytes。
`--memory-mode`は`--coordinate-memory`の別名。

v2主比較の設定は**T＝1024、W＝256、shared**。CLIの既定W＝1024とは異なる。

```sh
implementations/full/rr_gpu/target/release/rr-gpu \
  --input data/v1_lcc/USpowerGrid.mtx --iterations 15 --epsilon 0.1 --seed 0 \
  --tile-size 1024 --workgroup-size 256 --coordinate-memory shared \
  --output-format json --output-dir output/rr-example
```

JSONにはT・W・メモリ方式・shared容量・shuffle設定の`rr_*`項目も記録される。
[実験手順](../../../experiments/docs/README.md) · [v2結果](../../../experiment_results/v2/README.md) · [実験一覧](../../../experiment_results/README.md)

## 更新順序を維持した反復の高速化

ラウンドごとのqueue write/submitは同じ順序で実行し、GPU完了待ちを各反復の末尾にまとめています。
反復時間は最後のGPU処理の完了までを含みます。更新式・内側/外側シャッフル・タイル設定は変更していません。
[Full/SparseのCPU比較と座標一致検証](../../../experiment_results/v2/validation/rr-iteration-20261030/README.md)を参照してください。

追加のバッチ化と反復間CPU待機の除去も検証しましたが、既定設定で安定した高速化を確認できず、標準には採用していません。
候補と現行RRの最終座標・stressは20組で完全一致しました。
[追加案の全測定・再現用patch](../../../experiment_results/validation/rr-full-batching-20261030/README.md)を参照してください。

## 内側ラウンドの分割と交互実行（2026-11-05）

標準は分割なし（`--phase-chunks 1`）。内側・外側のshuffleを維持し、各外側ラウンドのタイル内matchingをまとめて処理する。全頂点対を1反復に一度ずつ処理する。

CPU完了待ちは反復末尾。JSONの`rr_phase_chunks`に実効分割数を記録する。品質診断用の`SGD_TRACE_DIR`／`SGD_TRACE_CHECKPOINTS`による座標保存費用は反復時間から除外する。

分割オプションは過去の診断を再現するために保持する。`experiment_results/full16/`および専用の旧実験ランナーは16分割の過去条件の記録であり、現在の標準設定の評価には流用しない。
