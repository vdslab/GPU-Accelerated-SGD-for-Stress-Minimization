# WGPU Round-robin Full SGD

正式比較のmethod IDは`rr_sgd`。リポジトリ直下から実行する。

```sh
cargo build --release --locked --manifest-path implementations/full/rr_gpu/Cargo.toml
```

| 引数 | 意味 | 既定値 |
|---|---|---|
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
