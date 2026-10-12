# Metal Atomic Full SGD

正式比較のmethod IDは `atomic_sgd`。リポジトリ直下から実行します。

```sh
cargo build --release --locked --manifest-path implementations/full/vram-lock-native/Cargo.toml
```

[実験手順](../../../experiments/docs/runbook.md)、結果は [実験一覧](../../../experiment_results/README.md)。既定入力はリポジトリ直下の `data/`、出力は `output/` です。
