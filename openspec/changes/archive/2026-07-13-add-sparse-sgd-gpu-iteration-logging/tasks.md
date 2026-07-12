## 1. GPU iteration進捗表示

- [x] 1.1 `sparse-sgd-gpu/src/gpu.rs` の学習率ループ先頭に、CPU版と同じ `Iteration: {iteration + 1}` 表示を追加する
- [x] 1.2 既存のSGD開始ログ、GPU dispatch、Timing集計、結果出力が変更されていないことを確認する

## 2. 検証

- [x] 2.1 iteration番号のフォーマットと1始まりを確認するテストを追加する
- [x] 2.2 `sparse-sgd-gpu` の `cargo fmt`、`cargo test`、`cargo clippy --all-targets -- -D warnings` を実行する
- [x] 2.3 小規模GPU実行で `Iteration: 1` から最終iterationまでが順番に表示されることを確認する
