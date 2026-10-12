# 共通コード

`experiment-common/` はCPU・Atomic・RR実装が共有する計測、seed、stress評価、JSON記録、環境情報を提供します。方式固有の座標更新は `implementations/` に置きます。

既定入力・出力の `repository_path()` はこのcheckoutのリポジトリ直下を参照するため、実行時の作業ディレクトリに左右されません。明示した `--input` / `--output-dir` の相対パスは実行時の作業ディレクトリが基準です。

```sh
cargo test --locked --manifest-path shared/experiment-common/Cargo.toml
```

Sparseのgraph/algorithmは現状CPU SparseのソースをGPU Sparseの2手法から共有します。今回の整理では数値計算を変更せず、共通crateへの切り出しは行いません。
