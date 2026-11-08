# 実行手順

リポジトリ直下で実行。Rust・Python 3。Atomic版はmacOS / Metal。

## 単発実行

ブラウザで設定を選ぶ場合は `python3 experiments/serve_single.py --open` を実行する。
Live Serverにも対応する。手順は[ローカル実行ページ](../../tools/single_run/README.md)を参照。

1グラフ・1手法を実行し、初期配置と最終配置の画像を保存する。warmupなし。

```sh
python3 experiments/run_single.py --input data/USpowerGrid.mtx --method rr_sparse_sgd \
  --iterations 15 --epsilon 0.1 --pivots 200 --seed 0
```

選択したSGD実装と描画ツールを自動でrelease buildし、設定と計測時間をコンソールに表示する。画像は既定で2048×2048。`--image-size 1024`または`--image-size 1920x1080`で変更できる。

| 系列 | `--method` |
|---|---|
| Full | `sgd`、`atomic_sgd`、`rr_sgd` |
| Sparse | `sparse_sgd`、`atomic_sparse_sgd`、`rr_sparse_sgd` |

結果は`output/<graph>/<日時>_<method>/`へ保存する。日時は日本時間で、時・分・秒を含む。成功時の出力は次の4ファイル。

```text
output/USpowerGrid/2026-10-13-12-34-56_rr_sparse_sgd/
├── 2026-10-13-12-34-56_0.png
├── 2026-10-13-12-34-56_1.png
├── 2026-10-13-12-34-56_info.txt
└── 2026-10-13-12-34-56_result.json
```

`*_0.png`は初期配置、`*_1.png`は実行後の配置。`*_info.txt`に設定・速度・辺・最終座標、`*_result.json`に機械可読な結果を保存する。描画時間はSGDの計測時間と分ける。同秒の再実行はフォルダ末尾に`_2`などを付けて保存する。

## 一括比較

[single-graph.json](../manifests/single-graph.json): USpowerGrid × Sparse 3手法 × seed 0。15反復・200 pivots。
測定3件＋warmup各1回。少数seedの動作確認用。

```sh
# Sparse 3手法のビルド
cargo build --release --locked --manifest-path implementations/sparse/baseline-sparse-sgd-non-gpu/Cargo.toml
cargo build --release --locked --manifest-path implementations/sparse/vram-lock-sparse/Cargo.toml
cargo build --release --locked --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml

# 条件確認 → 実行 → 中断後の再開
python3 experiments/run_experiments.py experiments/manifests/single-graph.json --dry-run
python3 experiments/run_experiments.py experiments/manifests/single-graph.json
python3 experiments/run_experiments.py experiments/manifests/single-graph.json --resume

# 集計
python3 experiments/aggregate_results.py \
  --experiment-dir output/experiments/single-graph \
  --profile validation --output-dir output/experiments/single-graph/tables
```

結果: `output/experiments/single-graph/`。
`results.jsonl`に測定、`artifacts/`に座標、`tables/`に時間・stressの表。
失敗・欠測は`stderr/`と`tables/validation-report.json`で確認。
`--dry-run`は測定せず、入力checksum・binary・実行コマンドを確認。

## 条件変更

```sh
cp experiments/manifests/single-graph.json experiments/manifests/my-graph.json
shasum -a 256 data/USpowerGrid.mtx
```

複製したmanifestの`experiment_id`・`datasets`（name/path/sha256）・`methods`・`seeds`を変更。
配列の組合せを実行。GPUと同じ系列のCPU基準を含め、入力checksumを転記。
既存結果を残して条件を変える場合は新しい`experiment_id`を使用。

[実験概要](README.md) · [v1](../../experiment_results/v1/README.md)
