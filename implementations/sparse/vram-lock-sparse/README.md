# Atomic Sparse SGD (Metal)

CPU Sparse SGDの制約ペアをMetal GPUへ転送し、両端点をatomicロックして更新する。
`vram-lock-native` と同じく、32スレッドのthreadgroupにつき1スレッドが1ペアを処理する。
macOSと、既存のnative atomic SGDが動作するMetal GPUが必要。

## 実行

リポジトリ直下から実行する場合：

```sh
cargo build --release --locked --manifest-path implementations/sparse/vram-lock-sparse/Cargo.toml
implementations/sparse/vram-lock-sparse/target/release/vram-lock-sparse \
  --input data/USpowerGrid.mtx --pivots 200 \
  --iterations 15 --epsilon 0.1 --seed 0 \
  --run-id atomic-sparse-check --output-format json \
  --output-dir output/atomic-sparse-check --verbose
```

method IDは `atomic_sparse_sgd`。既存の `experiments/run_experiments.py` のmanifestに指定できる。
初期・最終座標と頂点対応表を保存し、JSONモードのstdoutは結果1行、進捗はstderrに出す。
最大連結成分抽出、pivot数の頂点数へのclamp、centeringはCPU Sparse版と同じ。

## 更新と競合処理

- CPU Sparse版の `graph.rs` を共有し、pivot・領域・距離・制約・方向別重み・学習率・初期配置を一致させる。
- 各反復でCPU版と同じRNGの続きからペアをシャッフルする。GPUの実際の更新順は非決定的。
- ペアの両端点を頂点番号の小さい順にロックする。片側の重みが0でも、座標の読取りを保護するため両端をロックする。
- ロック取得後に両端の座標を読み、`mu_u=min(eta*weight_u,1)` と `mu_v=min(eta*weight_v,1)` を用いて別々の移動量をatomic加算する。
- 座標と学習率はGPU上ではf32。出力・centering・共通stress評価はf64。
- 座標が一致した場合はnative atomic SGDと同じ固定方向へ微小変位を作る。CPU版のランダム方向とは異なる。

1回の投入でのロック試行は128回までとし、失敗時は両端を解放して未完了に残す。
GPU完了後、ホストが未完了ペアだけを再投入する。完了したペアはその反復で再更新しない。
全ペア完了前に学習率を次へ進めない。進捗がなくなるか、1反復で1024回の投入でも完了しない場合はエラーにし、成功JSONを出さない。

`attempted_updates` は制約数×反復数、`completed_updates` は完了制約数を表す。
成功時の `retry_failures`（未完了数）は0となる。これはロック競合がなかったという意味ではない。
再投入は `dispatches` に含め、`--verbose` で各反復の投入回数を確認できる。

時間区間は既存の共通形式に従う。`iteration_time_ms` にCPUシャッフル、反復内の転送、GPU投入、待機、未完了ペアの抽出と再投入を含める。
最初のバッファ転送は `upload_time_ms`、最終座標の取得は `readback_time_ms`、centeringは `postprocess_time_ms`。
GPU timestampの個別計測は未対応。

## 検証

```sh
cargo test --locked --manifest-path implementations/sparse/vram-lock-sparse/Cargo.toml
python3 -B -m unittest discover -s experiments/tests
```

GPUテストで方向別重み・片側固定・muのclamp・座標一致・centering・高競合時の更新回数を確認する。
正式な速度・品質比較は別の実験結果として保存する。`experiment_results/v1` はAtomic Sparseを含む6手法の暫定結果を保持する。
