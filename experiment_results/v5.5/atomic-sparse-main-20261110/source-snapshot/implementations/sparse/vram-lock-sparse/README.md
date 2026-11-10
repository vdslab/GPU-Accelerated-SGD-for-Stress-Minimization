# Atomic Sparse SGD (Metal)

CPU Sparse SGDの制約ペアをMetal GPUへ転送し、両端点をatomicロックして更新する。
`vram-lock-native` と同じく、32スレッドのthreadgroupにつき1スレッドが1ペアを処理する。
Metal 3.2以降のdevice-scope atomic fenceが使えるApple silicon環境が必要（検証環境：macOS 15.7.2、M4 Pro）。対応しない言語版ではshaderコンパイル時に明示的に失敗する。

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
- 標準動作では準備時に一度だけ制約をシャッフルして隣接する制約の競合を分散し、各反復ではその投入順を固定する。Atomic Fullと同じく反復中のCPUシャッフルは行わない。初回シャッフルは `method_setup_time_ms` に含める。更新順はGPUのロック獲得順に依存し、一様ランダムな順序は保証しない。
- 比較用の `--cpu-shuffle` を指定した場合のみ、従来どおり各反復でCPU版と同じRNGの続きからペアをシャッフルする。出力座標ファイルの `CPU shuffle` ヘッダーに設定を保存する。JSONのmethod IDは共通で、`atomic_pair_order` に初回のみ／毎反復の別、`atomic_schedule` に実装版を記録する。
- ペアの両端点を頂点番号の小さい順にロックする。片側の重みが0でも、座標の読取りを保護するため両端をロックする。
- Atomic Fullと同じく、個々のロック取得成功後と解放前にdevice-scopeのseq_cst fenceを置く。2番目のロック取得に失敗して1番目を解放するrollback経路にも適用する。Metal 3.2の `atomic_thread_fence` を使い、座標の可視性をロックの引継ぎに同期させる。
- ロック取得・fence後に両端の座標を読み、`mu_u=min(eta*weight_u,1)` と `mu_v=min(eta*weight_v,1)` を用いて別々の移動量をatomic加算する。
- 座標と学習率はGPU上ではf32。出力・centering・共通stress評価はf64。
- 座標が一致した場合はnative atomic SGDと同じ固定方向へ微小変位を作る。CPU版のランダム方向とは異なる。

Atomic Fullと同じGPU内の再試行ループを使う。2番目のロックを取得できなければ1番目を解放して再試行する。
1反復につき1回dispatchし、GPU完了後に全制約の完了epochが現在の反復番号と一致することを検証する。
過去の反復の完了フラグを誤って再利用しない。未完了やMetalエラーは成功JSONを出さずに失敗する。
各threadgroupが一意の1制約を一度だけ更新するため、CPUによる未完了制約の抽出・再投入は不要。
無制限のGPU内再試行はFullと同じ設計であり、実験runner側にはプロセス時間制限を設ける。

`attempted_updates` と `completed_updates` は成功時に制約数×反復数、`dispatches` は反復数となる。
`retry_failures=0` は欠落した更新がないという意味で、ロック競合がないという意味ではない。

`iteration_time_ms` は反復内のGPU投入、待機、完了epoch検証を含み、`--cpu-shuffle` 時にはシャッフルと転送も含む。
`gpu_device_time_ms` と `iteration_times_ms` はMetal command bufferのGPU timestampで計測する。
後者はGPU時間のみであり、CPUとの速度比にはホスト処理を含む `iteration_time_ms` を使う。
最初のバッファ転送は `upload_time_ms`、最終座標取得は `readback_time_ms`、centeringは `postprocess_time_ms`。

[2026-11-06の原因調査・CPU比較](../../../experiment_results/validation/atomic-sparse-20261106/README.md)に
旧実装の保存、条件、反復時間、初回シャッフルを含む時間、stressをまとめる。
図の旧Atomicは別チェックアウトの毎反復CPUシャッフル版であり、このリポジトリの変更前の固定順版とも区別する。

## 検証

```sh
cargo test --locked --manifest-path implementations/sparse/vram-lock-sparse/Cargo.toml
python3 -B -m unittest discover -s experiments/tests
```

GPUテストで方向別重み・片側固定・muのclamp・座標一致・centering・高競合時の更新回数を確認する。
正式な速度・品質比較は別の実験結果として保存する。`experiment_results/v1` はAtomic Sparseを含む6手法の暫定結果を保持する。
