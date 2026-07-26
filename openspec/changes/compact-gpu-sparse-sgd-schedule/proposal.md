## Why

`sparse-sgd-gpu` の two-sided スケジューラは、全 matching round の辺配列と頂点集合を同時に保持しながら、各辺を既存 round へ first-fit で配置している。最大次数 38,625 の web-Stanford では 38,625 round の一時構造と辺の複製が大きなメモリ負荷となるため、同じ競合回避契約を保ったまま、スケジュール生成時の常駐データをコンパクトにする必要がある。

## What Changes

- GPU execution records at most 256 compute passes in one command encoder, submits each batch in invocation order, and waits for completion before recording the next batch.
- The flat schedule, invocation order, per-round dispatches, and total dispatch count remain unchanged; only command recording and submission are bounded.

- two-sided 制約ごとに決定的な round ID を割り当て、全 round の `Vec<Vec<EdgeInfo>>` と round 別 `HashSet<usize>` を同時構築しないスケジューラへ変更する。
- round ID の件数集計と prefix sum を使う2-pass処理で、既存GPU実行形式の平坦な `two_sided` 配列と `round_offsets` を直接構築する。
- pivot–pivot roundを初期色として扱い、seed付きgraph edge順序、first-fit結果、matchingの競合回避、round上限、再現性を維持する。
- 完成したコンパクトスケジュールを全iterationで再利用し、iterationごとの再スケジューリングやroundごとのGPU再アップロードを行わない。
- 大規模・高次数グラフ向けに、制約の完全性、round内競合、決定性、補助メモリ構造を検証するテストと計測を追加する。
- round数そのものの削減、two-sided更新方式の変更、dense one-sided `O(nh)` 配列の削減はこの変更の対象外とする。

## Capabilities

### New Capabilities

なし。

### Modified Capabilities

- `gpu-sparse-sgd`: two-sided matching scheduleを、全roundの辺・頂点集合を同時保持しないコンパクトな生成方式へ変更する。

## Impact

- `sparse-sgd-gpu/src/gpu.rs` changes to bounded command encoding and adds batch-boundary tests.
- Large-round graphs use multiple ordered submissions per iteration; graphs with at most 256 dispatches still use one submission.

- 主な変更対象は `sparse-sgd-gpu/src/schedule.rs` とその単体テストである。
- `Schedule` の公開結果形式、WGSL shader、CLI引数、結果ファイル形式、GPU dispatch順序の契約は維持する。
- 必要に応じてスケジュール生成統計を追加するが、新しい外部依存は導入しない。
- web-Stanfordのような高次数グラフではスケジュール生成時の一時メモリを削減する一方、38,625 roundに由来するdispatch数と計算時間は残る。
