## 1. Compact round assignment

- [x] 1.1 `build_schedule` のtwo-sided分類を、graph edgeの `EdgeInfo` コピーではなく `prepared.pairs` 内のchecked index列をshuffleする形へ変更する
- [x] 1.2 pivot circle methodの各制約へbase round IDを付与し、頂点ごとの使用roundリストを初期化する
- [x] 1.3 generation mark配列で両端点の使用roundを統合し、seed付きgraph edge順に最小の利用可能round IDを決めるfirst-fit割当を実装する
- [x] 1.4 round ID、制約index、件数に対する `usize` / `u32` のchecked conversionとoverflowエラーを追加する

## 2. Two-pass flat schedule construction

- [x] 2.1 制約ごとのround IDを数えてchecked prefix sumから `round_offsets` とround別write cursorを構築する
- [x] 2.2 正確な長さで確保した `Vec<GpuPair>` へstable scatterし、同一round内のseed付き制約順を維持する
- [x] 2.3 `Schedule` の公開フィールドとbase/spill/max-degree統計を維持したまま、全roundの `Vec<Vec<EdgeInfo>>`、round別 `HashSet`、graph edge値コピーをproduction経路から削除する
- [x] 2.4 一時的な制約index・round ID・cursorをflat schedule完成後に解放でき、完成scheduleを全iterationで再利用することを確認する

## 3. Memory-bounded validation and tests

- [x] 3.1 flat `two_sided` と `round_offsets` を頂点generation markで検証する、有界なround競合validatorへ置き換える
- [x] 3.2 小規模グラフの現行referenceスケジューラとcompactスケジューラを比較し、同一seedのpair分類・round ID・offset・pair順が一致するテストを追加する
- [x] 3.3 one-sidedとtwo-sidedの完全性、欠落・重複なし、round上限、複数seedの再現性を検証するテストを追加する
- [x] 3.4 高次数star graphでround数が `Δ`、round内端点が互いに素、頂点round-membership値がtwo-sided制約数の2倍であることを検証する
- [x] 3.5 既存の実GPU品質・canary・iteration再利用テストを実行し、WGSLとGPU実行結果が変更されないことを確認する

## 4. Verification

- [x] 4.5 Define a fixed 256-pass submission limit and overflow-safe contiguous batch ranges
- [x] 4.6 Refactor `GpuContext::execute` to submit and complete one batch at a time while preserving global uniform offsets and invocation order
- [x] 4.7 Report submissions per iteration and add boundary tests for 1, 256, 257, and 38,626 invocations
- [x] 4.8 Run automated checks plus USPowerGrid, luxembourg_osm, and web-Stanford verification; record web-Stanford peak private memory and validate OpenSpec strictly

- [x] 4.1 `sparse-sgd-gpu` で `cargo fmt --check`、`cargo test --release`、`cargo clippy --all-targets -- -D warnings`を成功させる
- [x] 4.2 USPowerGrid・200 pivotsで既存round統計、制約件数、GPU stress品質が維持されることを確認する
- [x] 4.3 luxembourg_osm・50 pivotsでround数、schedule生成時間、GPU実行が退行しないことを確認する
- [x] 4.4 web-Stanford・1 pivotで38,625 roundの完全なscheduleを生成し、変更前後のschedule時間・ピークメモリ・GPU実行結果を計測して `verification.md` に記録する
