## Why

`baseline-sparse-sgd-non-gpu` は各SGD iterationの開始時に `Iteration: N` を表示するが、`sparse-sgd-gpu` は開始時の総iteration数と終了時の集計時間しか表示しない。そのため、GPU計算が進行中か、どのiterationで停止しているかをターミナルから確認できない。

## What Changes

- `sparse-sgd-gpu` のGPU SGDループで、各iterationの開始前に `Iteration: N` を表示する。
- 表示形式をCPU版Sparse SGDと合わせ、1始まりのiteration番号とする。
- 既存の `SGD 開始`、集計Timing、出力ファイル形式、GPU計算順序は変更しない。
- iteration表示を確認する単体または統合テストを追加する。

## Capabilities

### New Capabilities

- `sparse-sgd-gpu-iteration-progress`: GPU Sparse SGDのiteration進捗表示契約を定義する。

### Modified Capabilities

なし。

## Impact

- 対象は `sparse-sgd-gpu/src/gpu.rs` のGPU実行ループと、そのテスト・実行ログである。
- CLI引数、GPU shader、結果TXTの数値内容、性能計測の意味は変更しない。
- 標準出力にiteration行が追加されるため、ログを機械解析する利用者は追加行を許容する必要がある。
