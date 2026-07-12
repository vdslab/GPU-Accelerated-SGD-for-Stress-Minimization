## ADDED Requirements

### Requirement: GPU Sparse SGDのiteration進捗表示

`sparse-sgd-gpu` は、GPU SGD計算ループで各iterationの開始時に、1始まりの番号を `Iteration: N` 形式で標準出力へ表示しなければならない（MUST）。表示順はGPU dispatch準備・submitの順序と一致し、既存のSGD開始ログおよび集計Timingを維持しなければならない（MUST）。

#### Scenario: 複数iterationの進捗を表示する

- **WHEN** `--iterations 3` のSparse SGD GPU実行を開始する
- **THEN** 標準出力に `Iteration: 1`、`Iteration: 2`、`Iteration: 3` がこの順で各1回表示される

#### Scenario: 1始まりで表示する

- **WHEN** 最初のGPU iterationのdispatch準備に入る
- **THEN** 標準出力に `Iteration: 1` が表示され、`Iteration: 0` は表示されない

#### Scenario: 既存ログと併存する

- **WHEN** GPU SGDが正常終了する
- **THEN** `SGD 開始: iterations=...` の開始ログ、iteration進捗行、既存のcompute/readback/total Timingがすべて表示される

#### Scenario: 計算順序を変更しない

- **WHEN** iteration進捗表示を追加したGPU実行を行う
- **THEN** 乱数seed、スケジュール、dispatch回数、最終座標、結果TXTのメタデータは表示追加前と同じになる
