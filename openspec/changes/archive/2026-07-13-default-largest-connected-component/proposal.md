## Why

SuiteSparse由来の行列には孤立頂点や小さな別成分が含まれることがあり、Sparse SGDは非連結入力をそのまま扱えません。毎回オプション指定や手動の前処理をせず、通常の`cargo run`で描画可能にするため、最大連結成分を標準入力グラフとして扱います。

## What Changes

- CPU Sparse SGDとGPU Sparse SGDは、全Matrix Market入力から最大連結成分を常に抽出して実行する。
- 採用頂点を連続した内部番号へ再番号付けし、元の頂点番号との対応を追跡する。
- 起動時のターミナルへ、元／採用頂点数・辺数、連結成分数、採用率を必ず表示する。
- 結果ファイルへ同じ成分統計を記録し、必要な場合に元頂点番号を確認できるmapを保存する。
- **BREAKING**: 非連結入力はエラーではなく最大連結成分のレイアウトとして実行される。
- 既存の位置引数、`--input`、pivot数・反復数・epsilon・seedなどのCLI指定は変更しない。

## Capabilities

### New Capabilities

- `default-largest-connected-component-input`: Matrix Market入力を最大連結成分へ自動縮約し、実行・端末表示・出力追跡を一貫させる。

### Modified Capabilities

なし。

## Impact

- `baseline-sparse-sgd-non-gpu`と`sparse-sgd-gpu`の共通グラフ読込、CLI起動、結果保存を変更する。
- `gpu_visualizer`の入力形式は維持する。描画対象は常に抽出後の連結グラフとなる。
