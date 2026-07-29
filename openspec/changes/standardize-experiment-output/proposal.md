## Why

現在の5手法は、入力・seed・計測区間・コンソール出力が揃っておらず、実行時間とストレスを同じ条件で再現可能に比較できない。論文実験を自動化する前に、1実行の記録形式と計測契約を共通化し、後からAtomicSparseSGDを同じ実験へ追加できる基盤が必要である。

## What Changes

- 実装済みのSGD、atomicSGD、RR-SGD、SparseSGD、RR-SparseSGDへ、共通の実験用CLI引数とseed指定を導入する。
- 実験モードの標準出力を、スキーマ版付きのJSONオブジェクト1行だけに統一する。進捗・デバッグ情報は既定で抑止し、必要時だけ標準エラーへ出力する。
- `Iteration time`、`Algorithm time`、`CLI total time`の3指標を、共通の区間定義と同期条件で計測する。RRのスケジュール構築時間は`method setup`として分離記録しつつ、Algorithm timeへ含める。
- タイミング計測終了後に、共通評価器で最終ストレスを測定し、厳密値と標本化推定を明示的に区別する。
- グラフ、実行パラメータ、GPU、更新統計、成果物パスを含む追跡可能な実験レコードを定義する。該当しない値は0ではなく`null`とする。
- JSON manifestから条件の直積を実行し、ウォームアップ、反復、結果JSONL追記、標準エラーログ保存、失敗記録、再開を行う実験ランナーを追加する。
- 未実装のAtomicSparseSGDが、実装後に同じCLI・出力スキーマ・自動実行へ参加するための契約を定義する。
- 出力形式と実験手順を`paper/docs/`へ文書化する。

## Capabilities

### New Capabilities

- `experiment-data-collection`: 比較対象手法の共通CLI、JSON出力、計測区間、ストレス評価、追跡情報、自動実験実行を定義する。

### Modified Capabilities

なし。

## Impact

- 対象crate: `baseline-sgd-non-gpu`、`vram-lock-native`、`rr_gpu`、`baseline-sparse-sgd-non-gpu`、`sparse-sgd-gpu`
- 共通化: 実験レコード、CLI設定、時間計測、ストレス評価を提供する新しい共通Rust crate
- 評価: `gpu_visualizer`にある厳密・標本化ストレス実装の共有ライブラリ化
- 自動化: `experiments/`配下のmanifest、ランナー、スキーマ、検証
- 出力: `output/experiments/<experiment-id>/`配下のJSONL、標準エラーログ、座標成果物
- 文書: `paper/docs/experiment-output-format.md`と`paper/docs/experiment-runbook.md`
- 既存の人間向け通常実行は維持し、厳密な出力契約は実験モードに適用する。
