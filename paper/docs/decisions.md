# 意思決定ログ

研究課題、実験条件、評価方法、論文構成に影響する判断を残します。過去の記述は消さず、新しい決定で置き換えます。

## D001: 論文ワークスペースを `paper/` に集約する

- 日付: 2026-07-28
- 状態: 採用
- 背景: 研究完了までの計画、実験記録、原稿、図表、参考文献が分散しない置き場が必要。
- 決定: 論文関連成果物を `paper/` に集約し、研究実装と生データは既存ディレクトリに置く。
- 理由: 論文に必要な履歴を保ちつつ、`git subtree split` で将来独立リポジトリへ切り出せる。
- 影響: 論文に採用する図表・集計値には、元ログと生成方法への参照が必要。

## D002: Full/Sparseの両方でCPU・atomic・round-robinを比較する

- 日付: 2026-07-28
- 状態: 採用
- 背景: GPU高速化による品質と時間の変化を、Full StressとSparse Stressの両方で同じ比較軸により評価したい。
- 選択肢: Sparse側を`RR-1`と`2sidedSGD`に分ける案、またはFull側と同じCPU/atomic/round-robinの3方式に揃える案。
- 決定: Full側は`SGD`, `atomicSGD`, `RR-SGD`、Sparse側は`SparseSGD`, `atomicSparseSGD`, `RR-SparseSGD`とする。
- 理由: 目的関数ごとにCPU基準・atomic競合処理・競合回避scheduleを対称に比較でき、GPU方式の差を説明しやすい。
- 影響: `atomicSparseSGD`を実装し、他のSparse手法と同じ前処理・制約・初期配置・学習率で検証する必要がある。`RR-1`と`2sidedSGD`は独立した主比較手法として扱わない。

## D003: 速度3指標とfinal full stressを報告する

- 日付: 2026-07-28
- 状態: 採用
- 背景: GPU kernel時間だけでは実用上の高速化を説明できず、全体時間だけではGPU並列化部分の効果を説明できない。
- 選択肢: 単一の全体時間だけを報告する案、または時間を複数区間に分解する案。
- 決定: `Iteration time`, `Algorithm time`, `CLI total time`の速度3指標と、`Final full stress`を全runで記録する。主結果はAlgorithm timeとFinal full stressとする。
- 理由: GPU更新処理、アルゴリズム全体、実利用の3つの観点を区別し、品質を保った高速化か判定できる。
- 影響: RRのschedule時間は`method_setup_time`として独立保存しつつAlgorithm timeへ含める。ファイル出力、描画、stress事後計算、デバッグ表示は速度計測から除外する。

## テンプレート

### D___: 決定の題名

- 日付:
- 状態: 提案 / 採用 / 置換 / 却下
- 背景:
- 選択肢:
- 決定:
- 理由:
- 影響:
- 置き換える決定:
