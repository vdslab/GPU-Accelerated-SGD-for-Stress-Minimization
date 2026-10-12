# 実験

CPU・Atomic GPU・Round-Robin GPUの実行時間と最終stressを比較する。
Full SGDとSparse SGDは、それぞれのCPU実装を基準とする。

- [実行手順](runbook.md)
- [v1結果](../../experiment_results/v1/README.md)
- [データ](../datasets/README.md)

## 実行

- 単発: `run_single.py`で1グラフ・1手法を実行し、before/after画像と設定・速度・座標を保存。
- 一括: `run_experiments.py`でmanifestの条件を実行し、`aggregate_results.py`で集計。

コマンドと出力先は[実行手順](runbook.md)を参照。

## v1

| 対象 | 条件 |
|---|---|
| 小・中規模10グラフ | Full・Sparse各3手法、seed 0–9 |
| 大規模6グラフ | Sparse 3手法、seed 0–2 |
| 共通パラメータ | 15反復、epsilon 0.1、Sparse 200 pivots |
| 測定回数 | 各条件1回、dataset/methodごとにwarmup 1回 |
| 実行結果 | 654件成功、失敗0件 |

入力は無向化・自己ループ除去・最大連結成分で統一。
Atomic Sparseは追加バッチで測定し、既存のCPU・RR結果と比較している。

## 指標

| 指標 | 内容 |
|---|---|
| Iteration time | SGDの15反復。同期・シャッフルを含み、前処理を除く。v1の主速度指標 |
| Algorithm time (cold) | 前処理・準備・GPU初期化・転送・SGD・読戻し・後処理の合計 |
| Stress ratio | GPUの最終stress ÷ 同じ系列・seedのCPUの最終stress。低いほど良い |
| Speedup | CPU時間 ÷ GPU時間。1より大きいほど高速 |

stressは8,000頂点以下でexact、超過は64始点のsampled評価。
評価方式の異なる値やFull・Sparse間の対応seed比は混ぜない。
v1は暫定結果。標本数とバッチ差を含む制約は[結果レポート](../../experiment_results/v1/report/RESULTS.md)に記録。
