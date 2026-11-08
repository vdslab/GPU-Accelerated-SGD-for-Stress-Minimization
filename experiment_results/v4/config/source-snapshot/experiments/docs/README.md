# 実験

CPU・Atomic GPU・Round-Robin GPUの実行時間と最終stressを比較する。
Full SGDとSparse SGDは、それぞれのCPU実装を基準とする。

- [実行手順](runbook.md)
- [v2結果](../../experiment_results/v2/README.md)
- [v1結果](../../experiment_results/v1/README.md)
- [データ](../datasets/README.md)

## 実行

- 単発: `run_single.py`で1グラフ・1手法を実行し、before/after画像と設定・速度・座標を保存。
- v2一括: `run_v2.py`で条件別に実行し、`summarize_v2.py`で図・CSV・考察を生成。
- 汎用一括: `run_experiments.py`でmanifestの条件を実行し、`aggregate_results.py`で集計。

コマンドと出力先は[実行手順](runbook.md)を参照。

## v2

| 対象 | 条件 | 正式実行数 |
|---|---|---:|
| 小・中規模10グラフ | Full・Sparse各3手法、seed 0–24 | 1,500 |
| 大規模4グラフ | Sparse 3手法、seed 0–9 | 120 |
| Full RRタイル・shared比較 | 4グラフ、T＝32 / 64 / 128 / 256 / 512 / 1024、W＝256、shared / global、10 seed | 480 |
| Full RR workgroup比較 | 同じ4グラフ、T＝1024、W＝128 / 256 / 512 / 1024、shared、10 seed | 160 |

共通条件は15反復、epsilon＝0.1、Sparseは200 pivots。主比較のFull RRはT＝1024・W＝256・shared。
各条件1回、実行を開始するたびにdataset / method / variantごとにwarmup 1回。
最大連結成分の入力をv1から再利用し、元ファイル・入力・頂点対応表のハッシュを確認。
**2,260件成功、失敗0件**。生JSON・座標・warmup・ログ・実行時ソースを保存。

リポジトリ直下から実行する。6手法のreleaseバイナリの準備は[実行手順](runbook.md)を参照。
保存済みv2を保持し、再実験には別の保存先を指定する。

```sh
python3 experiments/prepare_v2.py --results-dir experiment_results/v2-rerun
python3 experiments/run_v2.py experiment_results/v2-rerun/config/quality-timing.json --results-dir experiment_results/v2-rerun
python3 experiments/run_v2.py experiment_results/v2-rerun/config/large-sparse.json --results-dir experiment_results/v2-rerun
python3 experiments/run_v2.py experiment_results/v2-rerun/config/rr-memory-tiling.json --results-dir experiment_results/v2-rerun
python3 experiments/run_v2.py experiment_results/v2-rerun/config/rr-workgroup.json --results-dir experiment_results/v2-rerun
python3 experiments/summarize_v2.py --results-dir experiment_results/v2-rerun
```

中断した実行は同じコマンドに`--resume`を追加して再開する。入力・バイナリ・実行ソースの一致を検証する。
保存済み結果の再集計のみなら`python3 experiments/summarize_v2.py`。
実験後のJSON互換性修正は[変更記録](../../experiment_results/v2/validation/post-run-change.json)に保存。実測したソースはraw内のsnapshotで確認できる。

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
| Iteration time | SGDの15反復。同期・シャッフルを含み、前処理を除く。v1・v2の主速度指標 |
| Algorithm time (cold) | 前処理・準備・GPU初期化・転送・SGD・読戻し・後処理の合計 |
| Stress ratio | GPUの最終stress ÷ 同じ系列・seedのCPUの最終stress。低いほど良い |
| Speedup | CPU時間 ÷ GPU時間。1より大きいほど高速 |

stressは8,000頂点以下でexact、超過は64始点のsampled評価。
評価方式の異なる値やFull・Sparse間の対応seed比は混ぜない。
v1は暫定結果。標本数とバッチ差を含む制約は[結果レポート](../../experiment_results/v1/report/RESULTS.md)に記録。
