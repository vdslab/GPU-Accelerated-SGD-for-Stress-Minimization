# 実験結果 v1

2026-10-11開始。SGD更新部分の時間と、同じ条件で得たfull stressを比較する初回実験。

| フォルダ | 内容 |
|---|---|
| `config/` | 実験条件・入力checksum・共通グラフの処理記録 |
| `raw/` | 元のJSONL、環境情報、stderr、最終座標と初期座標 |
| `tables/` | CSV・Markdown・LaTeXの集計表と検証結果 |
| `figures/` | 論文向けのPNG・SVG図と代表レイアウト |
| `report/` | 結果の解釈、限界、追加実験の優先順位 |

大きな座標ファイル・実行中の出力はGit管理対象外。
完了後に元JSONL・環境情報・表・図・報告の必要ファイルだけを選んでコミットする。
次の条件変更ではv1を上書きせず、v2を作る。

## 条件

- 実装済み5手法：SGD、atomic SGD、RR-SGD、Sparse SGD、RR-Sparse SGD。
- 小・中規模10グラフ × 5手法 × seed 0〜9 = 500正式実行。
- 大規模6グラフ × Sparseの2手法 × seed 0〜2 = 36正式実行。
- 各15反復、epsilon=0.1、Sparseのpivot数200。
- 無向化、自己ループ除去、最大連結成分抽出を事前に共通化する。
  派生MTXと頂点対応表はGit対象外の `prod_data/v1_lcc/` に置く。
- 同じdataset/methodにつき最初にwarmupを1回行う。手法の実行順はseedごとに回す。
- 主速度指標は `iteration_time_ms`：15反復、CPUシャッフル、GPU投入・同期を含み、
  距離計算、事前schedule、初回転送、stress評価、描画を除く。
- exact stressとsampled stressを分ける。8,000頂点以下はexact、超過は64始点の推定値
  （評価seed=0）。大規模の品質は探索的な補助評価とする。
- 品質は25 seedの正式な非劣性評価に先立つ10 seedの暫定結果。
  大規模3 seedの速度も探索的結果である。Sparse atomicは未実装。
- 各実行のtimeout：小・中規模120秒、大規模180秒。失敗を0秒として集計しない。

## 再現

リポジトリ直下で実行する。Pythonにnumpy/scipy/matplotlibが必要。

```sh
python3 experiments/prepare_v1.py
python3 experiments/run_v1.py experiment_results/v1/config/quality-timing.json
python3 experiments/run_v1.py experiment_results/v1/config/large-sparse.json
python3 experiments/summarize_v1.py
```

中断後はrunnerに `--resume` を付ける。生データは改変せず、図表を再生成する。
SGD論文のFigure 7・10・14、DRGraphのTable 2・5・Figure 6を参考に、
白背景、控えめな補助線、統一色、平均と範囲または中央値とIQRを使う。
