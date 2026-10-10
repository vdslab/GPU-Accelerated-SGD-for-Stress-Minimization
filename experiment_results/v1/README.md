# 実験結果 v1

2026-10-11開始。SGD更新部分の時間と、同じ条件で得たfull stressを比較する初回実験。
同日、Atomic Sparse SGDを同じグラフ・seed・反復条件で追加した。

**対象：16グラフ、6手法。小・中規模600実行、大規模Sparse54実行、合計654正式実行。**

- [結果と考察](report/RESULTS.md)
- [グラフ別の集計表](tables/results.md)
- [CPU版に対するstress比](figures/01_stress_ratio.png)
- [SGD部分の高速化率](figures/02_iteration_speedup.png)
- [大規模Sparseの比較](figures/06_sparse_scaling.png)
- [代表レイアウト](figures/07_representative_layouts.png)
- [RRのラウンド数](figures/08_rr_sparse_rounds.png)
- [整合性検証とソース一致の証拠](tables/validation.json)

正式な非劣性評価に必要な25 seedは未収集。これらはv1の暫定図表であり、
標本数を増やして最終論文用の結果を確定する。

| フォルダ | 内容 |
|---|---|
| `config/` | 実験条件・入力checksum・共通グラフの処理記録 |
| `raw/` | 元のJSONL、環境情報、stderr、最終座標と初期座標 |
| `tables/` | CSV・Markdown・LaTeXの集計表と検証結果 |
| `figures/` | Matplotlibの同じ図から直接保存したPNG・SVGと代表レイアウト |
| `report/` | 結果の解釈、限界、追加実験の優先順位 |

大きな座標ファイル・実行中の出力はGit管理対象外。
元JSONL・環境情報・表・図・報告の必要ファイルを選んでコミットする。

## 条件

- Full：SGD、Atomic SGD、RR-SGD。Sparse：Sparse SGD、Atomic Sparse SGD、RR-Sparse SGD。
- 小・中規模10グラフ × 6手法 × seed 0〜9 = 600正式実行。
- 大規模6グラフ × Sparseの3手法 × seed 0〜2 = 54正式実行。
- 各15反復、epsilon=0.1、Sparseのpivot数200。
- 無向化、自己ループ除去、最大連結成分抽出を事前に共通化する。
  派生MTXと頂点対応表はGit対象外の `prod_data/v1_lcc/` に置く。
- 同じdataset/methodにつき最初にwarmupを1回行う。
- 主速度指標は `iteration_time_ms`：15反復、CPUシャッフル、GPU投入・同期を含み、
  距離計算、事前schedule、初回転送、stress評価、描画を除く。
  Atomic Sparseの未完了ペアの抽出と再投入も含む。
- exact stressとsampled stressを分ける。8,000頂点以下はexact、超過は64始点の推定値
  （評価seed=0）。大規模の品質は探索的な補助評価とする。
- 品質は25 seedの正式な非劣性評価に先立つ10 seedの暫定結果。
  大規模3 seedの速度も探索的結果である。
- 各実行のtimeout：小・中規模120秒、大規模180秒。失敗を0秒として集計しない。

## 追加測定と過去結果の扱い

既存5手法の536実行は元のJSONLを改変せず再利用し、Atomic Sparseの118実行を追加した。
既存バッチはseedごとに手法順を回しているが、Atomic Sparseは同日の追加バッチである。
Git commitは異なるため、既存5手法と共通処理のディレクトリが同一Git treeであることを検証し、
入力・初期配置・Sparse前処理・評価器・機器・反復条件の一致と合わせて対応比較を許可する。
検証結果は `tables/validation.json`、出所とchecksumは `tables/provenance.json` に保存する。
同じ測定バッチで全6手法を実行した結果ではないため、機器の時間変動は今後の再実験で確認する。

追加測定はcleanな実験用checkoutで行った。入力リンクによりdirtyとなった初期記録は
`discarded-dirty-records.jsonl` に残して全件再測定し、最終集計から除外している。
`environment.json` は初期バッチ開始時のsnapshotであり、入力リンクによるdirtyを含む。
最終採用runのcommitとdirty状態は `results.jsonl` の各記録を参照する。
追加バッチの成果物はこのv1へ移動しており、元の記録は変更せず、
`artifact-relocation.json` で元の実行場所と現在の保存場所を対応させる。

## 再現

リポジトリ直下のcleanなcheckoutで実行する。Pythonにnumpy/scipy/matplotlibが必要。

```sh
python3 -B experiments/prepare_v1.py
# 既存の5手法のバッチ
python3 -B experiments/run_v1.py experiment_results/v1/config/quality-timing.json
python3 -B experiments/run_v1.py experiment_results/v1/config/large-sparse.json
# Atomic Sparseの追加バッチ
cargo build --release --locked --manifest-path vram-lock-sparse/Cargo.toml
python3 -B experiments/run_v1.py experiment_results/v1/config/atomic-sparse-quality-timing.json
python3 -B experiments/run_v1.py experiment_results/v1/config/atomic-sparse-large-sparse.json
python3 -B experiments/summarize_v1.py
```

中断後はrunnerに `--resume` を付ける。成功済みのrunは省略する。
既存5手法のmanifestは当時の条件を保持し、追加手法は別manifestで管理する。
別commitの記録を再利用する場合、集計時にソース一致を再検証する。
図表はrawを変更せず再生成できる。代表配置の再生成にはローカルの座標ファイルも必要。

SGD論文のFigure 7・10・14、DRGraphのTable 2・5・Figure 6を参考に、
白背景、控えめな補助線、統一色、平均と範囲または中央値とIQRを使う。
