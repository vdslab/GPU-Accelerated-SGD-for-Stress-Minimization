# 実験結果 v1

2026-10-11開始。Full / Sparse SGDのCPU・Atomic・RR方式を比較。

| 対象 | 条件 | 正式実行 |
|---|---|---:|
| 小・中規模10グラフ | 6手法 × seed 0–9 | 600 |
| 大規模6グラフ | Sparse 3手法 × seed 0–2 | 54 |

15反復、epsilon=0.1、Sparseは200 pivots。各dataset/methodにwarmup 1回。

- 速度: `iteration_time_ms`（シャッフル・GPU投入・同期を含む。前処理・初回転送・stress評価は除外）
- 品質: final full stress。8,000頂点以下はexact、超過は64始点のsampled評価。
- 成功654件、失敗0件。標本数不足の暫定評価（`publication_ready=false`）。
- Atomic Sparseは追加バッチ。既存5手法との入力・初期配置・共通ソースの一致を検証済み。

[考察](report/RESULTS.md) · [表](tables/results.md) · [図](figures/) · [検証](tables/validation.json) · [出所](tables/provenance.json)

`config/`・`raw/`・`tables/`・`figures/`・`report/`は保存済み結果。旧`prod_data/`参照は移動記録を通して現在の`data/`へ解決。

[実験手順](../../experiments/docs/runbook.md)
