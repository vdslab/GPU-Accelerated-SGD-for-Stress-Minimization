# v5.6 大規模実験：germany_osm / italy_osm

公開対象の正式測定：成功 10/10。warmupは除外。

前処理は既存CPU経路を`--bfs-engine cpu`で指定。15反復、epsilon0.1、phase_chunks1、頂点別pivot順序、W128。各pivotでseed0〜4各1回。Stressは固定64始点・評価seed0の推定Full Stress。GPU timestampは無効。

SGD・前処理・総計算はnativeタイマーで、Stress評価・保存・圧縮を含まない。中央値と最小〜最大をJSON/CSVに保存。v5比は同一入力・pivot・seedの双方成功した組の旧/新（1超で高速）。Stress比は新/旧。旧v5はW256の歴史データであり、同時測定ではない。

| 入力 | pivot | 成功/5 | 前処理 ms | SGD ms | 総計算 ms | v5総計算速度比 | v5 SGD速度比 | Stress比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| germany_osm | 50 | 5/5 | 61749.957 | 1569.333 | 65347.920 | — | — | — |
| italy_osm | 50 | 5/5 | 33232.651 | 950.722 | 34770.477 | 1.020 | 0.743 | 1.000 |

未実行・容量停止・実測失敗を0秒として集計しない。germanyと追加pivotにはv5の成功値がなく、速度比を算出しない。GPUハード上限・RSS18GiB・空きRAM512MiB未満10秒・ディスク空き1GiBの条件を保持する。

## 実装の範囲

main `20f530952` を基に、既存の4バッファ構成の各容量をadapter上限へ拡張し、dispatchを2次元化。大容量転送の一時stagingを64MiBに制限。先に試作した頂点行タイル方式は不使用。今回の追加変更はこの実験のためにCodexが加えた変更であり、likr氏の変更そのものとして扱わない。ソース・差分・テストログ・実行バイナリhashを保存。

## 入力復元

italyの入力がローカルに存在しなかったため、SuiteSparseのUniversity of Floridaミラーから元データを再取得し、従来のnormalizeで復元。正規化後SHA256はv5記録と完全一致。証跡はvalidation/italy-input-restoration.json。
