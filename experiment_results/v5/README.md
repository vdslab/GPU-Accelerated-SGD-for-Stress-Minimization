# v5 実験結果・総覧

更新：2026-11-09。初回測定に加え、大規模・grid/workgroup・EX1・EX2まで測定と報告が完了。
**正式測定10,180成功、実測失敗0、GPU上限による未実行220条件。** warmup・予備検証・再試行前の記録はこの成功件数に含めない。図表集は既存測定の再集計で、追加の正式測定として数えない。

## 最初に見るファイル

- **現在のRR128とCPU/Atomicの比較図**：[チャート集](ex-chart-collection-rr128-20261108/README.md)。CPU/Atomicは初回v5、RR128はEX2の別セッション測定。
- **RR128の結果と考察**：[EX2](ex2-rr128-speed-stress-20261108/reports/RESULTS.md)。Full38・Sparse56グラフ、計1,410成功。
- **128対1024の比較**：[グラフ別図表](rr128-vs1024-charts-20261108/README.md)。SparseのW1024との対応比較は実測のある3グラフのみ。
- **初回v5の全体像・論文用図**：[paper](paper/README.md)。初回6,030＋大規模130成功の集計で、後続EX1/EX2を含めない。
- **再現情報の所在と現存確認**：[整理時の検証記録](validation/repository-inventory-20261109.json)。

## 実験の一覧と完了根拠

| 実験・保存先 | 条件 | 成功 | 未実行 | 結果・完了根拠 |
|---|---|---:|---:|---|
| 初回タイル `raw/tile/` | 10グラフ、CPU＋RR Fullの11T、W1024固定、seed10〜14×3回 | 1,800 | 0 | [結果](reports/tile.md)・[完了照合](validation/final-completion.json) |
| 初回通常比較 `raw/comparison/` | Full38・Sparse56グラフ、6方式、seed0〜4×3回、RR Full T=W1024／RR Sparse W256 | 4,230 | 0 | [速度](reports/speed.md)・[Stress](reports/stress.md)・[条件照合](validation/comparison-completion.json) |
| [大規模](large/README.md) | 14グラフ×5pivot×5seed、RR Sparseのみ、各1回 | 130 | 220 | [結果](large/reports/large.md)・[最終QA](large/validation/final-qa.json)・[進捗](large/progress.json) |
| [追加grid/workgroup](additional-grid-workgroup/README.md) | Full6グラフ×5T、Sparse6グラフ×2W×2p、各5seed×3回 | 810 | 0 | [結果](additional-grid-workgroup/reports/results.md)・[完了状態](additional-grid-workgroup/progress.json) |
| [EX1：W=T](ex1-tile-w-equal-t-20261107/README.md) | 初回と同じ10グラフ・11T、CPUも再測定、全件厳密Stress | 1,800 | 0 | [結果](ex1-tile-w-equal-t-20261107/reports/tile.md)・[完了](ex1-tile-w-equal-t-20261107/completion.json) |
| [EX2：RR128](ex2-rr128-speed-stress-20261108/README.md) | RR Full T=W128、RR Sparse W128、38/56グラフ×5seed×3回 | 1,410 | 0 | [結果](ex2-rr128-speed-stress-20261108/reports/RESULTS.md)・[完了](ex2-rr128-speed-stress-20261108/completion.json) |
| **合計** | warmup・予備検証・再集計を除外 | **10,180** | **220** | 実測失敗0、残り0 |

初回の `progress.json`・`validation/final-completion.json` は**初回6,030件のみ**の完了証跡。`large_graphs_executed: 0` はその時点の記録であり、後続実験を未完了とする意味ではない。各追加実験の完了記録は上表を参照する。

## 結果の要点

- 初回通常比較のRR Fullは、全38グラフでCPU比の幾何平均がSGD **1.383倍**・総計算 **0.630倍**、Stress/CPU **1.0656**。RR Sparseは全56グラフで **1.871倍・1.021倍・1.0171**。小規模の固定費とグラフごとの品質差を含む全対象の集計である。
- 大規模は130成功・220条件を実装/GPUの上限により確保前に未実行。最大は英国道路網7,733,822頂点・p50・5seed成功。CPU比較は測定していない。
- 追加gridでは最速Tが128で5件、256で1件、1024で0件。SparseのW1024にも一律の速度優位はなく、W256との180対応組では最終Stressが一致した。
- EX1のW=Tでは最速Tが128で7件、256で2件、64で1件。共通T128は全入力で最速という意味ではない。
- EX2は初回RRに対し、SGD速度比の幾何平均がFull **1.592倍**・Sparse **1.155倍**、総計算は **1.195倍・1.116倍**。Sparseは840実行で以前のStressと完全一致。Fullは品質差がグラフに依存する。異なるセッション・timestamp条件の新旧構成比較なので、速度差をT/W変更だけの因果効果としない。

## 共通条件と比較上の注意

全測定は15反復・epsilon0.1、RRはphase_chunks1。通常Sparseはp=min(200,n)。原則として対応graph・seed・repetitionの比を先に計算し、seed内3回中央値→5seed中央値・最小〜最大で集計する。大規模は各seed1回。同seed3回を15独立seedと数えず、min-maxを信頼区間と呼ばない。

| 項目 | 初回・追加grid | EX1 | EX2 |
|---|---|---|---|
| Full Stress | n≤8000で厳密、超過は256始点の推定 | 全10グラフ厳密 | Full全38グラフ厳密 |
| Sparse Stress | 通常256始点（n≤8000は厳密）、大規模64始点、評価seed0 | 対象外 | 初回通常比較と同じ |
| GPU timestamp | 無効、SGDはホスト経過時間 | 無効 | 有効、GPU pass時間も取得 |
| 描画計測 | なし | なし | あり。結果取得から初回描画完了までに評価・引渡し等を含む |

SGD経過時間・純GPU pass時間・総計算時間・描画込み時間は別指標。`total_compute_time_ms` は入力読込からCPU上の最終座標までで、Stress評価・保存・描画は含めない。Full/Sparseの目的関数・初期配置、p間の初期配置も区別する。

比較図では、初回推定だったFullのStressを保存座標から厳密再評価した値を**別オーバーレイ**として使っている。EX2単体報告の厳密値のみ34グラフの比較と、再評価を含むチャートの38グラフ比較は対象が異なる。元recordは変更しない。

## 再集計・図表集の使い分け

| 保存先 | 用途・対象 |
|---|---|
| [paper/](paper/README.md) | 初回＋大規模の論文候補9図、計時カバレッジと限界 |
| [execution-cost/](execution-cost/README.md) | 初回＋大規模の準備＋SGD＋読戻し、転送込みも別集計 |
| [ex-chart-collection-rr128-20261108/](ex-chart-collection-rr128-20261108/README.md) | 旧CPU/Atomic対EX2 RR128、16図。Fullは厳密再評価込み |
| [rr128-vs1024-charts-20261108/](rr128-vs1024-charts-20261108/README.md) | Full38・Sparse3グラフの128/1024比較。旧Sparse W256をW1024に代用しない |

各実験内の `config/` は入力ハッシュ・予定・条件、`raw/` はrun別記録・コマンド・出力ログ、`tables/` は集計、`figures/` はPNG/PDF、`source-snapshot/` は実行時ソース、`validation/` は当時の検証証跡。warmup・検証は正式測定と分離する。実行時バイナリ等の大容量ローカル資産はGit管理外。

## 現在の保存状態（2026-11-09確認）

**2026-11-08の容量整理で座標・辺共有データ・キャッシュ等を削除済み。** これは測定後の保存容量整理で、時間・Stressなどの数値record、設定、ログ、集計表、報告書、既存の図表は保持している。[容量整理の一覧と証跡](../storage-cleanup-20261108/README.md)を参照。

過去のREADME・report・archive検証にある「圧縮座標を保存」「復元可能」は測定完了時の状態であり、現在そのファイルが存在する保証ではない。recordに残る座標パス・当時の復元検証は来歴として保持する。**既存の数値を使う集計と、座標からのStress再評価・レイアウト再描画は区別する。** 後者は座標の再取得または再測定が必要であり、既存数値だけでは再生成できない。

初回の数値集計・タイル図は `python3 experiments/report_v5_comparison.py` と `python3 experiments/report_v5_tile.py` が入口。座標を必要とするレイアウト図まで含めた全レポートの再生成は、容量整理後の現存データだけで可能とはしていない。追加実験のrunnerを再実行すると測定・報告出力を更新し得るため、閲覧・確認には完了記録と保存済み図表を使う。
