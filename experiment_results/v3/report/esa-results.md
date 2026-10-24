# v3 実験結果

データ 100 グラフ（ready: 99、unavailable: 1）。
実験適格 98 グラフ。未取得 1、代替配布との同一性未確認 0、LCC規模による除外 1 グラフ。未取得・同一性未確認・規模除外は実行前の区別として扱う。
計画条件の記録 3,210 件：success: 2,426、memory_limit: 46、warmup_failed: 720、unavailable: 15、timeout: 3。 計画 3,210 件。

条件レコードとして確定した本測定の実起動 2,370 件（success: 2,321、memory_limit: 46、timeout: 3）。既存結果の再利用 120 条件、warmup失敗による未起動 690 条件。未確定の中断attemptはこの実起動数に含めず、最終監査で別枠に数える。
再利用のうち失敗結果の参照は 15 条件。新たな実起動やwarmup失敗による新規未起動には重複計上しない。
別枠のwarmupは実起動 468 件（success: 342、memory_limit: 117、timeout: 9）。warmupは性能・stressの本集計に混ぜない。

速度は前処理を除いたSGD反復の実時間。速度比は同じグラフ・seed・family・pivot数のCPU時間÷各手法時間。stress比は各手法÷同じCPU。点はseedの中央値、範囲は最小〜最大。失敗・未取得の条件は0に置き換えない。
FullとSparseの絶対stressは同じ評価定義を用いる。頂点数8,000以下は全点対の厳密値、8,000超は固定256起点（stress seed=0）による推定値とする。小中規模20件の中にも推定値のグラフがある。SGDの制約がFullであることと、比較用stressを厳密評価することは別である。
座標等のテキストはlossless gzipで保管する。圧縮時間はnative process_wall_seconds・SGD・cold総時間の計測外で、archive_secondsとして別記録する。圧縮の有無で座標値やstressは変化しない。
SGD時間は各反復の処理区間の合計で、CPUのshuffle、GPUへの投入と完了待ちを含む。前後の初期化やcheckpoint座標取得・保存は含まない。CPUの反復前ペアclone等はこの区間の外。cold総時間は別に保存する。反復別の図はtraceの反復番号・累積和・本測定record時間の一致を検査し、不一致をiteration-time-validation.csvに残して除外する。
Sparse全グラフの補足は制約数/頂点数・RRラウンド・dispatch数（sparse-schedule.csv）。さらに代表 5 グラフは同一input/preprocess/initial SHAを照合したseed0/P200の独立診断で、頂点別の厳密な予定回数分布を補足する。予定回数はGPU実store・lock retry・待ち時間の実測ではない。
以下は現在保存されている記録の集計。実行中の再集計では未完了条件が欠ける。

SNAP原配布復旧による追加測定は、core原本 3,210 件と独立recovery原本 90 件（計 3,300 件）を保持し、未起動unavailableだった90条件だけを補完した。有効な計画条件は 3,210 件。元90件はsupersededとして追跡し、実起動済み条件を上書きしていない。
対象はcom-Amazon、com-DBLP、com-Youtube、com-LiveJournal、soc-Pokec、loc-Gowalla。公式SNAP原配布と公開ノード数・unique edge数・変換/LCCの証拠を検証した、同名の原配布からの補完である。SuiteSparse MTXとのbyte/edgewise一致は未検証で、既存Network Repository代替データを検証済み扱いに変えたものではない。power197kの未取得とG_n_pin_poutの規模除外は保持する。
coreとrecoveryはそれぞれのexecution.jsonのnative commitに照合した。commitは異なるが、native/sharedの67ファイル集合・SHA、runner 1ファイルのSHA、6実行binaryのSHA、実行制限は一致する。共通の15反復・seed・P200の元basic90 ID/設定を照合しており、新条件として二重計上しない。
有効データ一覧（保存先: tables/effective-datasets.json） · 補完検証provenance（保存先: tables/cohort.json） · 原記録と有効/置換状態（保存先: tables/cohort-source-records.csv） · 置換前の未起動90条件（保存先: tables/cohort-superseded.csv） · 追加測定の固定設定（保存先: snap-origin-recovery/config/execution.json）。

## 1. 代表グラフの描画

![代表グラフの描画](../figures/01_layout_full.png)

![代表グラフの描画](../figures/01_layout_sparse.png)

![代表グラフの描画](../figures/01_layout_sparse_partial.png)

成功CPUがある場合はCPUの最終stress中央値に最も近いseedを選び、GPUも同じseedを表示する。良好なGPU実行だけを選抜していない。CPU成功seedが0件の場合だけ、事前に計画されたseed 0を固定で使い、同seedで成功し座標が存在する手法だけを表示する。成功したGPU seedを探して置き換えない。このfallbackはNo.1の描画選択だけを対象とする事後補足で、CPU比や主結果の対応ペアには使わない。
今回のCPU基準欠如による固定seed 0の対象：Sparse: delaunay_n19。欠測枠には選択seedと失敗理由を表示し、warmup失敗による未起動も区別する。seed 0自身が失敗していれば、別seedの成功座標があっても描画しない。
描画seed・選択規則・欠測理由（保存先: tables/layout-selection.csv）に全24枠を記録する。CPU成功があるのに有効なstress値がない場合はseed 0へ移らず、選択不能として記録する。
同じグラフの全手法は座標のunion範囲から同じx/y縮尺で描画する。25万辺を超える描画は辺のみ等間隔で間引き、表示割合とSGD秒を併記する。stressは間引く前の全グラフで評価する。

別図 `01_layout_sparse_partial` は **CPU基準なし・利用可能な全layoutを示す記述的補足**。本図の固定seed 0を成功seedで差し替えるものではなく、対応比較や中央値代表とは呼ばない。CPU成功のない対象グラフについて、成功かつ保存先のあるlayoutをseedや品質で絞らず全件表示する。今回の利用可能分：delaunay_n19 / RR GPU: seed 2,3。図を読めない保存先は欠測枠と事故記録に残す。
補足図の全条件・利用可否（保存先: tables/layout-sparse-partial-availability.csv）。本図が全欠測でも、別seedの成功layoutが存在する場合がある。

## 2. 小規模20グラフ：Full SGD

![小規模20グラフ：Full SGD](../figures/02_full.png)

![小規模20グラフ：Full SGD](../figures/02_full_iteration_times.png)

- Atomic GPU：対応比較できた 19 グラフ中 13 グラフでCPUより高速。グラフ別速度比 0.06〜3.36倍、stress比 1.001〜2.921。
- RR GPU：対応比較できた 19 グラフ中 19 グラフでCPUより高速。グラフ別速度比 1.03〜15.02倍、stress比 0.954〜1.671。

Full20のID・名前対応（保存先: tables/02_full-graph-order.csv） · 対応seed数（保存先: tables/02_full-counts.csv）

上下2段は全グラフの共通x軸。背景は名前から推定したカテゴリで、分類内を頂点数順に並べる。graph-order.csvにID・名前を、counts.csvに対応seed数を保存。下端の×と数は欠測・CPU基準欠如による未対応seed数（5件中）で、0や1の測定値ではない。比は対応する成功seedのみで計算する。

## 3. 大規模80グラフ：Sparse SGD

![大規模80グラフ：Sparse SGD](../figures/03_sparse.png)

![大規模80グラフ：Sparse SGD](../figures/03_sparse_small_aux.png)

![大規模80グラフ：Sparse SGD](../figures/03_sparse_components.png)

- Atomic GPU：対応比較できた 39 グラフ中 2 グラフでCPUより高速。グラフ別速度比 0.87〜1.04倍、stress比 0.971〜1.139。
- RR GPU：対応比較できた 37 グラフ中 34 グラフでCPUより高速。グラフ別速度比 0.45〜40.67倍、stress比 1.000〜1.341。

Sparse80 overviewにはRR GPUの代表描画4枠を添える。seedは実験1と同じCPU stress中央値に近いものを選び、接続線は対応するグラフのRR速度比中央値を指す。描画seed単独の速度比を指す線ではない。取得・成功していない描画は欠測枠として残す。小規模20グラフのSparse補助比較も別図で表示する。大規模80グラフの集計とは混ぜない。時間内訳図の積み上げは計測できた成分の平均で、cold総時間との差には区間外の準備処理が含まれる。

Sparse80のID・名前対応（保存先: tables/03_sparse-graph-order.csv） · 対応seed数（保存先: tables/03_sparse-counts.csv） · Sparse小20補助のID対応（保存先: tables/03_sparse_small_aux-graph-order.csv）

上下2段は全グラフの共通x軸。背景は名前から推定したカテゴリで、分類内を頂点数順に並べる。graph-order.csvにID・名前を、counts.csvに対応seed数を保存。下端の×と数は欠測・CPU基準欠如による未対応seed数（5件中）で、0や1の測定値ではない。比は対応する成功seedのみで計算する。

## 4. グラフ構造と速度

![グラフ構造と速度](../figures/04_structure.png)

![グラフ構造と速度](../figures/04_structure_quality.png)

![グラフ構造と速度](../figures/04_structure_bins.png)

![グラフ構造と速度](../figures/04_size_series.png)

![グラフ構造と速度](../figures/04_sparse_write_load.png)

頂点数V・辺数E・次数の平均・最大値・変動係数の5指標を、Full/Sparseの上下2段で速度比と比較する。品質比も同じ2×5構成の補助図で示す。各グラフは対応seed比の中央値1点とし、Spearman CSVはmetric列で速度と品質を区別する。CV=0はsymlog軸で保持する。次数だけで実行時間は決まらず、頂点数・制約数・タイル数との交絡がある。相関係数はstructure-correlations.csvに保存し、因果関係として扱わない。点形状の構造カテゴリはグラフ名からの推定であり、確認済みの用途分類ではない。元のSuiteSparseグループはsource_groupとしてgraph-structure.csvに保存する。平均次数4/16とCV0.5/2で分けた図はグラフごとの速度比中央値を再集計し、グラフを等重みにする。同系列のサイズ曲線もグラフ名prefixからの推定系列で、規模以外の構造差を制御した実験ではない。

代表グラフの予定書込回数CSV（保存先: analysis/sparse-write-load/summary.csv）・選択とSHA照合の証拠（保存先: analysis/sparse-write-load/selection.json）。Atomicは重み0の端点もlockするため、正重みの端点更新予定数と両端lock対象予定数を分ける。いずれも1反復あたりの静的回数で、実際のGPU store、競合率、retry、待ち時間を表さない。図は全頂点CV・上位1%頂点の割合・pivot群の総回数割合。表はall群の総数・最大値を併記する。

| Graph | 予定回数の定義 | 総数 / 反復 | 最大 / 頂点 | CV | 上位1%割合 | pivot群割合 |
|---|---|---:|---:|---:|---:|---:|
| luxembourg_osm | 正重み端点更新 | 23,158,518 | 206 | 0.002 | 1.01% | 0.17% |
| luxembourg_osm | Atomic lock対象 | 46,037,904 | 114,598 | 11.886 | 50.20% | 49.78% |
| web-Stanford | 正重み端点更新 | 54,934,849 | 38,814 | 0.813 | 3.10% | 0.08% |
| web-Stanford | Atomic lock対象 | 105,946,046 | 255,264 | 17.199 | 49.74% | 48.19% |
| coAuthorsCiteseer | 正重み端点更新 | 47,091,146 | 1,572 | 0.051 | 1.33% | 0.09% |
| coAuthorsCiteseer | Atomic lock対象 | 92,514,224 | 227,319 | 16.545 | 49.76% | 49.14% |
| preferentialAttachment | 正重み端点更新 | 20,998,006 | 1,182 | 0.074 | 1.47% | 0.20% |
| preferentialAttachment | Atomic lock対象 | 40,956,244 | 99,999 | 10.885 | 49.46% | 48.83% |
| kron_g500-logn17 | 正重み端点更新 | 31,796,316 | 30,090 | 1.406 | 11.16% | 0.16% |
| kron_g500-logn17 | Atomic lock対象 | 53,324,870 | 107,900 | 9.403 | 46.64% | 40.47% |

## 5. 収束曲線と目標stress到達

![収束曲線と目標stress到達](../figures/05_convergence.png)

![収束曲線と目標stress到達](../figures/05_iteration_times.png)

`05_convergence` は座標保存を伴う収束診断の計画10seedから、反復0・1・2・3・5・8・10・15のstressを後計算した図。GPUとCPUで同じ起点を使う。曲線はcheckpointごとの各手法の成功seed中央値であり、実際の単一実行の軌跡ではない。失敗により手法間の成功seed集合が異なる場合がある。計画した全24 graph/family枠を残し、全観測欠測の枠も表示する。図のobservedは有効なcheckpointが1点以上あるseed数、missingはないseed数。併記statusは計画条件の分類で、warmup_failedを実起動の事故件数と数えない。図の観測・欠測・status対応（保存先: tables/convergence-panel-coverage.csv）。
`05_iteration_times` は通常の `study=basic` benchmarkのseed 0〜4のうち、成功し反復時間trace検査を通過した実行だけを使う別図。線は反復ごとの中央値、帯は最小〜最大、凡例n/5は利用できたseed数。座標コピー・trace IOはSGD時間から除外している。10seedの収束診断やその目標到達時間を示す図ではない。
目標は同じseed・familyのCPU 15反復後stressの1.05倍と1.01倍。最初に到達を観測したcheckpointと直前観測点の幅を記録する。これは実際の初到達を保証する区間ではない。stressが非単調になり、未観測の反復で到達後に再上昇する場合、真の初到達はこの幅より前の可能性がある。upper_ms時点には目標を満たす保存座標があるが、lower_msは真の初到達の下限と断定しない。正確な初到達時刻や未到達条件の外挿は行わない。診断の同期・保存による影響があるため、図2・3の通常実行時間とは分けて読む。

最終stress評価の整合監査（保存先: tables/convergence-stress-audit.csv）は、各成功runのoffline反復15とnative stressを比較する。snapshotは重心移動前、nativeは移動後で、数学的には並進でstressは不変だがf32/f64の丸めによりbit一致は要求しない。stress定義・起点数・seedも照合し、相対差0.1%以上を要調査として記録する。監査flagだけで成功runを除外しない。

- sgd / 1.05倍目標：必要checkpointとCPU基準が揃った 60 実行中 60 実行で到達を観測。観測またはCPU基準の欠測 0 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 0 実行、最終反復15で目標以下だったのは 60/60 実行。
- atomic_sgd / 1.05倍目標：必要checkpointとCPU基準が揃った 60 実行中 30 実行で到達を観測。観測またはCPU基準の欠測 0 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 0 実行、最終反復15で目標以下だったのは 30/60 実行。
- rr_sgd / 1.05倍目標：必要checkpointとCPU基準が揃った 60 実行中 36 実行で到達を観測。観測またはCPU基準の欠測 0 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 0 実行、最終反復15で目標以下だったのは 36/60 実行。
- sparse_sgd / 1.05倍目標：必要checkpointとCPU基準が揃った 160 実行中 160 実行で到達を観測。観測またはCPU基準の欠測 20 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 1 実行、最終反復15で目標以下だったのは 160/160 実行。
- atomic_sparse_sgd / 1.05倍目標：必要checkpointとCPU基準が揃った 160 実行中 126 実行で到達を観測。観測またはCPU基準の欠測 20 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 5 実行、最終反復15で目標以下だったのは 123/160 実行。
- rr_sparse_sgd / 1.05倍目標：必要checkpointとCPU基準が揃った 157 実行中 97 実行で到達を観測。観測またはCPU基準の欠測 23 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 6 実行、最終反復15で目標以下だったのは 91/157 実行。
- sgd / 1.01倍目標：必要checkpointとCPU基準が揃った 60 実行中 60 実行で到達を観測。観測またはCPU基準の欠測 0 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 0 実行、最終反復15で目標以下だったのは 60/60 実行。
- atomic_sgd / 1.01倍目標：必要checkpointとCPU基準が揃った 60 実行中 4 実行で到達を観測。観測またはCPU基準の欠測 0 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 0 実行、最終反復15で目標以下だったのは 4/60 実行。
- rr_sgd / 1.01倍目標：必要checkpointとCPU基準が揃った 60 実行中 14 実行で到達を観測。観測またはCPU基準の欠測 0 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 0 実行、最終反復15で目標以下だったのは 14/60 実行。
- sparse_sgd / 1.01倍目標：必要checkpointとCPU基準が揃った 160 実行中 160 実行で到達を観測。観測またはCPU基準の欠測 20 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 1 実行、最終反復15で目標以下だったのは 160/160 実行。
- atomic_sparse_sgd / 1.01倍目標：必要checkpointとCPU基準が揃った 160 実行中 103 実行で到達を観測。観測またはCPU基準の欠測 20 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 4 実行、最終反復15で目標以下だったのは 99/160 実行。
- rr_sparse_sgd / 1.01倍目標：必要checkpointとCPU基準が揃った 157 実行中 72 実行で到達を観測。観測またはCPU基準の欠測 23 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは 5 実行、最終反復15で目標以下だったのは 67/157 実行。

計画上の全10seedを残し、観測なし・CPU基準なし・checkpoint不完全を未達と区別する。下表の欠測は到達率の分母に含めない。観測幅は評価可能で到達を観測したseedだけで、直前観測時刻と初到達を観測した時刻をそれぞれ中央値にした幅（秒）。seed間の隣接観測時刻の幅中央値であり、単一実行の初到達時刻でも、真の初到達を挟む保証区間でもない。CSVには到達観測後の再上昇と最終checkpoint・反復15で目標以下かのflagsも残す。全seedの時刻幅（保存先: tables/target-arrival.csv）・集計CSV（保存先: tables/target-arrival-coverage.csv）で確認できる。

| Graph | Family | Method | 計画seed | 評価可能 | 1.05倍到達 | 1.01倍到達 | 欠測 | 1.05倍 観測幅中央値 s | 1.01倍 観測幅中央値 s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| EVA | full | atomic_sgd | 10 | 10 | 0 | 0 | 0 | — | — |
| EVA | full | rr_sgd | 10 | 10 | 0 | 0 | 0 | — | — |
| EVA | full | sgd | 10 | 10 | 10 | 10 | 0 | 1.345–1.678 | 1.678–2.510 |
| EVA | sparse | atomic_sparse_sgd | 10 | 10 | 8 | 5 | 0 | 0.205–0.291 | 0.184–0.217 |
| EVA | sparse | rr_sparse_sgd | 10 | 10 | 6 | 2 | 0 | 0.239–0.352 | 0.111–0.160 |
| EVA | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 0.097–0.148 | 0.101–0.152 |
| USpowerGrid | full | atomic_sgd | 10 | 10 | 10 | 0 | 0 | 1.237–1.845 | — |
| USpowerGrid | full | rr_sgd | 10 | 10 | 10 | 0 | 0 | 0.292–0.446 | — |
| USpowerGrid | full | sgd | 10 | 10 | 10 | 10 | 0 | 1.638–2.046 | 2.046–3.062 |
| USpowerGrid | sparse | atomic_sparse_sgd | 10 | 10 | 10 | 10 | 0 | 0.189–0.272 | 0.189–0.272 |
| USpowerGrid | sparse | rr_sparse_sgd | 10 | 10 | 10 | 9 | 0 | 0.099–0.140 | 0.098–0.138 |
| USpowerGrid | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 0.113–0.171 | 0.113–0.171 |
| bcsstk16 | full | atomic_sgd | 10 | 10 | 0 | 0 | 0 | — | — |
| bcsstk16 | full | rr_sgd | 10 | 10 | 7 | 4 | 0 | 0.265–0.395 | 0.267–0.404 |
| bcsstk16 | full | sgd | 10 | 10 | 10 | 10 | 0 | 1.940–2.907 | 1.940–2.907 |
| bcsstk16 | sparse | atomic_sparse_sgd | 10 | 10 | 8 | 8 | 0 | 0.200–0.288 | 0.200–0.288 |
| bcsstk16 | sparse | rr_sparse_sgd | 10 | 10 | 9 | 8 | 0 | 0.143–0.203 | 0.143–0.202 |
| bcsstk16 | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 0.136–0.205 | 0.136–0.205 |
| bmw7st_1 | sparse | atomic_sparse_sgd | 10 | 10 | 10 | 10 | 0 | 6.282–9.415 | 6.290–9.416 |
| bmw7st_1 | sparse | rr_sparse_sgd | 10 | 10 | 10 | 10 | 0 | 0.288–0.382 | 0.293–0.413 |
| bmw7st_1 | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 6.344–9.503 | 6.349–9.506 |
| coAuthorsCiteseer | sparse | atomic_sparse_sgd | 10 | 10 | 5 | 1 | 0 | 10.021–15.024 | 10.021–15.017 |
| coAuthorsCiteseer | sparse | rr_sparse_sgd | 10 | 10 | 0 | 0 | 0 | — | — |
| coAuthorsCiteseer | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 9.272–13.925 | 9.288–13.963 |
| coPapersDBLP | sparse | atomic_sparse_sgd | 10 | 10 | 8 | 6 | 0 | 29.391–43.766 | 29.338–43.766 |
| coPapersDBLP | sparse | rr_sparse_sgd | 10 | 7 | 1 | 0 | 3 | 2.017–2.759 | — |
| coPapersDBLP | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 25.932–38.803 | 25.932–38.803 |
| delaunay_n17 | sparse | atomic_sparse_sgd | 10 | 10 | 10 | 10 | 0 | 5.192–7.787 | 5.234–7.848 |
| delaunay_n17 | sparse | rr_sparse_sgd | 10 | 10 | 10 | 7 | 0 | 0.180–0.254 | 0.183–0.256 |
| delaunay_n17 | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 5.191–7.787 | 5.196–7.800 |
| delaunay_n20 | sparse | atomic_sparse_sgd | 10 | 0 | 0 | 0 | 10 | — | — |
| delaunay_n20 | sparse | rr_sparse_sgd | 10 | 0 | 0 | 0 | 10 | — | — |
| delaunay_n20 | sparse | sparse_sgd | 10 | 0 | 0 | 0 | 10 | — | — |
| dwt_2680 | full | atomic_sgd | 10 | 10 | 10 | 4 | 0 | 1.038–1.546 | 1.040–1.552 |
| dwt_2680 | full | rr_sgd | 10 | 10 | 9 | 9 | 0 | 0.098–0.144 | 0.098–0.144 |
| dwt_2680 | full | sgd | 10 | 10 | 10 | 10 | 0 | 0.538–0.807 | 0.538–0.807 |
| dwt_2680 | sparse | atomic_sparse_sgd | 10 | 10 | 9 | 9 | 0 | 0.152–0.223 | 0.152–0.223 |
| dwt_2680 | sparse | rr_sparse_sgd | 10 | 10 | 9 | 7 | 0 | 0.107–0.150 | 0.108–0.152 |
| dwt_2680 | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 0.048–0.072 | 0.048–0.072 |
| fe_4elt2 | full | atomic_sgd | 10 | 10 | 10 | 0 | 0 | 3.302–4.926 | — |
| fe_4elt2 | full | rr_sgd | 10 | 10 | 10 | 1 | 0 | 0.719–1.124 | 0.702–1.109 |
| fe_4elt2 | full | sgd | 10 | 10 | 10 | 10 | 0 | 9.282–11.593 | 11.593–17.381 |
| fe_4elt2 | sparse | atomic_sparse_sgd | 10 | 10 | 10 | 10 | 0 | 0.377–0.549 | 0.377–0.549 |
| fe_4elt2 | sparse | rr_sparse_sgd | 10 | 10 | 10 | 8 | 0 | 0.111–0.157 | 0.111–0.157 |
| fe_4elt2 | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 0.354–0.530 | 0.354–0.530 |
| kron_g500-logn18 | sparse | atomic_sparse_sgd | 10 | 10 | 9 | 6 | 0 | 3.400–5.677 | 3.278–5.431 |
| kron_g500-logn18 | sparse | rr_sparse_sgd | 10 | 10 | 5 | 4 | 0 | 5.358–8.775 | 8.794–13.987 |
| kron_g500-logn18 | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 1.063–2.138 | 3.196–5.327 |
| luxembourg_osm | sparse | atomic_sparse_sgd | 10 | 10 | 10 | 10 | 0 | 4.484–6.719 | 4.484–6.719 |
| luxembourg_osm | sparse | rr_sparse_sgd | 10 | 10 | 10 | 10 | 0 | 0.145–0.173 | 0.168–0.235 |
| luxembourg_osm | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 4.479–6.722 | 4.479–6.722 |
| preferentialAttachment | sparse | atomic_sparse_sgd | 10 | 10 | 7 | 5 | 0 | 3.938–5.866 | 3.938–5.866 |
| preferentialAttachment | sparse | rr_sparse_sgd | 10 | 10 | 0 | 0 | 0 | — | — |
| preferentialAttachment | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 2.653–3.684 | 3.229–4.025 |
| roadNet-CA | sparse | atomic_sparse_sgd | 10 | 0 | 0 | 0 | 10 | — | — |
| roadNet-CA | sparse | rr_sparse_sgd | 10 | 0 | 0 | 0 | 10 | — | — |
| roadNet-CA | sparse | sparse_sgd | 10 | 0 | 0 | 0 | 10 | — | — |
| smallworld | sparse | atomic_sparse_sgd | 10 | 10 | 0 | 0 | 0 | — | — |
| smallworld | sparse | rr_sparse_sgd | 10 | 10 | 1 | 0 | 0 | 0.171–0.239 | — |
| smallworld | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 3.241–4.052 | 4.037–6.055 |
| web-NotreDame | sparse | atomic_sparse_sgd | 10 | 10 | 6 | 3 | 0 | 14.908–22.338 | 14.908–22.382 |
| web-NotreDame | sparse | rr_sparse_sgd | 10 | 10 | 2 | 0 | 0 | 3.847–5.846 | — |
| web-NotreDame | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 13.512–20.281 | 13.512–20.281 |
| web-Stanford | sparse | atomic_sparse_sgd | 10 | 10 | 6 | 4 | 0 | 11.465–17.140 | 11.461–17.138 |
| web-Stanford | sparse | rr_sparse_sgd | 10 | 10 | 7 | 5 | 0 | 13.108–19.743 | 12.875–19.260 |
| web-Stanford | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 10.588–15.877 | 10.588–15.877 |
| wiki-Vote | full | atomic_sgd | 10 | 10 | 0 | 0 | 0 | — | — |
| wiki-Vote | full | rr_sgd | 10 | 10 | 0 | 0 | 0 | — | — |
| wiki-Vote | full | sgd | 10 | 10 | 10 | 10 | 0 | 4.463–6.691 | 4.463–6.691 |
| wiki-Vote | sparse | atomic_sparse_sgd | 10 | 10 | 10 | 6 | 0 | 0.280–0.400 | 0.282–0.400 |
| wiki-Vote | sparse | rr_sparse_sgd | 10 | 10 | 7 | 2 | 0 | 0.364–0.442 | 0.399–0.542 |
| wiki-Vote | sparse | sparse_sgd | 10 | 10 | 10 | 10 | 0 | 0.156–0.195 | 0.187–0.261 |

## 6. タイルサイズ

![タイルサイズ](../figures/06_tiles.png)

![タイルサイズ](../figures/06_tiles_pareto.png)

- EVA：5seed完備条件のうち今回測れた最短は rr_sgd（5/5成功）、tile_size=512、SGD 0.348 秒、stress 1,097,464.6。
- USpowerGrid：5seed完備条件のうち今回測れた最短は rr_sgd（5/5成功）、tile_size=512、SGD 0.325 秒、stress 712,678.1。
- bcsstk16：5seed完備条件のうち今回測れた最短は rr_sgd（5/5成功）、tile_size=1024、SGD 0.391 秒、stress 172,083.8。
- dwt_2680：5seed完備条件のうち今回測れた最短は rr_sgd（5/5成功）、tile_size=1024、SGD 0.182 秒、stress 45,611.1。
- fe_4elt2：5seed完備条件のうち今回測れた最短は rr_sgd（5/5成功）、tile_size=512、SGD 0.915 秒、stress 2,626,016.0。
- wiki-Vote：5seed完備条件のうち今回測れた最短は rr_sgd（5/5成功）、tile_size=1024、SGD 0.572 秒、stress 5,359,121.8。
- EVA / T=1：同seed CPU stress比の中央値 1.000、SGD 117.673 秒（対応 5/5 seed）。
- EVA / T=1024：同seed CPU stress比の中央値 1.122、SGD 0.417 秒（対応 5/5 seed）。
- USpowerGrid / T=1024：同seed CPU stress比の中央値 1.019、SGD 0.418 秒（対応 5/5 seed）。
- bcsstk16 / T=1024：同seed CPU stress比の中央値 0.954、SGD 0.391 秒（対応 5/5 seed）。
- dwt_2680 / T=1：同seed CPU stress比の中央値 1.000、SGD 56.602 秒（対応 5/5 seed）。
- dwt_2680 / T=1024：同seed CPU stress比の中央値 1.004、SGD 0.182 秒（対応 5/5 seed）。
- fe_4elt2 / T=1024：同seed CPU stress比の中央値 1.017、SGD 1.147 秒（対応 5/5 seed）。
- wiki-Vote / T=1024：同seed CPU stress比の中央値 1.671、SGD 0.572 秒（対応 5/5 seed）。
全予定パラメータをx軸に残し、全欠測は軸下端の× 0/5、部分成功は手法色のn/5で示す。欠測はNaNとして線を切り、0の測定値へ置換しない。条件別・指標別のcoverage（保存先: tables/tiling-parameter-coverage.csv）で成功・欠測・未記録数を確認できる。


時間だけで選ぶと品質を落とす可能性があるため、同じ条件のstressも併記する。時間–品質散布図はseed 0–4の5件すべてが成功したTだけを描き、Tラベルを付ける。CPU水平線と散布図のCPU点はbasic Full CPUの5seedが完備したときだけ表示する別枠の参照であり、T比較の測定点ではない。欠測・部分成功条件はtile-pareto.csvに残し、散布図に値を補わない。破線は測定した中央値に対して非劣位の条件を結ぶもので、未測定条件を含む最適性の証明ではない。最短条件は今回の測定範囲での値。

## 7. pivot数

![pivot数](../figures/07_pivots.png)

- bmw7st_1：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.320 秒、stress 399,650,838.7。 cold総時間 2.990 秒。
- coAuthorsCiteseer：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.813 秒、stress 5,575,457,825.7。 cold総時間 2.824 秒。
- delaunay_n20：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.278 秒、stress 47,803,912,362.0。 cold総時間 9.140 秒。
- luxembourg_osm：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.067 秒、stress 163,311,293.5。 cold総時間 0.815 秒。
- preferentialAttachment：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.573 秒、stress 1,114,683,340.2。 cold総時間 1.496 秒。
- web-Stanford：5seed完備条件のうち今回測れた最短は sparse_sgd（5/5成功）、pivots=50、SGD 4.190 秒、stress 4,952,670,900.7。 cold総時間 6.275 秒。
全予定パラメータをx軸に残し、全欠測は軸下端の× 0/5、部分成功は手法色のn/5で示す。欠測はNaNとして線を切り、0の測定値へ置換しない。条件別・指標別のcoverage（保存先: tables/pivots-parameter-coverage.csv）で成功・欠測・未記録数を確認できる。


pivot増加は制約数・前処理・メモリと品質の交換になる。主図の時間は前処理・GPU初期化・SGD・readbackを含むcoldアルゴリズム総時間で、入力読込・stress評価・描画ファイル保存を含まない。summary.csvにはSGD単独・前処理・process wall・peak RSSも保存。Peak RSSはGPU専用VRAM量ではない。

## 事故・欠測と調査

- memory_limit 46 件：記録参照
- timeout 3 件：記録参照
- warmupの実失敗 126 件：memory_limit 117、timeout 9。
- warmup 333SP / atomic_sparse_sgd：memory_limit、
- warmup 333SP / rr_sparse_sgd：memory_limit、
- warmup 333SP / sparse_sgd：memory_limit、
- warmup AS365 / atomic_sparse_sgd：memory_limit、
- warmup AS365 / rr_sparse_sgd：memory_limit、
- warmup AS365 / sparse_sgd：memory_limit、
- warmup Emilia_923 / atomic_sparse_sgd：memory_limit、
- warmup Emilia_923 / rr_sparse_sgd：memory_limit、
- warmup Emilia_923 / sparse_sgd：memory_limit、
- warmup M6 / atomic_sparse_sgd：memory_limit、
- warmup M6 / rr_sparse_sgd：memory_limit、
- warmup M6 / sparse_sgd：memory_limit、
- warmup NACA0015 / atomic_sparse_sgd：memory_limit、
- warmup NACA0015 / rr_sparse_sgd：memory_limit、
- warmup NACA0015 / sparse_sgd：memory_limit、
- 未起動の原因別条件数：memory_limit 645、timeout 45。対応するwarmup_run_idは 126 件。
- warmup_failed 690 条件はwarmupの失敗を受けて起動しなかった条件。warmup_run_idで失敗した実行に対応する。同じwarmupが5seedを止めても、実事故は1件として数える。
- LCCがlarge基準未満のため規模除外：1 グラフ（G_n_pin_pout）、未起動 15 計画条件。実行時のtimeoutやGPU事故とは区別する。
- 入力データ未取得：1 グラフ（power197k）、未起動 15 計画条件。実行時のtimeoutやGPU事故とは区別する。

timeoutは計画した時間上限で停止した実行で、完了時間の実測値ではない。memory_limitは資源制限であり、そのグラフに対する性能やstressを0とみなさない。取得不能データの類似グラフへの差し替えは行わない。

- CPU対応比較から 11 記録を除外。理由はpair-exclusions.csv。


