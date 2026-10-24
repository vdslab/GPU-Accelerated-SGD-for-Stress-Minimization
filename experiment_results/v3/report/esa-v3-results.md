## 実験結果（v3）

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
主計画と補完計画の処理を完了した最終集計。成功・上限停止・未起動・入力欠測を区別する。

SNAP原配布復旧による追加測定は、core原本 3,210 件と独立recovery原本 90 件（計 3,300 件）を保持し、未起動unavailableだった90条件だけを補完した。有効な計画条件は 3,210 件。元90件はsupersededとして追跡し、実起動済み条件を上書きしていない。
対象はcom-Amazon、com-DBLP、com-Youtube、com-LiveJournal、soc-Pokec、loc-Gowalla。公式SNAP原配布と公開ノード数・unique edge数・変換/LCCの証拠を検証した、同名の原配布からの補完である。SuiteSparse MTXとのbyte/edgewise一致は未検証で、既存Network Repository代替データを検証済み扱いに変えたものではない。power197kの未取得とG_n_pin_poutの規模除外は保持する。
coreとrecoveryはそれぞれのexecution.jsonのnative commitに照合した。commitは異なるが、native/sharedの67ファイル集合・SHA、runner 1ファイルのSHA、6実行binaryのSHA、実行制限は一致する。共通の15反復・seed・P200の元basic90 ID/設定を照合しており、新条件として二重計上しない。
[有効データ一覧](../tables/effective-datasets.json) · [補完検証provenance](../tables/cohort.json) · [原記録と有効/置換状態](../tables/cohort-source-records.csv) · [置換前の未起動90条件](../tables/cohort-superseded.csv) · [追加測定の固定設定](../snap-origin-recovery/config/execution.json)。

**実施条件。** 2026年10月24〜25日、Apple M4 Pro・24 GiB unified RAM・Metalで実施。15反復、epsilon=0.1、基本・T・P比較はseed 0〜4、収束診断は0〜9。Sparse基本P=200、Full RR基本T=1024/W=1024。入力は無重み・無向化した最大連結成分で、数値0を先に除去し、loop/重複を整理した。1実行ずつ測り、同設定のwarmupは別枠とした。native全プロセスの上限は120秒・観測RSS 8 GiB。

**比較範囲。** CPUはf64、GPUはf32。Full Atomicは明示pair shuffleなし・lock再試行上限なしで、更新順IDと更新前座標の監査buffer確保・書込も計測負荷に含む。Sparse Atomicにはshader内128回とhost再投入1024pass等の制限が残る。両者を一括して「上限なし」とはしない。精度・更新順・微小距離処理・コンパイル経路が違うため、品質差をlockや順序だけの原因へ帰属しない。[6実装の比較範囲](implementation-scope.md)。

### 1. 代表グラフの描画

![代表グラフの描画](../figures/01_layout_full.png)

![代表グラフの描画](../figures/01_layout_sparse.png)

![代表グラフの描画](../figures/01_layout_sparse_partial.png)

**考察。** 同seedでも速さと描画品質は両立しない。Fullの代表EVAはCPU stress約95.5万に対しAtomic約113.0万、RR約113.7万で、18〜19%高かった。一方dwt_2680は向きが大きく違ってもstressは近い。SparseのcoAuthorsCiteseerはRRが速いがstressが高い。見た目だけで正誤を判定せず、以下の全seed比較と併せて読む。delaunay_n19のCPU基準欠如は欠測として残し、成功した別seedの全2枚だけを補足図へ分けた。

成功CPUがある場合はCPUの最終stress中央値に最も近いseedを選び、GPUも同じseedを表示する。良好なGPU実行だけを選抜していない。CPU成功seedが0件の場合だけ、事前に計画されたseed 0を固定で使い、同seedで成功し座標が存在する手法だけを表示する。成功したGPU seedを探して置き換えない。このfallbackはNo.1の描画選択だけを対象とする事後補足で、CPU比や主結果の対応ペアには使わない。
今回のCPU基準欠如による固定seed 0の対象：Sparse: delaunay_n19。欠測枠には選択seedと失敗理由を表示し、warmup失敗による未起動も区別する。seed 0自身が失敗していれば、別seedの成功座標があっても描画しない。
[描画seed・選択規則・欠測理由](../tables/layout-selection.csv)に全24枠を記録する。CPU成功があるのに有効なstress値がない場合はseed 0へ移らず、選択不能として記録する。
同じグラフの全手法は座標のunion範囲から同じx/y縮尺で描画する。25万辺を超える描画は辺のみ等間隔で間引き、表示割合とSGD秒を併記する。stressは間引く前の全グラフで評価する。

別図 `01_layout_sparse_partial` は **CPU基準なし・利用可能な全layoutを示す記述的補足**。本図の固定seed 0を成功seedで差し替えるものではなく、対応比較や中央値代表とは呼ばない。CPU成功のない対象グラフについて、成功かつ保存先のあるlayoutをseedや品質で絞らず全件表示する。今回の利用可能分：delaunay_n19 / RR GPU: seed 2,3。図を読めない保存先は欠測枠と事故記録に残す。
[補足図の全条件・利用可否](../tables/layout-sparse-partial-availability.csv)。本図が全欠測でも、別seedの成功layoutが存在する場合がある。

### 2. 小規模20グラフ：Full SGD

![小規模20グラフ：Full SGD](../figures/02_full.png)

![小規模20グラフ：Full SGD](../figures/02_full_iteration_times.png)

**考察。** Full RRは比較可能19/20 graph・95組すべてのgraphでSGD速度比中央値が1を超え、graph等重み中央値は7.384倍。ただし19中9 graphでstress比中央値が1.05を超えた。wiki-Voteは11.646倍でもstress 1.671倍、ukのRR seed0は3.022倍のstressとなった。AtomicのSGD比中央値は1.298倍。RRのcold比は3.769倍、process wall比は2.597倍で、SGDの利得がそのまま全工程の利得にはならない。[Full20の詳細考察](full-basic-observations.md)。

- Atomic GPU：対応比較できた 19 グラフ中 13 グラフでCPUより高速。グラフ別速度比 0.06〜3.36倍、stress比 1.001〜2.921。
- RR GPU：対応比較できた 19 グラフ中 19 グラフでCPUより高速。グラフ別速度比 1.03〜15.02倍、stress比 0.954〜1.671。

[Full20のID・名前対応](../tables/02_full-graph-order.csv) · [対応seed数](../tables/02_full-counts.csv)

上下2段は全グラフの共通x軸。背景は名前から推定したカテゴリで、分類内を頂点数順に並べる。graph-order.csvにID・名前を、counts.csvに対応seed数を保存。下端の×と数は欠測・CPU基準欠如による未対応seed数（5件中）で、0や1の測定値ではない。比は対応する成功seedのみで計算する。

### 3. 大規模80グラフ：Sparse SGD

![大規模80グラフ：Sparse SGD](../figures/03_sparse.png)

![大規模80グラフ：Sparse SGD](../figures/03_sparse_small_aux.png)

![大規模80グラフ：Sparse SGD](../figures/03_sparse_components.png)

大規模80候補のうち78件が入力条件を満たした。Sparse basic1,200条件は成功565、正式RSS停止27、正式timeout3、warmup_failed590、未取得15だった。warmup_failedにはサイズ除外15と、115件の実warmup停止による575件の未起動が含まれる。公式SNAP6件を補完し、Gowalla・DBLP・Amazonの45正式条件が成功した一方、YouTube・LiveJournal・Pokecの45正式条件は9件のwarmup RSS停止で未起動となった。これらを元のunavailable90件と二重計上しない。

同じ入力・初期配置・前処理・評価定義を満たす同seed成功組で比較すると、Atomicは39 graph/187組、RRは37 graph/181組が対応した。各graph内の対応比中央値を等重みで集計したSGD速度比中央値はAtomic0.944倍、RR25.116倍、stress比中央値は1.001145、1.006002だった。RRで速度改善とstress+5%以内を両立したのは21/37 graph、+1%以内は19/37 graph。全5seedが揃った35 graphではそれぞれ20/35、18/35で、成功部分集合での結果である。

RRのSGD25.116倍を処理全体の高速化とは呼ばない。同じ対応集合のcold algorithm比中央値は2.192倍、入力・stress評価・座標保存も含む全プロセスwall比は1.743倍だった。追加3件でもRRはDBLP29.70倍、Amazon22.02倍、Gowalla1.48倍と速いが、stressはそれぞれ+16.50%、+11.64%、+9.63%。追加によって5%品質基準を満たす高速graph数は増えなかった。

Luxやrgg18は高速化と近い品質を両立したが、coAuthorsCiteseerやsmallworldではRRのstressが約19%悪化し、kron17/18とweb-StanfordではRR自体がCPUより遅かった。RRのround・dispatchが多い例は時間増加と整合する一方、入力最大次数17・CV0.134のsmallworldでも品質が悪化するため、次数分布だけでは説明できない。頂点別の予定更新・lock対象数は補助指標で、実store、衝突率、retry、待ち時間の実測とは区別する。

出典と限界：core3210＋補完90原記録を保持した有効3210条件。追加6件はSNAP公式原配布から再構成し、SuiteSparse MTXとの辺・行番号の直接照合は未実施。CPU f64 / GPU f32、更新順・微小距離処理・実装経路も異なるため、品質差をlock又は更新順だけの原因へ帰属しない。小20補助は別集計で、Atomic/RRのSGD比中央値0.511/0.804倍、300条件全成功だった。


[全graphの対応比と欠測](sparse-basic-observations.md)。

- Atomic GPU：対応比較できた 39 グラフ中 2 グラフでCPUより高速。グラフ別速度比 0.87〜1.04倍、stress比 0.971〜1.139。
- RR GPU：対応比較できた 37 グラフ中 34 グラフでCPUより高速。グラフ別速度比 0.45〜40.67倍、stress比 1.000〜1.341。

Sparse80 overviewにはRR GPUの代表描画4枠を添える。seedは実験1と同じCPU stress中央値に近いものを選び、接続線は対応するグラフのRR速度比中央値を指す。描画seed単独の速度比を指す線ではない。取得・成功していない描画は欠測枠として残す。小規模20グラフのSparse補助比較も別図で表示する。大規模80グラフの集計とは混ぜない。時間内訳図の積み上げは計測できた成分の平均で、cold総時間との差には区間外の準備処理が含まれる。

[Sparse80のID・名前対応](../tables/03_sparse-graph-order.csv) · [対応seed数](../tables/03_sparse-counts.csv) · [Sparse小20補助のID対応](../tables/03_sparse_small_aux-graph-order.csv)

上下2段は全グラフの共通x軸。背景は名前から推定したカテゴリで、分類内を頂点数順に並べる。graph-order.csvにID・名前を、counts.csvに対応seed数を保存。下端の×と数は欠測・CPU基準欠如による未対応seed数（5件中）で、0や1の測定値ではない。比は対応する成功seedのみで計算する。

### 4. グラフ構造と速度

![グラフ構造と速度](../figures/04_structure.png)

![グラフ構造と速度](../figures/04_structure_quality.png)

![グラフ構造と速度](../figures/04_structure_bins.png)

![グラフ構造と速度](../figures/04_size_series.png)

![グラフ構造と速度](../figures/04_sparse_write_load.png)

**考察。** FullはVとSGD速度比のSpearman相関がAtomic 0.998、RR 0.921（各n=19）。Sparse RRは最大次数と速度比が−0.711、品質比が0.558（各n=37）で、次数の大きい成功例ほど遅く・stress比が高い傾向があった。成功部分集合での探索的相関であり、同系列やV/E/制約数の交絡がある。

予定lock対象の上位1%頂点への集中は代表5件で46.6〜50.2%だった。一方、正重みの更新予定数の上位1%割合はLux約1.01%、web約3.10%、kron17約11.16%。高速なLuxにもlock対象の集中があり、この値だけで実競合やRR遅延を説明できない。GPU store、CAS retry、待ち時間は測っていない。[構造・予定負荷の詳細](structure-esa-discussion.md)。

**推定stressの感度。** 保存済みseed0のLux・web-Stanford・144を64/256/1024起点で再評価した。delaunay_n20は成功CPUがないため144を補足選択。256対1024の絶対stress差は最大2.669%、CPUとの品質比変化は最大3.908%。web-Stanford Atomicの品質比は1.0472→1.0898となり、+5%境界を跨いだ。固定256評価による分類にはこの限界があり、1024も真値ではない。3 graph以外の誤差保証とはしない。[感度の詳細とnative照合](sampling-sensitivity-observations.md)。

頂点数V・辺数E・次数の平均・最大値・変動係数の5指標を、Full/Sparseの上下2段で速度比と比較する。品質比も同じ2×5構成の補助図で示す。各グラフは対応seed比の中央値1点とし、Spearman CSVはmetric列で速度と品質を区別する。CV=0はsymlog軸で保持する。次数だけで実行時間は決まらず、頂点数・制約数・タイル数との交絡がある。相関係数はstructure-correlations.csvに保存し、因果関係として扱わない。点形状の構造カテゴリはグラフ名からの推定であり、確認済みの用途分類ではない。元のSuiteSparseグループはsource_groupとしてgraph-structure.csvに保存する。平均次数4/16とCV0.5/2で分けた図はグラフごとの速度比中央値を再集計し、グラフを等重みにする。同系列のサイズ曲線もグラフ名prefixからの推定系列で、規模以外の構造差を制御した実験ではない。

[代表グラフの予定書込回数CSV](../analysis/sparse-write-load/summary.csv)・[選択とSHA照合の証拠](../analysis/sparse-write-load/selection.json)。Atomicは重み0の端点もlockするため、正重みの端点更新予定数と両端lock対象予定数を分ける。いずれも1反復あたりの静的回数で、実際のGPU store、競合率、retry、待ち時間を表さない。図は全頂点CV・上位1%頂点の割合・pivot群の総回数割合。表はall群の総数・最大値を併記する。

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

### 5. 収束曲線と目標stress到達

![収束曲線と目標stress到達](../figures/05_convergence.png)

![収束曲線と目標stress到達](../figures/05_iteration_times.png)

**考察。** 計画720条件のうち657成功、60未起動、3 memory_limit。成功657件の8保存点、計5,256点を評価した。1.05目標は509/657で到達を観測したが、反復15でも以下なのは500件。1.01目標は413件から終端404件へ減った（CPUも含むため、手法別は下表を参照）。

Full Atomicは1.05目標30/60、1.01目標4/60、Full RRは36/60・14/60。Sparse Atomicは126/160・103/160、RRは97/157・72/157。kron18のSparse RRは1.05目標5/10、1.01目標4/10で一度hitしたが、そのすべてで後に目標超へ戻り、終端は両目標0/10だった。最初のhitを品質維持の証拠とはしない。[母集団・代表例・再上昇の考察](convergence-esa-discussion.md)。

反復15のoffline/native stress照合は657/657件が0.1%以内（最大差0.00000318%）。No5の時間traceも657/657件で整合した。別図の通常benchmark時間は基本5 seedで、No5の10 seed診断とは分けた。

`05_convergence` は座標保存を伴う収束診断の計画10seedから、反復0・1・2・3・5・8・10・15のstressを後計算した図。GPUとCPUで同じ起点を使う。曲線はcheckpointごとの各手法の成功seed中央値であり、実際の単一実行の軌跡ではない。失敗により手法間の成功seed集合が異なる場合がある。計画した全24 graph/family枠を残し、全観測欠測の枠も表示する。図のobservedは有効なcheckpointが1点以上あるseed数、missingはないseed数。併記statusは計画条件の分類で、warmup_failedを実起動の事故件数と数えない。[図の観測・欠測・status対応](../tables/convergence-panel-coverage.csv)。
`05_iteration_times` は通常の `study=basic` benchmarkのseed 0〜4のうち、成功し反復時間trace検査を通過した実行だけを使う別図。線は反復ごとの中央値、帯は最小〜最大、凡例n/5は利用できたseed数。座標コピー・trace IOはSGD時間から除外している。10seedの収束診断やその目標到達時間を示す図ではない。
目標は同じseed・familyのCPU 15反復後stressの1.05倍と1.01倍。最初に到達を観測したcheckpointと直前観測点の幅を記録する。これは実際の初到達を保証する区間ではない。stressが非単調になり、未観測の反復で到達後に再上昇する場合、真の初到達はこの幅より前の可能性がある。upper_ms時点には目標を満たす保存座標があるが、lower_msは真の初到達の下限と断定しない。正確な初到達時刻や未到達条件の外挿は行わない。診断の同期・保存による影響があるため、図2・3の通常実行時間とは分けて読む。

[最終stress評価の整合監査](../tables/convergence-stress-audit.csv)は、各成功runのoffline反復15とnative stressを比較する。snapshotは重心移動前、nativeは移動後で、数学的には並進でstressは不変だがf32/f64の丸めによりbit一致は要求しない。stress定義・起点数・seedも照合し、相対差0.1%以上を要調査として記録する。監査flagだけで成功runを除外しない。

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

計画上の全10seedを残し、観測なし・CPU基準なし・checkpoint不完全を未達と区別する。下表の欠測は到達率の分母に含めない。観測幅は評価可能で到達を観測したseedだけで、直前観測時刻と初到達を観測した時刻をそれぞれ中央値にした幅（秒）。seed間の隣接観測時刻の幅中央値であり、単一実行の初到達時刻でも、真の初到達を挟む保証区間でもない。CSVには到達観測後の再上昇と最終checkpoint・反復15で目標以下かのflagsも残す。[全seedの時刻幅](../tables/target-arrival.csv)・[集計CSV](../tables/target-arrival-coverage.csv)で確認できる。

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

### 6. タイルサイズ

![タイルサイズ](../figures/06_tiles.png)

![タイルサイズ](../figures/06_tiles_pareto.png)

**考察。** T=1で5seedが完走したのはdwt_2680とEVA。CPU stress比中央値はほぼ1だが、SGD中央値は56.602秒・117.673秒だった。USpowerGrid/bcsstk16/fe_4elt2/wiki-VoteのT1とfe_4elt2のT2は、計5回のwarmupが120秒上限で停止し、正式25条件が未起動。途中反復は進んでおり、無限待ちを確認した結果ではない。

USpowerGrid・fe_4elt2ではT512がT1024を時間・stress両中央値で上回った。一方、最短でもEVAはCPU stress比1.149、wiki-Voteは1.671。小Tはouter roundとhost submit/pollを増やす実装だが、時間の成分ごとの原因は分離測定していない。[T/Pの詳細](sensitivity-observations.md)・[T上限停止と実装上の回数](tile-limit-incidents.md)。

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
全予定パラメータをx軸に残し、全欠測は軸下端の× 0/5、部分成功は手法色のn/5で示す。欠測はNaNとして線を切り、0の測定値へ置換しない。[条件別・指標別のcoverage](../tables/tiling-parameter-coverage.csv)で成功・欠測・未記録数を確認できる。


時間だけで選ぶと品質を落とす可能性があるため、同じ条件のstressも併記する。時間–品質散布図はseed 0–4の5件すべてが成功したTだけを描き、Tラベルを付ける。CPU水平線と散布図のCPU点はbasic Full CPUの5seedが完備したときだけ表示する別枠の参照であり、T比較の測定点ではない。欠測・部分成功条件はtile-pareto.csvに残し、散布図に値を補わない。破線は測定した中央値に対して非劣位の条件を結ぶもので、未測定条件を含む最適性の証明ではない。最短条件は今回の測定範囲での値。

### 7. pivot数

![pivot数](../figures/07_pivots.png)

**考察。** 5seed完備条件の最短cold時間は6 graphすべてP50だった（web-StanfordのみCPU、他5件RR）。ただし最短が最低stressではない。LuxのRRはP50→400でcold 0.815→8.797秒、RSS 0.644→5.039 GiB、stress 163.311→150.672百万となった。web-Stanford RRはP200→400で絶対stressが増え、品質改善は単調ではなかった。

P400のAtomicはweb-Stanford 4/5、coAuthorsCiteseer 1/5のみ成功で、完備条件の最短比較から除外した。P変更はpivot/制約/重み/学習率に加えて初期座標生成前のRNG消費も変えるため、「同じ初期配置でPだけ変更」の対照実験ではない。[P別の時間・stress・RSSと欠測](sensitivity-observations.md)。

- bmw7st_1：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.320 秒、stress 399,650,838.7。 cold総時間 2.990 秒。
- coAuthorsCiteseer：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.813 秒、stress 5,575,457,825.7。 cold総時間 2.824 秒。
- delaunay_n20：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.278 秒、stress 47,803,912,362.0。 cold総時間 9.140 秒。
- luxembourg_osm：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.067 秒、stress 163,311,293.5。 cold総時間 0.815 秒。
- preferentialAttachment：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.573 秒、stress 1,114,683,340.2。 cold総時間 1.496 秒。
- web-Stanford：5seed完備条件のうち今回測れた最短は sparse_sgd（5/5成功）、pivots=50、SGD 4.190 秒、stress 4,952,670,900.7。 cold総時間 6.275 秒。
全予定パラメータをx軸に残し、全欠測は軸下端の× 0/5、部分成功は手法色のn/5で示す。欠測はNaNとして線を切り、0の測定値へ置換しない。[条件別・指標別のcoverage](../tables/pivots-parameter-coverage.csv)で成功・欠測・未記録数を確認できる。


pivot増加は制約数・前処理・メモリと品質の交換になる。主図の時間は前処理・GPU初期化・SGD・readbackを含むcoldアルゴリズム総時間で、入力読込・stress評価・描画ファイル保存を含まない。summary.csvにはSGD単独・前処理・process wall・peak RSSも保存。Peak RSSはGPU専用VRAM量ではない。

### 事故・欠測と調査

**最終状態。** 正式実起動2,370件は成功2,321、memory_limit 46、timeout 3。別枠warmup468件は成功342、memory_limit 117、timeout 9。再利用120条件と未起動720条件を物理起動数に加えない。macOS再起動で中断した未確定attempt1回を別途保持し、正式の総起動attemptは2,371回。[最終監査](final-audit.md)・[資源停止と観測段階](resource-incidents.md)。

**メモリ・時間。** 120秒は入力/前処理/SGD/stress評価/保存を含むnative全体の上限で、SGD単体の完了時間ではない。RSSは専用VRAM量ではなく、約200 ms間隔の監視なので厳密な瞬間上限でもない。成功formal/warmupの24件で終了後peak RSSが8 GiBを超えたことも明記した。No6の小T停止は上記の5実warmup、Youtube/Pokec/LiveJournalは9実warmupのRSS停止から正式45条件が未起動だった。

**保存IOと再起動。** 当初の非bufferedテキスト保存が測定を圧迫したため、buffer化・明示flushを実装しcommit後に正式一巡を開始した。修正前662条件とpilotの停止監視事故は別に保存し、今回の最終値に混ぜていない。CPU/RRの座標一致と一部stress不変を検証した。[保存IOの調査](io-export-incident.md)・[pilot](pilot.md)。途中のmacOS再起動では確定1,074条件を保持し、未完の1 attemptを退避して再開した。前後でweb-NotreDameのCPU/RRのstressが一致した例はあるが、OS停止・3件の旧timeoutの単独原因は特定できていない。[再起動と再開](reboot-incident.md)・[当時の観測](web-notredame-timeout-system-snapshot.md)。

**後処理の事故。** NumPy/Accelerateのdotは有限vectorでも誤った浮動小数点例外flagを残した。非BLAS二乗和へ変更し全27件を再解析、警告0・旧値との差は最大相対4×10⁻¹⁶だった。[切り分け](sampling-dot-warning-investigation.md)。また再描画時にCSVのseedが文字列になり357件の定義不一致を誤検出したが、共通型parserへ修正した。stress値と到達集計は変わらず、最終657件はすべて整合した。[修正後の収束監査](../tables/convergence-stress-audit.csv)。

**データ欠測。** SNAP公式原配布6件は取得・CRC・SHA・N/E/ID・LCC検証を終えた。SuiteSparse MTXとの辺/行番号直接照合は未実施で、同一性の主張範囲を限定する。100候補中99件ready、98件適格（小20・大78）。power197kは公式配布エラーで未取得、G_n_pin_poutはLCC99,995頂点で規模除外。[取得再調査と補完](data-availability-recheck.md)・[source receipt一覧](../snap-origin-recovery/source-receipts/index.json)。

[全結果・条件別数値](RESULTS.md)・[実測metadata](../raw/results.jsonl)・[補完metadata](../snap-origin-recovery/raw/results.jsonl)。巨大な原本・座標・距離cacheはローカルに保持し、共有先には図・集計表・固定設定・hash/検証receiptを保存した。
