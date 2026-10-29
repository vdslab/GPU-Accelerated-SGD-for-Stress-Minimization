Full・SparseそれぞれのCPU／Atomic／Round-robin（RR）を比較し、**描画の見た目、反復速度、品質、グラフの特性による性能差、設定による速度と品質の変化**を調べる。そのため、**7種類の実験**を行う。

グラフはSuiteSparseから**小中規模20個と10万頂点以上の大規模80個、計100個**を選定した。Fullは小中規模20個、大規模Sparseは80個を中心に比較する。100個で基本性能を広く測り、代表例で収束や設定の影響を詳しく調べる。

2026年10月25日：v3の実験1〜7と集計を完了した。100候補中99個を準備でき、規模条件を満たすのは98個（小20・大78）。以下の「実験準備・実験概要」は実施前の計画と構成例であり、その後の「実験結果（v3）」に実測18画像・考察・欠測と事故調査を掲載する。

## 実験準備

手法間で入力・初期条件・計測条件を揃え、100個での基本比較と代表例での詳細比較を行う。

### 比較手法と実験一覧

Fullは全頂点ペア、Sparseは辺と代表点（pivot）に関するペアを使う。まず各系列内でCPU・Atomic・RRを比較する。

| No. | 実験 | 目的 | 使用グラフ |
|---:|---|---|---|
| 1 | Full・Sparseの描画比較 | 各系列の3手法で形状・構造の見え方を確認 | 小中規模4個＋大規模4個 |
| 2 | Fullの速度・品質比較 | 高速化と品質維持を評価 | 小中規模20個 |
| 3 | Sparseの速度・品質比較 | 大規模での実用性を評価 | 大規模80個＋補助比較の小中規模20個 |
| 4 | 規模・次数と性能の関係 | RRが有利・不利になる条件を整理 | 実験2・3の100個の結果を再分析 |
| 5 | 同じ品質への到達時間 | 良い配置を早く得られるか評価 | 小中規模6個＋大規模12個 |
| 6 | Full RRのタイル幅比較 | タイル幅による速度・品質の変化を評価 | 小中規模6個 |
| 7 | pivot数の比較 | 時間・品質・メモリの関係を評価 | 大規模6個 |

同じ実行から得られる速度・品質・座標は共用し、実験ごとに全グラフを再実行しない。

### グラフの選定基準

**規模・平均次数・次数の偏り**が異なるグラフを用意する。次数は頂点につながる辺の数であり、平均が同じでも、一部のハブに辺が集中するかどうかで処理負荷は変わり得る。

| 原データの頂点数 | 件数 |
|---|---:|
| 約1,000〜16,000 | 20 |
| 10万以上30万未満 | 28 |
| 30万以上100万未満 | 23 |
| 100万以上300万未満 | 21 |
| 300万以上約420万 | 8 |

道路・地理、幾何・メッシュ、Web・引用、社会・協働、生成グラフ、工学行列由来の構造を含める。道路のような低次数の構造、Webのようなハブのある構造、メッシュのような局所的な結合を比較する。これらは選定の狙いであり、次数は前処理後に実測する。

Delaunay・RGG・Kroneckerなどのサイズ系列は規模の影響を見るために使う。同系列の多数の結果を、独立した実例と同一視しない。

Fullは10万頂点で1反復あたり約50億ペアになり、距離の記憶量も大きいため、大規模80個への一律適用は行わない。

全100個の名前・サイズ・特徴は [[SuiteSparse実験用100グラフの選定と特性]] を参照。

### 詳細比較に使う代表例

以下は100個の中から選ぶ候補。前処理後の規模・次数を確認し、本計測前に確定する。

| セット | グラフ |
|---|---|
| 小中規模6個 | USpowerGrid、dwt_2680、EVA、fe_4elt2、bcsstk16、wiki-Vote |
| 大規模12個 | luxembourg_osm、roadNet-CA、delaunay_n17、delaunay_n20、web-Stanford、web-NotreDame、coAuthorsCiteseer、coPapersDBLP、preferentialAttachment、smallworld、kron_g500-logn18、bmw7st_1 |
| pivot比較用6個 | luxembourg_osm、delaunay_n20、web-Stanford、coAuthorsCiteseer、preferentialAttachment、bmw7st_1 |

設定調整に使ったグラフは記録し、残りのグラフでも固定した設定で性能を確認する。

### 入力と共通条件

数値ゼロ・自己ループ・重複辺を除き、方向と重みを取り除いて無向化し、最大連結成分を使う。実験で報告する規模は、この処理後の頂点数・辺数とする。10万頂点未満になった候補の大規模枠からの差し替えは、本計測前に理由を記録して行う。

| 項目 | 条件案 |
|---|---|
| 基本設定 | 15反復、ε=0.1、Sparseのpivot数200 |
| 初期条件 | 手法間で同じ初期座標を使用。Sparseはpivot・距離・重み・更新対象も揃える |
| 繰り返し | 基本比較は5 seed、詳細な収束比較は10 seedを目安とする |
| 計測 | 同一機器・実装版で試運転後に測定。前処理・反復・品質評価を分ける |
| 実行上限 | 試運転で時間・メモリを見積もり、本計測前に固定する |

基本比較は **80個×3手法×5回＋20個×6手法×5回＝1,800回**。メモリ不足・時間切れ・品質目標未達も結果として残す。

## 実験概要

見た目、速度・品質、性能差の要因を順に確認する。各実験の図は構成例であり、実測結果ではない。

### 1. Full・Sparseの描画結果を比較する

Full・SparseそれぞれのCPU・Atomic・RRで、全体形状や局所構造の見え方がどの程度変わるかを確認する。Fullでは小中規模、Sparseでは大規模の配置を比較する。

| 系列 | 使用グラフ | 選ぶ理由 |
|---|---|---|
| Full | USpowerGrid、dwt_2680、EVA、fe_4elt2 | 小中規模の異なる構造で、全体形状と線の密集を目視比較するため |
| Sparse | luxembourg_osm、web-Stanford、coAuthorsCiteseer、delaunay_n19 | 道路・Web・共著・幾何構造を含め、大規模での局所的な結合や密集部分の見え方を比較するため |

15反復後の画像を、**各系列で4行×3列（CPU／Atomic／RR）**に並べ、時間とstressを添える。各グラフではCPUのstressが中央値に近いseedを全手法で共有し、背景・線幅・点サイズ・透過率・色・縮尺を統一する。辺を間引く場合も同じ辺を使い、割合を明記する。全体図で違いが分かりにくい場合は、同じ頂点集合の拡大図を添える。次数の特徴は前処理後に確認する。

**出力：Full・Sparseそれぞれの描画比較図と必要な拡大図。**

<img width="710.55" alt="merged-1.png (77.9 KB)" src="https://img.esa.io/uploads/production/attachments/8704/2026/10/24/164555/f348cdca-7673-450f-ad54-f8690b18deb6.png">

<img width="710.55" alt="merged-2.png (92.6 KB)" src="https://img.esa.io/uploads/production/attachments/8704/2026/10/24/164555/4c0cd6e5-9aec-4ecb-aeda-2f344f2fa01f.png">

### 2. Fullの速度と品質を比較する

FullのGPU実装がCPUに対してどれだけ速くなり、最終的な品質を維持できるかを評価する。

**小中規模20個**を使う。全ペアを扱うFullを実行できる範囲で、特定の構造に偏らず比較するためである。

各反復の時間、15反復の合計時間、最終stressを測る。stressはグラフ上と描画上の距離のずれを表し、小さいほど目的に合った配置となる。

| 指標 | 解釈 |
|---|---|
| 速度比＝CPU時間÷GPU時間 | 2ならGPUが2倍速い |
| 品質比＝GPU stress÷CPU stress | 1なら同等、1.05ならstressが5%大きい |

**出力：反復時間の折れ線、20個それぞれの速度比・品質比とseedによるばらつき。**

<img width="710.1" alt="merged-3.png (101.1 KB)" src="https://img.esa.io/uploads/production/attachments/8704/2026/10/24/164555/b790c900-ae5f-48bd-9e96-8e84cc825456.png">

### 3. Sparseの速度と品質を比較する〈主実験〉

Sparseの高速化と品質維持が大規模でも成り立つかを調べ、前処理やメモリを含めた実用性を評価する。

**10万頂点以上の80個を中心に、小中規模20個も補助比較に使う。** 規模と構造を広くカバーし、小規模だけでは見えない処理時間・メモリの制約を確認するためである。

実験2と同じ指標に加え、前処理込みの総時間と最大メモリ使用量を記録する。大規模のstressは共通の評価用始点から推定する。始点はpivotとは独立に選び、Sparse内部の近似目的と比較用stressを区別する。

**出力：80グラフの速度比・品質比一覧、代表例の各反復時間、前処理と反復の時間内訳。**

**80グラフの一覧図をメイン図とする。** 横軸の並びを揃えた上下2段で、上段に速度比（CPU時間÷GPU時間）、下段に品質比（GPU stress÷CPU stress）を示す。AtomicとRRを2色で並べ、同じseedのCPUとGPUから求めた比の中央値を点、最小〜最大を縦線で表す。速度比は対数軸とし、上下とも1の基準線を引く。

道路・メッシュ・Web・社会ネットワークなどの構造別に背景を色分けし、各分類内は前処理後の頂点数順に並べる。全グラフのIDと名前の対応表を添え、代表例の描画サムネイルを対応する点へ線で結ぶ。サムネイルには手法・seedを明記する。

速度比は**15反復の合計時間（前処理を除く）**で求め、前処理込みの総時間は別図で補足する。時間切れ・メモリ不足・欠測は1や0に置き換えず、該当位置に記号と件数を示す。品質比と合わせて、高速化と品質維持を確認する。Fullの20個も実験2で同じ形式の一覧図にする。

<img width="963.9" alt="sparse-80-overview.png (206.3 KB)" src="https://img.esa.io/uploads/production/attachments/8704/2026/10/24/164555/f8901250-fab7-4eee-b1b1-1c574de0c308.png">

### 4. 規模・次数と性能の関係を調べる

RRが有利・不利になる条件を、頂点数・辺数・平均次数・最大次数・次数の偏りから説明する。

**実験2・3の100個の結果を再利用する。** 幅広い構造で傾向を調べ、Delaunay・RGG・Kroneckerなどのサイズ系列では同系統での規模の影響を確認するためである。同系列の結果を独立した実例と同一視しない。

平均次数は「4未満／4以上16未満／16以上」、偏りを示すCV（標準偏差÷平均）は「0.5未満／0.5以上2未満／2以上」を目安に分類し、連続値も示す。Fullは全ペア、Sparseはpivotの更新も含むため、入力次数だけで競合を説明せず、Sparseの頂点ごとの書込負荷とRRのラウンド数も補足する。

**出力：規模・次数・偏りと速度比・品質比の散布図、同系列の規模による変化の図。**

<img width="710.55" alt="merged-5.png (96.5 KB)" src="https://img.esa.io/uploads/production/attachments/8704/2026/10/24/164555/8a7ed89c-1a28-4751-bb7a-68c4c3baecef.png">

### 5. 同じ品質に到達する時間を比較する

1反復の速さだけでなく、同じ品質の配置を得るまでに必要な時間を比較する。

**代表の小中規模6個・大規模12個**を使う。構造と規模の違いを残しつつ、途中のstressを詳しく測る計算量を抑えるためである。

対応するCPUの15反復後のstressを基準に、その**1.05倍以下・1.01倍以下**へ初めて到達する時間と到達率を測る。FullとSparseはそれぞれのCPUを基準とする。未到達も記録し、必要なら共通の学習率設定で30反復の追加比較を行う。

**出力：stress–時間の折れ線、目標品質への到達時間・到達率の表。**

<img width="710.1" alt="merged-6.png (104.4 KB)" src="https://img.esa.io/uploads/production/attachments/8704/2026/10/24/164555/3c80e4a9-d389-402a-8914-867bad8e9b13.png">

### 6. Full RRのタイル幅を変える

タイル幅Tによって速度と品質がどう変わるかを調べる。**T=1も比較に含め、内側のシャッフルをなくした場合にCPU Fullの品質へ近づくか**を確認する。

**代表の小中規模6個**を使う。構造の異なる例で、タイル幅の効果がグラフに依存するかを見るためである。

workgroup内のスレッド数は**現在の既定値W=1024に固定**し、**T=1・2・4・8・16・32・64・128・256・512・1024**を比較する。初期座標・seed・反復数・学習率など、その他の条件も揃える。

T=1では1段階目のラウンドシャッフルだけで全頂点間の更新順が決まる。ただし、CPU Fullの全ペアを個別に並べ替える方法とは異なり、RRでは頂点が重複しないペアの組を保つ。

Tが小さいと有効にペアを更新するスレッドが減り、ラウンド数も増える。その影響を含めて時間と品質を測り、時間切れや品質差がない場合も記録する。

**出力：タイル幅に対する反復時間・品質の折れ線と、時間–品質の散布図。**

<img width="782.0999999999999" alt="tile-only.png (93.9 KB)" src="https://img.esa.io/uploads/production/attachments/8704/2026/10/24/164555/654373f8-eb8c-4f40-b9a6-ce3fd95da520.png">

### 7. Sparseのpivot数を変える

pivot数を増やすことで得られる品質改善と、時間・メモリの増加の関係を調べる。

**luxembourg_osm、delaunay_n20、web-Stanford、coAuthorsCiteseer、preferentialAttachment、bmw7st_1の6個**を使う。道路・幾何・Web・共著・生成・工学由来の構造を含め、pivot数の影響が特定の構造に限られないかを見るためである。

pivot数を**50・100・200・400**に変え、SparseのCPU・Atomic・RRを比較する。

**出力：pivot数に対する総時間・最終stress・最大メモリ使用量の折れ線。**

<img width="710.1" alt="merged-8.png (89.0 KB)" src="https://img.esa.io/uploads/production/attachments/8704/2026/10/24/164555/47b8af89-190d-44a4-84e9-f2b5c9e46f44.png">

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
[有効データ一覧](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/effective-datasets.json) · [補完検証provenance](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/cohort.json) · [原記録と有効/置換状態](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/cohort-source-records.csv) · [置換前の未起動90条件](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/cohort-superseded.csv) · [追加測定の固定設定](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/snap-origin-recovery/config/execution.json)。
**実施条件。** 2026年10月24〜25日、Apple M4 Pro・24 GiB unified RAM・Metalで実施。15反復、epsilon=0.1、基本・T・P比較はseed 0〜4、収束診断は0〜9。Sparse基本P=200、Full RR基本T=1024/W=1024。入力は無重み・無向化した最大連結成分で、数値0を先に除去し、loop/重複を整理した。1実行ずつ測り、同設定のwarmupは別枠とした。native全プロセスの上限は120秒・観測RSS 8 GiB。
**比較範囲。** CPUはf64、GPUはf32。Full Atomicは明示pair shuffleなし・lock再試行上限なしで、更新順IDと更新前座標の監査buffer確保・書込も計測負荷に含む。Sparse Atomicにはshader内128回とhost再投入1024pass等の制限が残る。両者を一括して「上限なし」とはしない。精度・更新順・微小距離処理・コンパイル経路が違うため、品質差をlockや順序だけの原因へ帰属しない。[6実装の比較範囲](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/implementation-scope.md)。
### 1. 代表グラフの描画
![01_layout_full.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/83de69a1-9352-4602-8b53-bd716988b089.png)
![01_layout_sparse.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/6eddde31-1885-4c91-9d8b-2bc08bbc32dc.png)
![01_layout_sparse_partial.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/c60afba1-9190-424d-a8f4-47f009f7af16.png)
**考察。** 同seedでも速さと描画品質は両立しない。Fullの代表EVAはCPU stress約95.5万に対しAtomic約113.0万、RR約113.7万で、18〜19%高かった。一方dwt_2680は向きが大きく違ってもstressは近い。SparseのcoAuthorsCiteseerはRRが速いがstressが高い。見た目だけで正誤を判定せず、以下の全seed比較と併せて読む。delaunay_n19のCPU基準欠如は欠測として残し、成功した別seedの全2枚だけを補足図へ分けた。
成功CPUがある場合はCPUの最終stress中央値に最も近いseedを選び、GPUも同じseedを表示する。良好なGPU実行だけを選抜していない。CPU成功seedが0件の場合だけ、事前に計画されたseed 0を固定で使い、同seedで成功し座標が存在する手法だけを表示する。成功したGPU seedを探して置き換えない。このfallbackはNo.1の描画選択だけを対象とする事後補足で、CPU比や主結果の対応ペアには使わない。
今回のCPU基準欠如による固定seed 0の対象：Sparse: delaunay_n19。欠測枠には選択seedと失敗理由を表示し、warmup失敗による未起動も区別する。seed 0自身が失敗していれば、別seedの成功座標があっても描画しない。
[描画seed・選択規則・欠測理由](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/layout-selection.csv)に全24枠を記録する。CPU成功があるのに有効なstress値がない場合はseed 0へ移らず、選択不能として記録する。
同じグラフの全手法は座標のunion範囲から同じx/y縮尺で描画する。25万辺を超える描画は辺のみ等間隔で間引き、表示割合とSGD秒を併記する。stressは間引く前の全グラフで評価する。
別図 `01_layout_sparse_partial` は **CPU基準なし・利用可能な全layoutを示す記述的補足**。本図の固定seed 0を成功seedで差し替えるものではなく、対応比較や中央値代表とは呼ばない。CPU成功のない対象グラフについて、成功かつ保存先のあるlayoutをseedや品質で絞らず全件表示する。今回の利用可能分：delaunay_n19 / RR GPU: seed 2,3。図を読めない保存先は欠測枠と事故記録に残す。
[補足図の全条件・利用可否](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/layout-sparse-partial-availability.csv)。本図が全欠測でも、別seedの成功layoutが存在する場合がある。
### 2. 小規模20グラフ：Full SGD
![02_full.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/da14af89-14ac-4c29-b6ef-89ac78e62386.png)
![02_full_iteration_times.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/fb59f083-3c75-463c-9d5f-3356c83bb3a4.png)
**考察。** Full RRは比較可能19/20 graph・95組すべてのgraphでSGD速度比中央値が1を超え、graph等重み中央値は7.384倍。ただし19中9 graphでstress比中央値が1.05を超えた。wiki-Voteは11.646倍でもstress 1.671倍、ukのRR seed0は3.022倍のstressとなった。AtomicのSGD比中央値は1.298倍。RRのcold比は3.769倍、process wall比は2.597倍で、SGDの利得がそのまま全工程の利得にはならない。[Full20の詳細考察](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/full-basic-observations.md)。
- Atomic GPU：対応比較できた 19 グラフ中 13 グラフでCPUより高速。グラフ別速度比 0.06〜3.36倍、stress比 1.001〜2.921。
- RR GPU：対応比較できた 19 グラフ中 19 グラフでCPUより高速。グラフ別速度比 1.03〜15.02倍、stress比 0.954〜1.671。
[Full20のID・名前対応](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/02_full-graph-order.csv) · [対応seed数](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/02_full-counts.csv)
上下2段は全グラフの共通x軸。背景は名前から推定したカテゴリで、分類内を頂点数順に並べる。graph-order.csvにID・名前を、counts.csvに対応seed数を保存。下端の×と数は欠測・CPU基準欠如による未対応seed数（5件中）で、0や1の測定値ではない。比は対応する成功seedのみで計算する。
### 3. 大規模80グラフ：Sparse SGD
![03_sparse.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/c9033cc6-da59-4a76-982b-a328903ffcf3.png)
![03_sparse_small_aux.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/9148dad2-d4a7-4237-ab4f-8a34b5395775.png)
![03_sparse_components.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/f1b94412-ec4c-4580-b647-c85cd9756763.png)
大規模80候補のうち78件が入力条件を満たした。Sparse basic1,200条件は成功565、正式RSS停止27、正式timeout3、warmup_failed590、未取得15だった。warmup_failedにはサイズ除外15と、115件の実warmup停止による575件の未起動が含まれる。公式SNAP6件を補完し、Gowalla・DBLP・Amazonの45正式条件が成功した一方、YouTube・LiveJournal・Pokecの45正式条件は9件のwarmup RSS停止で未起動となった。これらを元のunavailable90件と二重計上しない。
同じ入力・初期配置・前処理・評価定義を満たす同seed成功組で比較すると、Atomicは39 graph/187組、RRは37 graph/181組が対応した。各graph内の対応比中央値を等重みで集計したSGD速度比中央値はAtomic0.944倍、RR25.116倍、stress比中央値は1.001145、1.006002だった。RRで速度改善とstress+5%以内を両立したのは21/37 graph、+1%以内は19/37 graph。全5seedが揃った35 graphではそれぞれ20/35、18/35で、成功部分集合での結果である。
RRのSGD25.116倍を処理全体の高速化とは呼ばない。同じ対応集合のcold algorithm比中央値は2.192倍、入力・stress評価・座標保存も含む全プロセスwall比は1.743倍だった。追加3件でもRRはDBLP29.70倍、Amazon22.02倍、Gowalla1.48倍と速いが、stressはそれぞれ+16.50%、+11.64%、+9.63%。追加によって5%品質基準を満たす高速graph数は増えなかった。
Luxやrgg18は高速化と近い品質を両立したが、coAuthorsCiteseerやsmallworldではRRのstressが約19%悪化し、kron17/18とweb-StanfordではRR自体がCPUより遅かった。RRのround・dispatchが多い例は時間増加と整合する一方、入力最大次数17・CV0.134のsmallworldでも品質が悪化するため、次数分布だけでは説明できない。頂点別の予定更新・lock対象数は補助指標で、実store、衝突率、retry、待ち時間の実測とは区別する。
出典と限界：core3210＋補完90原記録を保持した有効3210条件。追加6件はSNAP公式原配布から再構成し、SuiteSparse MTXとの辺・行番号の直接照合は未実施。CPU f64 / GPU f32、更新順・微小距離処理・実装経路も異なるため、品質差をlock又は更新順だけの原因へ帰属しない。小20補助は別集計で、Atomic/RRのSGD比中央値0.511/0.804倍、300条件全成功だった。

[全graphの対応比と欠測](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/sparse-basic-observations.md)。
- Atomic GPU：対応比較できた 39 グラフ中 2 グラフでCPUより高速。グラフ別速度比 0.87〜1.04倍、stress比 0.971〜1.139。
- RR GPU：対応比較できた 37 グラフ中 34 グラフでCPUより高速。グラフ別速度比 0.45〜40.67倍、stress比 1.000〜1.341。
Sparse80 overviewにはRR GPUの代表描画4枠を添える。seedは実験1と同じCPU stress中央値に近いものを選び、接続線は対応するグラフのRR速度比中央値を指す。描画seed単独の速度比を指す線ではない。取得・成功していない描画は欠測枠として残す。小規模20グラフのSparse補助比較も別図で表示する。大規模80グラフの集計とは混ぜない。時間内訳図の積み上げは計測できた成分の平均で、cold総時間との差には区間外の準備処理が含まれる。
[Sparse80のID・名前対応](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/03_sparse-graph-order.csv) · [対応seed数](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/03_sparse-counts.csv) · [Sparse小20補助のID対応](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/03_sparse_small_aux-graph-order.csv)
上下2段は全グラフの共通x軸。背景は名前から推定したカテゴリで、分類内を頂点数順に並べる。graph-order.csvにID・名前を、counts.csvに対応seed数を保存。下端の×と数は欠測・CPU基準欠如による未対応seed数（5件中）で、0や1の測定値ではない。比は対応する成功seedのみで計算する。
### 4. グラフ構造と速度
![04_structure.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/24c59cca-f6c5-403a-8de1-1dd51feb1e4d.png)
![04_structure_quality.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/ae729271-8510-4c22-82a3-ade667b8bf87.png)
![04_structure_bins.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/08b6d4b0-f733-4057-ac6e-a0b9434636b0.png)
![04_size_series.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/2b199e69-ea1f-4b99-a3e0-8fe6ebb01f7f.png)
![04_sparse_write_load.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/7e6b71c0-228a-441e-beed-f3cff47efa0b.png)
**考察。** FullはVとSGD速度比のSpearman相関がAtomic 0.998、RR 0.921（各n=19）。Sparse RRは最大次数と速度比が−0.711、品質比が0.558（各n=37）で、次数の大きい成功例ほど遅く・stress比が高い傾向があった。成功部分集合での探索的相関であり、同系列やV/E/制約数の交絡がある。
予定lock対象の上位1%頂点への集中は代表5件で46.6〜50.2%だった。一方、正重みの更新予定数の上位1%割合はLux約1.01%、web約3.10%、kron17約11.16%。高速なLuxにもlock対象の集中があり、この値だけで実競合やRR遅延を説明できない。GPU store、CAS retry、待ち時間は測っていない。[構造・予定負荷の詳細](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/structure-esa-discussion.md)。
**推定stressの感度。** 保存済みseed0のLux・web-Stanford・144を64/256/1024起点で再評価した。delaunay_n20は成功CPUがないため144を補足選択。256対1024の絶対stress差は最大2.669%、CPUとの品質比変化は最大3.908%。web-Stanford Atomicの品質比は1.0472→1.0898となり、+5%境界を跨いだ。固定256評価による分類にはこの限界があり、1024も真値ではない。3 graph以外の誤差保証とはしない。[感度の詳細とnative照合](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/sampling-sensitivity-observations.md)。
頂点数V・辺数E・次数の平均・最大値・変動係数の5指標を、Full/Sparseの上下2段で速度比と比較する。品質比も同じ2×5構成の補助図で示す。各グラフは対応seed比の中央値1点とし、Spearman CSVはmetric列で速度と品質を区別する。CV=0はsymlog軸で保持する。次数だけで実行時間は決まらず、頂点数・制約数・タイル数との交絡がある。相関係数はstructure-correlations.csvに保存し、因果関係として扱わない。点形状の構造カテゴリはグラフ名からの推定であり、確認済みの用途分類ではない。元のSuiteSparseグループはsource_groupとしてgraph-structure.csvに保存する。平均次数4/16とCV0.5/2で分けた図はグラフごとの速度比中央値を再集計し、グラフを等重みにする。同系列のサイズ曲線もグラフ名prefixからの推定系列で、規模以外の構造差を制御した実験ではない。
[代表グラフの予定書込回数CSV](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/analysis/sparse-write-load/summary.csv)・[選択とSHA照合の証拠](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/analysis/sparse-write-load/selection.json)。Atomicは重み0の端点もlockするため、正重みの端点更新予定数と両端lock対象予定数を分ける。いずれも1反復あたりの静的回数で、実際のGPU store、競合率、retry、待ち時間を表さない。図は全頂点CV・上位1%頂点の割合・pivot群の総回数割合。表はall群の総数・最大値を併記する。
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
![05_convergence.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/4559332f-6304-4608-82b5-92f52a14fda4.png)
![05_iteration_times.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/ccf78499-9a1a-4c56-9fd8-49199a76c05f.png)
**考察。** 計画720条件のうち657成功、60未起動、3 memory_limit。成功657件の8保存点、計5,256点を評価した。1.05目標は509/657で到達を観測したが、反復15でも以下なのは500件。1.01目標は413件から終端404件へ減った（CPUも含むため、手法別は下表を参照）。
Full Atomicは1.05目標30/60、1.01目標4/60、Full RRは36/60・14/60。Sparse Atomicは126/160・103/160、RRは97/157・72/157。kron18のSparse RRは1.05目標5/10、1.01目標4/10で一度hitしたが、そのすべてで後に目標超へ戻り、終端は両目標0/10だった。最初のhitを品質維持の証拠とはしない。[母集団・代表例・再上昇の考察](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/convergence-esa-discussion.md)。
反復15のoffline/native stress照合は657/657件が0.1%以内（最大差0.00000318%）。No5の時間traceも657/657件で整合した。別図の通常benchmark時間は基本5 seedで、No5の10 seed診断とは分けた。
`05_convergence` は座標保存を伴う収束診断の計画10seedから、反復0・1・2・3・5・8・10・15のstressを後計算した図。GPUとCPUで同じ起点を使う。曲線はcheckpointごとの各手法の成功seed中央値であり、実際の単一実行の軌跡ではない。失敗により手法間の成功seed集合が異なる場合がある。計画した全24 graph/family枠を残し、全観測欠測の枠も表示する。図のobservedは有効なcheckpointが1点以上あるseed数、missingはないseed数。併記statusは計画条件の分類で、warmup_failedを実起動の事故件数と数えない。[図の観測・欠測・status対応](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/convergence-panel-coverage.csv)。
`05_iteration_times` は通常の `study=basic` benchmarkのseed 0〜4のうち、成功し反復時間trace検査を通過した実行だけを使う別図。線は反復ごとの中央値、帯は最小〜最大、凡例n/5は利用できたseed数。座標コピー・trace IOはSGD時間から除外している。10seedの収束診断やその目標到達時間を示す図ではない。
目標は同じseed・familyのCPU 15反復後stressの1.05倍と1.01倍。最初に到達を観測したcheckpointと直前観測点の幅を記録する。これは実際の初到達を保証する区間ではない。stressが非単調になり、未観測の反復で到達後に再上昇する場合、真の初到達はこの幅より前の可能性がある。upper_ms時点には目標を満たす保存座標があるが、lower_msは真の初到達の下限と断定しない。正確な初到達時刻や未到達条件の外挿は行わない。診断の同期・保存による影響があるため、図2・3の通常実行時間とは分けて読む。
[最終stress評価の整合監査](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/convergence-stress-audit.csv)は、各成功runのoffline反復15とnative stressを比較する。snapshotは重心移動前、nativeは移動後で、数学的には並進でstressは不変だがf32/f64の丸めによりbit一致は要求しない。stress定義・起点数・seedも照合し、相対差0.1%以上を要調査として記録する。監査flagだけで成功runを除外しない。
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
計画上の全10seedを残し、観測なし・CPU基準なし・checkpoint不完全を未達と区別する。下表の欠測は到達率の分母に含めない。観測幅は評価可能で到達を観測したseedだけで、直前観測時刻と初到達を観測した時刻をそれぞれ中央値にした幅（秒）。seed間の隣接観測時刻の幅中央値であり、単一実行の初到達時刻でも、真の初到達を挟む保証区間でもない。CSVには到達観測後の再上昇と最終checkpoint・反復15で目標以下かのflagsも残す。[全seedの時刻幅](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/target-arrival.csv)・[集計CSV](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/target-arrival-coverage.csv)で確認できる。
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
![06_tiles.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/f9f39e3e-f236-49d1-a39c-f868949bece6.png)
![06_tiles_pareto.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/5db9386c-ab25-4272-aff7-30752c090577.png)
**考察。** T=1で5seedが完走したのはdwt_2680とEVA。CPU stress比中央値はほぼ1だが、SGD中央値は56.602秒・117.673秒だった。USpowerGrid/bcsstk16/fe_4elt2/wiki-VoteのT1とfe_4elt2のT2は、計5回のwarmupが120秒上限で停止し、正式25条件が未起動。途中反復は進んでおり、無限待ちを確認した結果ではない。
USpowerGrid・fe_4elt2ではT512がT1024を時間・stress両中央値で上回った。一方、最短でもEVAはCPU stress比1.149、wiki-Voteは1.671。小Tはouter roundとhost submit/pollを増やす実装だが、時間の成分ごとの原因は分離測定していない。[T/Pの詳細](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/sensitivity-observations.md)・[T上限停止と実装上の回数](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/tile-limit-incidents.md)。
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
全予定パラメータをx軸に残し、全欠測は軸下端の× 0/5、部分成功は手法色のn/5で示す。欠測はNaNとして線を切り、0の測定値へ置換しない。[条件別・指標別のcoverage](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/tiling-parameter-coverage.csv)で成功・欠測・未記録数を確認できる。

時間だけで選ぶと品質を落とす可能性があるため、同じ条件のstressも併記する。時間–品質散布図はseed 0–4の5件すべてが成功したTだけを描き、Tラベルを付ける。CPU水平線と散布図のCPU点はbasic Full CPUの5seedが完備したときだけ表示する別枠の参照であり、T比較の測定点ではない。欠測・部分成功条件はtile-pareto.csvに残し、散布図に値を補わない。破線は測定した中央値に対して非劣位の条件を結ぶもので、未測定条件を含む最適性の証明ではない。最短条件は今回の測定範囲での値。
### 7. pivot数
![07_pivots.png](https://img.esa.io/uploads/production/attachments/8704/2026/10/25/164555/8bb2cc8e-385a-46cc-aa07-78b110242bec.png)
**考察。** 5seed完備条件の最短cold時間は6 graphすべてP50だった（web-StanfordのみCPU、他5件RR）。ただし最短が最低stressではない。LuxのRRはP50→400でcold 0.815→8.797秒、RSS 0.644→5.039 GiB、stress 163.311→150.672百万となった。web-Stanford RRはP200→400で絶対stressが増え、品質改善は単調ではなかった。
P400のAtomicはweb-Stanford 4/5、coAuthorsCiteseer 1/5のみ成功で、完備条件の最短比較から除外した。P変更はpivot/制約/重み/学習率に加えて初期座標生成前のRNG消費も変えるため、「同じ初期配置でPだけ変更」の対照実験ではない。[P別の時間・stress・RSSと欠測](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/sensitivity-observations.md)。
- bmw7st_1：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.320 秒、stress 399,650,838.7。 cold総時間 2.990 秒。
- coAuthorsCiteseer：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.813 秒、stress 5,575,457,825.7。 cold総時間 2.824 秒。
- delaunay_n20：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.278 秒、stress 47,803,912,362.0。 cold総時間 9.140 秒。
- luxembourg_osm：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.067 秒、stress 163,311,293.5。 cold総時間 0.815 秒。
- preferentialAttachment：5seed完備条件のうち今回測れた最短は rr_sparse_sgd（5/5成功）、pivots=50、SGD 0.573 秒、stress 1,114,683,340.2。 cold総時間 1.496 秒。
- web-Stanford：5seed完備条件のうち今回測れた最短は sparse_sgd（5/5成功）、pivots=50、SGD 4.190 秒、stress 4,952,670,900.7。 cold総時間 6.275 秒。
全予定パラメータをx軸に残し、全欠測は軸下端の× 0/5、部分成功は手法色のn/5で示す。欠測はNaNとして線を切り、0の測定値へ置換しない。[条件別・指標別のcoverage](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/pivots-parameter-coverage.csv)で成功・欠測・未記録数を確認できる。

pivot増加は制約数・前処理・メモリと品質の交換になる。主図の時間は前処理・GPU初期化・SGD・readbackを含むcoldアルゴリズム総時間で、入力読込・stress評価・描画ファイル保存を含まない。summary.csvにはSGD単独・前処理・process wall・peak RSSも保存。Peak RSSはGPU専用VRAM量ではない。
### 事故・欠測と調査
**最終状態。** 正式実起動2,370件は成功2,321、memory_limit 46、timeout 3。別枠warmup468件は成功342、memory_limit 117、timeout 9。再利用120条件と未起動720条件を物理起動数に加えない。macOS再起動で中断した未確定attempt1回を別途保持し、正式の総起動attemptは2,371回。[最終監査](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/final-audit.md)・[資源停止と観測段階](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/resource-incidents.md)。
**メモリ・時間。** 120秒は入力/前処理/SGD/stress評価/保存を含むnative全体の上限で、SGD単体の完了時間ではない。RSSは専用VRAM量ではなく、約200 ms間隔の監視なので厳密な瞬間上限でもない。成功formal/warmupの24件で終了後peak RSSが8 GiBを超えたことも明記した。No6の小T停止は上記の5実warmup、Youtube/Pokec/LiveJournalは9実warmupのRSS停止から正式45条件が未起動だった。
**保存IOと再起動。** 当初の非bufferedテキスト保存が測定を圧迫したため、buffer化・明示flushを実装しcommit後に正式一巡を開始した。修正前662条件とpilotの停止監視事故は別に保存し、今回の最終値に混ぜていない。CPU/RRの座標一致と一部stress不変を検証した。[保存IOの調査](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/io-export-incident.md)・[pilot](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/pilot.md)。途中のmacOS再起動では確定1,074条件を保持し、未完の1 attemptを退避して再開した。前後でweb-NotreDameのCPU/RRのstressが一致した例はあるが、OS停止・3件の旧timeoutの単独原因は特定できていない。[再起動と再開](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/reboot-incident.md)・[当時の観測](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/web-notredame-timeout-system-snapshot.md)。
**後処理の事故。** NumPy/Accelerateのdotは有限vectorでも誤った浮動小数点例外flagを残した。非BLAS二乗和へ変更し全27件を再解析、警告0・旧値との差は最大相対4×10⁻¹⁶だった。[切り分け](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/sampling-dot-warning-investigation.md)。また再描画時にCSVのseedが文字列になり357件の定義不一致を誤検出したが、共通型parserへ修正した。stress値と到達集計は変わらず、最終657件はすべて整合した。[修正後の収束監査](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/tables/convergence-stress-audit.csv)。
**データ欠測。** SNAP公式原配布6件は取得・CRC・SHA・N/E/ID・LCC検証を終えた。SuiteSparse MTXとの辺/行番号直接照合は未実施で、同一性の主張範囲を限定する。100候補中99件ready、98件適格（小20・大78）。power197kは公式配布エラーで未取得、G_n_pin_poutはLCC99,995頂点で規模除外。[取得再調査と補完](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/data-availability-recheck.md)・[source receipt一覧](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/snap-origin-recovery/source-receipts/index.json)。
[全結果・条件別数値](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/RESULTS.md)・[実測metadata](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/raw/results.jsonl)・[補完metadata](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/snap-origin-recovery/raw/results.jsonl)。巨大な原本・座標・距離cacheはローカルに保持し、共有先には図・集計表・固定設定・hash/検証receiptを保存した。

## まとめ

**v3の全計画3,210条件の処理と実験1〜7の可視化を完了した。** 成功・資源上限停止・未起動・入力欠測を区別し、代表例の描画・収束・設定感度を確認した。RRは多くの成功例でSGDを高速化したが、品質維持はグラフと設定に依存した。

大規模Sparse RRの対応比較37グラフでは、SGD速度比のgraph等重み中央値25.116倍に対しcold総時間比は2.192倍だった。高速かつCPU stress比+5%以内は21/37グラフ。成功部分集合と固定256起点の評価に対する結果であり、全候補やすべてのseedへの保証ではない。

### 計測・解釈で押さえる点

- **計測の準備：** 各反復ログを追加し、SGD区間・前処理込みcold時間・全プロセスwall時間を区別して記録した。
- **時間の公平性：** GPUは処理完了まで測る。細かな計測や品質評価の費用を通常実行の反復時間と混ぜない。
- **品質評価：** 全手法を同じ評価器で測る。8,000頂点以下は厳密計算、超過は固定256起点で評価し、代表3グラフで64／256／1024起点の感度を確認した。
- **収束の記録：** 大規模で毎反復の評価が重い場合は0・1・2・3・5・8・10・15反復で測る。目標到達時刻は観測点間の幅を持つものとして扱う。今回は15反復で測定し、任意の30反復追加比較は実施していない。
- **結果の解釈：** CPUはf64、GPUはf32、AtomicはMetal、RRはWGPU。差をすべてRRの更新順だけの効果としない。平均だけでなくグラフ別の結果・ばらつき・失敗件数を示す。
- **再現性：** 入力・前処理・初期座標・pivot・seed・設定・時間・stress・実装版・機器情報を保存する。

### 参考資料

| 資料 | 計画への反映 |
|---|---|
| [Zheng et al., Graph Drawing by Stochastic Gradient Descent](https://arxiv.org/html/1710.04626v3) | 15反復・ε=0.1を出発点とし、反復と品質の関係を調べる |
| [Ortmann et al., A Sparse Stress Model](https://arxiv.org/abs/1608.08909) | pivotによる近似と速度・品質の関係を評価する |
| [Börsig et al., SGD Works Really Well for Stress Minimization](https://arxiv.org/html/2008.10376v1) | 初期配置を揃え、品質の比較条件を明確にする |
| [Mooney et al., Stress in Graph Drawings](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2025.38) | stressと実際の描画を組み合わせて説明する |

80／20個の配分、次数の分類、品質目標の1%／5%は本研究向けの提案であり、先行研究の標準値ではない。

関連：[[SuiteSparse実験用100グラフの選定と特性]] ／ [[v2 実験結果と図の読み方]]

[データ取得状況](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/244602fdbd82964c5986ad2c88d3cc527284a8e4/experiment_results/v3/report/data-availability-recheck.md)