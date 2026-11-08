## Full SGDの既存実験と再検討事項
















以下は調査開始時に整理した既存実験の記録。後半にwiki-Voteを用いた実装の原因調査と新しい計測結果を追記した。
















### 既存の実験条件
















| 項目 | 条件 |
| --- | --- |
| 実行環境 | Apple M4 Pro、統合メモリ24 GiB、macOS 15.7.2。各実装を個別に実行 |
| 比較手法 | CPU逐次SGD（Rust・f64）、Atomic Full（native Metal）、RoundRobin Full（Rust・wgpu・WGSL、Metal backend・f32） |
| 共通条件 | 15反復、epsilon=0.1。同一入力・seed・初期配置・学習率で比較 |
| 入力処理 | 無向・非重み付き化、ゼロ要素・自己ループ・重複辺の除去、最大連結成分を抽出。入力と初期配置をSHAで照合 |
| RR基本設定 | タイルサイズT=1024、workgroup 1024スレッド。ブロック間とタイル内をRRで処理 |
| 打切り | プロセス120秒、観測RSS 8 GiB。事前試行で失敗した条件は本計測を見送り、欠測は対応比較から除外 |
| 時間指標 | SGD時間÷15を平均反復時間とする。shuffle、dispatch、完了待ち、再投入を含み、前処理・初期化・stress評価・座標保存を除く |
| 品質指標 | 共通のfull stress。8,000頂点以下は厳密評価、それより大きい場合は固定256始点による推定 |
| 集計 | 同じseedで成功したCPU/GPUの比を計算し、グラフごとに中央値と最小・最大を報告。速度比=CPU時間/GPU時間、stress比=GPU stress/CPU stress |
















### 実施済みの比較
















| 実験 | 対象・条件 | 保存済み結果 |
| --- | --- | --- |
| 基本比較 | 20グラフ×3手法×seed 0〜4。各手法100条件 | CPU 95件、Atomic 95件、RR 99件完了。CPUとRRで比較可能な19グラフすべてで反復時間短縮、速度比1.025〜15.017倍。9グラフでstress比1.05超 |
| タイルサイズ | USpowerGrid、dwt_2680、bcsstk16、fe_4elt2、EVA、wiki-Vote。T=1,2,4,…,1024、各5 seed、workgroup=1024。T=1のみinner shuffleなし | 66設定中61設定で全5 seed完了、5設定は事前タイムアウトで未実施。wiki-Voteは最小完了T=2からT=1024でstressが65.69%増加 |
| 目標stress到達 | 上記6グラフ、各10 seed、15反復。目標は同じseedのCPU最終stress×1.05。保存点で最初に目標以下となった時刻を採用 | RRは60件中36件到達。USpowerGrid 10/10、dwt_2680 9/10、bcsstk16 7/10、fe_4elt2 10/10、EVA 0/10、wiki-Vote 0/10 |
| 初期配置の追加診断 | wiki-Vote、RRのseed 0〜99、T=1024、15反復、厳密stress評価 | 最良seed 31でも、同じ初期配置のCPUに対するstress比1.318 |
| 合成グラフの追加診断 | BAグラフ3個、2,000頂点・9,975辺、各5初期配置、T=1024、15反復 | グラフ別stress比中央値1.170、1.172、1.175。計15件すべてでCPU比1.05超 |
















基本比較の代表例は、fe_4elt2が速度比14.955倍・stress比1.017、wiki-Voteが速度比11.646倍・stress比1.671。これらは再検証対象の既存値として扱う。
















### 調査開始時の確認点
















- Fullのstress悪化が大きいため、まず既存測定時と現行実装のソース・バイナリ・実行引数を照合する。保存記録はgit_dirty=trueを含むため、commit IDだけでは測定コードを特定しきれない。
- 同一入力・初期配置・学習率・評価処理で、既存の品質差が再現するか確認する。再実験の対象グラフ、seed数、反復数、タイルサイズは未決定。
- 更新式、全頂点対の重複・欠落、タイル境界、同期を点検し、RRの更新順序とf32/f64の影響を切り分ける。現時点で品質差の原因は確定していない。
- Atomic Fullには更新順序と更新前座標の記録処理も含まれる。時間比較の解釈と再計測時の計測範囲を確認する。
















### 記録の出所
















現行原稿のFull結果は以前の実験を引き継いでいる。Sparse再測定時にFullを再計測していないことをprovenance.jsonで確認した。
















- 原記録ディレクトリ　`/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3`
- 継承済み集計　`GPU-Accelerated-SGD-for-Stress-Minimization/experiment_results/v4/reference/full-v3/` の runs.csv、summary.csv、convergence.csv、provenance.json
- 原稿　`vis-paper-2027-kose/section/04.experiment.tex`
- [既存の実験結果記事](https://vdslab.esa.io/posts/9029)・[論文記事](https://vdslab.esa.io/posts/9028)
























## wiki-Voteの品質悪化を実装から調査（2026-11-05追記）








**同じ相手ブロックへの更新を連続させる実行順序が、今回の品質悪化の主要因と判断する。** RRを保ってタイル内更新を16分割し、外側ラウンドを巡回する診断用GPUを試したところ、3 seedすべてでstressがCPU比1.03以内になった。








既存実験と同じ最大連結成分（7,066頂点・100,736辺）、同じ初期配置、15反復、epsilon=0.1、T=1024、workgroup=1024。全配置を同じf64の厳密full stressで評価した。入力SHAと初期配置SHAは旧記録と一致。








| seed | CPU逐次 f64 | 現行GPU RR | 現行のCPU比 | GPU RR・16分割 | 16分割のCPU比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3,239,648 | 6,142,906 | 1.896 | 3,281,099 | 1.013 |
| 1 | 3,208,765 | 5,048,197 | 1.573 | 3,291,619 | 1.026 |
| 2 | 3,210,218 | 5,134,448 | 1.599 | 3,305,760 | 1.030 |








現行カーネルは1つの異ブロックタイルのmatchingを最後まで処理するため、各頂点が同じ相手ブロックに対して最大1024回連続で更新される。ラウンド順をshuffleしても、この集中は残る。16分割版では約64 matchingごとに別の外側ラウンドへ移る。更新式・学習率・反復数・全頂点対・RRのmatching・内外の置換列を保持し、処理順序を変えた。








確認した対照実験は次のとおり。








- **精度**　現行RR順序をCPUのf64で再生してもstressはseed順に5,976,151／5,028,689／5,068,248。全対shuffleをf32で行うと3,205,501／3,207,349／3,209,825。精度を上げるだけでは品質差を解消できなかった。
- **メモリと同期の確認**　旧実験と現行のWGSLは一致。旧stressもほぼ再現した。seed 0でshared版とglobal版のGPU stressは一致した。
- **対の網羅性**　24,960,645対を数え上げ、現行順序・16分割順序とも1反復に各対を一度ずつ処理することを確認。CPU再生で計測した近接ゼロ距離の分岐は0回。
- **分割数**　seed 0では4分割が3,774,086、64分割が3,249,978（CPU比1.003）。16分割をseed 0で選び、seed 1・2で追加確認した。
- **検証**　既存Fullテスト6件と診断用テスト5件が通過。部分ブロックを含む小規模例で1・4・16分割のshared/global実行をCPU再生と座標誤差2e-5以内で照合した。








16分割版は診断用の試作。今回の確認範囲はwiki-Voteの3 seedで、他グラフへの一般化・速度評価・標準実装への採用は今後検討する。後続ブロックへの集中更新が先行ブロックへの適合を崩す機序が考えられるが、ブロック別誤差推移は未計測。CPUのRR f32再生とGPUには演算差による結果差もある。








詳細・生データ・再現コードはローカルの `/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/experiment_results/validation/wiki-full-cause-20261105/` に保存した。診断時間にはGPU初期化などを含むため、正式な速度比較には使用しない。












## wiki-Voteの反復時間比較（2026-11-05追記）




現行Full RRと16分割GPU試作を同じセッションで逐次実行した。Apple M4 Pro、T=1024、workgroup=1024、15反復、epsilon=0.1。seed 0〜2それぞれで各方式を1回warmup後、5回ずつ計測し、方式の実行順を交互に入れ替えた。各測定の15反復合計を15で割り、その中央値を示す。




| seed | 現行（ms/反復） | 16分割（ms/反復） | 差（ms） | 時間倍率 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 17.87 | 26.34 | +8.47 | 1.474 |
| 1 | 18.15 | 25.72 | +7.58 | 1.418 |
| 2 | 17.54 | 25.27 | +7.73 | 1.441 |
| all | 17.82 | 25.53 | +7.71 | 1.432 |




allは各方式15測定の中央値。16分割は現行より約43.2%長い。15反復の合計では中央値が267.3 msから382.9 msとなった。




計測範囲はシャッフル、反復中のqueue write、dispatch、GPU完了待ちを含む。前処理、GPU初期化、初期転送、最終読戻し、stress評価、保存を除く。現行は標準src/gpu.rsを直接呼び出し、16分割は診断用モジュールを呼び出した。GPUコンテキストを再利用し、各実行で初期配置とバッファを用意し直した。




dispatchは1反復10回から160回へ増える。座標の共有メモリへの読込み・書戻しも増える。これらが時間増加に寄与すると考えられるが、個別の費用は切り分けていない。全seedで各方式のstressは6実行すべて一致し、原因調査時の値を再現した。




品質と時間を併せると、今回の16分割試作は約1.43倍の反復時間でCPU比1.03以内のstressを得た。今回の時間はwiki-Vote単一グラフ・単一セッションの測定で、一般化や最適化後の速度は未検証。




測定コードは implementations/full/rr_gpu/examples/wiki_full_timing.rs。生記録は同調査ディレクトリの timing/runs.jsonl、集計は timing/summary.csv、ソース・バイナリSHAは timing/metadata.json。








## 完了待ち位置の対照実験




同じseed・入力・更新順序で、標準GPUコードの完了待ちを外側ラウンドごとに戻した場合と、現行の反復末尾待ちを比較。各seedでwarmup後5回ずつ、交互の順序で測定。




| seed | ラウンドごと待機 ms/反復 | 反復末尾待機 ms/反復 |
| --- | ---: | ---: |
| 0 | 26.909 | 18.669 |
| 1 | 34.098 | 17.291 |
| 2 | 25.954 | 17.206 |




各seedで最終stressは全測定で一致した。旧記録との差は同じseed内でも存在する。




加えて、旧測定時とSHAが一致する実バイナリをseed 0で再実行した（各条件1回、参考測定）。




current-cli: 17.001794 ms/iteration, stress=6142905.898276006, binary=/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/implementations/full/rr_gpu/target/release/rr-gpu




old-no-trace: 35.615364 ms/iteration, stress=6142905.898276006, binary=/Users/kose/master/atomic-scheduling-investigation/target/v3-release/release/rr-gpu




old-trace: 38.315297 ms/iteration, stress=6142905.898276006, binary=/Users/kose/master/atomic-scheduling-investigation/target/v3-release/release/rr-gpu




旧測定のseed 0は39.479 ms/反復。今回の旧バイナリ再実行でも当時の時間とは差があり、待機位置の変更だけですべての差を説明できない。実行時負荷・電力状態・warmupなど残る測定条件の寄与は未分離。CPU比24.8倍は旧CPU記録と今回のGPU測定を組み合わせた参考値として扱う。








## Full 16分割の採用と論文実験の再実行




Fullの標準実装を16分割へ変更。分割数はmin(16,T)。--phase-chunks 1で分割前の方式を指定できる。完了待ちは各反復の末尾。収束記録を復元し、既存テスト6件を確認した。




論文の基本比較100条件、収束60条件、タイル330条件（うちT=1024の30条件は基本比較の再利用）を実行中。CPU・AtomicとSparseは既存結果を参照する。wiki-Voteの100 seed診断とBAグラフの追加診断も追って更新する。時間上限120秒・RSS8 GiB・空きディスク8 GiBを継承する。




実験終了まで暫定1〜2時間、その後に集計・図・本文・PDF・和訳を更新する。アブストラクトは編集対象外。PDFを正としてesa #9028の和訳を合わせる。変更点と残り時間はチャットでも報告する。




保存先は `/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/experiment_results/full16/`。進捗はcheckpoint.json、生記録はraw/results.jsonl、実測バイナリと計画のSHAはconfig/execution.json。






## 再実験の範囲変更（2026-11-05）


依頼によりタイルサイズ実験330条件は後日に延期し、今回の計画から除外。完了済み89条件を保持して、基本比較100条件と収束60条件のみ継続する。旧計画・実行ハッシュはconfig/*original-with-tiling*に保存。変更後はconfig/execution-speed-stress.json。


次数の偏りとサイズに対する確認として、BAモデル（m=5、2,000／4,000／8,000頂点、グラフseed 0・1）の6グラフを追加する。各グラフの初期配置seed 0〜4をCPU・分割前・16分割で比較し、計90条件。測定は直列に実行し、同一seedの入力・初期配置のハッシュを照合する。100 seed探索は今回は実施しない。


残りの実験時間は暫定20〜30分。既存CPU・Atomic・Sparse結果は再利用し、追加BA実験はCPUも同期間に新規測定する。タイルの旧結果は新実装の結果として示さず、本文と図から一旦外す。アブストラクトは変更しない。



## Full 16分割の再実験完了（2026-11-05 14:36計測終了）

### 範囲と実装変更

- Fullの既定値を16分割へ変更（実効値min(16,T)、`--phase-chunks 1`で分割前）。内側のshuffle済み順列を連続区間に分け、区間ごとに全外側ラウンドを巡回する。全対を1反復に一度ずつ処理し、更新式・学習率・15反復は維持する。
- CPU側のGPU完了待ちは各反復末尾。区間ごとに座標を共有メモリへ読み、処理後にグローバルメモリへ書き戻す。Full概念図とアクセス数の説明を修正。
- Sparseは既存の各相4分割を維持し、実装・測定結果を再利用。片側更新は更新先非pivotごとの担当、両側更新はpivot対のRRと実辺のmatching。
- タイルサイズ実験330条件は依頼により延期。旧タイル図・旧実験節と100 seed探索の旧結果は今回の本文から外した。旧測定記録は保持。
- アブストラクトは英語・和訳とも据え置き。そこに残る旧Full数値と更新後本文の不一致は意図的な未更新箇所。

### 実験と検証

- 既存20グラフ×5 seedの基本比較100条件、6グラフ×10 seedの収束60条件を再測定し全160条件成功。CPU・Atomicは同じ実験機の旧記録を使用するため、主比較は測定日の異なる値を含む。
- BAモデルm=5、頂点数2,000／4,000／8,000、各グラフseed 0・1の6入力を追加。各5初期配置でCPU・分割前・16分割を同期間に直列測定し、全90条件成功。追加計測は372.85秒（約6分13秒、warmup・保存を含む）。
- 入力・初期配置・seed・反復数・epsilon・stressの評価条件を照合し、対応比較の不一致は0件。T=W=1024、epsilon=0.1。
- 収束は0,1,2,3,5,8,10,15反復で保存。8,000頂点以下は厳密、その他は固定256始点から評価。全60件の最終保存配置を再評価。
- 保存配置は最終f32センタリングの前なので、ネイティブ最終stressとは最大3.96e-9の相対差がある。同じセンタリングを再現すると相対差は最大4.80e-15で一致。詳細はtables/convergence-stress-audit.csv。
- 追加のGPU回帰検証は1・4・16分割、T未満の部分ブロック、shared/globalを逐次RR参照と照合。単体6件とJSON統合1件が通過。実測後のソース変更はテストのみで、実行コードは保存したソースと一致。

### 主比較の結果

|項目|更新前|16分割の更新後|
|---|---:|---:|
|FullでCPUより速いグラフ|19/19|17/19|
|Fullでstress比中央値1.05以下|10/19|18/19|
|両条件を満たすFullグラフ|10/19|16/19|
|6グラフの品質目標到達|36/60|55/60|
|wiki-Voteのstress比中央値|1.671|1.026|
|wiki-VoteのCPU速度比|11.646|16.928|

旧Full速度との差には、分割だけでなく以前の完了待ち位置の変更と測定時期の差も含む。分割単独の速度・品質の対照は、次の同期間BA実験を用いる。

Fullの速度比範囲は0.855〜23.469倍。dwt_1005と1138_busはCPUより遅い。bcsstk17はstress比1.208が残る。dwt_2680の収束到達は旧9/10から6/10となり、個々の初期配置での改善は保証されない。EVAは9/10、他の4グラフは10/10。

|グラフ|Hit|End|CPU到達秒|16分割到達秒|
|---|---:|---:|---:|---:|
|EVA|9|9|1.678|0.281|
|USpowerGrid|10|10|2.046|0.199|
|bcsstk16|10|10|2.907|0.285|
|dwt_2680|6|6|0.807|0.160|
|fe_4elt2|10|10|11.593|0.463|
|wiki-Vote|10|10|6.691|0.385|

### 次数に偏りのあるBA入力の同期間比較

各行5初期配置の対応比率の中央値。反復時間は15反復合計÷15。stressは全入力で厳密評価。

|頂点数/graph seed|辺数|最大次数|CPU ms/反復|分割前 ms/反復|16分割 ms/反復|16分割CPU速度比|分割前stress比|16分割stress比|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|BA-2000-g0|9975|156|24.250|7.252|8.305|3.004|1.170|1.010|
|BA-2000-g1|9975|143|24.792|7.146|8.417|2.955|1.172|1.007|
|BA-4000-g0|19975|218|127.506|9.227|13.089|9.746|1.503|1.017|
|BA-4000-g1|19975|218|127.609|9.570|12.941|9.907|1.329|1.013|
|BA-8000-g0|39975|295|569.205|18.986|25.868|21.997|1.519|1.037|
|BA-8000-g1|39975|311|599.715|18.792|25.935|22.953|1.407|1.038|

分割前は0/30、16分割後は30/30配置がCPU差5%以内。分割後の反復時間は分割前の1.16〜1.42倍。BAの次数の偏りとサイズを変えた確認であり、次数分布だけの因果効果や全ネットワークへの一般化を示すものではない。

### 記録と再現

- 主計測: `experiments/run_full16.py`、固定計画 `config/execution-speed-stress.json`、生記録 `raw/results.jsonl`。
- 追加計測: `experiments/run_full16_degree.py`、`degree/config/execution.json`、`degree/results.jsonl`。NetworkXバージョン・入力・実行バイナリのSHAを記録。
- 集計: `python3 experiments/summarize_full16.py`。生記録と既存CPU/Atomicを統合した図用入力は `paper-data/`。
- 論文repoで `python3 figs/build_compact_schematics.py --full-only`、`python3 figs/build_comparison_figures.py`、`python3 figures/plot_iteration_times.py <このディレクトリ>/paper-data`、`python3 figs/build_degree_comparison.py`、`make check`。
- 凍結ソース: `config/source-snapshot/`、旧原稿・和訳: `config/manuscript-before/`、GPUテスト: `production-tests.log`。
- 旧CPU/Atomic/Sparse原記録: `../v4/reference/full-v3/`、`../v4/tables/`。


### 論文・和訳への反映完了

英語本文（はじめに・提案手法・評価・結論）と、Full概念図・反復時間図・品質到達時間図・BA比較図を更新。9ページのPDFはmake checkを通過し、描画結果も確認した。アブストラクトは元の英語ファイルと完全一致。和訳のAbstractも本文を維持した。

[更新した和訳 #9028](https://vdslab.esa.io/posts/9028)には同じ図と[更新版英語PDF](https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/8704/2026/11/05/164555/774a1fb8-1624-40e1-9e02-872ca1ce7fa6.pdf)を添付した。PDFを正とする。Sparse・道路網の測定値は再利用し、タイル感度実験は延期した。

![追加BAグラフの速度・stress比較](https://img.esa.io/uploads/production/attachments/8704/2026/11/05/164555/e0adf921-6edc-4cef-aca7-8c56ce810648.png)

計測は14:36までに全250条件完了。集計・対応ハッシュ照合・最終配置のstress照合・GPUテスト7件・PDF確認まで完了した。git commitやpushは行っていない。

## アブストラクトの関連箇所のみ更新（2026-11-05追記）

追加依頼により、当初据え置いていたアブストラクトも、今回のFullの変更に関わる箇所のみ英語・和訳とも更新した。上記の「アブストラクト据え置き」は追記前の作業履歴であり、この追記で解消した。

- Fullの手法説明に、内側ラウンド列の16区間を外側ラウンド間で交互に進め、各タイル区間で座標を再利用することを明記。
- CPUより反復が速いFullグラフを「全19グラフ」から「17/19」に修正。
- FullのCPU比速度向上率の中央値を7.38倍から9.98倍に修正。
- Fullの最終stress増加率の中央値を4.5%から0.04%、最大のグラフ別中央値を67.1%から20.8%に修正（19グラフの対応比率を集計）。
- 和訳ではFullのstress比中央値1.05以下も18/19に修正。Sparseと道路網の既存記述・結果は維持。

新しい実験は不要で、完了済みのFull16集計を使用した。英語原稿はAbstract以外の本文・図を変更せず、`make check`（textlint・厳密な再ビルド）を通過。9ページのPDFを再描画し、アブストラクトと全体のレイアウトを確認した。[和訳 #9028](https://vdslab.esa.io/posts/9028)と添付PDFを更新済み。

[アブストラクト更新後の英語PDF](https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/8704/2026/11/05/164555/38292f28-6df1-4f7c-bf31-ef1b1d74db9f.pdf)

## アブストラクトの結果を手法ごとに整理（2026-11-05）

読みやすさの修正依頼を受け、結果の順番を「Fullの速度・stress → Sparseの速度・stress → 40 pivotでの大規模描画」に変更した。英語の実験値は変更せず、各手法の速度と品質を隣接させた。和訳も英語PDFにそろえ、stress比1.05以下の件数や道路網の反復部分4.48秒の詳細は本文に残し、Abstractではstress増加率の中央値と前処理込みの道路網2件の結果を示す。

追加実験なし。`make check`と9ページのPDFの表示確認を完了。[和訳 #9028](https://vdslab.esa.io/posts/9028)と[英語PDF](https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/8704/2026/11/05/164555/82fc6170-f311-4346-9122-88521410f521.pdf)を更新した。
