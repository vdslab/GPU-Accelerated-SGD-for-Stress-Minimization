# No.5 収束結果の考察

正式720条件と、成功657runの8保存点、計5,256 checkpointのstress再評価を完了した。以下は到達判定・coverage・stress照合・時間traceのscalar記録に基づく。native成功657件と目標到達657件は同義ではなく、目標1.05/1.01ごとの観測到達・終端状態・欠測を分けて示す。

## 対象と観測の母集団

No.5は、18種類のグラフを24組の「graph × family」で比較する。小規模6件にはFullとSparseの両方（12組）、大規模12件にはSparse（12組）を実行し、各組のCPU / Atomic / RRをseed 0〜9で測った。合計は **24組 × 3手法 × 10seed = 720条件**。SNAP原配布の追加6グラフはbasicだけの補完であり、このNo.5母集団へは追加していない。

nativeの結果は **success 657、実warmup失敗による本測定未起動60、memory_limit 3**。Full180条件は全件成功。Sparse540条件は477成功、60未起動、3資源停止である。正式本測定を実際に起動したのは660条件で、No.5内に再利用条件はない。

### 全24組の成功数

各セルは成功数 / 計画10seed。最終確認では成功657runすべてに8保存点とCPU基準が揃い、各目標の評価可能数も657だった。

| Graph | Family | CPU | Atomic | RR | 欠測の理由 |
|---|---|---:|---:|---:|---|
| EVA | Full | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| USpowerGrid | Full | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| bcsstk16 | Full | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| dwt_2680 | Full | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| fe_4elt2 | Full | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| wiki-Vote | Full | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| EVA | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| USpowerGrid | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| bcsstk16 | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| bmw7st_1 | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| coAuthorsCiteseer | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| coPapersDBLP | Sparse | 10/10 | 10/10 | 7/10 | RR seed 0/3/8が本測定memory_limit。CPU基準は10seed存在 |
| delaunay_n17 | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| delaunay_n20 | Sparse | 0/10 | 0/10 | 0/10 | 3手法ともwarmup memory_limit。本測定30条件は未起動 |
| dwt_2680 | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| fe_4elt2 | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| kron_g500-logn18 | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| luxembourg_osm | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| preferentialAttachment | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| roadNet-CA | Sparse | 0/10 | 0/10 | 0/10 | 3手法ともwarmup memory_limit。本測定30条件は未起動 |
| smallworld | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| web-NotreDame | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| web-Stanford | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |
| wiki-Vote | Sparse | 10/10 | 10/10 | 10/10 | 10seedすべてnative成功 |

手法別では、CPU 220/240、Atomic 220/240、RR 217/240がnative成功した。CPUの最終stressが存在する参照枠は、graph/family/seedで **220/240枠**。成功したGPU条件437件は、同seed・同familyの成功CPUと入力/初期配置/前処理SHA、反復数、epsilon、P、stress評価定義が一致していた。

## 「未到達」と「基準・観測がない」を分ける

目標は、同じgraph/family/seedの **CPUの反復15保存配置をオフライン再評価したstressの1.05倍または1.01倍**。FullはFull CPU、SparseはSparse CPUを基準にする。全seedの中央値や別familyのCPU、GPU自身の最終値で目標を置き換えない。

- roadNet-CAとdelaunay_n20はCPUを含む全手法が未起動で、CPU基準そのものがない。20参照枠が欠け、その3手法分の60条件は到達率の「未到達」へ混ぜない。
- coPapersDBLPのRR seed 0/3/8はCPU基準が存在する一方、GPU本測定がRSSで停止した。これは測定欠測であり、15反復しても目標へ到達しなかったという観測ではない。RRは計画10seedと、native成功7seedの両方の母数が分かる形で記載する。
- CPU基準と成功runのcheckpoint観測が揃ったうえで、保存したどのcheckpointにも目標以下の値がなければ「15反復の保存点では未観測」とする。基準欠如や資源停止とは別の分類である。

目標到達率を示す際は、計画10seed、基準があるseed、利用可能な観測seed、実際に目標を観測したseedを分ける。欠測がグラフ規模やメモリ負荷に依存するため、成功例だけの比較を大規模12件全体の一般的な傾向へ広げない。

## 時間とcheckpointの解釈

保存反復は0/1/2/3/5/8/10/15であり、全更新後にstressを観測しているわけではない。図の最初の到達確認時刻は、そのcheckpointで目標を満たす配置が得られている時刻である。前の保存点との間を、実際の「初回到達」の保証区間とはしない。stressは非単調に動き、一度下回ってから上回る可能性もあるため、観測到達率と15反復後にも目標を満たす率を別々に確認する。

時間はtrace copy/IOを除いたSGD区間の累積で、プロセス全体のwall時間とは異なる。No.5は座標checkpointを保存するため、basic計測と付加処理が異なる。特にweb-NotreDameの同seed後続完走を、以前のbasic timeoutの置き換えや、再起動だけの厳密な介入効果として扱わない。

## 目標別の観測結果

1.05倍目標では **509/657** 条件で到達を観測し、反復15でも目標以下だったのは **500/657**。1.01倍では **413/657** の到達を観測し、反復15で目標以下は **404/657** だった。各目標で残る63/720条件は欠測で、未観測148件（1.05倍）・244件（1.01倍）と分ける。これらの合計には、定義上自分自身の反復15目標を満たすCPU220条件が含まれるため、手法比較には次の内訳を使う。

| Family | 手法 | 評価可能 / 計画 | 1.05到達観測 | 1.05反復15で以下 | 1.01到達観測 | 1.01反復15で以下 |
|---|---|---:|---:|---:|---:|---:|
| Full | CPU | 60/60 | 60 | 60 | 60 | 60 |
| Full | Atomic | 60/60 | 30 | 30 | 4 | 4 |
| Full | RR | 60/60 | 36 | 36 | 14 | 14 |
| Sparse | CPU | 160/180 | 160 | 160 | 160 | 160 |
| Sparse | Atomic | 160/180 | 126 | 123 | 103 | 99 |
| Sparse | RR | 157/180 | 97 | 91 | 72 | 67 |

Fullでは、1.05倍の到達率はAtomic30/60、RR36/60だが、1.01倍へ厳しくするとAtomic4/60、RR14/60まで下がる。SparseではAtomic126/160→103/160、RR97/157→72/157だった。欠測CPU20・Atomic20・RR23条件を分母へ混ぜて「未到達」とはしていない。この集計は今回選んだグラフと完了runに対する結果であり、全グラフへの一般化やlock方式だけの因果比較ではない。

coverage CSVの`reference_unavailable`は0だが、CPU基準が240枠すべて存在したという意味ではない。実装は自手法の観測欠如を先に判定するため、`measurement_unavailable`63件に、CPU基準もないroadNet-CA/delaunay_n20の60条件と、CPU基準があるcoPapersDBLP RRの3失敗条件が一緒に入る。母集団表と合わせて読む。

## 代表3グラフの比較

下表の時間は、到達を観測したseedだけの「最初に確認できたcheckpoint時刻」の中央値（SGD秒）。前のcheckpointとの間を真の初回到達の保証区間とはしない。両目標では到達したseed集合が異なるため、中央値の大小を同じrun内の順序とは解釈しない。

| Graph / Family | 手法 | 1.05到達→反復15以下 / 10 | 1.05観測時刻 s | 1.01到達→反復15以下 / 10 | 1.01観測時刻 s |
|---|---|---:|---:|---:|---:|
| dwt_2680 / Full | CPU | 10→10 | 0.807 | 10→10 | 0.807 |
| dwt_2680 / Full | Atomic | 10→10 | 1.546 | 4→4 | 1.552 |
| dwt_2680 / Full | RR | 9→9 | 0.144 | 9→9 | 0.144 |
| luxembourg_osm / Sparse | CPU | 10→10 | 6.722 | 10→10 | 6.722 |
| luxembourg_osm / Sparse | Atomic | 10→10 | 6.719 | 10→10 | 6.719 |
| luxembourg_osm / Sparse | RR | 10→10 | 0.173 | 10→10 | 0.235 |
| kron_g500-logn18 / Sparse | CPU | 10→10 | 2.138 | 10→10 | 5.327 |
| kron_g500-logn18 / Sparse | Atomic | 9→6 | 5.677 | 6→2 | 5.431 |
| kron_g500-logn18 / Sparse | RR | 5→0 | 8.775 | 4→0 | 13.987 |

**Full / dwt_2680。** RRは両目標で9/10seedが到達し、終端でも目標以下だった。到達を確認した時刻は中央値0.144秒、同じ到達seedにおけるCPU時間/RR時間の中央値は5.54倍。一方で残り1seedは保存点で未観測である。Atomicは1.05倍なら10/10だが、1.01倍は4/10で、到達を観測した4seedでも中央値1.552秒だった。短時間だけを比較せず、満たした品質とseed数を併記する必要がある。

**Sparse / luxembourg_osm。** 両GPUが両目標を10/10で観測し、全seedが反復15でも目標以下、観測されたreboundも0だった。RRの1.05倍確認時刻は中央値0.173秒、1.01倍は0.235秒。CPUの確認時刻は両目標とも6.722秒で、同seedのCPU/RR時刻比の中央値はそれぞれ38.84倍、28.67倍だった。これはSGDの目標確認までの時間であり、前処理込みの全体時間の速度比ではない。

**Sparse / kron_g500-logn18。** RRは1.05倍を5/10、1.01倍を4/10で一度は観測したが、到達した全seedで後の保存点が目標を上回り、反復15で目標以下は両目標とも0/10だった。Atomicも1.05倍は9/10到達から終端6/10、1.01倍は6/10到達から終端2/10へ減る。初回の目標確認だけを「良好に収束した」と扱うと、この終端悪化を見落とす。

## reboundと終端を分けた解釈

Fullの到達例では観測されたreboundは0。Sparseでは1.05倍目標でAtomic5件・RR6件・CPU1件、1.01倍でAtomic4件・RR5件・CPU1件に、初到達後の再上昇があった。1.05倍のAtomicでは5件中2件が反復15で再び目標以下に戻り、CPUの再上昇例も両目標とも終端では目標以下へ戻った。rebound数と「終端で目標を失った数」は同義ではなく、終端で目標以下でも初到達後ずっと以下だったという意味ではない。

1.05倍・1.01倍のどちらも、到達観測数から終端で目標以下の数へ9件減った。これを評価器の異常やlock不具合と特定するものではなく、保存した配置から観測したstressの非単調性として報告する。チェックポイントの間に一時的に到達・再上昇していた可能性は残る。

## stress評価と時間記録の整合性

最終再描画後の[stress照合CSV](../tables/convergence-stress-audit.csv)は657件すべてでnative/offlineの評価定義が一致し、0.1%の調査閾値を超えた行・欠けた反復15・無効値は0だった。最大相対差はbcsstk16 / Full RR / seed9の **0.000003178%**（offline 185,977.66296313007、native 185,977.65705262683）。保存配置はcentering前、native最終評価はcentering後で、f32/f64の丸めによりbit単位の一致までは求めていない。

**再描画時のmetadata解析事故（修正・再確認済み）：** CSVの`stress_seed`が文字列`"0"`のまま数値`0`と比較され、sampled357件に誤った`definition_mismatch`が出た。CSV読み戻しの型変換が原因で、通常cache経路と`--skip-convergence`経路を共通parserへ変更し、整数復元と小さいCSV往復テストを追加した。修正済みsource `bbddcfe` による最終再描画は正常終了し、657件すべてが`within_0_1_percent`、評価定義一致へ戻った。[修正前の監査CSV](cache-type-audit-before-fix.csv)との全657行照合でもnative/offlineのstress数値と差率は不変で、native再実験・stress再計算は行わず、到達集計も変わらない。更新式やGPU同期の無誤りまで、この照合で証明するものではない。

[iteration-time-validation.csv](../tables/iteration-time-validation.csv)の1,154行はすべてpassだが、これはbasic成功runの監査であり、No.5自身の監査とは分ける。No.5については657runの小さいtrace JSONだけで、反復0〜15の連続性、各反復durationと累積和、最終累積値とrecordのSGD時間を別途照合し、**657/657 pass、最終累積値との差は最大0 ms**だった。[No.5時間照合CSV](convergence-trace-time-validation.csv)に保存した。座標・巨大stressの再計算やcache変更は行っていない。

数値の根拠：[全seed・全目標の到達判定](../tables/target-arrival.csv)、[graph/family/method別の母数・到達・終端集計](../tables/target-arrival-coverage.csv)、[保存点の評価値](../tables/convergence.csv)。母集団の根拠は[固定plan](../config/plan.json)の`study=convergence`720条件と[core監査](../tables/audit-v3-core.json)、欠測段階は[実失敗CSV](../tables/audit-v3-core-failures.csv)。
