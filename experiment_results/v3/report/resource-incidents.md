# v3 資源制限・中断事故の調査（補完後の確定値）

更新：2026-10-24T23:12:30+09:00。**core 3,210 記録と SNAP 原配布の補完 90 記録、計 3,300 原記録を保持し、有効な 3,210 条件を確定した最終監査**である。元の未起動 unavailable 90 条件は履歴として保存し、集計時に補完条件へ一度だけ対応付けた。実起動済み条件を上書きしたものではない。

[最終監査 JSON](../tables/audit-v3.json) は adopted / success、整合性エラー 0、欠けた・余分な・重複した有効 ID は 0。[core 単独監査](../tables/audit-v3-core.json) と [補完単独監査](../tables/audit-v3-recovery.json) は別に保持する。native/shared source・runner・binary SHA と測定制限は一致し、native commit は base 別に照合した。

| 記録の出所 | 原記録 | 本測定の実起動 | warmup 実起動 | warmup 実失敗 |
|---|---:|---:|---:|---:|
| core | 3,210 | 2,325 | 450 | 117 |
| SNAP 補完 | 90 | 45 | 18 | 9 |
| 最終有効 cohort | 3,210 条件 | 2,370 | 468 | 126 |

補完 90 条件は success 45、warmup_failed 45。Amazon / DBLP / Gowalla は各 15 条件が成功した。YouTube / Pokec / LiveJournal は各手法の warmup が RSS 制限に達し、各 15 条件の本測定は未起動となった。データを取得・検証できたことと、15 反復の本測定を完走できたことを区別する。

## 条件数と実起動数

| 区分 | 件数 | 意味 |
|---|---:|---|
| 計画・確定条件 | 3,210 | 再利用・未起動も含む条件 ID 数 |
| 確定 record を持つ本測定の起動 | 2,370 | 成功 2,321、memory_limit 46、timeout 3 |
| 再利用 | 120 | 成功の再利用 105、warmup_failed の再利用 15 |
| 本測定の未起動 | 720 | 実 warmup 失敗で未起動 690、規模除外 15、power197k unavailable 15 |
| 再起動による追加中断 attempt | 1 | 同じ条件を後に完了。新たな計画条件や timeout には加算しない |
| 本測定の総起動 attempt | 2,371 | 確定起動 2,370 + 追加中断 1。warmup は別枠 |
| warmup の実起動 | 468 | 成功 342、memory_limit 117、timeout 9 |

したがって **2,370 + 120 + 720 = 3,210 条件**、本測定の実失敗は **49 件**、warmup の実失敗は **126 件**。pilot と prebuffer-aborted は加算しない。warmup が 1 回失敗して 5/10 seed が未起動となることと、5/10 回の本測定が実際に失敗したことを区別する。

| 条件 record の status | 件数 |
|---|---:|
| success | 2,426 |
| memory_limit | 46 |
| timeout | 3 |
| warmup_failed | 720 |
| unavailable | 15 |

この status 表は再利用を含む **条件数**。success 2,426 を実起動成功数 2,321 と混同しない。warmup_failed 720 も 720 回の実失敗ではなく、690 条件の実 warmup 失敗による未起動、15 条件の規模除外、15 条件の再利用から成る。入力 unavailable は未取得 power197k の 15 条件のみ。元 core に残る 90 件の unavailable は [superseded 履歴](../tables/cohort-superseded.csv) として保存し、有効件数へ重複加算しない。

## 資源停止と観測された段階

全体 wall 制限は 120 秒、RSS 監視閾値は 8 GiB、最低空き容量は 8 GiB。120 秒は入力・前処理・初期化・SGD・stress 評価・native 保存までを含む。Full Atomic の lock 試行上限とは異なる。trace の最終反復は観測段階であり、最後に動いた命令を特定するものではない。

| 区分 | status | 観測段階 | 件数 |
|---|---|---|---:|
| formal | memory_limit | 反復 0（第1反復未完了） | 10 |
| formal | memory_limit | trace なし | 36 |
| formal | timeout | 反復 1〜14（途中） | 3 |
| warmup | memory_limit | trace なし | 117 |
| warmup | timeout | trace なし | 3 |
| warmup | timeout | 反復 1〜14（途中） | 6 |

本測定の memory_limit 46 件は trace なし 36 件・反復 0 まで 10 件。warmup の memory_limit 117 件はすべて trace なしだった。資源監視が観測 RSS により停止した記録であり、OS の OOM や lock の deadlock を確認したという意味ではない。途中停止に最終 stress や 15 反復分の完了時間を割り当てない。

詳細：[全実失敗と trace 段階](../tables/audit-v3-failures.csv)、[status 別時間・RSS](../tables/audit-v3-status.csv)。

### delaunay_n14 の Full 方式

16,384 頂点・49,122 辺。core で確定した関連停止は以下の 7 起動であり、補完では再実行していない。

| 起動 | status | 全体秒 | peak RSS GiB | 最終反復 |
|---|---|---:|---:|---:|
| v3-basic-delaunay_n14-sgd-s0-pNone-tNone | memory_limit | 3.868 | 8.445 | 0 |
| v3-basic-delaunay_n14-rr_sgd-s0-pNone-t1024 | memory_limit | 3.856 | 8.148 | なし |
| v3-basic-delaunay_n14-sgd-s1-pNone-tNone | memory_limit | 3.827 | 8.320 | 0 |
| v3-basic-delaunay_n14-sgd-s2-pNone-tNone | memory_limit | 3.858 | 8.196 | 0 |
| v3-basic-delaunay_n14-sgd-s3-pNone-tNone | memory_limit | 3.857 | 8.321 | 0 |
| v3-basic-delaunay_n14-sgd-s4-pNone-tNone | memory_limit | 4.053 | 8.320 | 0 |
| warmup-delaunay_n14-atomic_sgd-pNone-tNone | memory_limit | 3.858 | 8.885 | なし |

Full CPU の 5 seed はいずれも反復 0 までで停止し、第1反復の shuffle 中か座標更新中かは未特定。Full RR seed 0 と Full Atomic warmup は trace 開始前の準備段階で停止した。Full Atomic の 1 warmup 停止により本測定の 5 条件は未起動になった。

全点対は 134,209,536 ペア。CPU は 32 bytes のペア配列だけで約 4 GiB、usize 距離行列が約 2 GiB、さらに実行時のペア clone がある。Full RR の距離行列・中間ペア・buffer、Full Atomic の GPU 用ペア・常時記録用配列も負荷となる。ただし allocation の理論サイズをそのまま実 RSS の内訳とは解釈しない。

## 補完 3 グラフの warmup RSS 停止

YouTube / Pokec / LiveJournal のデータ取得・構造検証・LCC 準備は成功した。一方、P=200 の 3 手法で各 1 回、計 9 warmup が memory_limit で停止し、対応する 45 本測定条件は開始しなかった。9 回の停止を 45 回の本測定失敗へ読み替えない。

| グラフ | 手法 | 全体秒 | 観測 peak RSS GiB | 最終反復 |
|---|---|---:|---:|---:|
| com-LiveJournal | atomic_sparse_sgd | 98.992 | 8.057 | なし |
| com-LiveJournal | rr_sparse_sgd | 100.228 | 8.068 | なし |
| com-LiveJournal | sparse_sgd | 105.583 | 8.020 | なし |
| com-Youtube | atomic_sparse_sgd | 14.467 | 8.028 | なし |
| com-Youtube | rr_sparse_sgd | 14.658 | 8.051 | なし |
| com-Youtube | sparse_sgd | 118.294 | 8.613 | なし |
| soc-Pokec | atomic_sparse_sgd | 64.463 | 8.031 | なし |
| soc-Pokec | rr_sparse_sgd | 73.392 | 8.109 | なし |
| soc-Pokec | sparse_sgd | 69.607 | 8.137 | なし |

全 9 起動に反復 trace の完了記録がなく、15 反復を完走したという証拠はない。記録上の停止理由は監視 RSS による memory_limit であり、timeout status や Metal error が記録された停止ではない。共通前処理・制約配列・clone・GPU buffer 等のどの allocation で閾値を超えたかは、既存の段階 metadata だけでは区別できない。lock deadlock や更新式の誤りを原因と特定した事例ではない。最終 stress と完了 SGD 時間は欠測のまま扱う。[補完の失敗・観測段階](../tables/audit-v3-recovery-failures.csv)。

## 120 秒で停止した起動

本測定の timeout は web-NotreDame の 3 起動。warmup の timeout は以下の 9 起動。すべて全体 wall の制限によるもので、lock 待ちを原因と特定した結果ではない。

| 区分 | 条件 | 全体秒 | 最終反復 |
|---|---|---:|---:|
| formal | v3-basic-web-NotreDame-atomic_sparse_sgd-s1-p200-tNone | 120.186 | 6 |
| formal | v3-basic-web-NotreDame-sparse_sgd-s2-p200-tNone | 120.131 | 6 |
| formal | v3-basic-web-NotreDame-atomic_sparse_sgd-s2-p200-tNone | 120.152 | 3 |
| warmup | warmup-USpowerGrid-rr_sgd-pNone-t1 | 120.143 | 13 |
| warmup | warmup-bcsstk16-rr_sgd-pNone-t1 | 120.152 | 14 |
| warmup | warmup-fe_4elt2-rr_sgd-pNone-t1 | 120.181 | 5 |
| warmup | warmup-fe_4elt2-rr_sgd-pNone-t2 | 120.168 | 11 |
| warmup | warmup-kron_g500-logn19-rr_sparse_sgd-p200-tNone | 120.174 | 1 |
| warmup | warmup-rgg_n_2_21_s0-rr_sparse_sgd-p200-tNone | 120.230 | なし |
| warmup | warmup-wiki-Vote-rr_sgd-pNone-t1 | 120.177 | 11 |
| warmup | warmup-wikipedia-20051105-atomic_sparse_sgd-p200-tNone | 120.154 | なし |
| warmup | warmup-wikipedia-20051105-rr_sparse_sgd-p200-tNone | 120.153 | なし |

反復途中まで進んだものと、trace 開始前のものを分けて保存する。途中まで更新されたことは分かっても、その後の停止が deadlock かどうかをこの trace だけで確定できない。

web-NotreDame では、再起動前の basic seed 1 Atomic、seed 2 CPU/Atomic が timeout、RR seed 1/2 は約 97 秒で成功した。後続の計画済み No.5 では同じ seed 0/1/2 の全手法が完走した。比較可能な成功組では入力・初期配置・前処理 SHA 等が一致し、CPU seed 1 の最終 stress は両計測で 7,311,566,741.47717 と一致した。ただし No.5 は座標 checkpoint を追加するため、再起動だけの介入対照ではない。以前の失敗・遅延値は置換せず残す。[システム snapshot と遅延調査](web-notredame-timeout-system-snapshot.md)、[同 seed 比較と再起動調査](reboot-incident.md)を参照。

## 入力除外と再起動の区別

`G_n_pin_pout` は正規データの LCC が 99,995 頂点で large 群の 100,000 頂点条件を満たさない。warmup 側は unavailable・実起動なし、本測定側は runner の流れにより warmup_failed となった。3 手法 × 5 seed の 15 条件を資源事故へ数えない。実 warmup 失敗が参照されている条件は、graph/method/P/T と失敗理由を監査し、別々に分類した。[未起動理由一覧](../tables/audit-v3-unexecuted.csv)。

2026-10-24 12:19:50 JST のホスト再起動で、web-NotreDame CPU seed 3 の未確定 attempt が 1 回中断した。途中 artifact と receipt を退避し、同一 protocol で再開して当該条件は成功した。原因は未特定で、OS panic・メモリ不足・手動再起動・GPU lock のどれかに断定する証拠はない。この中断は上の memory_limit/timeout や確定 ID に追加せず、追加起動 attempt としてのみ数える。[中断 receipt](../interruption-reboot-20261024/receipt.json)。

## 閾値を超える終了後 peak RSS を記録した成功起動

補完後の全成功起動のうち該当するものは **24 件（本測定 21、warmup 3）**。24 件すべてが core 由来で、補完で新たな超過成功はなかった。[report 側の一覧](success-rss-threshold-exceedances.csv) は [最終監査の一覧](../tables/audit-v3-success-rss-exceedances.csv) と byte 単位で同じ確定表。origin_base / origin_scope により出所も区別できる。

| 区分 | 手法 | 件数 | RSS 最小 GiB | RSS 最大 GiB |
|---|---|---:|---:|---:|
| formal | atomic_sparse_sgd | 15 | 8.020 | 8.544 |
| formal | rr_sgd | 2 | 8.047 | 8.484 |
| formal | rr_sparse_sgd | 4 | 8.003 | 8.074 |
| warmup | atomic_sparse_sgd | 2 | 8.019 | 8.162 |
| warmup | rr_sgd | 1 | 9.130 | 9.130 |

監視は 200 ms 間隔であり、実行中の厳密な最大 RSS を 8 GiB 以下へ保証する方式ではない。保存値は監視で観測したプロセス群 RSS と、終了後の macOS `time -l` の最大 RSS の大きい方である。短い peak を監視が捉えず、終了後の最大値だけが閾値を超えることがある。成功 status を閾値超過だけで後から失敗へ書き換えない。RSS は GPU 専用 VRAM の量ではない。

## 既知の過去事故と出典

最初の一巡は大量の unbuffered text 出力により保存時間が大きかったため、中止して保管した。1 MiB の buffered 出力と、native 計測外での検証付き gzip 保管を導入して最初から測り直した。CPU の検証では最終 stress が不変で、全 6 手法の正常完了を確認した。これは下記 prebuffer-aborted の履歴であり、上の core 件数へ合算しない。[出力事故調査](io-export-incident.md)。

固定 source の説明：[Full CPU 前処理](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/implementations/full/baseline-sgd-non-gpu/src/graph.rs)、[Full Atomic の準備と trace](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/implementations/full/vram-lock-native/src/metal.rs)、[Full RR の GPU 準備](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/implementations/full/rr_gpu/src/gpu.rs)、[監視・warmup 判定](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/experiments/run_v3.py)。

## 付録：prebuffer-aborted の歴史的な途中調査

**以下は測り直す前の中断計測の保存記録であり、現在の有効 3,210 条件や成功 RSS 24 件とは別集計である。数値を現在の最終値として使用しない。**

### delaunay_n14：Full方式が8 GiBのRSS制限へ到達

確定済みのrecord.jsonがある実行だけを調査した。計測・実装変更・再実行は行っていない。対象グラフは16,384頂点・49,122辺。制限はプロセス群RSS 8 GiB（8,589,934,592 bytes）。

| 実行 | 結果 | 全体秒 | peak RSS (GiB) | traceの最後 |
|---|---|---:|---:|---|
| Full CPU warmup | success | 42.862 | 7.882 | 15 |
| Full Atomic warmup | memory_limit | 4.296 | 8.321 | traceなし |
| Full CPU seed 0 | memory_limit | 3.418 | 8.230 | 0 |
| Full CPU seed 1 | memory_limit | 3.401 | 8.342 | traceなし |
| Full CPU seed 2 | memory_limit | 3.413 | 8.136 | traceなし |
| Full CPU seed 3 | memory_limit | 3.427 | 8.050 | traceなし |
| Full CPU seed 4 | memory_limit | 3.633 | 8.195 | 0 |

#### 停止した段階

- Full CPUのtrace生成は、前処理完了後、execute_sgd内で初期座標と全ペア配列をcloneした後に始まる。seed 0・4は反復0だけ記録され、1反復目の完了記録がない。初回反復の開始前後で停止しており、shuffle中か実座標更新中かは既存の記録だけでは分からない。更新が0件だったとも断定できない。
- trace自体がないFull CPU実行は、少なくとも反復ループの計測開始前で停止した。前処理中と全ペアclone中のどちらかは、詳細な段階ログがないため特定できない。
- Full Atomic warmupはtraceがない。trace作成はCPU前処理・GPU用配列準備・Metal buffer作成の後、最初のdispatchより前である。したがって、この停止時点でSGD kernelの反復は始まっていない。CPU前処理/GPU用配列/buffer準備のどの確保で制限を越えたかは未特定。
- 成功したFull CPU warmupは反復0〜15が揃い、SGD時間38.343秒、全体42.862秒だった。同じ設定がメモリ境界の近くにあり、warmupの成功だけでは本測定の成功を保証しなかった。

#### ソースから確認できるメモリ負荷

- Fullは全点対を保持する。16,384頂点ではペア数134,209,536。CPUの1ペアはusize×2とf64×2で32 bytesとなり、配列本体だけで約4 GiBになる。
- CPU前処理はn×nのusize距離行列（約2 GiB）とペア配列を作る。execute_sgdは所有する元ペア配列に加えて、そのcloneを作るため、ソース上で約4 GiBずつの配列2本が存在する。容量の余裕やその他のメモリも加わる。
- Full Atomicも全ペアをGPU用へ変換する。GPU用ペア配列の要素は16 bytes（約2 GiB）。さらに更新順ID約0.5 GiBと更新前4座標約2 GiBのhost配列を準備し、それぞれのMetal bufferを作る。大きな配列・bufferの準備だけで強いメモリ負荷がある。
- これらはソースから求めた配列サイズであり、そのままRSSの合計ではない。Vecの容量、変換時のallocation再利用、解放、ページの常駐状態と監視頻度で実測RSSは変わる。今回の記録だけでは、その内訳までは確定できない。

#### 解釈

今回確定しているのは、runnerが観測RSSによって明示的に停止したこと。OSのメモリ枯渇や、Atomicのlock待ちによる停止が確認された事例ではない。Full CPUは第1反復完了前、Full Atomicはdispatch前であり、この失敗に最終stressやSGD完了時間を割り当てない。成功したFull RRと比較する際も、対応するCPU seedが欠測なので、そのseedの速度比は計算できない。

参照：
- [Full CPU前処理](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/implementations/full/baseline-sgd-non-gpu/src/graph.rs)、[全ペアcloneとtrace開始](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/implementations/full/baseline-sgd-non-gpu/src/algorithm.rs)
- [Full Atomic配列・buffer・trace・dispatchの順序](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/implementations/full/vram-lock-native/src/metal.rs)
- [制限による停止のrunner](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/experiments/run_v3.py)

調査に使った確定record（公開していないローカル記録。中断計測の移動後の保存先）：
- warmup-delaunay_n14-sgd-pNone-tNone（ローカル保存先：`/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/prebuffer-aborted/warmups/warmup-delaunay_n14-sgd-pNone-tNone/record.json`）
- warmup-delaunay_n14-atomic_sgd-pNone-tNone（ローカル保存先：`/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/prebuffer-aborted/warmups/warmup-delaunay_n14-atomic_sgd-pNone-tNone/record.json`）
- v3-basic-delaunay_n14-sgd-s0-pNone-tNone（ローカル保存先：`/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/prebuffer-aborted/raw/artifacts/v3-basic-delaunay_n14-sgd-s0-pNone-tNone/record.json`）
- v3-basic-delaunay_n14-sgd-s1-pNone-tNone（ローカル保存先：`/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/prebuffer-aborted/raw/artifacts/v3-basic-delaunay_n14-sgd-s1-pNone-tNone/record.json`）
- v3-basic-delaunay_n14-sgd-s2-pNone-tNone（ローカル保存先：`/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/prebuffer-aborted/raw/artifacts/v3-basic-delaunay_n14-sgd-s2-pNone-tNone/record.json`）
- v3-basic-delaunay_n14-sgd-s3-pNone-tNone（ローカル保存先：`/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/prebuffer-aborted/raw/artifacts/v3-basic-delaunay_n14-sgd-s3-pNone-tNone/record.json`）
- v3-basic-delaunay_n14-sgd-s4-pNone-tNone（ローカル保存先：`/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/prebuffer-aborted/raw/artifacts/v3-basic-delaunay_n14-sgd-s4-pNone-tNone/record.json`）
