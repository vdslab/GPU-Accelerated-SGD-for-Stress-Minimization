# esa最終更新に向けた進捗・照合メモ（解析完了・esa公開待ち）

正式v3の**core主計画3,210条件とSNAP補完90条件は終了**し、[最終有効cohort監査](../tables/audit-v3.json)は有効3,210条件・整合性エラー0を確認した。元の未起動90枠だけを補完し、原記録3,300行を保持している。取得・native実験、offline解析、1〜7の18図生成・目視QA、各表の監査、最終考察・事故件数の照合は完了した。残るのはesa公開工程（公開用成果物のGitHub保存、実測画像添付、本文反映、公開後確認を含む）。core単独値と最終有効cohortの値を区別する。以下は掲載までの確認事項。

## 現在の進捗

- [x] core主計画3,210条件を確定。履歴としてのcore単独statusは success 2,381、memory_limit 46、timeout 3、warmup_failed 675、unavailable 105。最終有効cohortは success 2,426、memory_limit 46、timeout 3、warmup_failed 720、unavailable 15。success条件数と実起動数は同一ではない。
- [x] 最終有効cohort監査はエラー0。正式実起動2,370件、再利用120条件、未起動720条件を分離した。正式実失敗49件、別枠の実warmup468件（失敗126件）、ホスト再起動による追加中断attempt1件を記録した。中断attemptを新しい計画条件や確定失敗に足さない。
- [x] SNAP公式原配布6件を取得・内容検証・LCC準備した。先にコミットした `c2f1dc7` の補完スクリプトで、同binary・同limitsの独立baseのbasic90条件を確定。Gowalla / DBLP / Amazonの45正式条件が成功し、YouTube / LiveJournal / Pokecは9実warmupのRSS停止で45正式条件が未起動となった。
- [x] loader adopted / errors=0を確認し、元の未起動unavailable90枠だけを置換。有効条件は3,210、保存する原記録は3,210+90=3,300行。Sparse basicのmetadata再集計と[考察メモ](sparse-basic-observations.md)・[esa考察案](sparse-esa-discussion.md)を更新した。補完後の有効cohortを1〜7の最終解析へ反映済み。
- [x] 頂点別の予定更新・lock対象負荷診断5件successの数値・図・考察を確認し、本文用のメモへ反映した。実測競合・retry・待ち時間の診断ではない。
- [x] 64/256/1024始点のstress感度計算3件successについて、差率・native整合flagと考察を確認した。調査中のNumPy dot警告と修正後再照合は[感度解析の事故記録](sampling-dot-warning-investigation.md)に残した。
- [x] No.5の成功657run・5,256 checkpointのoffline評価を完了。最終[stress照合](../tables/convergence-stress-audit.csv)は657件すべて定義一致・0.1%以内で、最大相対差は0.000003178071675456068%。CSV再読込のseed型不一致による誤flagも修正後に解消した。[No.5考察](convergence-esa-discussion.md)
- [x] 最終再描画はsource `bbddcfe` で正常終了。1〜7の18図について目視QAを完了し、2,415件の対応比較・目標到達・coverage・構造/P/Tの各表と最終考察を照合した。欠測枠、失敗段階、補助図と基本測定の区別を残した。
- [x] 時間監査は[基本測定1,154件](../tables/iteration-time-validation.csv)と[No.5 trace657件](convergence-trace-time-validation.csv)を別scopeで確認し、どちらも全件pass。No.5の最終累積SGD時間とrecordとの差は最大0 ms。最終資源監査・再起動・timeout等の事故記録も更新済み。
- [ ] esa公開工程：公開用の設定・表・18図・報告をGitHubへ保存して参照先を確定し、esaへ画像添付・1〜7の結果/考察/事故概要を反映する。公開後に保存内容・画像表示・根拠リンクを確認する。

**最終有効cohortはデータ99件ready・98件適格（小20＋大78）で確定した。** power197k1件は未取得、G_n_pin_pout1件はサイズ除外。core確定時の93件ready・92件適格（小20＋大72）と、未取得4＋同一性未確認3の状態は取得当初の履歴として保持する。補完6件は公式SNAP原配布から再構成したもので、SuiteSparse配布物との辺・行番号の直接照合済みとは扱わない。

## 実験番号と図の対応

| 計画番号 | 主な図 | 掲載時に残す説明 |
|---|---|---|
| 1 描画 | `01_layout_full`、`01_layout_sparse`、`01_layout_sparse_partial` | 各4グラフ×3手法。CPU stress中央値に近い同seed、共通縮尺・表示辺、辺の表示割合。CPU成功なしの場合だけ事後補足の固定seed0を使い、成功GPU seedの探索はしない。欠測した図枠も残す |
| 2 Full | `02_full`、`02_full_iteration_times` | 小中規模20候補。対応seed比の中央値・最小最大、成功対応seed数。CPU不成功時は速度比を作らない |
| 3 Sparse | `03_sparse`、`03_sparse_components` | 大規模80候補の位置を残し、最終有効cohortの適格78件と区別（core単独では72件だった）。`03_sparse_small_aux`は小規模20件の補助比較と明記 |
| 4 構造 | `04_structure`、`04_structure_quality`、`04_structure_bins`、`04_size_series`、診断成功時だけ`04_sparse_write_load` | V・E・平均次数・最大次数・CV、速度と品質の両方。同系列の点を独立な証拠と数えない。静的予定回数を実測競合と呼ばない |
| 5 収束 | `05_convergence`、`05_iteration_times`、目標到達表 | 代表18グラフ×10 seedの計画。CPU15反復stressの1.05/1.01倍。初めて観測した到達checkpointと前観測の時刻、到達数/評価可能数と欠測数。反復時間図と収束条件の測定を混同しない |
| 6 T | `06_tiles`、`06_tiles_pareto` | 代表6グラフ、T=1〜1024の2冪、W=1024。T=1のみinner shuffle無効。T=1024はbasic再利用 |
| 7 P | `07_pivots` | 指定6グラフ、P=50/100/200/400。cold総時間・最終stress・RSS。P=200はbasic再利用 |

1・4はbasic結果の再利用であり、独立した再測定回数に加えない。基本1,800・収束720・T330・P360の計3,210計画条件と、実起動数・再利用120条件・warmup・未起動を分けて報告する。SNAP追加90件は新しい計画条件として加算せず、元の欠測90枠の補完として扱う。最終件数は補完後の有効実記録を使う。

## 自動生成esa本文へ補足する情報

`summarize_v3.py`の`report()`は図・対応比・到達表・失敗数を生成するが、次の補足は別文書から明示的に取り込む。

- **データ採否：** 最終は99件ready・98件適格、power197k未取得1件・G_nサイズ除外1件。core確定時の93件ready・7欠測・1規模除外は取得時点の履歴として残す。SNAP公式6件を取得・変換・LCC検証し、追加90条件の成功45・実warmupでの遮断45を確認して有効cohortへ反映した。Network Repositoryの代替3件が辺同一性検証済みに変わったという説明はしない。[データ監査](data-availability.md)・[公式原配布の再確認](data-availability-recheck.md)
- **推定stressの感度：** `sampling-sensitivity.json/csv`の実測値・採用3グラフ・欠測状況を掲載する。64/256/1024起点比較と数値照合は完了済み。256対1024の最大差率を添え、3グラフ以外の誤差保証とはしない。
- **実装の比較範囲：** Full Atomicは明示shuffleなし・無制限lock再試行・更新順IDと更新前座標の常時記録あり。Sparse Atomicは毎反復shuffleと未完了ペア再投入を使う。両者を一括して「上限なしAtomic」としない。Full Atomicの監査buffer確保・書込も測定負荷に含まれる。[実装差](implementation-scope.md)
- **精度と順序：** CPU f64、GPU f32、順序・乱数消費・微小距離処理・コンパイル経路が異なる。品質差をlockや更新順の単一要因だけへ帰属しない。
- **P比較の交絡：** Pが変わると初期座標生成前の乱数消費と制約・学習率も変わる。「同じ初期配置でPだけ変更した対照実験」と書かない。
- **計画との差：** Sparseの頂点別予定負荷診断は計算済みだが、実測書込・競合集中度は計測していない。静的予定回数の補足であり、制約数/頂点数・round・dispatchと同様、実競合・待ち時間の直接測定の代わりにはならない。30反復追加は実施していないため、結果があるように書かない。
- **事故の時系列：** 60秒pilotのtimeoutと、正式120秒wall上限を区別する。停止監視EPERM事故・テキスト保存負荷・buffer修正前662条件の中断・正式計測中のホスト再起動を簡潔に説明し、別試行や追加中断attemptを最終条件の成否件数へ混ぜない。補完取得・追加計測中に事故が起きた場合も別の来歴として追記する。[pilot](pilot.md)・[保存事故](io-export-incident.md)・[資源事故](resource-incidents.md)・[再起動事故](reboot-incident.md)・[遅延時の補足](web-notredame-timeout-system-snapshot.md)

## 指標と欠測の読み方

- 速度比は同グラフ・seed・family・PのCPU時間÷GPU時間。stress比はGPU÷同CPU。1より大きい速度比は速く、1より大きいstress比は悪化。
- SGD時間は15反復のhost処理区間合計。shuffle・dispatch・GPU完了待ちを含み、snapshot読出し/保存・stress評価・入力読込を含まない。kernel単独時間と呼ばない。
- P比較のcold総時間は前処理・初期化・SGD・readback等。process wallは入力読込・stress評価・保存も含む。gzip保管時間は別計測。
- exact stressは8,000頂点以下、超過は共通256起点・seed0の推定値。Sparse内部の近似目的と比較用full stressを区別する。絶対stress値を異なるグラフ間の品質尺度として比較しない。
- 最大RSSはプロセスの観測値で、専用VRAM量・実装の理論必要メモリ・GPU物理使用量と同一ではない。120秒timeoutは全体wall上限による停止で、SGDだけの完了時間ではない。
- `warmup_failed`は未起動条件。同じwarmupで5 seedが保留されても実事故5回と数えない。timeout・memory_limit・unavailable・CPU基準欠如・目標未達を区別する。
- 対応比の成功例だけを見ると、重いグラフや不安定条件の失敗を落とす。80/20候補の母集団と欠測件数を図・本文の両方で残す。
- 収束図の手法別中央値は単一実行の軌跡ではない。checkpoint間でstressが非単調なら初到達の厳密時刻は不明。1%/5%は本研究の基準で、一般標準ではない。

## esaへ反映した後の確認

- 原文の構成例画像を実測結果と誤認しないよう、実験概要の後に「実験結果（v3）」を置く。
- 1〜7の実測PNGをesaへ添付し、ローカルファイルパスの画像参照を残さない。各番号に少なくとも考察または欠測理由を添える。
- 番号3・4など複数の補助図が必要な節は、主図1枚だけの掲載で計画上の情報を落とさない。
- 根拠リンクは他の閲覧者が開けるGitHub/esa添付等へ置換する。ローカル絶対パスのままでは共有できない。
- 「計測中」「現在保存されている記録」など生成本文の途中経過表現を、全計画処理が終了したことと成否を示す表現へ整える。
- 既存本文・構成例・画像を不用意に削除せず、追加部分に最終結果、指標定義、事故概要を含める。
