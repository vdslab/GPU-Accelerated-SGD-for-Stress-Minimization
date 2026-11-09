# v5 超大規模グラフ実験

> 2026-11-09整理：130成功・GPU上限による未実行220・実測失敗0、報告・QA完了。 [v5総覧](../README.md)。
> 保存状態：2026-11-08の容量整理で座標等は削除済み。以下の座標保存・復元の説明は実行当時の記録。数値結果・既存図表は保持。再描画には座標の再取得が必要。

2026-11-06のユーザー指示で、手動実行予定からCodexによる実行へ変更。通常6,030件は完了済みで保持する。

| 項目 | 条件 |
|---|---|
| 対象 | 14グラフ、RR Sparseのみ |
| pivot | 1000 → 500 → 200 → 100 → 50 |
| seed・回数 | 0〜4、各1回（計画350件） |
| SGD | 15反復、epsilon0.1、分割なし、頂点別pivot順序 |
| 最終Stress | 同じ最終座標、64始点、評価seed0 |
| 計時 | SGDとtotal_compute_time_ms、Stress・保存を除外 |
| 制限 | 時間打切りなし。RSS18GiB、空きRAM512MiBが10秒継続、ディスク空き1GiBで容量停止 |
| 開始前 | 空きディスク4GiB＋160n＋48m bytes、空きRAM2GiBを確保 |
| ウォームアップ | graph×p×起動セッションごとseed999、集計対象外 |

## 実装上の適用可能性

Apple M4 Pro/Metalでstorage buffer bindingの上限4,294,967,295 bytes、頂点dispatchの上限65,535 workgroups（256頂点/group）を実測照会した。one-sided配列は8np bytes。確定する上限超過220条件は確保前に未実行として保存し、130実行・26設定の全5seedが成功した。これはGPU一般の不可能性ではなく、現実装とこの機材の上限である。メモリ予測値だけで未実行にはしない。

| グラフ | n | m | 実測対象のp |
|---|---:|---:|---|
| auto | 448,695 | 3,314,611 | 1000, 500, 200, 100, 50 |
| NACA0015 | 1,039,183 | 3,114,818 | 500, 200, 100, 50 |
| belgium_osm | 1,441,295 | 1,549,970 | 200, 100, 50 |
| Flan_1565 | 1,564,794 | 56,300,289 | 200, 100, 50 |
| netherlands_osm | 2,216,688 | 2,441,238 | 200, 100, 50 |
| com-Orkut | 3,072,441 | 117,185,083 | 100, 50 |
| AS365 | 3,799,275 | 11,368,076 | 100, 50 |
| com-LiveJournal | 3,997,962 | 34,681,189 | 100, 50 |
| italy_osm | 6,686,493 | 7,013,978 | 50 |
| great-britain_osm | 7,733,822 | 8,156,517 | 50 |
| germany_osm | 11,548,845 | 12,369,181 | なし（GPU上限） |
| asia_osm | 11,950,757 | 12,711,603 | なし（GPU上限） |
| road_usa | 23,947,347 | 28,854,312 | なし（GPU上限） |
| europe_osm | 50,912,018 | 54,054,660 | なし（GPU上限） |

## 保存と比較

`raw/`に全350件の成功・実測失敗・GPU上限の未実行を区別して保存。`warmups/`は集計対象外。`config/`に全job・入力/バイナリハッシュ・機材上限・事前判定、`progress.json`に進捗。`tables/figures/reports`に集計・図表・考察を保存済み。

各pにつき5seedの中央値・最小〜最大と成功数/5を示す。失敗を0で埋めない。p間の初期座標はnativeの乱数消費により異なる可能性があり、同seedだけで一致とみなさない。fingerprintを保存し、pivot列を人為的に揃えない。各seed1回なので通常比較の3回反復とは精度が異なる。

Codexを残して実行する。Chrome・Discord等を通常終了したが、OSとCodex等は常駐する。OSのみの最小メモリ環境とは扱わず、開始前と実行中のメモリ・swap・RSSを記録する。


### 辺リスト保存を含む容量計算の修正（20:25以降）

com-Orkutは初期・最終の各ファイルに約1.17億辺を複製保存し、各ファイル約1.8GBとなる。従来の頂点数のみの保存容量見積もりは不十分だったため、次の実行前に4GiB＋160n＋48m bytesを予約する。保存は可逆XZ（preset0、CRC64。初回の復旧試行のみpreset3）へ変更し、復元後のSHA256・サイズを検証後に元ファイルを置き換える。レイアウト・Stress・計測バイナリは変えず、圧縮処理は計時の対象外とする。最初のcom-Orkut p100 seed0は保存中の容量停止として失敗試行をvalidationに保全し、容量確保後に同じ条件で再試行する。

保存用のcom-Orkut先頭64MiBでpreset0/1/3を検証したところ、サイズは約14MBでほぼ同等、圧縮時間はpreset0で約1.2秒、preset3で約7.9秒だった。このため以後preset0を使う（実験本体の計時とは別）。検証記録はlarge/validation/xz-preset-benchmark.json。


### 容量停止からの再開と履歴領域（21:30以降）

Codex自動差分が測定ペイロードをGitのloose objectとして複製し、.git/objectsが約28GiBに増えていた。ローカルの.git/info/excludeでv5のraw/warmups/validationのみ自動差分から一時除外した。実験ファイルは移動・削除していない。Git内部は既存objectをpackへ圧縮し、verify-packで全入力objectの格納を確認してからprune-packedで重複looseコピーだけを除く。コミット・ref・reflog・checkpoint・unreachable objectを削除するpruneやexpireは行わない。履歴保存状態と検証記録はlarge/validation/git-compaction-*、進捗はlarge/maintenance.json。

容量停止からの再開時のみ、同じバイナリ・条件で成功した30分以内のウォームアップは検証記録を残して再利用できる。それ以外はseed999を新たに実行する。SGD・Stress・seed・反復条件は変更しない。


### 容量に収まるグラフを先に処理（22:50以降）

保存容量の予約条件を満たせない場合、そのグラフの残り条件は未測定のまま保留し、予約条件を満たす後続グラフを先に処理する。各グラフのp降順・seed順・回数と全350条件のscopeは維持し、保留を失敗や確定未実行へ数えない。実際の順序変更と容量の証拠はvalidation/schedule-deferred-*.jsonに残す。容量回復後に同条件で再開でき、完了済み実行を重複しないことをfixtureによる統合確認で検証済み。


### 同じ辺データの物理共有（2026-11-07）

com-Orkutの各初期・最終ファイルは同じ巨大な辺リストを含むため、ヘッダ・辺・座標のgzip memberを連結し、APFS clonefileで同じ位置の辺memberだけをCOW共有する。各ファイル単体を普通のgzipで展開すれば、元のnativeテキスト全体を復元でき、共有テンプレートは復元時に不要。新ファイルの復元SHA256・長さ・EOF CRCとテンプレート不変を検証し、manifestとrecordを永続化してから元圧縮ファイルを置き換える。

合成2seed/2stageと2MiB境界をまたぐマーカーのテスト、実データ1件で全byte・計測値の一致を確認した。検証結果はvalidation/shared-gzip-*.json。通常6030件は変更しない。既存runのarchive_secondsは保持し、再保存時間はshared_repack_secondsへ分離。今後のOrkutもnative計測完了後に同じ共有保存を行う。共有対象は保存用の辺データであり、GPU実行時の配列・メモリ条件は従来どおり。

<!-- V5_LARGE_REPORT_START -->

## 超大規模v5の測定・保存完了

更新: 2026-11-07T13:49:24+09:00。350条件のうち成功130、GPU上限の未実行220、実測失敗0、保全した再試行前試行1。結果・19図のPNG/PDF・考察をv5/largeとObsidianへ保存。通常6030件の結果は保持。[結果と考察](reports/large.md)参照。

<!-- V5_LARGE_REPORT_END -->

## 結果ファイルの入口

| 保存先 | 内容 |
|---|---|
| [reports/large.md](reports/large.md) | 最終結果、グラフ別の考察、測定の限界、全19図 |
| [tables/runs.csv](tables/runs.csv) | 全350条件の状態と全native測定値 |
| [tables/summary.csv](tables/summary.csv) | 70設定の5seed中央値・最小最大・成功数 |
| [tables/paired-pivot-ratios.csv](tables/paired-pivot-ratios.csv) | 対応seedごとに計算したp間Stress比 |
| [tables/retried-attempts.csv](tables/retried-attempts.csv) | 再試行前の容量停止1試行（本測定とは別） |
| figures/ | PNG/PDF、完了可否・時間・Stress・RSS・10グラフのレイアウト |
| validation/ | 圧縮復元、容量整理、中断復旧、最終整合性と目視QAの証拠 |

生データと配置はこのリポジトリ内のraw/に保持する。巨大データのGit自動複製を防ぐためraw/warmups/validation/shared-edgesはローカルの.git/info/excludeで一時除外しており、Gitへの追加・push済みという意味ではない。

## 超大規模グラフ入力の容量整理（2026-11-07）

ユーザー指示により、超大規模14グラフの入力MTX（元ファイル・正規化版・既存LCCコピー）と対応するダウンロードZIPを削除。実行可能な10グラフ130条件の最終座標は取得済み、4グラフ220条件はGPU上限で未実行。初期・最終座標、頂点対応、測定結果、図表は保持。入力を使う再実行には再取得が必要。取得元・入力ハッシュは既存manifest/configに保持し、削除ファイル・SHA256・容量は `validation/input-data-cleanup-20261107.json` に記録。
