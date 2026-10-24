# v3 SNAP 原配布からの補完準備

SNAP公式原配布から再構成（SuiteSparse配布物との辺・行番号直接照合未実施）

正式一巡の欠測90条件を独立baseで補完する。元の3210条件・欠測statusは変更しない。

| Graph | Prepared N | Prepared E | Validation |
|---|---:|---:|---|
| com-Amazon | 334863 | 925872 | `data/v3_snap_origin/graphs/com-Amazon/20261024T132257219978Z/validation.json` |
| com-DBLP | 317080 | 1049866 | `data/v3_snap_origin/graphs/com-DBLP/20261024T132305047946Z/validation.json` |
| com-Youtube | 1134890 | 2987624 | `data/v3_snap_origin/graphs/com-Youtube/20261024T132314193702Z/validation.json` |
| com-LiveJournal | 3997962 | 34681189 | `data/v3_snap_origin/graphs/com-LiveJournal/20261024T132407609759Z/validation.json` |
| soc-Pokec | 1632803 | 22301964 | `data/v3_snap_origin/graphs/soc-Pokec/20261024T132551156123Z/validation.json` |
| loc-Gowalla | 196591 | 950327 | `data/v3_snap_origin/graphs/loc-Gowalla/20261024T132631312241Z/validation.json` |

取得済みgzipをSHA確認し、EOFまで読んでCRCを検証した。元ID・有向unique・無向unique・selfloop・重複を別計数し、公式N/E/nnzと照合した。
昇順の元IDから行番号を決定し、既存prepare_v3の同じLCC処理を再利用した。SuiteSparse配布物と行番号・辺・同seedの初期配置割当が同一とは主張しない。
raw gzip、全体のmatrix→元ID対応、LCC→matrix行→元ID対応と各SHAを保持する。失敗したpartial原本・中間ファイルは自動削除しない。

次の実行：`python3 experiments/run_v3.py --results-dir experiment_results/v3/snap-origin-recovery`。同じ実行ファイル・native制限のSparse3手法×5seed×6graph=90条件、各設定のwarmupを別途実行する。
