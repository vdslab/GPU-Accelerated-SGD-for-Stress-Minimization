# v3 の SNAP 原配布からの補完

`recover_v3_snap.py` は、正式一巡で未取得だった6件を公式 SNAP 原配布から準備する。正式一巡の3,210条件が確定し、runnerが終了するまで実行できない。計測中のdownload・前処理は行わない。

使用する出典名は **「SNAP公式原配布から再構成（SuiteSparse配布物との辺・行番号直接照合未実施）」**。SuiteSparseのファイルが復旧した、同じ行番号である、同seedの初期配置が同じ頂点に割り当たる、といった主張はしない。Network Repositoryの別配布版は使用しない。

## 実行順序

1. この準備スクリプトをコミットする。正式一巡のrunner・nativeソース・実行ファイルは変更しない。
2. 正式一巡が完了しrunner終了後、`python3 experiments/recover_v3_snap.py` を実行する。
3. `experiment_results/v3/snap-origin-recovery/preparation-report.md` と `config/recovery.json`、各graphの`validation.json`を確認する。
4. 6件すべてが検証成功したら、`python3 experiments/run_v3.py --results-dir experiment_results/v3/snap-origin-recovery` を実行する。Sparse3手法×5seed×6件の90条件と各設定のwarmupを独立baseに記録する。nativeの120秒/8GiB等の制限は正式一巡の`limits.json`をbyte単位でコピーする。
5. 元の3,210条件には欠測の履歴を残し、補完90条件の来歴を検証してから集計時に明示的に対応付ける。

## 検証とメモリ

- gzipは公式HTTPSから1回取得し、SHA-256と受信byte数・Content-Lengthを保存する。gzipをEOFまで展開してCRCを検証する。途中ファイルは`.partial-時刻`のまま保持する。
- gzipの行を順に読み、64bit packed ID pairsをディスクへ書く。巨大なPython `set`や全辺listを作らず、memmapのsortと固定サイズchunkで有向unique・無向uniqueを分けて計数する。
- 元IDの最小/最大、unique頂点数、0始まりか、ID範囲内のgap、raw物理行、selfloop、有向重複、reciprocal pairを別々に残す。公式N/E/nnzを厳密照合する。Pokecは無向化前の有向30,622,564辺、他5件は無向unique Eとexpanded nnzを照合する。
- Gowallaなどの逆向き行を、同じ有向pairの重複と混同しない。ヘッダのEはraw行・有向unique・無向unique・公式公開Eのどれと一致したか記録し、どれにも合わなければ停止する。ヘッダが存在しなければ欠如を明示し、公式公開統計と実測unique数を照合する。
- 頂点数や元ID範囲が合わない場合、孤立頂点やID対応を黙って補正せず停止する。全6件は公式資料の最大弱連結成分が全Nであることも確認し、LCC結果が一致しなければ停止する。
- 昇順unique元IDをcompact matrix行へ対応させ、辞書順で各無向辺を一度だけ記したpattern Matrix Marketを生成する。`matrix-node-ids.npy`を保存し、同一の既存`prepare_v3.py`を別workerで再利用してLCCを作る。全方向の重複を入力するより一時SciPy配列を抑える。
- `vertex-map.npy`はLCC行→変換MTXの0-based行、`lcc-origin-node-ids.npy`はLCC行→元SNAP IDを示す。元gzip・変換MTX・両mapping・prepared MTX・検証JSONのSHAを保存する。
- 準備workerは既定8GiB/1,800秒、downloadはsocket60秒/全体900秒、空きdiskは最低8GiBを確保する。これらはデータ準備の制限であり、native測定の制限を変更しない。

## 保存先と再開

原本と中間成果物は`data/v3_snap_origin/`へ保存する。このworktreeでは`data`自体はsymlinkではないことを確認済み。実行時にもsymlinkになっていないことを確認し、変わっていれば停止する。元の`data/v3_lcc`や既存MTXを上書きしない。

成功した各graphの`config/prepared/<name>.json`をcheckpointとし、再開時にはraw gzip・変換MTX・mapping・prepared MTX・検証JSONの全SHAを再照合してから再利用する。途中の原本や失敗した中間ファイルは削除しない。SHAを持たない既存raw原本は再利用・上書きせず、検証エラーで停止する。

失敗時は`preparation-failure-時刻.json`と`preparation-failure.md`に対象と理由を保存する。6件すべてが成功するまで実行用`datasets.json`を作らない。補完runnerの`execution.json`が存在した後は再準備を拒否し、実験途中のconfig変更を防ぐ。

## 公式メタデータ

| Graph | N | 公式E | SuiteSparse nnz | 元ID範囲 |
|---|---:|---:|---:|---|
| [com-Amazon](https://sparse.tamu.edu/SNAP/com-Amazon) | 334,863 | 925,872（無向） | 1,851,744 | 1..548,551、gapあり |
| [com-DBLP](https://sparse.tamu.edu/SNAP/com-DBLP) | 317,080 | 1,049,866（無向） | 2,099,732 | 0..425,956、gapあり |
| [com-Youtube](https://sparse.tamu.edu/SNAP/com-Youtube) | 1,134,890 | 2,987,624（無向） | 5,975,248 | 1..1,157,827、gapあり |
| [com-LiveJournal](https://sparse.tamu.edu/SNAP/com-LiveJournal) | 3,997,962 | 34,681,189（無向） | 69,362,378 | 0..4,036,537、gapあり |
| [soc-Pokec](https://sparse.tamu.edu/SNAP/soc-Pokec) | 1,632,803 | 30,622,564（有向） | 30,622,564 | 1..1,632,803 |
| [loc-Gowalla](https://sparse.tamu.edu/SNAP/loc-Gowalla) | 196,591 | 950,327（無向） | 1,900,654 | 0..196,590 |

公開統計は2026-10-24に公式ページで再確認した。元のカタログにN/nnzが記録されている3件は、その固定カタログ値とも照合する。残る3件のnnzは上表の公式ページ値を固定し、既存datasetのsource Nとも照合する。
