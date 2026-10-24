# v3 計測中の正規データ配布状況の再確認

> 最終更新：下記16:46の記録は到達性を再確認した時点の履歴。末尾に、公式6ファイルの実取得・検証・補完90条件まで完了した確定結果を追記した。

確認日時：**2026-10-24 16:46:01 JST**。正式 v3 の計測中に、既存の [取得監査](../../../experiments/datasets/v3-acquisition.json) に記録済みの公式 URL だけを、軽量な HEAD リクエストで再確認した。データ本体のダウンロード、前処理、実験対象・config・制限・runner・native の変更は行っていない。

**SuiteSparse の配布停止は継続している。一方、SNAP 原配布の既知 6 ファイルは、以前の接続拒否から HTTP 200 に変わった。** これは配布先が今回の確認環境から応答するようになったという証拠であり、ファイル取得・内容検証・前処理の完了を意味しない。

## SuiteSparse 公式

[公式トップページ](https://sparse.tamu.edu/) には、10 月 16 日付の配布停止告知が引き続き表示された。行列ファイルは利用できず 403 を返し、既知のミラーがなく、復旧時期は未定という説明に変更はなかった。

| 正規 Matrix Market ファイル URL | HEAD 応答 |
|---|---|
| https://sparse.tamu.edu/MM/SNAP/com-Youtube.tar.gz | 403 Forbidden |
| https://sparse.tamu.edu/MM/SNAP/com-LiveJournal.tar.gz | 403 Forbidden |
| https://sparse.tamu.edu/MM/SNAP/soc-Pokec.tar.gz | 403 Forbidden |
| https://sparse.tamu.edu/MM/PowerSystem/power197k.tar.gz | 403 Forbidden |
| https://sparse.tamu.edu/MM/SNAP/com-Amazon.tar.gz | 403 Forbidden |
| https://sparse.tamu.edu/MM/SNAP/com-DBLP.tar.gz | 403 Forbidden |
| https://sparse.tamu.edu/MM/SNAP/loc-Gowalla.tar.gz | 403 Forbidden |

## SNAP 原配布

以下の 6 URL は、取得監査時には接続拒否だったが、今回はいずれもリダイレクトなしで HTTP 200 を返した。Content-Length は gzip 本体のヘッダー値であり、実ファイルを受信して確認したサイズではない。

| 原配布ファイル URL | HEAD 応答 | Content-Length（bytes） |
|---|---:|---:|
| https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz | 200 | 10,590,328 |
| https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz | 200 | 124,262,769 |
| https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz | 200 | 132,454,730 |
| https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz | 200 | 4,501,435 |
| https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz | 200 | 4,138,463 |
| https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz | 200 | 6,351,523 |

## 現在の扱い

- 未取得 4 件のうち YouTube / LiveJournal / Pokec は、公式原ファイルへの到達性が確認できた。power197k の正規配布先は引き続き 403 で、取得可能な正規ファイルは今回確認できていない。
- 同一性未確認の Network Repository 版が手元にある Amazon / DBLP / Gowalla についても、公式原ファイルへの到達性が確認できた。ただし、まだファイルを取得・変換していないため、手元の別配布版との辺単位の一致や頂点対応は検証できていない。
- この再確認ではデータ本体を取得せず、正式計測開始時に固定した cohort と全 3,210 条件の状態は変更していない。取得開始時点の欠測理由と今回の公開状況を区別して記録する。
- 追加取得する場合は、取得日時・公式原ファイル SHA・変換手順・頂点対応・前処理後の構造を別途検証する必要がある。HTTP 200 だけを根拠に既存 v3 の欠測を成功へ変更しない。

今回の確認は既知の 13 ダウンロード URL への HEAD 各 1 回と、公式停止告知の閲覧に限定した。旧 UFL や不明な候補ホスト、第三者配布サイトの再探索は行っていない。

## power197k の追加探索

2026-10-24 18:33 JST に、未取得の power197k について追加で公開配布先を検索した。[Simula / SparCity の行列一覧](https://datasets.simula.no/sparcity/sparse_matrices.html) には、SuiteSparse / PowerSystem / power197k の原順序 MTX と RCM 並べ替え版へのリンクが掲載されていた。ただし、一覧が指す [原順序 MTX](https://datasets.simula.no/downloads/SparCity/datasets/SuiteSparse/PowerSystem/power197k/power197k.mtx) に対する HEAD は **404 Not Found** だった。並べ替え版を原データの代わりとして採用していない。

一覧ページの存在や検索結果だけを、ファイル取得や同一性検証が完了した証拠とは扱わない。この探索でも利用可能な power197k 本体は得られておらず、未取得の扱いを維持する。計測中にデータ本体の取得・前処理は行っていない。

20:19:59 JST の再確認でも、SuiteSparse の power197k URL は公式ファイルホスト `sparse-files.engr.tamu.edu` の同じパスへ301転送された後、Amazon S3の `403 AccessDenied` を返した。配布元の公開状態が変わった場合を考慮して再確認したが、本体の取得には至っていない。

転送先の同じパスをHTTPSで確認したHEADも15秒で接続timeoutとなった。証明書検証の無効化や別ファイルによる代用はしていない。

## 実取得と補完実験の完了（最終更新）

更新：2026-10-24T23:13:14+09:00。

到達性の復旧を確認した公式 SNAP gzip 6 件を、core の全 3,210 条件の計測終了後に実際に取得し、構造検証と LCC 準備に成功した。原本 gzip・SHA-256・CRC 検証結果、昇順元 ID→matrix 行→LCC 行の対応と各 SHA を保持した。[取得・変換 manifest](../snap-origin-recovery/config/recovery.json)、[補完データ一覧](../snap-origin-recovery/config/datasets.json)、[準備結果](../snap-origin-recovery/preparation-report.md)を参照。

出典名は **「SNAP公式原配布から再構成（SuiteSparse配布物との辺・行番号直接照合未実施）」**。元の Network Repository 版は使用していない。SuiteSparse が配布する MTX との byte/辺/行番号の直接比較は行えておらず、同 seed の初期配置が同じ元頂点に割り当たるとは主張しない。取得 gzip の SHA は原本の固定・再開時の照合用であり、SuiteSparse 版との同一性を証明する値ではない。

| Graph | 準備済み N | 準備済み無向 E | 本測定15条件の結果 |
|---|---:|---:|---|
| com-Amazon | 334,863 | 925,872 | 15 success |
| com-DBLP | 317,080 | 1,049,866 | 15 success |
| com-Youtube | 1,134,890 | 2,987,624 | 3 warmup が memory_limit、15条件は未起動 |
| com-LiveJournal | 3,997,962 | 34,681,189 | 3 warmup が memory_limit、15条件は未起動 |
| soc-Pokec | 1,632,803 | 22,301,964 | 3 warmup が memory_limit、15条件は未起動 |
| loc-Gowalla | 196,591 | 950,327 | 15 success |

全 6 件で gzip CRC と公開 N/E/nnz に対する実測構造の照合が成功し、LCC の頂点数は元の全 N と一致した。Pokec は有向 unique 30,622,564 辺を照合してから無向化し、22,301,964 unique 辺になった。Gowalla は raw 1,900,654 行・ordered unique 1,900,654 対を、unordered unique 950,327 辺へ統合した。逆向きの行を入力破損や同一有向 pair の重複とは誤認していない。Pokec/Gowalla はファイルに N/E ヘッダ値がなく、その欠如を明示して公式公開統計と観測 unique 数を照合した。

データ準備の最終状態は **99 件 ready・98 件 eligible（小規模20、大規模78）**。G_n_pin_pout は ready だが LCC 99,995 頂点のため large 条件未達、**power197k 1 件は未取得**のまま。データが ready でも、YouTube/Pokec/LiveJournal の P=200 本測定は資源制限のため未起動であり、描画や最終 stress が得られた扱いにはしない。

補完は同じ native/shared source・runner・binary SHA と測定制限の独立 base で 90 条件を確定した。結果は success 45、warmup_failed 45。実 warmup の RSS 停止は 9 起動で、45 起動の本測定失敗ではない。[補完監査](../tables/audit-v3-recovery.json)、[事故・停止段階](resource-incidents.md)を参照。

最終解析では **core 3,210 + 補完90 = 3,300 原記録を保存し、有効3,210条件として一度だけ集計**する。元の unavailable 90 条件は [superseded 履歴](../tables/cohort-superseded.csv)に残し、source が復旧する前の取得状況や失敗記録を書き換えていない。[採用検証](../tables/cohort.json)、[最終監査](../tables/audit-v3.json)。
