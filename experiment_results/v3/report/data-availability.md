# v3 実行前のデータ監査

> この文書は SNAP 原配布の復旧・補完前の取得状況を残した歴史的記録である。最終状態は [有効データ一覧](../tables/effective-datasets.json) と [再取得・補完の結果](data-availability-recheck.md) を参照。補完後は 99 件 ready・98 件 eligible、power197k 1 件は未取得。

2026年10月24日。選定100グラフのうち、正規配布データ93件の前処理が完了した。計画の規模条件を満たすのは **小規模20件・大規模72件、計92件**。取得・同一性確認ができない7件と、大規模条件を満たさない1件は比較対象から除外する。

| 状態 | 件数 | 扱い |
|---|---:|---|
| 正規配布・前処理完了・規模条件適合 | 92 | 小規模20件・大規模72件を実行対象とする |
| 正規配布・前処理完了・大規模条件未達 | 1 | `G_n_pin_pout`を大規模比較から除外 |
| 正規配布ファイルを取得できない | 4 | 欠測として保存 |
| 別配布版との同一性が未確認 | 3 | 正規比較では欠測として保存 |

## 正規配布ファイルを取得できない4件

- [SNAP/com-Youtube](https://sparse.tamu.edu/SNAP/com-Youtube)
- [SNAP/com-LiveJournal](https://sparse.tamu.edu/SNAP/com-LiveJournal)
- [SNAP/soc-Pokec](https://sparse.tamu.edu/SNAP/soc-Pokec)
- [PowerSystem/power197k](https://sparse.tamu.edu/PowerSystem/power197k)

公式メタデータページは参照できるが、正規ファイルの取得先はHTTP 403、旧University of Floridaの配布先はHTTP 404だった。SNAP由来3件の原配布先も接続拒否だった。`power197k`のGroup・名称は公式ページと一致しており、名称違いによる取得失敗ではない。

[SuiteSparse公式サイト](https://sparse.tamu.edu/)は、2026年10月16日から行列ファイルの配布を停止し、HTTP 403になることと、既知のミラーがないことを告知している。確認したURLと応答は[取得監査JSON](../../../experiments/datasets/v3-acquisition.json)に記録した。

## 別配布版との同一性が未確認の3件

`SNAP/com-Amazon`、`SNAP/com-DBLP`、`SNAP/loc-Gowalla`には、Network Repository由来のローカルファイルがある。公開頂点数・非零要素数は一致するが、正規配布ファイルとの辺単位の同一性・頂点対応を確認できていないため、v3の正規比較には使用しない。原配布先の確認URLは以下のとおりで、いずれも接続拒否だった。

- [SNAP com-Amazon原ファイル](https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz)
- [SNAP com-DBLP原ファイル](https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz)
- [SNAP loc-Gowalla原ファイル](https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz)

**取得できなかった正規データを、類似名の別グラフや同一性未確認のデータで代替していない。** 7件とも`datasets.json`の`status: unavailable`として理由を残した。

## 最大連結成分の規模条件

`DIMACS10/G_n_pin_pout`は正規配布データの前処理に成功したが、最大連結成分は **99,995頂点**だった。大規模群の条件「10万頂点以上」を満たさないため、`status: ready`・`eligible: false`とし、大規模比較から除外した。小規模群への移し替えや別グラフとの交換は行っていない。

前処理では明示的ゼロ・自己ループを除去し、重みと方向を取り除いて重複辺を統合し、最大連結成分を抽出した。各グラフの原ファイル・前処理後ファイル・元頂点対応のSHA-256、頂点数・辺数、採否理由は[実行用データ一覧](../config/datasets.json)に保存した。元の取得一覧は[固定した出典カタログ](../config/source-catalog.json)を参照できる。
