# 本実験用データ

SuiteSparse Matrix Collectionで検索し、各詳細ページの **Matrix Market**
リンクから取得したアーカイブを展開する。主行列の `.mtx` のみをリポジトリ直下の
`prod_data/` に保存する。行列の値や構造は取得時に変更しない。

取得元ページ、アーカイブURL、展開元のメンバー名、行列サイズ、要素数、
SHA-256は `prod-data-catalog.json` に記録している。
`prod_data/` は `.gitignore` の対象であり、MTX本体はコミットしない。

## 取得済み

計画の18グラフ中、SuiteSparseに掲載されている16グラフを取得した。

- qh882、1138_bus、dwt_1005、poli、dwt_2680、USpowerGrid、3elt
- dwt_419、cage8、EVA
- G65、fe_4elt2、bcsstk31、venkat50、luxembourg_osm、web-NotreDame

既存の実行時の既定値・実験manifest・インポート確認が使用する
bcspwr08、bcspwr10、bcsstk29、bcspwr01も取得した。合計20ファイル。

## 未取得

`btree9` と `block_2000` はSuiteSparseの全件一覧で見つからなかった。
同名データを取得できていないため、代替グラフへの置き換えや自動生成は行っていない。
計画上の18グラフを全て揃えるには、この2件の別の配布元の確認が必要。

## 実行

各Rust実装のディレクトリから実行する際の既定パスは `../prod_data/...`。
実験runnerをリポジトリ直下から使う際はmanifestの `prod_data/...` を使用する。

```sh
python3 experiments/run_experiments.py experiments/manifests/e0-uspowergrid.json --dry-run
```

既存manifestのパスと入力checksumも取得したMTXに合わせて更新した。
過去の実験台帳と集計結果は当時の記録として保持する。
