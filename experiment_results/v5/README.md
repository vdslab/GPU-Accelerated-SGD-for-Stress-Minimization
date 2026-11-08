# v5 実験結果

**2026-11-06完了：6,030実行がすべて成功。失敗・欠測0。**

| 実験 | 対象・条件 | 成功/予定 | 結果・考察 |
|---|---|---:|---|
| タイルサイズ | 10グラフ、CPU基準＋RR Fullの11T、seed10〜14×各3回 | 1,800/1,800 | [タイル](reports/tile.md) |
| 速度・最終Stress共用 | 共通38件×6方式＋Sparse専用18件×3方式、seed0〜4×各3回 | 4,230/4,230 | [速度](reports/speed.md)・[Stress](reports/stress.md) |
| 超大規模 | 元8件＋移管6件、後日RR Sparseのみ | 今回未実行 | Obsidianの大規模実験計画を参照 |

通常比較は56グラフ・282設定、各設定5seed×3回が揃った。Sparse3方式比較はユーザー指定でcoPapersCiteseerまでとした。速度と最終Stressは同じ実行・最終座標から取得し、追加レイアウトは0。途中収束測定は行わない。

## 最初に見るファイル

- [速度の結果と考察](reports/speed.md)：SGD時間・総計算時間・対応CPU速度比・グラフ別の表。
- [最終Stressの結果と考察](reports/stress.md)：同じrunの絶対StressとCPU比、再現性、制約。
- [タイルサイズの結果と考察](reports/tile.md)：全Tの速度・品質比較、選定履歴。
- [最終照合結果](validation/final-completion.json)：予定runとの一致とObsidian保存確認。
- [通常比較の条件照合](validation/comparison-completion.json)：seed・ハッシュ・評価・圧縮保存の確認。

## フォルダ

| 保存先 | 内容 |
|---|---|
| config/ | 入力パス・ハッシュ、最終予定6,030run、実行順、T選択、環境・バイナリSHA、移管理由、変更前の予定 |
| raw/tile/ | タイル全1,800runのJSON、コマンド、stdout/stderr、圧縮座標、圧縮検証 |
| raw/comparison/ | 通常比較全4,230runの同様の記録 |
| warmups/ | seed999のウォームアップ。集計対象外 |
| tables/ | run別CSV、対応比、seed内3回→5seedの階層集計、規模別・再現性の表 |
| figures/ | 全図のPNG・PDF。Full/Sparse、時間/速度比/Stress/再現性でファイル名を統一 |
| reports/ | タイル・速度・Stressの結果と考察 |
| logs/ | ビルド、テスト、実行、報告作成、容量回復の履歴 |
| source-snapshot/ | 実行・報告コード、計測器、実装dirty差分 |
| validation/ | 完了照合、機能確認、変更前の検証結果。検証runは本測定に混ぜない |

## 集計条件

- 全実験15反復、epsilon0.1。RR FullはT1024（ユーザー指定）、W1024、shared、両shuffle有効、phase_chunks1。RR Sparseもphase_chunks1。
- 通常Sparse p=min(200,n)。Stressはn≤8000で厳密、それ以外は評価seed0・256始点の推定Full Stress。
- 同じgraph・seed・repetitionのCPU値から比を計算し、seed内3回の中央値→5seedの中央値・最小〜最大を取る。
- 主なSGD時間はiteration_time_ms、入力読込〜CPU上の最終座標完成の総計算時間はtotal_compute_time_ms。Stress評価・ファイル保存は計時外。
- GPU pass時間の補助計測は全GPU方式で無効（SGD_GPU_TIMESTAMPS=0）。実経過時間は取得している。
- 小規模を含む全体と大きいグラフ群では速度傾向が異なる。SGDの高速化と処理全体の高速化を混同しない。同seed内のAtomicのStress変動もreports/stress.mdに記載。

## 大規模側へ移管したグラフ

AS365・great-britain_osm・asia_osmはCPU/Atomic前処理の必要メモリ下限を理由に移管。auto・NACA0015・Flan_1565はユーザーの実験範囲指定で移管した。後者3件のメモリ不足を実測したという意味ではない。

元の大規模8件と合わせ14件を、後日メモリ使用を抑えたターミナル手動実行でRR Sparseのみ評価する予定。p=1000→500→200→100→50、seed0〜4を各1回、最大350実行。今回の成功・失敗・欠測には数えない。詳細はconfig/deferred-to-large.jsonとObsidianの大規模実験計画。

## 再生成

リポジトリ直下で `python3 experiments/report_v5_comparison.py`。完了済み生データから通常比較の集計・図表・考察を再生成する。タイルの原集計は `python3 experiments/report_v5_tile.py`。通常比較のTと移管履歴はconfig/plan.json・deferred-to-large.jsonを正とする。

Obsidianの実験概要・タイル・速度・Stress・各考察にも結果と図を保存済み。


## 超大規模の追加実行

2026-11-06の追加指示により、RR Sparseのみ14グラフ・5pivot・5seed（各1回）を開始する。通常実験6,030件の完了記録は維持する。計画・機材制約・進捗・全結果は [large/README.md](large/README.md) に分離して保存する。
