# 追加検証：FullのgridとSparseのworkgroup

更新: 2026-11-07T14:49:31+09:00

既存v5とは別の追加実験。結果を見てグラフを追加・除外せず全結果を保存する。

| 項目 | 条件 |
|---|---|
| Full | 2D grid 64×64, 96×96, 128×128, 128×192、3D grid 16³, 24³ |
| Full T / W | T=64,128,256,512,1024、W1024固定 |
| Sparse | 2D grid 64×64,128×128、3D grid24³、luxembourg_osm、la2010、coPapersCiteseer |
| Sparse W / p | W256と1024、p1000と200 |
| 実行 | 各5seed×3回、15反復、epsilon0.1、分割なし |
| 評価 | 同じ最終配置のFull Stress、n≤8000厳密、それ以上256始点seed0 |
| 計時 | 主指標は平均1反復のSGD経過時間。準備・転送・読戻し込みも別に集計 |
| 予定 | Full450＋Sparse360＝810実行、warmupは除外 |

SparseのWはFullのTとは別の設定。既存のCPU/Atomicや本測定バイナリは変更しない。実験用Sparseバイナリを別にビルドし、W256との数値一致を検証してから測定する。1024が最速という結論は先に置かない。Fullは全頂点対を扱うためgrid構造そのものの効果と頂点数の効果を混同しない。大きいFullはRAM制約に触れる可能性があり、容量停止・未実行と速度結果を区別する。

## 保存容量による中断と再開

2026-11-07 15:07、226成功・失敗0の時点で次実行前のディスク4 GiB基準を下回り安全停止。非稼働ビルドキャッシュ等を整理し、実験バイナリ8個のSHA不変と全入力・結果の保全を確認した。履歴のprune/expireは実行していない。証拠はvalidation/disk-cache-recovery-before.json・after.json。

以後coPapersの巨大な保存辺リストのみ、既存大規模実験で検証済みの標準gzip連結member＋APFS共有保存を使う。各ファイルは単独で復元可能で、復元SHA256・長さ・CRCと計測scalar不変を確認する。native測定後に逐次変換し、archive_secondsとshared_repack_secondsを計測時間から分離する。他graphはXZのまま。小さい複製で保存・復旧・元ファイル不変を確認済み（validation/storage-revision.json）。warmup後もRAM2 GiB・disk4 GiBを再確認する。数値条件・810件のscope・入力・バイナリは不変。

<!-- FINAL_COMPLETION_START -->

## 完了

2026-11-07T18:01:32+09:00：全810件成功、失敗0・未測定0、54設定完了。tablesに全run CSVと階層集計、対応W比・p比、fingerprint確認、figuresにPNG/PDF4図、reports/results.mdに詳細考察を保存した。1980個の圧縮ファイルのSHA、8バイナリ・全入力SHA、元measurement scalar不変、階層集計を照合済み。4図を目視確認し、Obsidianの実験・考察・概要・論文図表_確認用と図表/CSVの保存・リンクを検証済み。証拠: validation/final-data-audit.json、final-qa.json、report-completion.json。

<!-- FINAL_COMPLETION_END -->
