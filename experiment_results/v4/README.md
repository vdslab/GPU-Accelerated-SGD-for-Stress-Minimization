# v4 Sparse SGD 論文図の再実験

現行CPU / Atomic / RR Sparseの比較。Fullは再実行せず、reference/full-v3に保存した既存値を使う。

- 基本比較：図5の37グラフ＋図7のdelaunay_n19、3手法×seed 0〜4 = 570実行
- 収束：luxembourg_osm / kron_g500-logn18 / coAuthorsCiteseer、3手法×seed 0〜9 = 90実行
- 全て15反復、epsilon 0.1、pivot 200。GPU単一f32、CPU f64、精度補償なし
- RR：頂点別pivot置換と4分割の相実行。Atomic：現行既定のCPU事前shuffleなし。CPU：全制約を毎反復shuffle
- 8,000頂点以下はexact、それ以外は固定256始点・評価seed 0。推定値をexactとみなさない
- 120秒のプロセス時間・観測RSS 8 GiB・空きディスク8 GiBの監視、1設定1回warmup後に正式実行
- warmup失敗・正式失敗は保存し、欠測を速度比0/1やstress0に置き換えない
- 基本計測は途中座標保存なし。収束診断は0,1,2,3,5,8,10,15の座標を保存し、その費用は反復時間から除く
- 入力はv3の前処理済み最大連結成分をSHA照合してコピー。取得不能だったSNAP3件は回復コホートを使用

## 構成

- config/: 計画、条件、入力SHA、実装・バイナリSHA、実行環境
- raw/: 正式記録、stdout/stderr、コマンド、座標（gzipと解凍SHA）、収束checkpoint
- warmups/: 事前試行と停止記録
- tables/: 集計値、対応seed、欠測、収束値
- figures/: 論文用PDF/SVGと確認用PNG
- report/: 結果本文・esa更新用本文
- reference/full-v3/: 再利用するFullの出所。v4 Sparseと混同しない
- validation/: テスト、traceによる配置不変性、集計照合

## 実行

```sh
python3 experiments/run_v4.py
```

中断後も同じコマンドで再開。バイナリ・runner・計画の変更を検出した場合は停止する。
現時点では測定準備済み。完了状態はcheckpoint.jsonで確認する。
