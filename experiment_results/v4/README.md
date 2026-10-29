# v4 RR Sparse SGD：論文図の再実験

RR Sparseのみ現行の通常f32実装で再測定する。CPU・Atomic Sparseは既存v3の記録を一貫して参照する。Fullもv3の値を継承する。

## 対象と条件

- 基本比較：既存図5の37グラフ＋図7のdelaunay_n19、RR × seed0〜4 =190条件
- 収束：luxembourg_osm / kron_g500-logn18 / coAuthorsCiteseer、RR × seed0〜9 =30条件
- RR計220条件。別枠のwarmupはグラフごとに1回、基本・収束間で再利用
- 全て15反復、epsilon0.1、pivot200。RRは単一f32、精度補償なし、頂点別pivot置換と4分割相実行
- CPU・Atomic参照440条件：reference/sparse-v3/。CPUはf64で全制約を毎反復shuffle、Atomicはv3のCPU事前shuffleありの測定版
- CPU・Atomicは2026年10月24〜25日、RR v4は10月30日。同一機種の別日時測定であり、同一セッションの時間比較とは区別する
- 共通f64 full stress評価。8000頂点以下はexact、それ以外は固定256始点・評価seed0。今回の38グラフはいずれもsampled
- 120秒のプロセス時間・観測RSS8GiB・空きディスク8GiBの監視。欠測を速度比0/1やstress0にしない
- 通常計測には途中保存を入れず、収束診断は0,1,2,3,5,8,10,15反復の配置を保存。コピー・保存費用は反復時間から除く
- 入力はv3と同じ前処理済み最大連結成分をSHA照合。対応比較はinput/preprocess/initial SHAと設定の一致を必須とする

## 計画変更と記録

当初は3手法660条件を再実行する計画だった。166条件の記録後、ユーザーの「CPU sparseとAtomic Sparseは同じなんだからRR Sparseだけでいい」という指示に従い、RRのみへ変更した。
変更前に完了したRR55条件を引き継ぎ、CPU・Atomic111条件は補助記録として残す。比較の基準は全グラフでv3に統一し、この111条件を混ぜない。
変更前のCPU56条件は、v3との初期SHA一致と最終stressの完全一致を確認した（validation/cpu-v3-reuse-audit.json）。

元の計画config/plan.json・execution.json・ソースSHAは保存し、新計画はconfig/rr-only-plan.json、新driverの固定SHA等はconfig/rr-only-execution.jsonに記録する。
モデル切替時の1中断attempt、範囲変更で停止したAtomic warmupはinterruptions/へ保存し、正式比較に含めない。

## 構成

- config/: 新旧計画、入力SHA、実装・バイナリSHA、環境
- raw/: 正式の原記録と座標gzip・SHA manifest（旧CPU・Atomic補助分も保持）
- warmups/: RRの事前試行と、範囲変更前の他手法事前試行
- tables/runs.csv: RR v4＋CPU・Atomic v3参照。各行のexperiment_versionに出所を記載
- tables/excluded-auxiliary-v4.csv: 比較から除いた変更前のCPU・Atomic v4
- tables/missing.csv: 対応比較の欠測と理由、paired-ratios.csv: 同seedの比と元run ID
- tables/convergence.csv: RR v4の再評価値とCPU・Atomic v3参照、arrivals.csv: 目標到達観測
- figures/: 論文用PDF/SVG・確認用PNG
- report/: 結果、論文草稿、esa本文・公開記録
- reference/full-v3/: Fullの既存値と出所
- reference/sparse-v3/: CPU・Atomicの既存値と出所
- validation/: traceによる配置不変性・集計監査

## 再開と集計

```sh
python3 experiments/run_v4_rr_only.py
python3 experiments/summarize_v4.py
python3 experiments/plot_v4.py
python3 experiments/report_v4.py
python3 experiments/paper_v4.py
python3 experiments/diagram_v4.py
```

計測は1実行ずつ行い、重い集計を並行実行しない。新driverはバイナリ・元helper・新driver・新計画のSHA変更を検出すると停止する。
旧run_v4.pyは当初計画を保存するhelperであり、RRのみの再開には使わない。
進捗はcheckpoint.jsonとrunner-rr-only.log、残り時間の概算はexperiments/progress_v4.pyで確認する。
