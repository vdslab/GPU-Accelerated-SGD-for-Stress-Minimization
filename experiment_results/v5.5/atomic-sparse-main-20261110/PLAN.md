# Atomic Sparse 最新main：速度・Stress

Obsidian「2_速度比較」「11_共通条件と出典」の通常比較56グラフ。Atomic Sparseのみ新規56×5seed×3回=840件。15反復、epsilon0.1、p=min(200,n)、seed0〜4、repetition0〜2。各グラフseed999のwarmupをセッションごとに除外。旧実行順から対象方式だけを抽出し維持。初期化から毎回実行。時間による打切りなし。14大規模グラフは対象外。

実験前にorigin/mainをfetchし、mainを839346091717f3928be83bbcdba8cf8783a48e4eに更新。更新前のmainはcodex/main-before-atomic-sparse-20261110として保存。mainのworktreeでrelease/lockedビルド。バイナリSHA256、ビルドログ、ソースsnapshot、機材・メモリ情報を保存。既定GPU permutation、cpu_shuffle=false、CSR CPU BFS auto、GPU timestamps=0。Atomic Sparseの旧固定順実験は使用しない。

主計時はSGD+readback（GPU完了待ちを含む）および入力読込〜CPU最終座標完成のtotal_compute_time_ms。nativeのSGD/準備/転送/読戻し等を別々に保存。Stress評価・座標保存・可逆圧縮は主計時外。描画は測定中に実行しない。Stressは同じ最終座標でn<=8000厳密、超過は固定256始点・評価seed0。速度とStressで別のレイアウトを生成しない。

各runで入力・初期座標・前処理fingerprintと旧正式測定の一致を検証。attempted=completed=15×constraints、retry_failures=0、main commitとclean実装を確認。CPU比較は過去正式測定を対応graph/seed/repetitionで参照し、評価方式も照合。各runのCPU/GPU比を先に計算し、seed内3回中央値→5seed中央値・最小最大。グラフ間は等重み幾何平均。過去CPU/RRとの比較は同時測定ではない。

逐次実行、実行中はビルド・描画・他実験を行わない。開始前available>=2GiB/free disk>=4GiB。実行中RSS<=18GiB、available>=512MiB（10秒継続）、disk>=1GiBの既存最新main再測定と同じ資源条件を固定。共通ノートの8GiBは案であり、coPapersCiteseerまで条件を変えず実施するため18GiBを採用。機材の温度・電源・常駐アプリは統制していない。swapとメモリは記録する。資源停止・失敗は成功値へ混ぜず記録。

実行・再開：`python3 experiments/run_atomic_sparse_main.py`。完了済み成功recordは再利用するが旧測定は新規測定へ混ぜない。中断時の未完成フォルダはvalidation/interruptedへ保存し、そのrunを再実行。失敗recordがある場合は自動で上書きせず停止。

集計：`python3 experiments/report_atomic_sparse_main.py`。
論文図表：`python3 experiments/build_atomic_sparse_paper_charts.py`。
完了840件が揃うまで正式集計しない。旧チャートは保護し、paper-charts以下へ新しい図6・表1・補足全56入力図、レビューPDF、CSV、差替用PDF・TeXを生成。図7はRR対CPUのためAtomic Sparse更新では数値不変。論文本文・PDF自体は変更せず、データ差替用成果を用意する。

## ユーザー指定による確定範囲と成果物の変更

web-Stanfordまでで終了。実行52グラフ×15件=780件、全件成功。後続com-DBLP・web-NotreDame・com-Amazon・coPapersCiteseerの60件は未実行・対象外であり失敗ではない。対象の全更新・artifact圧縮が完了した境界でrunnerを停止し、後続warmupも開始していない。元の840件のjob一覧・計画はoriginal-jobs.jsonとoriginal-plan.jsonへ保存。

後続のユーザー指示により図表生成は行わず、数値データをv5.5としてcommit・pushする。上記の図表生成計画は取り消し。正本の保存先は`experiment_results/v5.5/atomic-sparse-main-20261110`。配置変更に伴う参照先とversionラベルだけを更新し、native stdout・実行時コマンド・数値・hashは保持した。詳細はvalidation/relocation.json。グラフ別所要時間にはv/eと過去v5・最新正式RR Sparseの対応所要時間も保存する。
