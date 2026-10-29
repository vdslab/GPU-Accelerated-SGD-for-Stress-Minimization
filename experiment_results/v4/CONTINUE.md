# 実施記録：完了

2026年10月30日、RR Sparse v4の220条件を完走し、集計・図・ローカル論文・esa更新を完了した。現在の状態と再現手順はREADME.md、数値はreport/RESULTS.md、監査はvalidation/final-audit.jsonを参照。

## 計画と記録の推移

1. 当初は3手法660条件を計画した。元のconfig/plan.json・execution.json・ソースSHAは変更せず保持した。
2. モデル切替で元runnerが59件記録後に終了した。親の記録処理が失われたRR1attemptはinterruptions/model-switch-*へ隔離し、品質値で選別せず同条件を再実行した。
3. 166件（RR55、CPU56、Atomic55）を記録した時点で、ユーザー指示によりRRのみへ絞った。停止したAtomic warmupはinterruptions/scope-change-rr-onlyへ保存した。
4. experiments/run_v4_rr_only.pyはRR55件を引き継ぎ、新計画config/rr-only-plan.jsonの220件を完了した。raw/results.jsonlは計331件で、CPU・Atomic111件は補助として残した。正式比較は一貫してv3のCPU・Atomic440参照を使う。
5. 固定した同じ入力・初期配置・15反復・epsilon0.1・pivot200で照合した。通常f32、精度補償なし。CPU56補助件はv3と初期SHA・最終stressが完全一致した。
6. 最新論文側で導入・章構成・wgpu説明・参考文献の編集が並行して入っていたため、reference/paper-latest-before-v4へ保存し、v4のSparse根拠だけを統合した。変更しない節の一致をコードで検査した。
7. esa9029 revision12、esa9028 revision34として保存し、保存後の本文と新画像URLを再読照合した。

## 主なスクリプト

- run_v4_rr_only.py：完了済みRR計画の再開（既存結果は再実行しない）
- summarize_v4.py：v3参照との対応監査、集計、途中stress
- plot_v4.py / diagram_v4.py：図の再生成
- report_v4.py / paper_v4.py / merge_latest_paper_v4.py：結果本文と最新構成への統合
- install_v4.py：ローカル論文・図への反映、Full図の不変性検査
- prepare_esa_v4.py：アップロード済みURLと変更前本文からesa用Markdownを構成
- progress_v4.py：実行中に用いた過去所要時間に基づく残り時間の概算

一時heartbeat sparse-v4-esaは完了後に削除した。計測・後処理のプロセスは終了済みであり、esa更新を再実行する必要はない。
