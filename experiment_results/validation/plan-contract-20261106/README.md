# 計画と実装の接続確認（2026-11-06）

本測定ではなく機能確認。pivotや初期配置の実装変更は行っていない。

- karate（34頂点）、6方式、seed=0、各3回、15反復。Sparseは有効p=34。18実行が成功し、既存ランナーの出力検証を通過。
- 全実行で時間と最終厳密Stressを同じJSONから取得。RR Full/Sparseはphase_chunks=1。
- 推定評価経路だけを確認する一時的なpath8001入力をRR Sparse・p=200・15反復で実行。SGD_STRESS_SAMPLES=256は反映されるが、SGD_STRESS_SEED=20261106は未対応で出力stress_seed=0。計画とは不一致。
- 時間制限はrun_methodのsubprocess.run(timeout=...)でプロセス全体へ適用。模擬子プロセスがlayout_completeを出した後、評価中のsleepで0.2秒のtimeoutとなることを確認。実際の長時間Stress評価による打切り試験ではない。
- ソース照合：共通measure_auto系はDEFAULT_SEED=0を渡す。Sparse CLIは最終座標取得後、同じプロセスでStress評価を実行し、評価後に座標・JSONを保存する。そのため評価中のtimeoutでレイアウト結果も保存されない可能性がある。

判定：基本実行と出力は正常。ただし評価seedと段階別時間制限は計画未対応。本測定開始可とは判定しない。

summary.jsonに実行結果、各実行フォルダにコマンド・stdout・stderr・座標、timeout-check.jsonにランナー境界試験を保存。
