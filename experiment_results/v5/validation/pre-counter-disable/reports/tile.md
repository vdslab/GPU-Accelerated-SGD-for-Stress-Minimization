# v5 tile 結果と考察

更新: 2026-11-06T03:08:51+09:00

完了記録540、成功540。測定値は `/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/experiment_results/v5/tables/tile-runs.csv`、階層集計は `/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/experiment_results/v5/tables/tile-summary.json`。

同seed3回の中央値を取り、完全な5seedの分布を比較する。成功率と欠測を併記し、失敗を0で埋めない。

- email: SGD時間の中央値が最小のTは64、15反復48.246 ms、最終Stress 95014.8。
- add20: SGD時間の中央値が最小のTは128、15反復51.148 ms、最終Stress 318358。
- CA-GrQc: SGD時間の中央値が最小のTは256、15反復72.920 ms、最終Stress 1.27639e+06。

Tは時間とStressを併せて比較し、通常比較の共通Tは結果確認後に固定する。構造群は候補の比較であり、スケールフリー性や未知グラフへの一般化は実証していない。
