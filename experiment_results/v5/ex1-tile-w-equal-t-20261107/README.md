# EX1: W=T additional tile experiment

> 2026-11-09整理：1,800/1,800成功、[完了記録](completion.json)・[結果](reports/tile.md)。 [v5総覧](../README.md)。
> 保存状態：2026-11-08の容量整理で座標等は削除済み。以下の座標保存・復元の説明は実行当時の記録。数値結果・既存図表は保持。再描画には座標の再取得が必要。

Previous v5 is read-only. Same datasets and execution order; binaries rebuilt for optional exact evaluation override, seeds10–14, three repetitions, 15 iterations, epsilon0.1; only RR W changes from 1024 to T. CPU150 and RR1650 are freshly measured. shared, inner/outer shuffle on, phase_chunks1, GPU timestamps off. Exact Full Stress for all ten graphs (SGD_STRESS_MODE=exact); evaluation time excluded from SGD/total-compute intervals. Warmup per graph/method/T per session. No time cutoff; stop between runs on <4GiB disk or <2GiB available RAM.

Run/resume: `python3 experiments/run_ex1_tile_w_equal_t.py --run`. Progress: progress.json and PROGRESS.md. Reports generated automatically after all runs. Source and binary snapshots retained.
