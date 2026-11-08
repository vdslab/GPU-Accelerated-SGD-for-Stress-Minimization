# EX1: W=T additional tile experiment

Previous v5 is read-only. Same datasets and execution order; binaries rebuilt for optional exact evaluation override, seeds10–14, three repetitions, 15 iterations, epsilon0.1; only RR W changes from 1024 to T. CPU150 and RR1650 are freshly measured. shared, inner/outer shuffle on, phase_chunks1, GPU timestamps off. Exact Full Stress for all ten graphs (SGD_STRESS_MODE=exact); evaluation time excluded from SGD/total-compute intervals. Warmup per graph/method/T per session. No time cutoff; stop between runs on <4GiB disk or <2GiB available RAM.

Run/resume: `python3 experiments/run_ex1_tile_w_equal_t.py --run`. Progress: progress.json and PROGRESS.md. Reports generated automatically after all runs. Source and binary snapshots retained.
