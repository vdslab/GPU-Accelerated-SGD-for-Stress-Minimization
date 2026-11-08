# EX2 RR128 speed and Stress experiment

The original v5 and EX1 results are read-only. This directory is an additional experiment.

Run/resume: `python3 experiments/run_ex2_rr128.py --run`.

The runner first validates all 10 timing fields on small/large Full and Sparse inputs, checks timestamp-on/off output equality and Sparse W256/W128 equality, then executes 1,410 formal runs sequentially. Formal records exclude validation and warmups. Full uses T=W128 and exact Full Stress. Sparse uses W128, p=min(200,n) and the previous evaluation protocol. Timings use actual pass timestamps and the first final-coordinate frame completion on the shared monotonic clock. Rendering: 2048x2048, all edges, radius1px, blue. Stress evaluation and handoff are included in result-to-frame elapsed time; PNG encoding is excluded.

`progress.json` and `PROGRESS.md` are the current live status. `validation/preflight-complete.json` records preflight success; `completion.json` is created only after all runs and report generation succeed. A failure stops the runner and preserves evidence in `failure.json`; resource limits stop between runs and preserve completed records. Do not treat the absence of completion.json as completion.

Raw native outputs, renderer logs, command lines and losslessly archived initial/final coordinates remain. Coordinates use independently readable gzip files with APFS-shared immutable edge data; manifests verify recovery. Per-run layout PNGs are omitted after recording their SHA, but report charts are retained in figures/. Source/binary/input evidence is frozen under source-snapshot/, bin/ and config/.

After all runs, the offline report writes summary tables, eight PNG/PDF charts, and Obsidian's EX2 result note and EX2_v5_図表 assets. See PLAN.md / EXECUTION.md for the Japanese definitions and boundaries.
