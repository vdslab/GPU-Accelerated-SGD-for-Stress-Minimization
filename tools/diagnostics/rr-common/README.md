# RR Sparse quality regression checks

`quality_probe` is the Rust example in `implementations/sparse/sparse-sgd-gpu`.
It reports preprocessing/initial/final hashes and sampled full stress with 256 BFS
sources (seed 0), including for small graphs. `cpu` runs the independent f64 CPU
reference with the same initialization. Phase splitting and affine pivot-stride
options are experimental and are not enabled by the production CLI.

`compare_quality.py AUDIT_DIR --v3-runs PATH` checks saved `suite-*.jsonl` against
`optimized-*.jsonl` for exact preprocessing, initial/final coordinate hashes and
stress. Adding `--max-cpu-ratio 1.05` separately rejects matched CPU comparisons
above that threshold. Only matching preprocessing/initial hashes and compatible
256-source sampled evaluation are compared to v3; small exact references are
excluded. Exit code 1 indicates a failed check, including missing outputs.

The 2026-10-30 audit passes the unchanged-RR checks but **fails** the CPU +5%
quality gate. Speed improvements must not be presented as resolving this issue.
