# Full RR tile-size cause diagnostics

Run from the repository root. The diagnostic uses the current production GPU
source through `include!`; production files are not edited. The full-graph
comparisons preserve the update order for each T and assert exact coordinate
hash equality across W, memory mode, declared capacity, and repetitions.

```sh
cargo build --release --locked --manifest-path implementations/full/rr_gpu/Cargo.toml --example tile_cause
python3 tools/diagnostics/rr-tile-cause/run.py --output output/rr-tile-cause-recheck
python3 tools/diagnostics/rr-tile-cause/summarize.py output/rr-tile-cause-recheck
```

The runner refuses to overwrite existing run files and runs all GPU workloads
serially. It records source/binary/input hashes and commands. Full runs use seed
10, 15 iterations, one warmup plus three measurements per configuration. The
cross-round microbenchmark uses synthetic d=1 distances, one warmup plus five
measurements. A separate timestamp-disabled full run checks measurement effects.

Interventions:

- Full SGD: T64/128/256/512/1024 × W32/128/256/1024, shared coordinates.
- Full SGD: T128/256 with the workgroup arrays declared as capacity 1024.
- Full SGD: T128/256/1024 with global coordinates and storage barriers.
- Micro: T128/256/1024 × 1/2/4/8/16 independent coordinate replicas.
- Micro distance source: production row-major matrix; constant (allows compiler
  folding); runtime uniform; packed cross-tile/offset/local layout. All values
  equal one, with identical output coordinates checked across variants.

The microbenchmark repeats one cross round eight times; it is not a full SGD
epoch, convergence test, or graph-quality comparison. It establishes available
throughput and sensitivity to distance layout. It does not measure per-core
occupancy, bank conflicts, or per-instruction stall attribution.

Recorded investigation:
`experiment_results/validation/rr-tile-cause-20261107/RESULTS.md`.
`micro.jsonl` is an initial exploratory run and is excluded from summaries;
`micro-controls.jsonl` is the final controlled run. `profile-stdout.jsonl` is
from Instruments recording and is also excluded. Raw trace files are ignored.
