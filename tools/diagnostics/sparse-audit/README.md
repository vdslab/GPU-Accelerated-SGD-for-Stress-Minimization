# Sparse correctness and quality audit

Standalone diagnostic executable; production implementations are not modified.

```sh
cargo test --locked --manifest-path tools/diagnostics/sparse-audit/Cargo.toml -- --test-threads=1
cargo build --release --locked --manifest-path tools/diagnostics/sparse-audit/Cargo.toml
# Re-evaluate saved layouts using 64 sources and four sets of 1024 sources.
tools/diagnostics/sparse-audit/target/release/sparse-audit eval data/delaunay_n17.mtx LAYOUT.final.txt
# Shared 4096 sources, seed 20261027; paired differences relative to first layout.
tools/diagnostics/sparse-audit/target/release/sparse-audit paired data/delaunay_n17.mtx BASELINE.final.txt OTHER.final.txt
# Seeds 0,1,2; six variants; writes layouts, per-run JSON and constraint statistics.
tools/diagnostics/sparse-audit/target/release/sparse-audit run data/delaunay_n17.mtx OUTPUT_DIRECTORY
```

The six variants are CPU f64 shuffled/fixed, diagnostic CPU f32 shuffled/fixed,
and production GPU shuffled/fixed. Each seed uses the same prepared inputs and
cloned post-preprocessing RNG. CPU f32 uses the GPU's deterministic tiny-distance
fallback but does not emulate Metal's fast math, FMA, or hardware sqrt bit-for-bit.

`run` holds prepared constraints and copies for diagnostic comparisons. Its elapsed
time is NOT comparable to the original benchmark's iteration time: GPU setup and
conversions are included, and other diagnostic jobs may run concurrently. Quality
is measured on identical source vertices, using 64 and 1024 sources with seed 0.
The CPU f64 seed-0 shuffled result was checked against the original saved run.

`paired` independently calculates source contributions with Graph BFS. The paired
95% interval is a normal approximation to the finite-population source-sampling
mean (sample without replacement), not a confidence interval over optimization
seeds and not proof of an exact full-stress ordering. Its baseline is the first
input layout. For `eval`, the four sample sets may overlap; they are not one
4096-distinct-source sample.

Tests include an independent Floyd–Warshall/brute-force derivation of all sparse
constraints for 3 graph sizes × 3 pivot counts × 5 seeds, including all-pivot
reduction to full weights. The GPU test moves one contested hub while 4096 leaves
remain fixed, checks a known serial recurrence across 20 shuffled/fixed runs, and
checks every update completed. Passing does not establish a memory-model proof.

`eval` and `paired` require saved positions to follow the production `# Positions`
header and the graph's largest-component vertex order. This tool is for the
unweighted, undirected graph model used by the repository.

## Smallworld order/visibility probe

`order-probe` is a diagnostic host fork of the production Metal runner. It compiles
production `shader.metal`, optionally removes its two fences, or injects tracing.
It uses seed 0, 200 pivots, 15 iterations, epsilon 0.1, and CPU shuffle enabled.

```sh
cargo build --release --locked --manifest-path tools/diagnostics/sparse-audit/Cargo.toml --bin order-probe
PROBE_FENCE=1 PROBE_BATCH=65536 PROBE_TRACE=0 tools/diagnostics/sparse-audit/target/release/order-probe data/smallworld.mtx OUTPUT_PREFIX
```

`PROBE_BATCH` limits the consecutive chunk of the same shuffled index permutation
submitted at a time; every chunk completes before the next. The default is all
constraints. Failed locks are retried within the chunk; this is not an exact
serial replay. `PROBE_FENCE=0` removes the fences for comparison. `PROBE_TRACE=1`
records, under both endpoint locks, the preceding writer ID at each endpoint,
four input coordinates, and four output coordinates. Records are inspected after
each complete iteration, then reused. The validator checks every input bit against
the prior writer (or the iteration's starting position), every final coordinate,
and every vertex's complete predecessor chain, rejecting cycles/missing links.
It also compares every arithmetic update with an independent f64 evaluation of
the same f32 inputs, using scaled tolerance 2e-5. This is not bit-exact arithmetic
replay: Metal fast math and f32 rounding differ from the f64 check.

The trace adds two relaxed atomic exchanges and log traffic, changing scheduling;
a passing trace does not prove all uninstrumented schedules correct. Chunking is
a controlled ordering intervention, not a proposed production default. Probe
elapsed time includes diagnostic CPU work and is not a production speed result.
`ORDER` reports the mean zero-based retry pass separately for true graph edges
(`dij == 1`) and other constraints. See the saved experiment scripts and raw logs
in `experiment_results/v2/validation/atomic-sparse-smallworld-order-20261027`.

For a validator negative control only, `PROBE_CORRUPT_LOG=1` changes one recorded
input coordinate on the CPU after the GPU finishes the first iteration. With
tracing enabled, the validator must reject that log with `inconsistent position
history`. It does not alter GPU positions. The normal and corrupted-log controls
were run on `dwt_419`; both produced their expected pass/fail outcomes.
