# RR Sparse f32: delayed phase interleaving (2026-10-30)

## Decision

Do not change the production/browser default. f32 is not a demonstrated lower
bound on achievable stress: changing only update order improves several cases.
However, the best screened candidate worsens Stanford seed 3 by 16.18% and seed 7
by 5.91%. This does not meet the requested broadly applicable quality preservation.
The new controls are diagnostic only; no graph-specific selector was introduced.

CPU and GPU use the same stress evaluator. Rounding affects the positions obtained
*during optimization*, not the definition of stress. A worse value means a worse
layout under that metric; rounding is a sensitivity mechanism, not an excuse to
accept the regression. This experiment does not isolate rounding as the sole cause.

## Method and controls

Apple M4 Pro / Metal; unchanged f32 shader, initial positions, pivot selection,
learning-rate schedule, 15 iterations, epsilon 0.1, 200 pivots, RNG consumption,
and constraint visit count. Each iteration still visits every constraint once.
Two-sided updates remain disjoint RR matchings. One-sided pivot interactions and
two-sided matching rounds are divided into chunks and interleaved after a fixed
iteration. No graph degree, graph name, or observed stress is used for selection.

`chunks@start` specifies the chunk count and zero-based first interleaved iteration.
`1@0` is the original schedule. `200@5` keeps the original order for 5 iterations
and interleaves 200 chunks in each of the last 10. The additional candidates
`4@10` and `200@10` change only the last 5 iterations.

Screening used 64 evaluation sources on five cases, including known adverse cases
from earlier interleaving experiments. `200@5` improved all five. The subsequent
validation uses 256 sources, evaluation seed 0, 8 graphs and optimization seeds
0, 1, 3, 7. Seed 7 was not used in the screen. Initial-position and preprocessing
SHA-256 hashes must match between variants. This is finite validation, not a
proof for every graph or seed. The 2% gate in the script is an investigation
criterion, not a user-specified tolerance or statistical confidence interval.

## Results: candidate 200@5

Stress ratios below are candidate / original RR, using the same 256 sources.

| Graph | Median ratio across 4 seeds | Worst ratio |
| --- | ---: | ---: |
| EVA | 0.9877 | 1.0068 |
| USpowerGrid | 1.0009 | 1.0036 |
| luxembourg_osm | 0.9976 | 0.9993 |
| preferentialAttachment | 0.8520 | 0.8685 |
| smallworld | 0.8534 | 0.9267 |
| coAuthorsCiteseer | 0.8256 | 0.9112 |
| web-NotreDame | 0.8684 | 0.9883 |
| web-Stanford | 0.9577 | 1.1618 |

Overall median ratio is 0.9364; two of 32 cases exceed 1.02. The median does not
justify accepting the two substantial regressions.

For Stanford seed 0, the 256-source stress falls from 6,286,796,052 to
5,383,782,695 (-14.36%). Iteration time is 1.572 s vs 1.623 s (+3.23%).
The earlier 64-source screen gives 6,212,335,104 vs 4,947,680,522 (-20.36%).
Those two reductions are from different sample sizes and must not be mixed.
The prior CPU f64 reference is 5,103,171,457 at 256 sources (see the preceding
cause investigation for full precision), so this candidate does not establish
CPU-equivalent quality even for seed 0. No CPU speedup claim is made here.

Timing columns are single ordered observations, with the baseline run first;
GPU warm-up and system load can bias these observations, especially small cases.
They are useful for spotting a large cost but are not repeated performance
benchmarks. Stress comparisons use deterministic inputs and identical sources.

## Reproduction

From the repository root:

```sh
cargo build --release --locked --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml --example quality_search
python3 tools/diagnostics/rr-common/validate_f32_quality.py --output /tmp/rr-quality-validation
python3 tools/diagnostics/rr-common/validate_f32_quality.py --output /tmp/rr-quality-late --variants '1@0,4@10,200@10'
```

The validator saves every result, checks common input hashes, and writes all
regressions to `summary.json`; process success means the experiment completed,
not that the candidate passed the quality gate. Use a fresh output directory
when changing code: existing complete results with matching CLI settings are
resumed, and the cache does not fingerprint the executable.

`screen-*.jsonl` contains the exploratory 64-source results; `validation/` contains
the 32-case test of `200@5`; `late-validation/` contains both later-switch variants.
See `../rr-stanford-cause-20261030/README.md` for CPU precision/order replay.

## Implementation boundary

`execute()` retains the original schedule. The new
`execute_quality_diagnostic()` permits a delayed switch; unused uniform/dispatch
slots are not executed before that switch. The diagnostic run's per-iteration
submission/dispatch fields describe capacity bounds, not totals for a variable schedule.
The normal production schedule has constant counts, so its existing reporting
remains valid. No browser option or production quality claim was added.

## Later switch results

Both later-switch candidates also fail the conservative 2% investigation gate.

| Variant | Median stress ratio (32 cases) | Worst ratio | Cases above 1.02 |
| --- | ---: | ---: | ---: |
| 4@10 | 0.982499 | 1.042493 | 2 |
| 200@10 | 0.979530 | 1.041875 | 2 |

Both improve all four Stanford seeds, but regress on preferentialAttachment seed 3
and coAuthorsCiteseer seed 7. For `4@10` those regressions are +3.20% and +4.25%;
for `200@10`, +4.19% and +3.42%. Delaying the switch reduces the worst observed
regression but does not remove it. No default change follows from these results.

## Checks

- `cargo test --release --locked --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml --all-targets`: 51 tests passed, including Metal execution, delayed dispatch boundaries, existing CPU replay checks, and CLI JSON output.
- `cargo clippy --release --locked --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml --all-targets -- -D warnings`: passed.
- `cargo fmt --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml --check`: passed.
- Python syntax and resuming the completed 32-case validation: passed.
