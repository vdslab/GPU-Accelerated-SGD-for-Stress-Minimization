## Verification

### Automated checks

- `cargo fmt --all -- --check`: passed
- `cargo test --release`: passed (30 library tests, 3 binary tests)
- `cargo clippy --all-targets -- -D warnings`: passed
- The real-GPU canary and CPU-quality comparison test passed on NVIDIA GeForce GTX 1660 Ti (Vulkan).

### USPowerGrid

Command parameters: 200 pivots, 15 iterations, epsilon 0.1, seed 0.

- one-sided constraints: 947,723
- two-sided constraints: 26,491
- base rounds: 199
- spill rounds: 8
- total rounds: 207
- dispatches per iteration: 208
- schedule time: 50.484 ms
- GPU compute time: 66.238 ms
- exact full stress: 721,013.515321

The round statistics match the existing seed-0 reference, and stress remains within the established quality range.

### luxembourg_osm

Command parameters: 50 pivots, 15 iterations, epsilon 0.1, seed 0.

- one-sided constraints: 5,727,346
- two-sided constraints: 120,891
- base rounds: 49
- spill rounds: 3
- total rounds: 52
- dispatches per iteration: 53
- schedule time: 266.849 ms
- GPU compute time: 66.540 ms

The previous measurement was 52 rounds and approximately 254.6 ms scheduling, so round behavior is unchanged and the timing difference is within ordinary run-to-run variation.

### web-Stanford

Command parameters: 1 pivot, 1 iteration, epsilon 0.1, seed 0.

| Metric | Previous scheduler | Compact scheduler |
|---|---:|---:|
| Selected vertices | 255,265 | 255,265 |
| Selected edges | 1,941,926 | 1,941,926 |
| One-sided constraints | 255,228 | 255,228 |
| Two-sided constraints | 1,941,926 | 1,941,926 |
| Total rounds | 38,625 | 38,625 |
| Dispatches per iteration | 38,626 | 38,626 |
| Schedule time | 261.521 s | 257.565 s |
| GPU compute time | 1.580 s | 1.580 s |
| Highest sampled private memory | 2,444.6 MiB | 2,560.6 MiB |

Before schedule completion, compact-scheduler private memory rose gradually to approximately 463.3 MiB. The jump above 2 GiB occurred only after `GPU:` / `Iteration: 1`, while one command encoder recorded 38,626 compute passes. The previous implementation showed the same transition.

This demonstrates that removing all-round CPU containers preserves the exact schedule and slightly improves schedule time, but does not reduce whole-process peak memory. The dominant peak is associated with recording tens of thousands of WGPU/Vulkan compute passes in one command buffer, not with the final compact `Schedule` or BFS state. Reducing that peak requires a follow-up GPU execution design, such as bounded pass batches with submission boundaries, and is outside this change's stated non-goals.

### Bounded command batching follow-up

The follow-up implementation limits each command encoder and submission to 256 compute passes. It submits and completes each contiguous batch before recording the next one, while retaining global dynamic-uniform offsets and the original invocation order.

| Metric | Compact, one submission | Compact, bounded batches |
|---|---:|---:|
| Selected vertices | 255,265 | 255,265 |
| Selected edges | 1,941,926 | 1,941,926 |
| One-sided constraints | 255,228 | 255,228 |
| Two-sided constraints | 1,941,926 | 1,941,926 |
| Total rounds | 38,625 | 38,625 |
| Dispatches per iteration | 38,626 | 38,626 |
| Submissions per iteration | 1 | 151 |
| Schedule time | 257.565 s | 4.929 s |
| GPU compute time | 1.580 s | 0.563 s |
| Highest sampled private memory | 2,560.6 MiB | 550.9 MiB |
| Highest sampled working set | not recorded | 471.5 MiB |

The bounded run used web-Stanford with 1 pivot, 1 iteration, epsilon 0.1, and seed 0. Memory was sampled every 50 ms on NVIDIA GeForce GTX 1660 Ti (Vulkan). The process completed GPU readback in 7.985 s and subsequently wrote both full result files successfully. Peak private memory fell by approximately 78.5%, confirming that the former peak was driver-side state associated with recording 38,626 passes in one command buffer.

USPowerGrid retained 207 rounds, 208 dispatches, and one submission per iteration; scheduling took 44.922 ms and GPU compute took 66.943 ms. luxembourg_osm retained 52 rounds, 53 dispatches, and one submission per iteration; scheduling took 262.446 ms and GPU compute took 66.212 ms. Both preserve the previous constraint counts and single-submission behavior.
