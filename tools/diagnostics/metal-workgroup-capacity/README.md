# Native Metal workgroup capacity probes

These probes measure the relationship between per-workgroup work and available
workgroup count on the local Apple GPU. They do not interpret launch limits,
occupancy, or a latency plateau as same-cycle instruction issue capacity.

From the repository root:

```sh
xcrun swiftc -O tools/diagnostics/metal-workgroup-capacity/benchmark.swift -o tools/diagnostics/metal-workgroup-capacity/benchmark
python3 tools/diagnostics/metal-workgroup-capacity/run.py --output output/metal-workgroup-capacity-recheck
python3 tools/diagnostics/metal-workgroup-capacity/analyze.py output/metal-workgroup-capacity-recheck
```

The runner executes `sweep 5 4096` followed by `boundary 5 16384` serially, using
fresh output files. Each configuration has one warmup and five measured trials;
configurations are shuffled each round with a fixed seed. A trial contains four
serial dispatches. Timing is command-buffer GPU elapsed time divided by four.
Whole-graph preprocessing, quality evaluation, and CPU validation are excluded.

`fma1` and `fma4` use one/four scalar recurrence chains per thread. Grouping
changes preserve each thread's input and operations. CPU fused-FMA checks and
output hashes verify this equivalence. RR-like probes use identical matching,
pair-update arithmetic, and shared-memory barriers. Same-T outputs must match
across W, G, uniform/matrix distance, and trials. All groups' output coordinates
must equal the first group's coordinates. RR distances are synthetic d=1; the
probe is not a full-graph SGD implementation.

`resident 3 32768` queues 131,072 threads for each W32/64/128/256/512/1024,
for both fma4 and the RR-like T=W kernel. This supplies enough queued work to
observe high residency. Its profiled elapsed times are not mixed into the
unprofiled sweep. Convert observed SIMD/core into WG equivalents explicitly;
do not label that conversion a count of same-cycle instruction issues.

## Fixed total work, without graph inputs

Run `benchmark benchmark.metal fixed 5 4096` and repeat with `16384` as the
last argument (use the full executable/source paths from the repository root).
The `fixed` mode uses only fma1/fma4. For each total thread count N in
1024/4096/8192/16384/32768/65536/131072/262144, it sweeps W32 through W1024
with G=N/W. Inputs, outputs, operations per thread, and pipeline are unchanged
across W. There is no shared-memory allocation or barrier in these kernels.
Run `python3 tools/diagnostics/metal-workgroup-capacity/analyze_fixed.py RESULT_DIR`
to validate and summarize the two `iterations*.jsonl` files. This measures
useful dispatch geometry for these arithmetic kernels, not universal optimal
geometry or same-cycle instruction issue width.

`rr-fixed 5 4096` compares shared-memory cross-tile updates at fixed T and
fixed WG count while changing W. It sweeps T64/128/256/512/1024 and
N8192/32768/131072 output slots, using G=N/T WGs. The `4096` argument is
unused by RR kernels; each runs four cycles of T matchings. Both uniform and
strided matrix distance modes are included. Analyze with
`python3 tools/diagnostics/metal-workgroup-capacity/analyze_rr_fixed.py RESULT_DIR`.
This separates W from T and keeps update and barrier counts fixed within a
comparison. It remains a synthetic native Metal diagnostic rather than the
production graph pipeline; see its RESULTS.md for the precise differences.

## Hardware counters

On this Mac, public `MTLDevice.counterSets` exposed only timestamps. In the
Instruments UI, the Metal System Trace template was configured with:

- Counter Set: **Performance Limiters**
- Enable Shader Timeline: **on**
- Performance State: **Default**

The saved template is `performance-limiters.tracetemplate` in the result
directory. It exposes 48 counters, including Compute SIMD Groups Inflight,
Kernel Occupancy, and F32 Utilization. Counter values are GPU-global samples,
correlated with labeled GPU execution intervals; they are not per-instruction
or per-workgroup physical-core assignments. The inflight count is averaged per
shader core. Inflight groups are not the same as issued instructions per cycle.

Example capture (use a new output name):

```sh
xcrun xctrace record --template experiment_results/validation/metal-workgroup-capacity-20261107/performance-limiters.tracetemplate --time-limit 10s --no-prompt --output output/profile2.trace --target-stdout /absolute/path/output/profile2.jsonl --launch -- /absolute/path/tools/diagnostics/metal-workgroup-capacity/benchmark /absolute/path/tools/diagnostics/metal-workgroup-capacity/benchmark.metal profile2 5 32768
```

Export the `gpu-counter-info` and `metal-gpu-intervals` schemas to
`profile2-counter-info.xml` and `profile2-gpu-intervals.xml`. Then extract only
the relevant counter series without storing the very large raw counter XML:

```sh
python3 tools/diagnostics/metal-workgroup-capacity/extract_counters.py output profile2 --stream-counters
```

The script correlates samples entirely within the middle 80% of labeled GPU
intervals, excludes repetition zero, and weights by sample duration. Profiled
timings are excluded from the unprofiled sweep. Keep the trace locally for
re-export, and gzip the extracted sample CSV to reduce storage.

Results: `experiment_results/validation/metal-workgroup-capacity-20261107/RESULTS.md`.
No production shader, scheduler, default T/W, or paper result was changed.
