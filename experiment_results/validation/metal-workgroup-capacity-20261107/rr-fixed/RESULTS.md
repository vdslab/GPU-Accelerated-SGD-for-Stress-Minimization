# RR-like shared-memory benchmark — M4 Pro, 20 GPU cores

180 configurations, one warmup plus five timed trials each (1,080 trials).
Each trial contains four serial dispatches; GPU command-buffer elapsed time / 4, median of five, in milliseconds.
The same pipeline is reused at all W values. No Instruments profiling ran concurrently.

## Controlled comparison

At fixed (T,N), G=N/T workgroups each handle two coordinate tiles of T vertices.
N counts update slots / output float4 elements, not all vertices in a real graph.
Each WG uses 16T bytes of threadgroup memory; four cycles of T cross-tile matchings perform 4T² pair updates per dispatch.
Each matching reads and updates both endpoints in shared memory and ends with a threadgroup barrier (4T+1 barriers including initialization).
Only W changes within a row: six sizes W32/64/128/256/512/1024, corresponding to 1/2/4/8/16/32 SIMD groups.
W<T makes threads process multiple disjoint pairs before the barrier. W>T leaves excess threads with no pair update, but participating in barriers.
WG count, shared bytes, update count, input data, and synchronization count are identical across all six W in a row.
All groups and all W values produce byte-identical output for each T, including both distance modes and all trials.

## Relation to the actual SGD shader

This native Metal diagnostic reproduces the cross-tile pair-update formula, shared coordinate reads/writes and per-matching synchronization.
It is not the production WGSL/wgpu pipeline. It omits self-tile handling, partial tiles, outer scheduling and randomized graph-specific data accesses.
rr_uniform uses a runtime constant d=1, removing distance-buffer reads.
rr_matrix reads an all-ones distance matrix with stride 8192; groups reuse the same read-only region. This tests strided reads but not the cache footprint of independent graph tiles.
Four cycles reuse shared coordinates without returning them to global memory between cycles, unlike separately dispatched production tile jobs.
The fixed initial input and distance values are synthetic. The result selects candidates for real-graph verification, not a universal SIMD-per-WG optimum.
Compare W within each row. T changes both arithmetic and barrier counts, so times across T do not represent equal total work.

## rr_uniform

| T | G | W32 | W64 | W128 | W256 | W512 | W1024 | W within 5% of best |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 64 | 128 | 0.1514 | 0.1259 | 0.1300 | 0.1466 | 0.1911 | 0.2765 | 64, 128 |
| 64 | 512 | 0.3720 | 0.3933 | 0.4109 | 0.4521 | 0.5727 | 0.9232 | 32 |
| 64 | 2048 | 1.3276 | 1.4533 | 1.5204 | 1.6789 | 2.1499 | 3.5571 | 32 |
| 128 | 64 | 0.4902 | 0.3128 | 0.2758 | 0.2894 | 0.3202 | 0.4034 | 128, 256 |
| 128 | 256 | 0.8032 | 0.7519 | 0.7841 | 0.8199 | 0.9108 | 1.1533 | 64, 128 |
| 128 | 1024 | 2.5895 | 2.6669 | 2.9498 | 3.0952 | 3.4462 | 4.4400 | 32, 64 |
| 256 | 32 | 1.8689 | 1.0025 | 0.6338 | 0.5634 | 0.5840 | 0.6187 | 256, 512 |
| 256 | 128 | 2.1333 | 1.6101 | 1.5770 | 1.6935 | 1.7743 | 1.9353 | 64, 128 |
| 256 | 512 | 5.3523 | 5.1981 | 5.3355 | 5.9702 | 6.2795 | 6.9155 | 32, 64, 128 |
| 512 | 16 | 7.3300 | 3.7897 | 2.0265 | 1.3038 | 1.2241 | 1.1942 | 512, 1024 |
| 512 | 64 | 7.3984 | 4.3449 | 3.5407 | 3.5342 | 3.9092 | 4.0202 | 128, 256 |
| 512 | 256 | 15.4280 | 10.7862 | 10.3128 | 10.7151 | 12.1161 | 12.5856 | 64, 128, 256 |
| 1024 | 8 | 28.8454 | 14.7564 | 7.6525 | 4.6769 | 4.1107 | 4.0963 | 512, 1024 |
| 1024 | 32 | 29.0454 | 14.8379 | 8.7226 | 7.0925 | 7.2098 | 7.5993 | 256, 512 |
| 1024 | 128 | 57.7872 | 31.2778 | 22.6712 | 22.0690 | 23.2980 | 25.3597 | 128, 256 |

## rr_matrix

| T | G | W32 | W64 | W128 | W256 | W512 | W1024 | W within 5% of best |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 64 | 128 | 0.1736 | 0.1438 | 0.1473 | 0.1625 | 0.2073 | 0.2951 | 64, 128 |
| 64 | 512 | 0.4313 | 0.4514 | 0.4679 | 0.5081 | 0.6271 | 0.9847 | 32, 64 |
| 64 | 2048 | 1.5563 | 1.6771 | 1.7432 | 1.8957 | 2.3589 | 3.7970 | 32 |
| 128 | 64 | 0.5576 | 0.3578 | 0.3160 | 0.3274 | 0.3593 | 0.4459 | 128, 256 |
| 128 | 256 | 0.9422 | 0.8722 | 0.9022 | 0.9371 | 1.0274 | 1.2795 | 64, 128 |
| 128 | 1024 | 3.0547 | 3.1258 | 3.4086 | 3.5531 | 3.9016 | 4.9221 | 32, 64 |
| 256 | 32 | 2.1187 | 1.1243 | 0.7260 | 0.6541 | 0.6753 | 0.8415 | 256, 512 |
| 256 | 128 | 2.4720 | 1.9019 | 1.8511 | 1.9778 | 2.0946 | 3.0198 | 64, 128 |
| 256 | 512 | 6.3162 | 6.1607 | 6.2909 | 6.9564 | 7.7669 | 12.4646 | 32, 64, 128 |
| 512 | 16 | 8.5097 | 4.7147 | 2.6483 | 1.8222 | 1.6511 | 1.6528 | 512, 1024 |
| 512 | 64 | 8.3795 | 5.2061 | 4.2197 | 4.2365 | 4.8270 | 6.5485 | 128, 256 |
| 512 | 256 | 18.3628 | 13.4877 | 12.7009 | 12.9713 | 15.1138 | 21.3653 | 128, 256 |
| 1024 | 8 | 39.7380 | 20.8593 | 11.5264 | 6.7226 | 5.2254 | 5.2411 | 512, 1024 |
| 1024 | 32 | 39.1913 | 20.7758 | 11.7358 | 8.8280 | 8.9586 | 10.4563 | 256, 512 |
| 1024 | 128 | 78.3885 | 40.6297 | 30.0776 | 28.3244 | 30.3388 | 36.5388 | 256 |
