# Fixed-work arithmetic benchmark — Apple M4 Pro (20 GPU cores)

No graph inputs. N total threads, G=N/W workgroups, SIMD width 32.
Each thread runs x=fma(x,a,b), either one recurrence (fma1) or four independent recurrences (fma4).
No threadgroup memory or barriers. Fast math disabled; same pipeline for all W.
Two loop lengths, eight N values, six W values, two kernels: 192 configurations.
Each has one warmup and five measured trials, in shuffled order: 1,152 trials total.
GPU command-buffer elapsed time / four serial dispatches; medians below, in microseconds.
CPU reference checks passed; full output hashes agree across all six W values and all trials for each N/kernel/loop length.

The previous residency benchmark measures inflight capacity; it does not determine the fastest WG size.
These measurements select dispatch geometry for these recurrence kernels. They do not measure same-cycle issue width or establish an optimal WG size for every kernel.

## fma1, 4096 iterations per thread

| N | W32 | W64 | W128 | W256 | W512 | W1024 | W within 5% of best |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1024 | 55.96 | 55.91 | 55.00 | 56.23 | 87.44 | 142.09 | 32, 64, 128, 256 |
| 4096 | 57.50 | 57.96 | 57.92 | 57.55 | 106.79 | 162.13 | 32, 64, 128, 256 |
| 8192 | 77.92 | 68.53 | 100.60 | 108.97 | 109.47 | 161.93 | 64 |
| 16384 | 111.44 | 113.34 | 126.04 | 161.06 | 163.40 | 163.82 | 32, 64 |
| 32768 | 218.72 | 218.94 | 218.42 | 219.49 | 268.49 | 270.44 | 32, 64, 128, 256 |
| 65536 | 412.62 | 409.75 | 406.83 | 404.94 | 429.07 | 482.25 | 32, 64, 128, 256 |
| 131072 | 752.59 | 759.91 | 753.16 | 764.12 | 767.82 | 816.91 | 32, 64, 128, 256, 512 |
| 262144 | 1470.10 | 1472.16 | 1477.56 | 1478.74 | 1488.96 | 1494.86 | 32, 64, 128, 256, 512, 1024 |

## fma4, 4096 iterations per thread

| N | W32 | W64 | W128 | W256 | W512 | W1024 | W within 5% of best |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1024 | 56.91 | 55.55 | 55.84 | 57.28 | 91.84 | 147.74 | 32, 64, 128, 256 |
| 4096 | 56.97 | 58.25 | 57.48 | 57.50 | 102.85 | 156.46 | 32, 64, 128, 256 |
| 8192 | 74.87 | 75.47 | 96.69 | 97.98 | 104.31 | 160.15 | 32, 64 |
| 16384 | 115.48 | 115.37 | 121.28 | 156.96 | 159.86 | 159.84 | 32, 64 |
| 32768 | 216.09 | 215.99 | 216.41 | 224.27 | 266.69 | 266.45 | 32, 64, 128, 256 |
| 65536 | 406.84 | 402.06 | 405.23 | 403.33 | 431.06 | 485.02 | 32, 64, 128, 256 |
| 131072 | 765.41 | 763.03 | 767.28 | 772.84 | 776.79 | 830.25 | 32, 64, 128, 256, 512 |
| 262144 | 1496.68 | 1498.39 | 1498.35 | 1511.14 | 1512.54 | 1517.66 | 32, 64, 128, 256, 512, 1024 |

## fma1, 16384 iterations per thread

| N | W32 | W64 | W128 | W256 | W512 | W1024 | W within 5% of best |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1024 | 211.67 | 210.80 | 211.58 | 218.41 | 345.42 | 542.00 | 32, 64, 128, 256 |
| 4096 | 222.16 | 221.94 | 222.06 | 222.00 | 424.28 | 634.13 | 32, 64, 128, 256 |
| 8192 | 246.50 | 284.36 | 374.57 | 424.56 | 427.03 | 631.26 | 32 |
| 16384 | 435.60 | 437.71 | 486.25 | 626.56 | 639.11 | 641.57 | 32, 64 |
| 32768 | 861.03 | 861.75 | 860.97 | 869.14 | 1063.94 | 1068.80 | 32, 64, 128, 256 |
| 65536 | 1672.89 | 1609.81 | 1632.58 | 1595.00 | 1705.44 | 1915.71 | 32, 64, 128, 256 |
| 131072 | 3005.56 | 3037.54 | 3015.22 | 3065.15 | 3041.95 | 3246.20 | 32, 64, 128, 256, 512 |
| 262144 | 5850.56 | 5854.07 | 5862.48 | 5879.80 | 5924.84 | 5991.00 | 32, 64, 128, 256, 512, 1024 |

## fma4, 16384 iterations per thread

| N | W32 | W64 | W128 | W256 | W512 | W1024 | W within 5% of best |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1024 | 211.50 | 210.83 | 211.47 | 217.75 | 349.56 | 585.53 | 32, 64, 128, 256 |
| 4096 | 219.52 | 219.50 | 219.37 | 222.13 | 401.16 | 616.61 | 32, 64, 128, 256 |
| 8192 | 284.41 | 288.79 | 368.26 | 416.88 | 419.69 | 625.28 | 32, 64 |
| 16384 | 450.44 | 451.43 | 492.93 | 617.73 | 624.07 | 629.00 | 32, 64 |
| 32768 | 852.72 | 850.92 | 850.41 | 880.16 | 1047.12 | 1050.23 | 32, 64, 128, 256 |
| 65536 | 1598.07 | 1582.09 | 1594.50 | 1606.37 | 1703.59 | 1920.16 | 32, 64, 128, 256 |
| 131072 | 3024.34 | 3033.38 | 3037.21 | 3054.07 | 3071.60 | 3286.76 | 32, 64, 128, 256, 512 |
| 262144 | 5908.39 | 5927.34 | 6017.58 | 5950.95 | 5977.88 | 6018.94 | 32, 64, 128, 256, 512, 1024 |

## Interpretation

W32/W64 (one/two SIMD groups per WG) are consistently competitive here. At N=8192, W1024 is about 2.1–2.4 times slower than W32/W64 for fma4. At N=262144, the difference is around 1–2%.
More SIMD groups per WG are not inherently faster. With enough total work, multiple configurations approach similar throughput. Near ties should not be treated as a universal ranking.
No profile capture ran concurrently with these timing runs. The machine remained an interactive desktop; five-trial ranges are recorded in summary.csv.
