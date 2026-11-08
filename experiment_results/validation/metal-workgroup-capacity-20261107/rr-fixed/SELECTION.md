# Measured workgroup configurations

Selection uses the lowest median GPU time in the existing rr_matrix benchmark, without interpolating unmeasured WG counts. W is threads per WG; SIMD width is 32. The near-best list uses an explicitly chosen 5% tolerance, not a statistical confidence interval.

| T | WG count | Selected W | SIMD/WG | Median ms | W1024 / selected time | SIMD/WG within 5% |
|---:|---:|---:|---:|---:|---:|:---|
| 64 | 128 | 64 | 2 | 0.1438 | 2.05 | 2, 4 |
| 64 | 512 | 32 | 1 | 0.4313 | 2.28 | 1, 2 |
| 64 | 2048 | 32 | 1 | 1.5563 | 2.44 | 1 |
| 128 | 64 | 128 | 4 | 0.3160 | 1.41 | 4, 8 |
| 128 | 256 | 64 | 2 | 0.8722 | 1.47 | 2, 4 |
| 128 | 1024 | 32 | 1 | 3.0547 | 1.61 | 1, 2 |
| 256 | 32 | 256 | 8 | 0.6541 | 1.29 | 8, 16 |
| 256 | 128 | 128 | 4 | 1.8511 | 1.63 | 2, 4 |
| 256 | 512 | 64 | 2 | 6.1607 | 2.02 | 1, 2, 4 |
| 512 | 16 | 512 | 16 | 1.6511 | 1.00 | 16, 32 |
| 512 | 64 | 128 | 4 | 4.2197 | 1.55 | 4, 8 |
| 512 | 256 | 128 | 4 | 12.7009 | 1.68 | 4, 8 |
| 1024 | 8 | 512 | 16 | 5.2254 | 1.00 | 16, 32 |
| 1024 | 32 | 256 | 8 | 8.8280 | 1.18 | 8, 16 |
| 1024 | 128 | 256 | 8 | 28.3244 | 1.29 | 8 |

Increasing W at fixed T and G does not distribute a WG onto more GPU cores. It changes how that WG schedules its pairs among threads and the number of SIMD groups participating in barriers. W>T cannot add more useful pair updates per matching. For W<T, decreasing W serializes multiple pairs per thread. The measured optimum balances these effects; the timings do not uniquely attribute costs to individual hardware mechanisms.

These results apply to the synthetic native Metal cross-tile probe, not all Apple GPU kernels. For production selection, graph size determines G and T determines work per WG; graph-specific memory behavior and the production WGSL pipeline remain unmeasured by this table.
