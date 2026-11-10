# Large Graph Verification

This document records large-graph verification prior to the full WebGPU renderer transition. For the current WebGPU rendering architecture and all-edge measurements, see [gpu-rendering.md](gpu-rendering.md).

2026-11-09. Google Chrome 156, hardware WebGPU, `seed = 0`, `epsilon = 0.1`, `4096 × 4096`.

| Graph                                | Active Vertices | Active Edges (Undirected, Deduplicated) | Pivots | Iterations | Stress Evaluation              |
| ------------------------------------ | --------------: | --------------------------------------: | -----: | ---------: | ------------------------------ |
| web-Stanford                         |         255,265 |                               1,941,926 |    200 |         15 | Sampled from 16 fixed sources  |
| germany_osm                          |      11,548,845 |                              12,369,181 |      5 |         15 | Sampled from 2 fixed sources   |
| germany_osm (128 MiB GPU buffer cap) |      11,548,845 |                              12,369,181 |      1 |          3 | Sampled from 2 fixed sources   |

For `web-Stanford`, the largest connected component (255,265 vertices) was extracted from the 281,903-vertex input. For `germany_osm`, all vertices and edges belong to a single connected component. Pivot counts were selected to fit within the target GPU buffer limits.

Non-white Before/After rendering, enlarged modal zoom/fit controls, and `4096 × 4096` PNG export were verified. Under the 128 MiB buffer limit, multi-buffer splitting of two-sided matchings was also verified.

## Reproduction

Run from `web/rr-lab`:

```sh
PIVOTS=200 ITERATIONS=15 npm run test:large
GRAPH_FILE=../../data/germany_osm.mtx ITERATIONS=15 npm run test:large
GPU_BUFFER_MB=128 GRAPH_FILE=../../data/germany_osm.mtx ITERATIONS=3 npm run test:large
```

## Input Datasets

- [SNAP/web-Stanford](https://sparse.tamu.edu/SNAP/web-Stanford): 30,579,901 bytes, SHA-256 `a514e9693666e9d063b3f0b3535f45706f31e524239194750785af737f8ad090`
- [DIMACS10/germany_osm](https://sparse.tamu.edu/DIMACS10/germany_osm): 198,530,725 bytes (~189 MiB), SHA-256 `a8de087a1262601537a77ca8acc0ce6d3f790f07621f0cef605a8c362303924c`
