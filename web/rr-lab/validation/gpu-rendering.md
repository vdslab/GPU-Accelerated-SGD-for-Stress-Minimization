# Transition to WebGPU Rendering and Measurements

2026-11-09. Investigated the single-run and Web implementations and verified them on `kron_g500-logn17`.

## State Before the Change

- **Single-run site**: `experiments/run_single.py` launches `tools/gpu_visualizer`, which draws edges and circular nodes on the GPU via a Rust/`wgpu` render pipeline and returns the generated PNG to the browser.
- **Previous Web version**: Only SGD ran on WebGPU. The render Worker used Canvas 2D with `willReadFrequently: true` on `OffscreenCanvas`. Moving rendering into a Worker prevented UI freezes but did not move drawing onto the GPU.
- Displaying all edges left ~5.11 million edges to be drawn on the CPU, and Before rendering could block the After render queue. Even after SGD reached 15 / 15, the status remained stuck on the SGD progress string while waiting for rendering.

## State After the Change

- Implemented a browser WebGPU renderer in `render/gpu-renderer.js` and `render/graph.wgsl`.
- Uses `OffscreenCanvas.getContext('webgpu')` (with offscreen `GPUTexture` fallback for headless/WSL2 environments) → vertex/fragment shaders → render pass to draw line-width edge quads and circular node quads on the GPU.
- Keeps coordinates, edge endpoints, and BFS colors in GPU buffers. Camera pan/zoom updates a 48-byte uniform buffer without re-uploading coordinate arrays.
- Splits drawing into short GPU submissions so a single long command does not monopolize the GPU.
- Before and After share a single rendered GPU image for the initial layout; starting a new run with the same seed does not redraw Before.
- Prioritizes After rendering and coalesces queued camera updates during dragging while keeping the latest completed frame visible.
- Displays `SGD complete · Rendering` while waiting for the final render queue, and records the active render backend in the UI, summary text, and exported JSON.
- Defaults to **All edges** and all vertices without vertex downsampling or pixel aggregation on the GPU, with an optional **Auto (thin large graphs)** edge mode.
- Restricts Canvas 2D to compositing finished GPU frames onto the page and acting as a fallback when WebGPU is unavailable.

## Local Measurements

Chrome 156, Apple / Metal, no software fallback. `seed = 0`, `epsilon = 0.1`, `4096 × 4096`, all edges displayed, 15 iterations (single verification run, not a cross-device benchmark).

| Dataset          |   Vertices |      Edges | Pivots | Before GPU Render | After GPU Render | Post-SGD Render Wait |
| ---------------- | ---------: | ---------: | -----: | ----------------: | ---------------: | -------------------: |
| kron_g500-logn17 |    107,901 |  5,113,981 |     48 |        3,568.8 ms |         750.6 ms |             780.5 ms |
| germany_osm      | 11,548,845 | 12,369,181 |      5 |        8,143.2 ms |         232.9 ms |             394.0 ms |

Render time is measured with a monotonic CPU clock covering GPU coordinate upload, GPU render completion wait, and `ImageBitmap` creation (excluding initial pipeline/graph buffer setup). Post-SGD render wait measures the time from receiving final SGD coordinates until the remaining render queue finishes presenting to the page canvas.

Because the Before layout is random, many long edges span the viewport; the After layout has shorter edges and different overlap density, resulting in much faster rendering on the same graph.

## Reproduction

Run from `web/rr-lab`:

```sh
GRAPH_FILE=../../data/kron_g500-logn17.mtx PIVOTS=48 ITERATIONS=15 EDGE_LIMIT=0 npm run test:large
GRAPH_FILE=../../data/germany_osm.mtx ITERATIONS=15 EDGE_LIMIT=0 npm run test:large
```

Each canvas records `data-renderer`, `data-render-ms`, `data-draw-calls`, `data-drawn-edges`, `data-drawn-nodes`, `data-gpu-vendor`, and `data-gpu-fallback`, and the verification script checks that WebGPU is active without software fallback, all edges are drawn, and the output canvases contain non-white pixels.
