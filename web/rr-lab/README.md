# RR-SGD

Full title: **RR-SGD: WebGPU-Accelerated Graph Layout with Round-Robin SGD**

Live URL: https://vdslab.github.io/GPU-Accelerated-SGD-for-Stress-Minimization/

The site name, browser title, and page heading are **RR-SGD**. Click the **ⓘ** button next to the heading to view an overview of the algorithms, the differences between Full and Sparse, and usage instructions.

This static web application performs Matrix Market (`.mtx`) preprocessing, RR Full / RR Sparse SGD, and stress evaluation directly in the browser. Both preprocessing and the RR Full / RR Sparse SGD GPU solvers share the Rust implementations (`implementations/full/rr_gpu`, `implementations/sparse/sparse-sgd-gpu`, `shared/experiment-common`) compiled to `wasm32-unknown-unknown` (`wgpu` WebGPU backend + `wasm-bindgen`) as a WebAssembly package (`web/rr-lab/wasm` → `dist/pkg/`). Graph rendering also uses a WebGPU render pipeline to draw edges and circular nodes onto an `OffscreenCanvas` inside a dedicated Web Worker. Canvas 2D is used only to composite finished frames onto the page and as a fallback when WebGPU rendering is unavailable.

The **Active Graph** card in the left sidebar displays the source type (`Sample` or `Upload`), graph name, active vertex count, and undirected edge count. Merely changing the sample dropdown or switching the source radio button does not replace the currently loaded graph; the active graph switches only after a new graph finishes loading, and if loading fails, the previous graph and results are preserved intact.

The interface mirrors `tools/single_run` in the research repository and is managed directly within the same monorepo.

## Running Locally

```sh
npm run dev
```

Open http://localhost:4173. No runtime dependencies or build steps are required to serve `dist/`. A WebGPU-capable browser and a secure context (HTTPS or `localhost`) are required.

Select a sample and click **Load**, or upload a `.mtx` file. Loading a graph only parses the file and checks its scale—it does not generate initial coordinates, render on the GPU, or start SGD. Configure the method, iterations, epsilon, seed, and (for Sparse) pivot count, then click **Compute Layout** to generate the initial layout, run SGD, and render the final result. After a run completes, you can edit parameters and recalculate without resetting; the previous result remains visible until you start the next run. Pausing and resuming with unchanged parameters continues from the paused epoch, while changing parameters recalculates from the initial layout. Pause/resume, preprocessing cancellation, interactive Before/After enlargement, result text copying, and JSON/PNG downloads are supported. Render and export resolutions can be set to 512, 1024, 2048, or 4096 square pixels (default `4096 × 4096`), independent of CSS display size or device pixel ratio. Changing the resolution after calculation re-renders without altering coordinates or optimization results.

## Samples

| Graph          | Vertices | Undirected Edges | Default Method on Load     |
| -------------- | -------: | ---------------: | -------------------------- |
| Community      |      240 |              550 | Full                       |
| USpowerGrid    |    4,941 |            6,594 | Full (Sparse also enabled) |
| Luxembourg OSM |  114,599 |          119,666 | Sparse                     |

The `.mtx` files in `dist/samples/` are unmodified copies of the datasets in the repository's `data/` directory. No vertices are truncated or downsampled. See [samples/README.md](dist/samples/README.md) for dataset attribution and license details. For all inputs, the largest connected component (LCC) is extracted upon loading.

## Architecture and Data Flow

```text
wasm/                       WebAssembly / wgpu bridge for the Rust crates (rr_gpu, sparse-sgd-gpu)
dist/pkg/                   Prebuilt WASM binary and wasm-bindgen JS bindings
dist/wasm/index.js          Shared WASM loader for browser and Node
dist/app.js                 UI controller and calculation state coordinator
  ui/                       Settings validation, result formatting/export, Worker clients
  render/                   WebGPU renderer and WGSL vertex/fragment shaders
  workers/graph-worker.js   Worker entry point for loading, preprocessing, and stress evaluation
    graph/                  MTX streaming parser, LCC extraction, CSR/BFS, samples, stress evaluation
    preprocess/full.js      All-pairs u16 shortest-path distances and weight range (Rust WASM)
    preprocess/sparse.js    Pivot selection, O(N) region CSR, directed weights, u64 bitmask matching (Rust WASM)
  sgd/context.js            WebGPU initialization, buffer ownership, coordinate readback
  sgd/initialization.js     Seeded initial layout generation (shared Rust WASM StdRng)
  sgd/learning-rate.js      Exponential learning-rate schedule
  sgd/full-schedule.js      Two-stage Round-Robin tile schedule for Full SGD
  sgd/full-engine.js        Full GPU solver driver (Rust wgpu FullStepper)
  sgd/sparse-engine.js      Sparse GPU solver driver (Rust wgpu SparseStepper)
  sgd/full.wgsl             Full coordinate update shader (u16 4-bank distances, shared coordinates)
  sgd/sparse.wgsl           Sparse coordinate update shader (shared pivot cache, 20B GpuPair, 4-bank table)
```

Preprocessing and stress evaluation have no DOM or GPU dependencies. The SGD engines are decoupled from file loading and UI state, and all Worker communication is encapsulated in `ui/graph-client.js`. Large CPU preprocessing buffers are released once uploaded to the GPU. To rebuild `dist/pkg/` after modifying the Rust solvers, run `npm run build:wasm`.

## RR Full SGD and RR Sparse SGD

**RR Full SGD** executes the inter-tile and intra-tile Round-Robin schedule from `implementations/full/rr_gpu` (Phase 1A optimizations) via WebAssembly + `wgpu` (`TILE_SIZE = 128`, workgroup size `64`, shared memory coordinate caching). Every vertex pair is updated once per iteration with shuffled outer and inner schedules:

- Lossless `u16` all-pairs shortest-path distance storage (`2 bytes/pair`, packed two per `u32`) split across 4 storage buffer bindings (`dist_bank0..3`).
- Pre-packed static tile-pair tables, single `write_buffer` inner permutation uploads, dynamic uniform offsets, and batched `ComputePass` submissions (up to 256 dispatches per pass) to minimize CPU–GPU dispatch overhead.

**RR Sparse SGD** executes the pivot-based preprocessing and two-phase conflict-free updates from `implementations/sparse/sparse-sgd-gpu` (Phase 1B / Phase 3 optimizations) via WebAssembly + `wgpu`:

- Distance-proportional max-min pivot sampling with deterministic tie-breaking (preferring earlier pivots) and single-pass $O(N)$ CSR region construction (Phase 3).
- Directed weights $s_{ip} / d_{ip}^2$, with original graph edges overwritten to weight $1$ in both directions.
- **One-Sided Phase**: Each invocation owns one non-pivot vertex and cooperatively caches pivot permutations, vertex IDs, and coordinates in workgroup shared memory (`wg_pivot_perm`, `wg_pivot_id`, `wg_pivot_pos`) to reduce global memory traffic (Phase 1B). The one-sided table is split across 4 storage buffer banks (`one_sided_bank0..3`).
- **Two-Sided Phase**: Conflict-free pivot circle matchings combined with fast `u64` bitmask First-Fit edge coloring (Phase 1B), packed into 20-byte padding-free `GpuPair` entries and dispatched in batched `ComputePass` chunks.
- Per-vertex pivot ordering via a 4-round Feistel network with cycle walking, plus per-epoch shuffling of phase order, matching order, and shared-pivot order.
- Exponential learning-rate decay from $1 / w_{\min}$ to $\varepsilon / w_{\max}$ using `f32` coordinate updates without atomics or locks.

## Scale and Memory Limits

After loading a graph, the application compares the required distance matrix size against the WebGPU adapter limits. If RR Full SGD exceeds the limit, the Full option is disabled and the method automatically switches to RR Sparse SGD, displaying both the required memory and the adapter limit.

Fixed 4,096-vertex cutoffs have been removed. By splitting large tables across 4 storage buffer bindings, the solver supports up to `4 × 128 MiB = 512 MiB` on standard WebGPU adapters (and up to `4 GiB` on adapters reporting larger limits):

- **Full distance matrix**: $2 \times n^2$ bytes (`u16` packed). `USpowerGrid` requires ~46.5 MiB. Standard WebGPU environments (`128 MiB` per binding, `512 MiB` across 4 banks) support RR Full SGD up to $N = 16,384$ vertices. `Luxembourg OSM` would require ~24.5 GiB for Full, so the UI directs users to Sparse before allocation.
- **Sparse one-sided table**: $8 \times n \times k$ bytes across 4 banks (`Luxembourg OSM` with 200 pivots uses ~174.9 MiB, fitting well within the 512 MiB 4-bank limit).
- **Streaming MTX parser**: Parses arbitrarily large files line-by-line via `File.stream()`, storing endpoints in `Uint32Array`, extracting the largest connected component with Union-Find, and building CSR adjacency tables without fixed file-size cutoffs.
- **WebGPU rendering**: Renders all vertices as instanced circular quads and all edges by default (`Auto` edge thinning regularly subsamples graphs above 500,000 edges down to 100,000 displayed edges while using all edges for optimization).

All graphs are treated as undirected and unweighted with self-loops and duplicate edges removed. Matrix entry values are ignored, and uploaded files never leave the browser.

## Stress Evaluation and Timing

Full stress is defined as:

$$\sum_{i < j} \left(\frac{\|x_i - x_j\|_2 - d_{ij}}{d_{ij}}\right)^2$$

Graphs with $n \le 512$ are evaluated exactly across all vertex pairs. Larger graphs are estimated using uniformly sampled BFS sources from a fixed, independent seed (64 sources for $n \le 10,000$; up to 16 sources for $n > 10,000$, capped to keep distance buffers within 128 MiB) and scaled by $n / (2 \times \text{sources})$. Both Before and After layouts use the identical source set, and the evaluation mode (`exact` or `sampled`) is recorded in the UI and exported JSON.

Elapsed SGD time measures CPU wall-clock time across GPU epoch execution, coordinate readback, and After render requests, excluding preprocessing, GPU buffer initialization, stress evaluation, and final render worker completion wait (`renderWaitMs`).

## Verification and Formatting

```sh
npm install
npm test
npm run test:layout
npm run test:load
npm run test:browser
npm run format:check
npm run format
```

- `npm test`: Verifies MTX parsing, BFS, Full all-pairs coverage, Sparse weight parity against an independent all-pairs reference, conflict-free matchings, Feistel bijectivity, memory limits, and stress evaluation.
- `npm run test:layout`: Verifies responsive layout geometry across desktop and mobile viewports.
- `npm run test:load`: Verifies that selecting or dropping files only triggers graph loading until an explicit calculation action is taken.
- `npm run test:browser`: Launches headless Google Chrome with WebGPU enabled to verify CPU/GPU numerical parity (RMSE $< 10^{-3}$), `USpowerGrid` Full, `Luxembourg OSM` Sparse, Before image preservation, pause/resume, cancellation recovery, and 4096×4096 PNG export.

## Deployment to GitHub Pages

`.github/workflows/pages.yml` runs the test suite on pushes to `main` and deploys `dist/` directly to GitHub Pages. All HTML, ES module, Web Worker, WASM, WGSL, and sample asset paths are resolved relatively so the app works seamlessly under repository subpaths:

```sh
RR_TEST_BASE_PATH=/GPU-Accelerated-SGD-for-Stress-Minimization/ npm run test:browser
RR_TEST_BASE_PATH=/GPU-Accelerated-SGD-for-Stress-Minimization/ npm run test:load
```
