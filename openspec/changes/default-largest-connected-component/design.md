## Context

Sparse SGD requires finite shortest-path distances between all vertices it lays out. SuiteSparse matrices are frequently nonconnected after diagonal entries are removed and nonzero off-diagonal entries are interpreted as undirected edges. Requiring a special invocation for the common case makes ordinary `cargo run` fail even when a useful dominant graph component exists.

The CPU and GPU executables must make the same graph-selection decision before pivot preprocessing, scheduling, or layout starts.

## Goals / Non-Goals

**Goals:**

- Every normal CLI invocation uses the largest connected component automatically.
- Existing input-path and SGD parameter arguments remain valid without additions.
- The terminal always states exactly what fraction of the input is being laid out.
- Result files retain enough provenance to relate internal vertices to original Matrix Market indices.

**Non-Goals:**

- Drawing every disconnected component in a single layout.
- Adding synthetic edges between components.
- Treating matrix values as weighted graph edges.
- Changing the `gpu_visualizer` parser or its input format.

## Decisions

### Select the largest component during normal graph preparation

After Matrix Market parsing and undirected edge normalization, the common graph-loading path enumerates components in `O(n+m)` and returns the largest component. The selected vertices are reindexed to `0..k-1`; a `local_id -> original_id` map is retained. Equal-size components are resolved deterministically by the smallest original vertex ID.

This moves the decision before both CPU and GPU preprocessing, guaranteeing equal input graphs. Retaining the old error-by-default behavior was rejected because it conflicts with the requested ordinary `cargo run` workflow.

### Always report selection statistics on the terminal

Before execution, both executables print a compact, explicit summary:

```text
Input graph: nodes=N, edges=M, components=C
Largest component used: nodes=k/N (p%), edges=e/M (q%)
```

The summary is printed even when `k=N`, so a successful command always makes the selection policy visible.

### Preserve command-line compatibility and emit provenance

Existing positional input, `--input`, `--pivots`, `--iterations`, `--epsilon`, `--seed`, and centering options keep their meanings. If a historical `--largest-component` flag is accepted, it is a no-op compatibility alias because selection is now unconditional.

Each output prefix saves a vertex-map sidecar and each initial/final result records source and selected graph statistics plus its map path. This avoids silently presenting a small component as if it were the full input.

### Fail only when the selected component cannot support layout

Malformed input still errors. A largest component with fewer than two vertices or no non-self-loop edges errors with the selection statistics; otherwise layout proceeds regardless of the number of discarded components.

## Risks / Trade-offs

- [A very small component is drawn without a command-line opt-in] → terminal and result metadata always show retained counts and percentages.
- [Users rely on the old nonconnected-input error] → retain a clear statistics line; offer a no-op alias for the old selection flag rather than rejecting existing scripts.
- [CPU/GPU graph mismatch] → implement the selection in shared graph preparation and test both executables against the same fixture.
- [Large vertex maps add output size] → use one sidecar per output prefix rather than duplicating the mapping in both result files.
