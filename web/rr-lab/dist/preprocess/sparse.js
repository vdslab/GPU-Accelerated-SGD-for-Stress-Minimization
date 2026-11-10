import { multiBankCapacity } from '../graph/method-support.js';
import { wasm_prepare_sparse, wasm_select_pivots } from '../wasm/index.js';

/** Distance-proportional pivot sampling + inverted O(N) prefix counting via Rust sparse-sgd-gpu. */
export function selectPivots(csr, count, random, report = () => {}) {
  const n = csr.n,
    h = Math.min(count, n);
  const edges = [];
  for (let u = 0; u < n; u++) {
    for (let i = csr.offsets[u]; i < csr.offsets[u + 1]; i++) {
      const v = csr.neighbors[i];
      if (u < v) edges.push(u, v);
    }
  }
  const seed = random?.seed ?? 0;
  report(0, 'Selecting pivots and computing shortest paths');
  const selected = wasm_select_pivots(n, new Uint32Array(edges), h, seed >>> 0);
  const pivots = selected.take_pivots();
  const flatDistances = selected.take_flat_distances();
  selected.free();
  const rows = Array.from({ length: h }, (_, p) => flatDistances.subarray(p * n, (p + 1) * n));
  const nearest = new Uint32Array(n).fill(0xffffffff),
    regions = new Uint32Array(n);
  for (let p = 0; p < h; p++) {
    const row = rows[p];
    for (let v = 0; v < n; v++) {
      if (row[v] < nearest[v]) {
        nearest[v] = row[v];
        regions[v] = p;
      }
    }
  }
  const prefix = rows.map((row) => {
    let max = 0;
    for (const d of row) max = Math.max(max, d);
    return new Uint32Array(max + 1);
  });
  for (let v = 0; v < n; v++) {
    const reg = regions[v];
    prefix[reg][rows[reg][v]]++;
  }
  for (let p = 0; p < h; p++) {
    for (let i = 1; i < prefix[p].length; i++) prefix[p][i] += prefix[p][i - 1];
  }
  report(1, 'Selecting pivots and computing shortest paths');
  return { pivots, rows, regions, prefix };
}

export function sparseMemoryBytes(n, pivots) {
  return n * pivots * 8;
}

/** Rust sparse-sgd-gpu preprocessing: O(N) inverted prefix counts, u64 bitmask First-Fit coloring, and 20-byte GpuPair packing. */
export function prepareSparse(graph, config, limits, report = () => {}) {
  const n = graph.n,
    h = Math.min(config.pivots, n),
    bytes = sparseMemoryBytes(n, h);
  const max = multiBankCapacity(limits);
  if (bytes > max)
    throw new Error(
      `Sparse pivot table requires ${Math.ceil(bytes / 2 ** 20)} MiB. Please set pivots to at most ${Math.floor(max / (8 * n))} on this GPU.`,
    );
  const endpoints = graph.edgeData ?? Uint32Array.from(graph.edges.flat());
  report(0.1, 'Selecting pivots and computing shortest paths');
  const out = wasm_prepare_sparse(n, endpoints, h, (config.seed ?? 0) >>> 0);
  const pivots = out.take_pivots();
  const oneSided = out.take_one_sided();
  const pairs = out.take_pairs();
  const roundOffsets = out.take_round_offsets();
  const weightMin = out.wmin();
  const weightMax = out.wmax();
  const oneSidedCount = out.one_sided_count();
  const twoSidedCount = out.two_sided_count();
  const constraints = out.constraint_count();
  out.free();
  report(1, 'Building Sparse constraints and RR matchings');
  return {
    method: 'sparse',
    n,
    pivots,
    oneSided,
    pairs,
    roundOffsets,
    weightMin,
    weightMax,
    stats: {
      pivots: h,
      oneSidedCount,
      twoSidedCount,
      constraints,
      rounds: roundOffsets.length - 1,
      bufferBytes: bytes + pairs.byteLength,
    },
  };
}
