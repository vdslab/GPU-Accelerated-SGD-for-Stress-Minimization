import { forEachEdge } from './edges.js';
import { wasm_prepare_full } from '../wasm/index.js';

/** Compact adjacency avoids allocating a JavaScript array for every BFS vertex. */
export function adjacency(graph) {
  if (graph.csr) return graph.csr;
  const offsets = new Uint32Array(graph.n + 1);
  forEachEdge(graph, (u, v) => {
    offsets[u + 1]++;
    offsets[v + 1]++;
  });
  for (let i = 1; i <= graph.n; i++) offsets[i] += offsets[i - 1];
  const neighbors = new Uint32Array(offsets[graph.n]);
  const cursor = offsets.slice();
  forEachEdge(graph, (u, v) => {
    neighbors[cursor[u]++] = v;
    neighbors[cursor[v]++] = u;
  });
  const csr = { offsets, neighbors, n: graph.n };
  Object.defineProperty(graph, 'csr', { value: csr, configurable: true });
  return csr;
}

/** Reuse the queue across traversals; the returned row belongs to the caller. */
export function bfs(csr, source, queue = new Uint32Array(csr.n)) {
  const row = new Uint32Array(csr.n).fill(0xffffffff);
  row[source] = 0;
  queue[0] = source;
  let head = 0,
    tail = 1;
  while (head < tail) {
    const u = queue[head++];
    for (let i = csr.offsets[u]; i < csr.offsets[u + 1]; i++) {
      const v = csr.neighbors[i];
      if (row[v] === 0xffffffff) {
        row[v] = row[u] + 1;
        queue[tail++] = v;
      }
    }
  }
  if (tail !== csr.n) throw new Error('距離計算には連結グラフが必要です。');
  return row;
}

export function distances(graph, onProgress = () => {}) {
  const endpoints = graph.edgeData ?? Uint32Array.from(graph.edges.flat());
  onProgress(0);
  const outcome = wasm_prepare_full(graph.n, endpoints);
  const dist = outcome.take_dist_u16();
  const diameter = outcome.dmax();
  const dmin = outcome.dmin();
  const constraintCount = outcome.constraint_count();
  outcome.free();
  onProgress(1);
  return { dist, diameter, dmin, constraintCount };
}
