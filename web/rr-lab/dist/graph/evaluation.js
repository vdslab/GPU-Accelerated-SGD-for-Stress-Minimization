import { adjacency, bfs } from './traversal.js';
import { rng } from '../sgd/random.js';

/** Fixed independent sources let Before/After use the identical full-stress estimator. */
export function prepareEvaluation(graph) {
  const count = Math.min(
    graph.n <= 512 ? graph.n : graph.n <= 10000 ? 64 : 16,
    Math.max(1, Math.floor((128 * 2 ** 20) / (4 * graph.n))),
  );
  const sources = new Uint32Array(count),
    chosen = new Set(),
    random = rng(0x53545245);
  for (let i = 0; i < count; i++) {
    let v = count === graph.n ? i : Math.floor(random() * graph.n);
    while (chosen.has(v)) v = Math.floor(random() * graph.n);
    chosen.add(v);
    sources[i] = v;
  }
  const csr = adjacency(graph),
    queue = new Uint32Array(graph.n),
    rows = new Uint32Array(count * graph.n);
  sources.forEach((source, i) => rows.set(bfs(csr, source, queue), i * graph.n));
  return { sources, rows, n: graph.n, exact: count === graph.n };
}
export function evaluateStress(positions, evaluation) {
  const { sources, rows, n } = evaluation;
  let sum = 0;
  for (let i = 0; i < sources.length; i++) {
    const u = sources[i];
    for (let v = 0; v < n; v++) {
      if (u === v) continue;
      const target = rows[i * n + v];
      const error =
        (Math.hypot(
          positions[u * 2] - positions[v * 2],
          positions[u * 2 + 1] - positions[v * 2 + 1],
        ) -
          target) /
        target;
      sum += error * error;
    }
  }
  return {
    value: (sum * n) / (2 * sources.length),
    exact: evaluation.exact,
    sources: sources.length,
    sourceIds: Array.from(sources),
  };
}
