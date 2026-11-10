import { distances } from '../graph/traversal.js';
import { fullMemoryBytes, graphMethodSupport } from '../graph/method-support.js';
export { fullMemoryBytes } from '../graph/method-support.js';

/** Enforce the adapter's actual storage limit before allocating a lossless u16 n² matrix. */
export function prepareFull(graph, config, limits, report = () => {}) {
  const bytes = fullMemoryBytes(graph.n);
  const support = graphMethodSupport(graph.n, limits);
  if (!support.full.available) {
    throw new Error(
      `Full RRの距離行列に${(bytes / 2 ** 30).toFixed(2)} GiB必要です。このGPUの上限は${Math.floor(support.full.limit / 2 ** 20)} MiBです。RR Sparseを選択してください。`,
    );
  }
  report(0, '全点対最短路を計算中');
  const { dist, diameter, dmin, constraintCount } = distances(graph, (value) =>
    report(value, '全点対最短路を計算中'),
  );
  return {
    method: 'full',
    n: graph.n,
    dist,
    weightMin: 1 / (diameter * diameter),
    weightMax: 1 / (dmin * dmin),
    stats: {
      constraints: constraintCount ?? (graph.n * (graph.n - 1)) / 2,
      bufferBytes: bytes,
      diameter,
    },
  };
}
