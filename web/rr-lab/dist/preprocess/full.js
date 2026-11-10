import { distances } from '../graph/traversal.js';
import { fullMemoryBytes, graphMethodSupport } from '../graph/method-support.js';
export { fullMemoryBytes } from '../graph/method-support.js';

/** Enforce the adapter's actual storage limit before allocating a lossless u16 n² matrix. */
export function prepareFull(graph, config, limits, report = () => {}) {
  const bytes = fullMemoryBytes(graph.n);
  const support = graphMethodSupport(graph.n, limits);
  if (!support.full.available) {
    throw new Error(
      `Full RR distance matrix requires ${(bytes / 2 ** 30).toFixed(2)} GiB, exceeding this GPU's ${Math.floor(support.full.limit / 2 ** 20)} MiB limit. Please select RR Sparse.`,
    );
  }
  report(0, 'Computing all-pairs shortest paths');
  const { dist, diameter, dmin, constraintCount } = distances(graph, (value) =>
    report(value, 'Computing all-pairs shortest paths'),
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
