import { distances } from '../graph/traversal.js';
import {
  atomicFullMemoryBytes,
  fullMemoryBytes,
  graphMethodSupport,
} from '../graph/method-support.js';
export { atomicFullMemoryBytes, fullMemoryBytes } from '../graph/method-support.js';

/** Enforce the adapter's actual storage limit before allocating a lossless u16 n² matrix or explicit pair banks. */
export function prepareFull(graph, config, limits, report = () => {}) {
  const isAtomic = config?.method === 'atomic_full';
  const method = isAtomic ? 'atomic_full' : 'full';
  const bytes = isAtomic ? atomicFullMemoryBytes(graph.n) : fullMemoryBytes(graph.n);
  const support = graphMethodSupport(graph.n, limits);
  const methodSupport = isAtomic ? support.atomic_full : support.full;
  if (!methodSupport.available) {
    throw new Error(
      `${isAtomic ? 'Atomic Full' : 'Full RR'} requires ${(bytes / 2 ** 30).toFixed(2)} GiB, exceeding this GPU's ${Math.floor(methodSupport.limit / 2 ** 20)} MiB limit. Please select RR Sparse or Atomic Sparse.`,
    );
  }
  report(0, 'Computing all-pairs shortest paths');
  const { dist, diameter, dmin, constraintCount } = distances(graph, (value) =>
    report(value, 'Computing all-pairs shortest paths'),
  );
  return {
    method,
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
