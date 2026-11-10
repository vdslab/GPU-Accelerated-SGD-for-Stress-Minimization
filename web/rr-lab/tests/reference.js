import { wasm_reference_atomic_full, wasm_reference_full } from '../dist/wasm/index.js';

/** Independent serial execution of the exact Rust GPU matching order, for numeric QA. */
export function reference(graph, dist, diameter, { iterations, seed, epsilon }) {
  const distU16 = dist instanceof Uint16Array ? dist : Uint16Array.from(dist);
  return wasm_reference_full(graph.n, distU16, diameter, iterations, seed >>> 0, epsilon);
}

/** Independent serial execution of the Atomic Full SGD pair sequence, for numeric QA. */
export function atomicFullReference(graph, dist, diameter, { iterations, seed, epsilon }) {
  const distU16 = dist instanceof Uint16Array ? dist : Uint16Array.from(dist);
  return wasm_reference_atomic_full(graph.n, distU16, diameter, iterations, seed >>> 0, epsilon);
}
