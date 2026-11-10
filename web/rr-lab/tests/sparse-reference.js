import { wasm_hash32, wasm_reference_sparse, wasm_shuffle_pivot } from '../dist/wasm/index.js';

export function hash32(value) {
  return wasm_hash32(value >>> 0) >>> 0;
}

export function shufflePivot(index, key, count) {
  return wasm_shuffle_pivot(index >>> 0, key >>> 0, count >>> 0) >>> 0;
}

export function sparseReference(prepared, config) {
  return wasm_reference_sparse(
    prepared.n,
    prepared.pivots,
    prepared.oneSided,
    prepared.pairs,
    prepared.roundOffsets,
    prepared.weightMin,
    prepared.weightMax,
    config.iterations,
    (config.seed ?? 0) >>> 0,
    config.epsilon,
  );
}
