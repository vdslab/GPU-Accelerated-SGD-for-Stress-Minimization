export function learningRate(prepared, config, epoch) {
  const start = 1 / prepared.weightMin,
    end = config.epsilon / prepared.weightMax;
  return config.iterations === 1
    ? start
    : start * Math.exp((-Math.log(start / end) * epoch) / (config.iterations - 1));
}
export const METHOD_LABELS = {
  full: 'RR Full SGD',
  sparse: 'RR Sparse SGD',
  atomic_full: 'Atomic Full SGD',
  atomic_sparse: 'Atomic Sparse SGD',
};

export function isSparseMethod(method) {
  return method === 'sparse' || method === 'atomic_sparse';
}

export function isFullMethod(method) {
  return method === 'full' || method === 'atomic_full';
}
