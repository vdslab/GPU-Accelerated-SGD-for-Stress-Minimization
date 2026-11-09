export function learningRate(prepared, config, epoch) {
  const start = 1 / prepared.weightMin,
    end = config.epsilon / prepared.weightMax;
  return config.iterations === 1
    ? start
    : start * Math.exp((-Math.log(start / end) * epoch) / (config.iterations - 1));
}
export const METHOD_LABELS = { full: 'RR Full SGD', sparse: 'RR Sparse SGD' };
