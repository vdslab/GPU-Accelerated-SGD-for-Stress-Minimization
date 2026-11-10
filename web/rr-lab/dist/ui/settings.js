import { edgeCount } from '../graph/edges.js';
import { runIntent } from './run-state.js';
import { graphMethodSupport, memoryLabel } from '../graph/method-support.js';
import { isSparseMethod } from '../sgd/learning-rate.js';

export const element = (id) => document.getElementById(id);
export function readSettings() {
  const config = {
    method: element('method').value,
    iterations: Number(element('iterations').value),
    epsilon: Number(element('epsilon').value),
    seed: Number(element('seed').value),
    pivots: Number(element('pivots').value),
  };
  if (!['full', 'sparse', 'atomic_full', 'atomic_sparse'].includes(config.method))
    throw new Error('Please select a method.');
  if (element('method').selectedOptions[0]?.disabled)
    throw new Error('The selected method is currently unavailable.');
  if (!Number.isInteger(config.iterations) || config.iterations < 1 || config.iterations > 500)
    throw new Error('Iterations must be between 1 and 500.');
  if (!Number.isInteger(config.seed) || config.seed < 0 || config.seed > 4294967295)
    throw new Error('Seed must be an integer between 0 and 4294967295.');
  if (!Number.isFinite(config.epsilon) || config.epsilon < 0.0001 || config.epsilon > 1)
    throw new Error('Epsilon must be between 0.0001 and 1.');
  const maxPivots = Number(element('pivots').max);
  if (
    isSparseMethod(config.method) &&
    (!Number.isInteger(config.pivots) || config.pivots < 1 || config.pivots > maxPivots)
  )
    throw new Error(`Pivots must be between 1 and ${maxPivots}.`);
  return config;
}
export function setStatus(message, error = false) {
  element('status').textContent = message;
  element('status').classList.toggle('error', error);
}
export function setProgress(value) {
  element('progress').value = value;
  element('percent').textContent = `${Math.round(value * 100)}%`;
}
/** Re-evaluate whenever the loaded graph or GPU changes, and restore Full for smaller graphs. */
export function updateMethodSupport(run) {
  const fullOption = element('method').querySelector('[value="full"]');
  const sparseOption = element('method').querySelector('[value="sparse"]');
  const atomicFullOption = element('method').querySelector('[value="atomic_full"]');
  const atomicSparseOption = element('method').querySelector('[value="atomic_sparse"]');
  const note = element('method-support');
  const pivotLimit = element('pivot-limit');
  if (!run.graph || !run.context) {
    fullOption.disabled = !run.context;
    sparseOption.disabled = !run.context;
    if (atomicFullOption) atomicFullOption.disabled = !run.context;
    if (atomicSparseOption) atomicSparseOption.disabled = !run.context;
    fullOption.textContent = 'RR Full SGD';
    if (atomicFullOption) atomicFullOption.textContent = 'Atomic Full SGD';
    note.hidden = true;
    pivotLimit.textContent = '';
    element('pivots').max = '1000';
    return;
  }
  const support = graphMethodSupport(run.graph.n, run.context.limits);
  fullOption.disabled = !support.full.available;
  sparseOption.disabled = support.sparse.maxPivots < 1;
  if (atomicFullOption) {
    atomicFullOption.disabled = !support.atomic_full.available;
    atomicFullOption.textContent = support.atomic_full.available
      ? 'Atomic Full SGD'
      : 'Atomic Full SGD (Unavailable)';
  }
  if (atomicSparseOption) {
    atomicSparseOption.disabled = support.atomic_sparse.maxPivots < 1;
  }
  fullOption.textContent = support.full.available ? 'RR Full SGD' : 'RR Full SGD (Unavailable)';
  const switchedFromFull = !support.full.available && element('method').value === 'full';
  const switchedFromAtomicFull =
    !support.atomic_full.available && element('method').value === 'atomic_full';
  if (switchedFromFull) element('method').value = 'sparse';
  if (switchedFromAtomicFull) element('method').value = 'atomic_sparse';
  if (!support.full.available) {
    note.hidden = false;
    note.textContent = `Full unavailable: distance matrix ${memoryLabel(support.full.requiredBytes)} > limit ${memoryLabel(support.full.limit)}. Use Sparse.`;
  } else if (!support.atomic_full.available) {
    note.hidden = false;
    note.textContent = `Atomic Full unavailable: pair table ${memoryLabel(support.atomic_full.requiredBytes)} > limit ${memoryLabel(support.atomic_full.limit)}. Use Sparse.`;
  } else {
    note.hidden = true;
  }
  const maxPivots =
    element('method').value === 'atomic_sparse'
      ? support.atomic_sparse.maxPivots
      : support.sparse.maxPivots;
  element('pivots').max = String(maxPivots);
  if (
    (switchedFromFull || switchedFromAtomicFull) &&
    maxPivots >= 1 &&
    Number(element('pivots').value) > maxPivots
  ) {
    element('pivots').value = String(maxPivots);
  }
  pivotLimit.textContent = maxPivots < 1000 ? `(max ${maxPivots})` : '';
}

export function updateControls(run) {
  updateMethodSupport(run);
  const busy = run.phase !== 'idle';
  for (const id of ['method', 'iterations', 'epsilon', 'seed', 'resolution'])
    element(id).disabled = busy;
  element('method').disabled = busy || !run.context;
  element('pivots').disabled = busy || !isSparseMethod(element('method').value);
  for (const id of ['file', 'sample', 'load-sample', 'source-sample', 'source-upload'])
    element(id).disabled = busy;
  let valid = true;
  try {
    const intent = runIntent(run, readSettings());
    element('run').textContent = intent.label;
    element('run-hint').textContent = intent.hint;
    element('run').title = intent.hint;
    element('run-hint').hidden = intent.label !== 'Recalculate';
  } catch (error) {
    valid = false;
    element('run-hint').hidden = false;
    element('run').textContent = 'Compute Layout';
    element('run-hint').textContent = error.message;
  }
  element('run').disabled = busy || !run.context || !run.graph || !valid;
  element('run-hint').classList.toggle('error', !valid);
  element('pause').disabled = run.phase !== 'running' || run.pauseRequested;
  element('reset').disabled = busy || !run.graph || !run.view.initial;
  element('cancel').disabled = !['loading', 'preparing'].includes(run.phase);
  for (const id of ['expand-before', 'expand-after', 'png'])
    element(id).disabled = busy || !run.positions;
  for (const id of ['json', 'copy-result', 'copy-json'])
    element(id).disabled = busy || !run.history.length;
  element('json').disabled ||= Boolean(run.exporting);
  const large = run.graph && run.graph.n + edgeCount(run.graph) > 2000000;
  element('copy-json').disabled ||= Boolean(large);
  element('copy-json').title = large ? 'For large graphs, use "Save Layout JSON" instead.' : '';
}
