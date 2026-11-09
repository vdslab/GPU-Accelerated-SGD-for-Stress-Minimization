import { edgeCount } from '../graph/edges.js';
import { runIntent } from './run-state.js';
import { graphMethodSupport, memoryLabel } from '../graph/method-support.js';

export const element = (id) => document.getElementById(id);
export function readSettings() {
  const config = {
    method: element('method').value,
    iterations: Number(element('iterations').value),
    epsilon: Number(element('epsilon').value),
    seed: Number(element('seed').value),
    pivots: Number(element('pivots').value),
  };
  if (!['full', 'sparse'].includes(config.method)) throw new Error('手法を選択してください。');
  if (element('method').selectedOptions[0]?.disabled)
    throw new Error('選択した手法は現在利用できません。');
  if (!Number.isInteger(config.iterations) || config.iterations < 1 || config.iterations > 500)
    throw new Error('反復回数は1–500です。');
  if (!Number.isInteger(config.seed) || config.seed < 0 || config.seed > 4294967295)
    throw new Error('seedは0–4294967295の整数です。');
  if (!Number.isFinite(config.epsilon) || config.epsilon < 0.0001 || config.epsilon > 1)
    throw new Error('epsilonは0.0001–1です。');
  const maxPivots = Number(element('pivots').max);
  if (
    config.method === 'sparse' &&
    (!Number.isInteger(config.pivots) || config.pivots < 1 || config.pivots > maxPivots)
  )
    throw new Error(`pivotsは1–${maxPivots}で指定してください。`);
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
  const note = element('method-support');
  const pivotLimit = element('pivot-limit');
  if (!run.graph || !run.context) {
    fullOption.disabled = !run.context;
    sparseOption.disabled = !run.context;
    fullOption.textContent = 'RR Full SGD';
    note.hidden = true;
    pivotLimit.textContent = '';
    element('pivots').max = '1000';
    return;
  }
  const support = graphMethodSupport(run.graph.n, run.context.limits);
  fullOption.disabled = !support.full.available;
  sparseOption.disabled = support.sparse.maxPivots < 1;
  fullOption.textContent = support.full.available ? 'RR Full SGD' : 'RR Full SGD（利用不可）';
  if (!support.full.available && element('method').value === 'full')
    element('method').value = 'sparse';
  note.hidden = support.full.available;
  note.textContent = `Full不可：距離行列 ${memoryLabel(support.full.requiredBytes)} > 上限 ${memoryLabel(support.full.limit)}。Sparseを使用してください。`;
  element('pivots').max = String(support.sparse.maxPivots);
  pivotLimit.textContent =
    support.sparse.maxPivots < 1000 ? `（上限 ${support.sparse.maxPivots}）` : '';
}

export function updateControls(run) {
  updateMethodSupport(run);
  const busy = run.phase !== 'idle';
  for (const id of ['method', 'iterations', 'epsilon', 'seed', 'resolution'])
    element(id).disabled = busy;
  element('method').disabled = busy || !run.context;
  element('pivots').disabled = busy || element('method').value !== 'sparse';
  for (const id of ['file', 'sample', 'load-sample', 'source-sample', 'source-upload'])
    element(id).disabled = busy;
  let valid = true;
  try {
    const intent = runIntent(run, readSettings());
    element('run').textContent = intent.label;
    element('run-hint').textContent = intent.hint;
    element('run').title = intent.hint;
    element('run-hint').hidden = intent.label !== 'この設定で再計算';
  } catch (error) {
    valid = false;
    element('run-hint').hidden = false;
    element('run').textContent = 'レイアウトを計算';
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
  element('copy-json').title = large
    ? '大規模グラフのJSONは「座標JSONを保存」から保存してください。'
    : '';
}
