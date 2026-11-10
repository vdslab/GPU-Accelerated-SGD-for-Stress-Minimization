import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runIntent, sameSolverSettings } from '../dist/ui/run-state.js';
const config = { method: 'sparse', pivots: 200, seed: 0, epsilon: 0.1, iterations: 15 };
const paused = () => ({
  engine: {},
  config: { ...config },
  epoch: 5,
  history: [{ epoch: 0 }, { epoch: 5 }],
  failed: false,
});

test('a completed layout can run again without reset', () => {
  const run = { ...paused(), epoch: 15 };
  const intent = runIntent(run, { ...config });
  assert.equal(intent.resume, false);
  assert.equal(intent.label, 'Run Again');
});
test('only an unchanged paused layout resumes its existing solver', () => {
  const run = paused();
  assert.equal(runIntent(run, { ...config }).resume, true);
  for (const change of [
    { seed: 1 },
    { iterations: 30 },
    { pivots: 100 },
    { epsilon: 0.2 },
    { method: 'full' },
  ]) {
    const intent = runIntent(run, { ...config, ...change });
    assert.equal(intent.resume, false);
    assert.equal(intent.label, 'Recalculate');
  }
  // Restoring a draft restores the option to resume; the saved run was never mutated.
  assert.equal(runIntent(run, { ...config }).resume, true);
  assert.equal(run.epoch, 5);
});
test('failed or disposed solvers restart, and irrelevant Full pivot settings do not invalidate a pause', () => {
  assert.equal(runIntent({ ...paused(), failed: true }, config).resume, false);
  assert.equal(runIntent({ ...paused(), engine: null }, config).resume, false);
  assert(
    sameSolverSettings({ ...config, method: 'full' }, { ...config, method: 'full', pivots: 17 }),
  );
  assert.equal(
    runIntent({ engine: null, config: null, epoch: 0, history: [], failed: false }, config).label,
    'Compute Layout',
  );
});
