import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bindRunTrigger } from '../dist/ui/run-trigger.js';
test('Run requires an explicit fresh gesture for the currently loaded graph', () => {
  const listeners = new Map();
  const button = {
    disabled: false,
    addEventListener(type, fn) {
      listeners.set(type, fn);
    },
    matches() {
      return false;
    },
  };
  const graph = {},
    state = { phase: 'idle', graph };
  let starts = 0;
  bindRunTrigger(
    button,
    () => state,
    () => {
      starts++;
    },
  );
  const emit = (type, props = {}) =>
    listeners.get(type)({ type, isTrusted: true, detail: 1, button: 0, ...props });
  emit('click');
  assert.equal(starts, 0);
  emit('pointerdown');
  state.graph = {};
  emit('click');
  assert.equal(starts, 0);
  state.phase = 'loading';
  emit('pointerdown');
  state.phase = 'idle';
  emit('click');
  assert.equal(starts, 0);
  emit('pointerdown');
  emit('click', { isTrusted: false });
  assert.equal(starts, 0);
  emit('pointerdown');
  emit('click');
  assert.equal(starts, 1);
  emit('keydown', { key: 'Enter', repeat: true });
  emit('click', { detail: 0 });
  assert.equal(starts, 1);
  emit('keydown', { key: 'Enter', repeat: false });
  emit('click', { detail: 0 });
  assert.equal(starts, 2);
  emit('pointerdown');
  emit('blur');
  emit('click');
  assert.equal(starts, 2);
});
