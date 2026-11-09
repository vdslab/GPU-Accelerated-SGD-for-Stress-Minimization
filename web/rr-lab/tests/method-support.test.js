import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fullMemoryBytes, graphMethodSupport, memoryLabel } from '../dist/graph/method-support.js';
const limits = { maxStorageBufferBindingSize: 256 * 2 ** 20, maxBufferSize: 256 * 2 ** 20 };
test('Full support follows both buffer limits, including the exact boundary', () => {
  assert(graphMethodSupport(4941, limits).full.available);
  assert(graphMethodSupport(8192, limits).full.available);
  assert(!graphMethodSupport(8193, limits).full.available);
  assert(!graphMethodSupport(4941, { ...limits, maxBufferSize: 64 * 2 ** 20 }).full.available);
  assert(!graphMethodSupport(114599, limits).full.available);
  assert(graphMethodSupport(257, limits).full.available);
});
test('large uploaded graphs get a feasible Sparse pivot ceiling without allocating a distance matrix', () => {
  const support = graphMethodSupport(914231, limits);
  assert.equal(support.full.requiredBytes, fullMemoryBytes(914231));
  assert.equal(support.full.available, false);
  assert.equal(support.sparse.maxPivots, 36);
  assert.match(memoryLabel(support.full.requiredBytes), /TiB/);
  assert(!graphMethodSupport(4941, null).full.available);
});
