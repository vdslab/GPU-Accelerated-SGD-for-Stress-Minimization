import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  atomicFullMemoryBytes,
  fullMemoryBytes,
  graphMethodSupport,
  memoryLabel,
} from '../dist/graph/method-support.js';
const limits = { maxStorageBufferBindingSize: 256 * 2 ** 20, maxBufferSize: 256 * 2 ** 20 };
test('Full support follows u16 4-bank buffer limits, including the exact N=16384 boundary', () => {
  assert(graphMethodSupport(4941, limits).full.available);
  assert(graphMethodSupport(8192, limits).full.available);
  assert(graphMethodSupport(16384, limits).full.available);
  assert(!graphMethodSupport(16385, limits).full.available);
  assert(!graphMethodSupport(4941, { ...limits, maxBufferSize: 32 * 2 ** 20 }).full.available);
  assert(!graphMethodSupport(114599, limits).full.available);
  assert(graphMethodSupport(257, limits).full.available);
  assert(graphMethodSupport(4941, limits).atomic_full.available);
  assert(graphMethodSupport(8192, limits).atomic_full.available);
  assert(!graphMethodSupport(8193, limits).atomic_full.available);
  assert.equal(atomicFullMemoryBytes(257), 8 * 257 * 256);
});
test('large uploaded graphs get a feasible Sparse pivot ceiling without allocating a distance matrix', () => {
  const support = graphMethodSupport(914231, limits);
  assert.equal(support.full.requiredBytes, fullMemoryBytes(914231));
  assert.equal(support.full.available, false);
  assert.equal(support.atomic_full.available, false);
  assert.equal(support.sparse.maxPivots, 73);
  assert.equal(support.atomic_sparse.maxPivots, 29);
  assert.match(memoryLabel(support.full.requiredBytes), /TiB/);
  assert(!graphMethodSupport(4941, null).full.available);
  assert(!graphMethodSupport(4941, null).atomic_full.available);
});
