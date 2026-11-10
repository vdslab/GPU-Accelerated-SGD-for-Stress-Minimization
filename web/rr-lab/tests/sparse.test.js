import { test } from 'node:test';
import assert from 'node:assert/strict';
import { prepareSparse, selectPivots } from '../dist/preprocess/sparse.js';
import { prepareFull, fullMemoryBytes } from '../dist/preprocess/full.js';
import { adjacency, bfs } from '../dist/graph/traversal.js';
import { prepareEvaluation, evaluateStress } from '../dist/graph/evaluation.js';
import { rng } from '../dist/sgd/random.js';
import { learningRate } from '../dist/sgd/learning-rate.js';
const limits = { maxStorageBufferBindingSize: 256 * 2 ** 20, maxBufferSize: 256 * 2 ** 20 };
const cycle = (n) => ({ n, edges: Array.from({ length: n }, (_, i) => [i, (i + 1) % n]) });

test('Sparse constraints match an independent all-pairs definition, including directional weights and edge overrides', () => {
  for (const [n, h] of [
    [7, 1],
    [7, 3],
    [7, 7],
    [17, 5],
    [129, 13],
  ]) {
    const graph = cycle(n),
      config = { seed: 42, pivots: h };
    const prepared = prepareSparse(graph, config, limits);
    const csr = adjacency(graph),
      selected = selectPivots(csr, h, rng(42));
    const all = Array.from({ length: n }, (_, s) => bfs(csr, s));
    const pivotSlots = new Map(Array.from(selected.pivots, (v, p) => [v, p]));
    const regions = Array.from({ length: n }, (_, v) => {
      let winner = 0;
      for (let p = 1; p < h; p++)
        if (all[v][selected.pivots[p]] < all[v][selected.pivots[winner]]) winner = p;
      return winner;
    });
    const edgeKeys = new Set(
      graph.edges.map(([u, v]) => [Math.min(u, v), Math.max(u, v)].join(',')),
    );
    const expected = new Map();
    const weight = (v, p) =>
      regions.filter(
        (region, x) =>
          region === p && all[selected.pivots[p]][x] <= Math.floor(all[v][selected.pivots[p]] / 2),
      ).length /
      all[v][selected.pivots[p]] ** 2;
    for (let u = 0; u < n; u++)
      for (let v = u + 1; v < n; v++) {
        const key = [u, v].join(','),
          edge = edgeKeys.has(key);
        if (!edge && !pivotSlots.has(u) && !pivotSlots.has(v)) continue;
        expected.set(key, [
          all[u][v],
          edge ? 1 : pivotSlots.has(v) ? weight(u, pivotSlots.get(v)) : 0,
          edge ? 1 : pivotSlots.has(u) ? weight(v, pivotSlots.get(u)) : 0,
        ]);
      }
    const seen = new Set();
    function check(u, v, d, wu, wv) {
      if (u > v) {
        [u, v] = [v, u];
        [wu, wv] = [wv, wu];
      }
      const key = [u, v].join(',');
      assert(!seen.has(key), 'duplicate ' + key);
      seen.add(key);
      const target = expected.get(key);
      assert(target, 'unexpected constraint ' + key);
      assert.equal(d, target[0]);
      assert(Math.abs(wu - target[1]) < 1e-6);
      assert(Math.abs(wv - target[2]) < 1e-6);
    }
    for (let v = 0; v < n; v++)
      for (let p = 0; p < h; p++) {
        const at = (v * h + p) * 2,
          d = prepared.oneSided[at],
          w = prepared.oneSided[at + 1];
        if (w > 0) {
          assert(!pivotSlots.has(v));
          check(v, prepared.pivots[p], d, w, 0);
        }
      }
    const ints = new Uint32Array(prepared.pairs.buffer),
      floats = new Float32Array(prepared.pairs.buffer);
    for (let r = 0; r < prepared.roundOffsets.length - 1; r++) {
      const touched = new Set();
      for (let i = prepared.roundOffsets[r]; i < prepared.roundOffsets[r + 1]; i++) {
        const at = i * 5,
          u = ints[at],
          v = ints[at + 1];
        assert(!touched.has(u));
        assert(!touched.has(v));
        touched.add(u);
        touched.add(v);
        check(u, v, floats[at + 2], floats[at + 3], floats[at + 4]);
      }
    }
    assert.equal(seen.size, expected.size);
    assert.equal(prepared.stats.constraints, expected.size);
  }
});
test('capacity checks use required bytes, not a fixed node cutoff', () => {
  assert.equal(fullMemoryBytes(4941), 48826962);
  assert.throws(() => prepareFull(cycle(114599), {}, limits), /RR Sparse/);
  assert.throws(
    () =>
      prepareSparse(
        cycle(100),
        { seed: 0, pivots: 50 },
        { maxStorageBufferBindingSize: 100, maxBufferSize: 100 },
      ),
    /pivots/,
  );
});
test('learning rate uses both directional weight extremes', () => {
  const p = { weightMin: 0.01, weightMax: 2 },
    config = { iterations: 3, epsilon: 0.1 };
  assert.equal(learningRate(p, config, 0), 100);
  assert(Math.abs(learningRate(p, config, 2) - 0.05) < 1e-12);
});
test('evaluation is exact on small graphs and deterministic/sampled on large graphs', () => {
  const graph = cycle(7),
    positions = new Float32Array(Array.from({ length: 14 }, (_, i) => i / 10));
  const evaluation = prepareEvaluation(graph);
  let exact = 0;
  for (let u = 0; u < 7; u++)
    for (let v = u + 1; v < 7; v++) {
      const d = Math.min(v - u, 7 - v + u);
      exact +=
        ((Math.hypot(
          positions[2 * u] - positions[2 * v],
          positions[2 * u + 1] - positions[2 * v + 1],
        ) -
          d) /
          d) **
        2;
    }
  assert(Math.abs(evaluateStress(positions, evaluation).value - exact) < 1e-10);
  const a = prepareEvaluation(cycle(513)),
    b = prepareEvaluation(cycle(513));
  assert.equal(a.sources.length, 64);
  assert.equal(a.exact, false);
  assert.deepEqual(a.sources, b.sources);
});

test('native Feistel pivot permutation has no duplicates for odd/even domains', async () => {
  const { shufflePivot } = await import('./sparse-reference.js');
  for (const count of [1, 2, 3, 5, 13, 128, 200, 257, 997])
    for (const key of [0, 42, 0xffffffff]) {
      const seen = new Set(Array.from({ length: count }, (_, i) => shufflePivot(i, key, count)));
      assert.equal(seen.size, count);
      assert([...seen].every((v) => v >= 0 && v < count));
    }
});
