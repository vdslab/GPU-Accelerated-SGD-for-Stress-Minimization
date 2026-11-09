import { edges } from '../dist/graph/edges.js';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseMTX } from '../dist/graph/mtx.js';
import { largestComponent } from '../dist/graph/components.js';
import { distances } from '../dist/graph/traversal.js';
import { schedule, selfRounds } from '../dist/sgd/full-schedule.js';
import { initialPositions } from '../dist/sgd/initialization.js';
import { rng } from '../dist/sgd/random.js';
import { stress } from './test-helpers.js';
test('MTX deduplicates, strips loops, extracts LCC and preserves original IDs', () => {
  const g = largestComponent(
    parseMTX(
      '%%MatrixMarket matrix coordinate real general\n% test\n5 5 5\n2 3 1\n3 2 1\n3 4 2\n4 4 1\n1 1 1',
    ),
  );
  assert.deepEqual([...g.ids], [2, 3, 4]);
  assert.deepEqual(
    [...edges(g)],
    [
      [0, 1],
      [1, 2],
    ],
  );
  const { dist, diameter } = distances(g);
  assert.equal(diameter, 2);
  assert.deepEqual([...dist], [0, 1, 2, 1, 0, 1, 2, 1, 0]);
});
test('rejects malformed input and permits components above 4096 vertices', () => {
  for (const text of [
    'wrong',
    '%%MatrixMarket matrix array real general\n2 2',
    '%%MatrixMarket matrix coordinate pattern general\n2 3 0',
    '%%MatrixMarket matrix coordinate pattern general\n2 2 1\n0 2',
    '%%MatrixMarket matrix coordinate pattern general\n2 2 2\n1 2',
  ])
    assert.throws(() => parseMTX(text));
  assert.equal(
    largestComponent({ n: 4097, edges: Array.from({ length: 4096 }, (_, i) => [i, i + 1]) }).n,
    4097,
  );
});
test('two-level RR covers every real pair exactly once with no concurrent writes', () => {
  for (const n of [2, 3, 63, 127, 128, 129, 240, 255, 256, 257, 387]) {
    const seen = new Set();
    for (const round of schedule(Math.ceil(n / 128))) {
      const blocks = new Set();
      for (const [i, j] of round) {
        assert(!blocks.has(i));
        blocks.add(i);
        if (i !== j) {
          assert(!blocks.has(j));
          blocks.add(j);
        }
        const li = Math.min(128, n - i * 128),
          lj = Math.min(128, n - j * 128),
          rounds = i === j ? selfRounds(li) : 128;
        for (let r = 0; r < rounds; r++) {
          const touched = new Set();
          const visit = (u, v) => {
            assert(!touched.has(u));
            assert(!touched.has(v));
            touched.add(u);
            touched.add(v);
            const key = [Math.min(u, v), Math.max(u, v)].join(',');
            assert(!seen.has(key));
            seen.add(key);
          };
          if (i !== j) {
            for (let u = 0; u < li; u++) {
              const v = (u + r) % 128;
              if (v < lj) visit(i * 128 + u, j * 128 + v);
            }
          } else {
            const effective = li + (li % 2);
            for (let k = 0; k < effective / 2; k++) {
              const u = k === 0 ? effective - 1 : (r + k) % rounds,
                v = k === 0 ? r : (r + rounds - k) % rounds;
              if (u < li && v < li) visit(i * 128 + u, i * 128 + v);
            }
          }
        }
      }
    }
    assert.equal(seen.size, (n * (n - 1)) / 2);
  }
});
test('seed reproducibility and zero stress at target distances', () => {
  assert.deepEqual(initialPositions(20, 42), initialPositions(20, 42));
  assert.notDeepEqual(initialPositions(20, 42), initialPositions(20, 43));
  assert.equal(rng(42)(), rng(42)());
  assert.equal(stress(new Float32Array([0, 0, 1, 0]), new Float32Array([0, 1, 1, 0]), 2), 0);
});
