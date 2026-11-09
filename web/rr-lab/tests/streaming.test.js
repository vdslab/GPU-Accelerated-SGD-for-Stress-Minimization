import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseMTX, parseMTXStream } from '../dist/graph/mtx.js';
import { edges } from '../dist/graph/edges.js';
import { largestComponent } from '../dist/graph/components.js';
import { prepareSparse } from '../dist/preprocess/sparse.js';
test('streamed MTX preserves split headers, UTF-8 comments, CRLF, duplicates and the final unterminated line', async () => {
  const text =
    '%%MatrixMarket matrix coordinate real general\r\n% 日本語\r\n4 4 5\r\n1 2 1\r\n2 1 1\r\n2 3 1\r\n3 4 1\r\n4 4 1';
  const bytes = new TextEncoder().encode(text);
  const stream = new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += 7) controller.enqueue(bytes.slice(i, i + 7));
      controller.close();
    },
  });
  const streamed = await parseMTXStream(stream, bytes.length);
  assert.deepEqual(streamed, parseMTX(text));
  assert.deepEqual(
    [...edges(largestComponent(streamed))],
    [
      [0, 1],
      [1, 2],
      [2, 3],
    ],
  );
});
test('packed graph matching and constraints are identical to array-backed input', () => {
  const graph = { n: 17, edges: Array.from({ length: 17 }, (_, i) => [i, (i + 1) % 17]) };
  const limits = { maxStorageBufferBindingSize: 2 ** 28, maxBufferSize: 2 ** 28 },
    config = { pivots: 5, seed: 42 };
  const a = prepareSparse(graph, config, limits),
    b = prepareSparse({ n: 17, edgeData: Uint32Array.from(graph.edges.flat()) }, config, limits);
  assert.deepEqual(b, a);
});
test('stream input rejects incomplete and extra rows', async () => {
  for (const text of [
    '%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2',
    '%%MatrixMarket matrix coordinate pattern general\n3 3 1\n1 2\n2 3',
  ]) {
    const stream = new Blob([text]).stream();
    await assert.rejects(parseMTXStream(stream));
  }
});

test('large coordinate JSON uses bounded chunks and retains IDs, coordinates and all edges', async () => {
  const { layoutBlob, resultData } = await import('../dist/ui/results.js');
  const n = 17000,
    run = {
      config: { method: 'sparse' },
      source: 'fixture',
      view: { resolution: 4096, edgeLimit: 100000 },
      prepared: { stats: {} },
      graph: {
        n,
        ids: Uint32Array.from({ length: n }, (_, i) => i + 1),
        edgeData: Uint32Array.from({ length: (n - 1) * 2 }, (_, i) => Math.floor(i / 2) + (i % 2)),
      },
      positions: Float32Array.from({ length: n * 2 }, (_, i) => i * 0.125),
    };
  const actual = JSON.parse(await (await layoutBlob(run)).text());
  assert.deepEqual(actual, JSON.parse(JSON.stringify(resultData(run))));
});
test('matching slices cover every pair once across buffer boundaries in shuffled round order', async () => {
  const { matchingDispatches } = await import('../dist/sgd/sparse-schedule.js');
  const offsets = Uint32Array.from([0, 3, 3, 11, 14]);
  const slices = matchingDispatches(offsets, [3, 1, 0, 2], 4);
  const actual = slices.flatMap(({ chunk, start, count }) =>
    Array.from({ length: count }, (_, i) => chunk * 4 + start + i),
  );
  assert.deepEqual(actual, [11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert(slices.every((slice) => slice.start + slice.count <= 4));
});
