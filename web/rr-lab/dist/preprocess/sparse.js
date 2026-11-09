import { edgeCount, edgeAt } from '../graph/edges.js';
import { adjacency, bfs } from '../graph/traversal.js';
import { rng, shuffle } from '../sgd/random.js';

/** Distance-proportional pivot sampling, with the native first-pivot tie rule. */
export function selectPivots(csr, count, random, report = () => {}) {
  const n = csr.n,
    queue = new Uint32Array(n);
  const pivots = new Uint32Array(count),
    rows = new Array(count);
  const nearest = new Uint32Array(n).fill(0xffffffff),
    regions = new Uint32Array(n);
  let next = Math.floor(random() * n);
  for (let p = 0; p < count; p++) {
    pivots[p] = next;
    const row = (rows[p] = bfs(csr, next, queue));
    let total = 0;
    for (let v = 0; v < n; v++) {
      if (row[v] < nearest[v]) {
        nearest[v] = row[v];
        regions[v] = p;
      }
      total += nearest[v];
    }
    report((p + 1) / count, 'pivot選択・最短路を計算中');
    if (p + 1 === count) break;
    let ticket = Math.floor(random() * total);
    for (let v = 0; v < n; v++) {
      if (ticket < nearest[v]) {
        next = v;
        break;
      }
      ticket -= nearest[v];
    }
  }
  const prefix = rows.map((row, p) => {
    let max = 0;
    for (const d of row) max = Math.max(max, d);
    const counts = new Uint32Array(max + 1);
    for (let v = 0; v < n; v++) if (regions[v] === p) counts[row[v]]++;
    for (let i = 1; i < counts.length; i++) counts[i] += counts[i - 1];
    return counts;
  });
  return { pivots, rows, regions, prefix };
}

/** Circle matching of pivots, then first-fit insertion of original graph edges. */
export function sparseMatching(graph, selected, seed) {
  const { pivots, rows, prefix } = selected,
    h = pivots.length,
    m = edgeCount(graph);
  const slots = new Int32Array(graph.n).fill(-1);
  pivots.forEach((v, p) => (slots[v] = p));
  const effective = h + (h % 2),
    baseRounds = h > 1 ? effective - 1 : 0;
  const csr = adjacency(graph);
  const usageOffsets = new Uint32Array(graph.n + 1);
  let degreeMax = 0;
  for (let v = 0; v < graph.n; v++) {
    const degree = csr.offsets[v + 1] - csr.offsets[v];
    degreeMax = Math.max(degreeMax, degree);
    usageOffsets[v + 1] = usageOffsets[v] + degree + (slots[v] >= 0 ? h : 0);
  }
  const used = new Uint32Array(usageOffsets[graph.n]),
    usedCounts = new Uint32Array(graph.n);
  const bound = 2 * degreeMax + h + 1,
    marks = new Uint32Array(bound),
    counts = new Uint32Array(bound);
  const assigned = new Uint32Array(m).fill(0xffffffff),
    clique = [];
  let rounds = baseRounds;
  const add = (r, u, v) => {
    counts[r]++;
    used[usageOffsets[u] + usedCounts[u]++] = r;
    used[usageOffsets[v] + usedCounts[v]++] = r;
    rounds = Math.max(rounds, r + 1);
  };
  for (let r = 0; r < baseRounds; r++)
    for (let k = 0; k < effective / 2; k++) {
      const a = k === 0 ? effective - 1 : (r + k) % baseRounds,
        b = k === 0 ? r : (r + baseRounds - k) % baseRounds;
      if (a >= h || b >= h) continue;
      const d = rows[a][pivots[b]],
        wa = d === 1 ? 1 : prefix[b][Math.floor(d / 2)] / (d * d),
        wb = d === 1 ? 1 : prefix[a][Math.floor(d / 2)] / (d * d);
      clique.push([r, pivots[a], pivots[b], d, wa, wb]);
      add(r, pivots[a], pivots[b]);
    }
  const order = new Uint32Array(m);
  for (let i = 0; i < m; i++) order[i] = i;
  shuffle(order, rng(seed ^ 0x53504152));
  let generation = 0;
  for (const index of order) {
    const [u, v] = edgeAt(graph, index);
    if (slots[u] >= 0 && slots[v] >= 0) continue;
    generation++;
    for (let i = 0; i < usedCounts[u]; i++) marks[used[usageOffsets[u] + i]] = generation;
    for (let i = 0; i < usedCounts[v]; i++) marks[used[usageOffsets[v] + i]] = generation;
    let r = 0;
    while (r < rounds && marks[r] === generation) r++;
    assigned[index] = r;
    add(r, u, v);
  }
  const offsets = new Uint32Array(rounds + 1);
  for (let r = 0; r < rounds; r++) offsets[r + 1] = offsets[r] + counts[r];
  const buffer = new ArrayBuffer(Math.max(24, offsets.at(-1) * 24));
  const integers = new Uint32Array(buffer),
    floats = new Float32Array(buffer),
    cursor = offsets.slice();
  const write = (r, u, v, d, wu, wv) => {
    const at = cursor[r]++ * 6;
    integers[at] = u;
    integers[at + 1] = v;
    floats[at + 2] = d;
    floats[at + 3] = wu;
    floats[at + 4] = wv;
  };
  for (const pair of clique) write(...pair);
  for (const index of order) {
    if (assigned[index] === 0xffffffff) continue;
    const [u, v] = edgeAt(graph, index);
    write(assigned[index], u, v, 1, 1, 1);
  }
  return { pairs: new Uint8Array(buffer), roundOffsets: offsets, slots };
}

export function sparseMemoryBytes(n, pivots) {
  return n * pivots * 8;
}

/** No n² allocation: only n×h distances/constraints and graph/pivot matchings. */
export function prepareSparse(graph, config, limits, report = () => {}) {
  const n = graph.n,
    h = Math.min(config.pivots, n),
    bytes = sparseMemoryBytes(n, h);
  const max = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  if (bytes > max)
    throw new Error(
      `Sparseのpivot表に${Math.ceil(bytes / 2 ** 20)} MiB必要です。このGPUではpivotsを${Math.floor(max / (8 * n))}以下にしてください。`,
    );
  const csr = adjacency(graph);
  const selected = selectPivots(csr, h, rng(config.seed), (value, label) =>
    report(value * 0.7, label),
  );
  const { pivots, rows, prefix } = selected;
  const matching = sparseMatching(graph, selected, config.seed);
  const entries = new Float32Array(n * h * 2);
  let weightMin = 1,
    weightMax = 1,
    oneSidedCount = 0;
  for (let p = 0; p < h; p++) {
    const row = rows[p];
    for (let v = 0; v < n; v++) {
      const d = row[v];
      if (d <= 1) continue; // Original edges are two-sided, with both weights = 1.
      const weight = prefix[p][Math.floor(d / 2)] / (d * d);
      if (weight > 0) {
        weightMin = Math.min(weightMin, weight);
        weightMax = Math.max(weightMax, weight);
      }
      if (matching.slots[v] >= 0) continue; // Pivots never write in the one-sided phase.
      const at = (v * h + p) * 2;
      entries[at] = d;
      entries[at + 1] = weight;
      if (weight > 0) oneSidedCount++;
    }
    report(0.7 + (0.3 * (p + 1)) / h, 'Sparse制約・RR matchingを構築中');
  }
  return {
    method: 'sparse',
    n,
    pivots,
    oneSided: entries,
    pairs: matching.pairs,
    roundOffsets: matching.roundOffsets,
    weightMin,
    weightMax,
    stats: {
      pivots: h,
      oneSidedCount,
      twoSidedCount: matching.roundOffsets.at(-1),
      constraints: oneSidedCount + matching.roundOffsets.at(-1),
      rounds: matching.roundOffsets.length - 1,
      bufferBytes: bytes + matching.pairs.byteLength,
    },
  };
}
