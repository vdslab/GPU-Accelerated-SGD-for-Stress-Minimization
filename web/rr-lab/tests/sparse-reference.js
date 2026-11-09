import { rng, shuffle } from '../dist/sgd/random.js';
import { initialPositions } from '../dist/sgd/initialization.js';
import { learningRate } from '../dist/sgd/learning-rate.js';
export function hash32(value) {
  let x = value >>> 0;
  x = Math.imul((x >>> 16) ^ x, 0x45d9f3b) >>> 0;
  x = Math.imul((x >>> 16) ^ x, 0x45d9f3b) >>> 0;
  return ((x >>> 16) ^ x) >>> 0;
}
export function shufflePivot(index, key, count) {
  if (count <= 1) return 0;
  let remaining = count - 1,
    bits = 0;
  while (remaining) {
    bits++;
    remaining >>>= 2;
  }
  const mask = (1 << bits) - 1;
  let value = index;
  do {
    let left = value >>> bits,
      right = value & mask;
    for (let round = 0; round < 4; round++) {
      const next = left ^ (hash32((right ^ key ^ Math.imul(round + 1, 0x85ebca6b)) >>> 0) & mask);
      left = right;
      right = next;
    }
    value = (left << bits) | right;
  } while (value >= count);
  return value;
}
export function sparseReference(prepared, config) {
  const { n, pivots, oneSided, pairs, roundOffsets } = prepared,
    h = pivots.length,
    random = rng(config.seed),
    p = initialPositions(n, config.seed);
  const ints = new Uint32Array(pairs.buffer),
    floats = new Float32Array(pairs.buffer);
  function delta(x, y, qx, qy, d, salt) {
    let dx = qx - x,
      dy = qy - y,
      len = Math.hypot(dx, dy);
    if (len < 1e-12) {
      const a = (hash32(salt) % 6283) * 0.001;
      dx = Math.cos(a) * 1e-6;
      dy = Math.sin(a) * 1e-6;
      len = Math.hypot(dx, dy);
    }
    const factor = (len - d) / (2 * len);
    return [dx * factor, dy * factor];
  }
  for (let iteration = 0; iteration < config.iterations; iteration++) {
    const eta = learningRate(prepared, config, iteration),
      order = shuffle(
        Array.from({ length: roundOffsets.length - 1 }, (_, i) => i),
        random,
      ),
      permutation = shuffle(
        Array.from({ length: h }, (_, i) => i),
        random,
      );
    function one() {
      for (let v = 0; v < n; v++) {
        let x = p[v * 2],
          y = p[v * 2 + 1],
          changed = false;
        for (let k = 0; k < h; k++) {
          const slot =
              permutation[
                shufflePivot(k, (v ^ config.seed ^ Math.imul(iteration, 0x9e3779b9)) >>> 0, h)
              ],
            at = (v * h + slot) * 2,
            w = oneSided[at + 1];
          if (w <= 0) continue;
          const pivot = pivots[slot],
            d = delta(
              x,
              y,
              p[pivot * 2],
              p[pivot * 2 + 1],
              oneSided[at],
              v ^ pivot ^ config.seed ^ iteration,
            ),
            mu = Math.min(w * eta, 1);
          x = Math.fround(x + mu * d[0]);
          y = Math.fround(y + mu * d[1]);
          changed = true;
        }
        if (changed) {
          p[v * 2] = x;
          p[v * 2 + 1] = y;
        }
      }
    }
    function two() {
      for (const r of order)
        for (let index = roundOffsets[r]; index < roundOffsets[r + 1]; index++) {
          const at = index * 6,
            u = ints[at],
            v = ints[at + 1],
            d = delta(
              p[u * 2],
              p[u * 2 + 1],
              p[v * 2],
              p[v * 2 + 1],
              floats[at + 2],
              u ^ v ^ config.seed ^ iteration,
            ),
            mu = Math.min(floats[at + 3] * eta, 1),
            mv = Math.min(floats[at + 4] * eta, 1);
          p[u * 2] += mu * d[0];
          p[u * 2 + 1] += mu * d[1];
          p[v * 2] -= mv * d[0];
          p[v * 2 + 1] -= mv * d[1];
        }
    }
    if (random() < 0.5) {
      one();
      two();
    } else {
      two();
      one();
    }
  }
  return p;
}
