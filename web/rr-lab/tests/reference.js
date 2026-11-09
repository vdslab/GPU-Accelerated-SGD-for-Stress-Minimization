// Independent serial execution of the exact GPU matching order, for numeric QA.
import { rng, shuffle } from '../dist/sgd/random.js';
import { schedule, selfRounds } from '../dist/sgd/full-schedule.js';
import { initialPositions } from '../dist/sgd/initialization.js';
export function reference(graph, dist, diameter, { iterations, seed, epsilon }) {
  const p = initialPositions(graph.n, seed),
    random = rng(seed),
    n = graph.n,
    rounds = schedule(Math.ceil(n / 128)),
    partial = n % 128 || 128;
  function update(u, v, eta) {
    const dx = p[2 * v] - p[2 * u],
      dy = p[2 * v + 1] - p[2 * u + 1],
      len = Math.max(Math.hypot(dx, dy), 1e-12),
      d = dist[u * n + v],
      factor = (Math.min(eta / (d * d), 1) * (len - d) * 0.5) / len;
    const x = dx * factor,
      y = dy * factor;
    p[2 * u] += x;
    p[2 * u + 1] += y;
    p[2 * v] -= x;
    p[2 * v + 1] -= y;
  }
  for (let epoch = 0; epoch < iterations; epoch++) {
    const eta =
      iterations === 1
        ? diameter ** 2
        : diameter ** 2 * Math.exp((-Math.log(diameter ** 2 / epsilon) * epoch) / (iterations - 1));
    for (const ri of shuffle(
      rounds.map((_, i) => i),
      random,
    )) {
      const cross = shuffle(
          Array.from({ length: 128 }, (_, i) => i),
          random,
        ),
        full = shuffle(
          Array.from({ length: 127 }, (_, i) => i),
          random,
        ),
        last = shuffle(
          Array.from({ length: selfRounds(partial) }, (_, i) => i),
          random,
        );
      for (const [i, j] of rounds[ri]) {
        const li = Math.min(128, n - i * 128),
          lj = Math.min(128, n - j * 128);
        if (i !== j) {
          for (const r of cross)
            for (let u = 0; u < li; u++) {
              const v = (u + r) % 128;
              if (v < lj) update(i * 128 + u, j * 128 + v, eta);
            }
        } else {
          const effective = li + (li % 2),
            m = effective - 1;
          for (const r of li === 128 ? full : last)
            for (let k = 0; k < effective / 2; k++) {
              const u = k === 0 ? effective - 1 : (r + k) % m,
                v = k === 0 ? r : (r + m - k) % m;
              if (u < li && v < li) update(i * 128 + u, i * 128 + v, eta);
            }
        }
      }
    }
  }
  return p;
}
