import { rng } from './random.js';

/** Same seeded initial positions for both solvers and Before rendering. */
export function initialPositions(n, seed) {
  const random = rng(seed),
    p = new Float32Array(n * 2);
  for (let i = 0; i < p.length; i++) p[i] = random() - 0.5;
  return p;
}
