import { wasm_initial_positions } from '../wasm/index.js';

/** Same seeded initial positions (via Rust experiment_common::seed::seeded_positions) for both solvers and Before rendering. */
export function initialPositions(n, seed) {
  return wasm_initial_positions(n, seed >>> 0);
}
