import init, { initSync } from '../pkg/rr_lab_wasm.js';
export * from '../pkg/rr_lab_wasm.js';

const wasmUrl = new URL('../pkg/rr_lab_wasm_bg.wasm', import.meta.url);
if (typeof process !== 'undefined' && process.versions?.node) {
  const { readFile } = await import('node:fs/promises');
  initSync({ module: await readFile(wasmUrl) });
} else {
  await init({ module_or_path: wasmUrl });
}
