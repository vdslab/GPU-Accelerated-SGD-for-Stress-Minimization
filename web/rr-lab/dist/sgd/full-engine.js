import { initialPositions } from './initialization.js';
import { WasmFullEngine, create_wasm_gpu_context } from '../wasm/index.js';

/** Two-level tile RR Full SGD via Rust wgpu WASM (u16 4-bank distance storage, static tile packing, batched ComputePass submissions). */
export class FullEngine {
  static async create(context, prepared, seed) {
    if (!context.wasmContext) {
      context.wasmContext = await create_wasm_gpu_context(
        context.limits.maxStorageBufferBindingSize,
        context.limits.maxBufferSize,
      );
    }
    const distU16 =
      prepared.dist instanceof Uint16Array ? prepared.dist : Uint16Array.from(prepared.dist);
    const initPos = initialPositions(prepared.n, seed >>> 0);
    const inner = new WasmFullEngine(context.wasmContext, prepared.n, distU16, initPos, seed >>> 0);
    return new FullEngine(context.wasmContext, inner);
  }
  constructor(wasmContext, inner) {
    this.wasmContext = wasmContext;
    this.inner = inner;
  }
  async epoch(eta) {
    return this.inner.epoch(this.wasmContext, eta);
  }
  dispose() {
    if (this.inner) {
      this.inner.free();
      this.inner = null;
    }
  }
}
