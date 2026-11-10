import { initialPositions } from './initialization.js';
import {
  WasmAtomicSparseEngine,
  WasmSparseEngine,
  create_wasm_gpu_context,
} from '../wasm/index.js';

/** RR Sparse SGD via Rust wgpu WASM (shared-memory pivot cache, 20-byte GpuPair, 4-bank one_sided storage, batched ComputePass submissions). */
export class SparseEngine {
  static async create(context, prepared, seed) {
    if (!context.wasmContext) {
      context.wasmContext = await create_wasm_gpu_context(
        context.limits.maxStorageBufferBindingSize,
        context.limits.maxBufferSize,
      );
    }
    const initPos = initialPositions(prepared.n, seed >>> 0);
    const inner = new WasmSparseEngine(
      context.wasmContext,
      prepared.n,
      prepared.pivots,
      prepared.oneSided,
      prepared.pairs,
      prepared.roundOffsets,
      initPos,
      seed >>> 0,
    );
    return new SparseEngine(context.wasmContext, inner);
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

/** Atomic Sparse SGD via Rust wgpu WASM (vram-lock-sparse: ordered pair spinlock acquisition, 20-byte GpuPair, multi-bank 2D dispatch). */
export class AtomicSparseEngine {
  static async create(context, prepared, seed) {
    if (!context.wasmContext) {
      context.wasmContext = await create_wasm_gpu_context(
        context.limits.maxStorageBufferBindingSize,
        context.limits.maxBufferSize,
      );
    }
    const initPos = initialPositions(prepared.n, seed >>> 0);
    const inner = new WasmAtomicSparseEngine(
      context.wasmContext,
      prepared.n,
      prepared.pairs,
      initPos,
    );
    return new AtomicSparseEngine(context.wasmContext, inner);
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
