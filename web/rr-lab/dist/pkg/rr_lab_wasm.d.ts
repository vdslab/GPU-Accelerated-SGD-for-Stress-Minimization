/* tslint:disable */
/* eslint-disable */

export class WasmAtomicFullEngine {
  free(): void;
  [Symbol.dispose](): void;
  epoch(ctx: WasmGpuContext, eta: number): Promise<Float32Array>;
  constructor(
    ctx: WasmGpuContext,
    n: number,
    dist_u16: Uint16Array | number[],
    initial_positions: Float32Array | number[],
    seed: number,
  );
  reset(ctx: WasmGpuContext): void;
}

export class WasmAtomicSparseEngine {
  free(): void;
  [Symbol.dispose](): void;
  epoch(ctx: WasmGpuContext, eta: number): Promise<Float32Array>;
  constructor(
    ctx: WasmGpuContext,
    n: number,
    pairs_u8: Uint8Array | number[],
    initial_positions: Float32Array | number[],
  );
  reset(ctx: WasmGpuContext): void;
}

export class WasmAtomicSparsePrepared {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  constraint_count(): number;
  take_pairs(): Uint8Array;
  take_pivots(): Uint32Array;
  wmax(): number;
  wmin(): number;
}

export class WasmFullEngine {
  free(): void;
  [Symbol.dispose](): void;
  epoch(ctx: WasmGpuContext, eta: number): Promise<Float32Array>;
  constructor(
    ctx: WasmGpuContext,
    n: number,
    dist_u16: Uint16Array | number[],
    initial_positions: Float32Array | number[],
    seed: number,
  );
  reset(ctx: WasmGpuContext): void;
}

export class WasmFullPrepared {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  constraint_count(): number;
  dmax(): number;
  dmin(): number;
  take_dist_u16(): Uint16Array;
}

export class WasmGpuContext {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
}

export class WasmSelectedPivots {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  take_flat_distances(): Uint32Array;
  take_pivots(): Uint32Array;
}

export class WasmSparseEngine {
  free(): void;
  [Symbol.dispose](): void;
  epoch(ctx: WasmGpuContext, eta: number): Promise<Float32Array>;
  constructor(
    ctx: WasmGpuContext,
    n: number,
    pivots: Uint32Array | number[],
    one_sided_f32: Float32Array | number[],
    pairs_u8: Uint8Array | number[],
    round_offsets: Uint32Array | number[],
    initial_positions: Float32Array | number[],
    seed: number,
  );
  reset(ctx: WasmGpuContext): void;
}

export class WasmSparsePrepared {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  constraint_count(): number;
  one_sided_count(): number;
  take_one_sided(): Float32Array;
  take_pairs(): Uint8Array;
  take_pivots(): Uint32Array;
  take_round_offsets(): Uint32Array;
  two_sided_count(): number;
  wmax(): number;
  wmin(): number;
}

export function create_wasm_gpu_context(
  max_storage_buffer_binding_size: number,
  max_buffer_size: number,
): Promise<WasmGpuContext>;

export function wasm_hash32(value: number): number;

export function wasm_initial_positions(n: number, seed: number): Float32Array;

export function wasm_prepare_atomic_sparse(
  n: number,
  edge_data: Uint32Array | number[],
  pivots: number,
  seed: number,
): WasmAtomicSparsePrepared;

export function wasm_prepare_full(n: number, edge_data: Uint32Array | number[]): WasmFullPrepared;

export function wasm_prepare_sparse(
  n: number,
  edge_data: Uint32Array | number[],
  pivots: number,
  seed: number,
): WasmSparsePrepared;

export function wasm_reference_atomic_full(
  n: number,
  dist_u16: Uint16Array | number[],
  diameter: number,
  iterations: number,
  seed: number,
  epsilon: number,
): Float32Array;

export function wasm_reference_atomic_sparse(
  n: number,
  pairs_u8: Uint8Array | number[],
  weight_min: number,
  weight_max: number,
  iterations: number,
  seed: number,
  epsilon: number,
): Float32Array;

export function wasm_reference_full(
  n: number,
  dist_u16: Uint16Array | number[],
  diameter: number,
  iterations: number,
  seed: number,
  epsilon: number,
): Float32Array;

export function wasm_reference_sparse(
  n: number,
  pivots: Uint32Array | number[],
  one_sided_f32: Float32Array | number[],
  pairs_u8: Uint8Array | number[],
  round_offsets: Uint32Array | number[],
  weight_min: number,
  weight_max: number,
  iterations: number,
  seed: number,
  epsilon: number,
): Float32Array;

export function wasm_select_pivots(
  n: number,
  edge_data: Uint32Array | number[],
  pivots: number,
  seed: number,
): WasmSelectedPivots;

export function wasm_shuffle_pivot(index: number, key: number, count: number): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmatomicfullengine_free: (a: number, b: number) => void;
  readonly __wbg_wasmatomicsparseengine_free: (a: number, b: number) => void;
  readonly __wbg_wasmatomicsparseprepared_free: (a: number, b: number) => void;
  readonly __wbg_wasmfullengine_free: (a: number, b: number) => void;
  readonly __wbg_wasmfullprepared_free: (a: number, b: number) => void;
  readonly __wbg_wasmgpucontext_free: (a: number, b: number) => void;
  readonly __wbg_wasmselectedpivots_free: (a: number, b: number) => void;
  readonly __wbg_wasmsparseengine_free: (a: number, b: number) => void;
  readonly __wbg_wasmsparseprepared_free: (a: number, b: number) => void;
  readonly create_wasm_gpu_context: (a: number, b: number) => any;
  readonly wasm_hash32: (a: number) => number;
  readonly wasm_initial_positions: (a: number, b: number) => [number, number];
  readonly wasm_prepare_atomic_sparse: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
  ) => [number, number, number];
  readonly wasm_prepare_full: (a: number, b: number, c: number) => [number, number, number];
  readonly wasm_prepare_sparse: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
  ) => [number, number, number];
  readonly wasm_reference_atomic_full: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
    f: number,
    g: number,
  ) => [number, number];
  readonly wasm_reference_atomic_sparse: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
    f: number,
    g: number,
    h: number,
  ) => [number, number];
  readonly wasm_reference_full: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
    f: number,
    g: number,
  ) => [number, number];
  readonly wasm_reference_sparse: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
    f: number,
    g: number,
    h: number,
    i: number,
    j: number,
    k: number,
    l: number,
    m: number,
    n: number,
  ) => [number, number];
  readonly wasm_select_pivots: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
  ) => [number, number, number];
  readonly wasm_shuffle_pivot: (a: number, b: number, c: number) => number;
  readonly wasmatomicfullengine_epoch: (a: number, b: number, c: number) => any;
  readonly wasmatomicfullengine_new: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
    f: number,
    g: number,
  ) => [number, number, number];
  readonly wasmatomicfullengine_reset: (a: number, b: number) => void;
  readonly wasmatomicsparseengine_epoch: (a: number, b: number, c: number) => any;
  readonly wasmatomicsparseengine_new: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
    f: number,
  ) => [number, number, number];
  readonly wasmatomicsparseengine_reset: (a: number, b: number) => void;
  readonly wasmatomicsparseprepared_constraint_count: (a: number) => number;
  readonly wasmatomicsparseprepared_take_pairs: (a: number) => [number, number];
  readonly wasmatomicsparseprepared_take_pivots: (a: number) => [number, number];
  readonly wasmatomicsparseprepared_wmax: (a: number) => number;
  readonly wasmatomicsparseprepared_wmin: (a: number) => number;
  readonly wasmfullengine_epoch: (a: number, b: number, c: number) => any;
  readonly wasmfullengine_new: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
    f: number,
    g: number,
  ) => [number, number, number];
  readonly wasmfullengine_reset: (a: number, b: number) => void;
  readonly wasmfullprepared_constraint_count: (a: number) => number;
  readonly wasmfullprepared_dmax: (a: number) => number;
  readonly wasmfullprepared_dmin: (a: number) => number;
  readonly wasmfullprepared_take_dist_u16: (a: number) => [number, number];
  readonly wasmselectedpivots_take_flat_distances: (a: number) => [number, number];
  readonly wasmselectedpivots_take_pivots: (a: number) => [number, number];
  readonly wasmsparseengine_epoch: (a: number, b: number, c: number) => any;
  readonly wasmsparseengine_new: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
    f: number,
    g: number,
    h: number,
    i: number,
    j: number,
    k: number,
    l: number,
    m: number,
  ) => [number, number, number];
  readonly wasmsparseengine_reset: (a: number, b: number) => void;
  readonly wasmsparseprepared_constraint_count: (a: number) => number;
  readonly wasmsparseprepared_one_sided_count: (a: number) => number;
  readonly wasmsparseprepared_take_one_sided: (a: number) => [number, number];
  readonly wasmsparseprepared_take_pairs: (a: number) => [number, number];
  readonly wasmsparseprepared_take_pivots: (a: number) => [number, number];
  readonly wasmsparseprepared_take_round_offsets: (a: number) => [number, number];
  readonly wasmsparseprepared_two_sided_count: (a: number) => number;
  readonly wasmsparseprepared_wmax: (a: number) => number;
  readonly wasmsparseprepared_wmin: (a: number) => number;
  readonly wasm_bindgen_7e5fa5872b8dba1c___convert__closures_____invoke___js_sys_cf86f6e409df1d5___Function_fn_wasm_bindgen_7e5fa5872b8dba1c___JsValue_____wasm_bindgen_7e5fa5872b8dba1c___sys__Undefined___js_sys_cf86f6e409df1d5___Function_fn_wasm_bindgen_7e5fa5872b8dba1c___JsValue_____wasm_bindgen_7e5fa5872b8dba1c___sys__Undefined_______true_: (
    a: number,
    b: number,
    c: any,
    d: any,
  ) => void;
  readonly wasm_bindgen_7e5fa5872b8dba1c___convert__closures_____invoke___wasm_bindgen_7e5fa5872b8dba1c___JsValue__core_9b3796e30d99ddb7___result__Result_____wasm_bindgen_7e5fa5872b8dba1c___JsError___true_: (
    a: number,
    b: number,
    c: any,
  ) => [number, number];
  readonly wasm_bindgen_7e5fa5872b8dba1c___convert__closures_____invoke___wasm_bindgen_7e5fa5872b8dba1c___JsValue______true_: (
    a: number,
    b: number,
    c: any,
  ) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init(
  module_or_path?:
    { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>,
): Promise<InitOutput>;
