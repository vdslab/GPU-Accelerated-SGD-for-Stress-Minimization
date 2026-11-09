import { matchingDispatches } from './sparse-schedule.js';
import { GpuBuffers, shaderModule } from './context.js';
import { rng, shuffle } from './random.js';
import { initialPositions } from './initialization.js';

const UNIFORM_ALIGNMENT = 256;
const WORKGROUP_SIZE = 256;

export class SparseEngine extends GpuBuffers {
  static async create(context, prepared, seed) {
    const device = context.device;
    const module = await shaderModule(device, new URL('./sparse.wgsl', import.meta.url));
    const bindLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 32 },
        },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ...[2, 3, 4].map((binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' },
        })),
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });
    const pipelines = await Promise.all(
      ['one_sided_phase', 'two_sided_round'].map((entryPoint) =>
        device.createComputePipelineAsync({ layout, compute: { module, entryPoint } }),
      ),
    );
    const engine = new SparseEngine(device);
    [engine.onePipeline, engine.twoPipeline] = pipelines;
    try {
      engine.setup(prepared, seed, bindLayout);
      return engine;
    } catch (error) {
      engine.dispose();
      throw error;
    }
  }
  setup(prepared, seed, layout) {
    this.n = prepared.n;
    this.h = prepared.pivots.length;
    this.offsets = prepared.roundOffsets;
    this.seed = seed;
    this.iteration = 0;
    this.random = rng(seed);
    this.setupPositions(initialPositions(this.n, seed));
    this.pairChunkCount = Math.max(
      1,
      Math.floor(
        Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize) /
          24,
      ),
    );
    const chunkCount = Math.max(1, Math.ceil(prepared.pairs.byteLength / 24 / this.pairChunkCount));
    this.uniform = this.buffer(
      (this.offsets.length + chunkCount + 4) * UNIFORM_ALIGNMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      'sparse dispatch uniforms',
    );
    this.oneSided = this.buffer(prepared.oneSided, GPUBufferUsage.STORAGE, 'one-sided constraints');
    this.pairBuffers = [];
    for (let chunk = 0; chunk < chunkCount; chunk++) {
      this.pairBuffers.push(
        this.buffer(
          prepared.pairs.subarray(
            chunk * this.pairChunkCount * 24,
            (chunk + 1) * this.pairChunkCount * 24,
          ),
          GPUBufferUsage.STORAGE,
          'two-sided matching constraints',
        ),
      );
    }
    this.pivotData = new Uint32Array(this.h * 2);
    this.pivotData.set(prepared.pivots, this.h);
    this.pivotPermutation = this.buffer(
      this.pivotData,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      'pivot permutation + IDs',
    );
    this.binds = this.pairBuffers.map((pairs) =>
      this.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform, size: 32 } },
          ...[this.positions, this.oneSided, this.pivotPermutation, pairs].map((buffer, i) => ({
            binding: i + 1,
            resource: { buffer },
          })),
        ],
      }),
    );
  }
  async epoch(eta) {
    const roundOrder = shuffle(
      Array.from({ length: this.offsets.length - 1 }, (_, r) => r),
      this.random,
    );
    this.pivotData.set(
      shuffle(
        Array.from({ length: this.h }, (_, p) => p),
        this.random,
      ),
    );
    const one = { one: true, chunk: 0, start: 0, count: this.h };
    const rounds = matchingDispatches(this.offsets, roundOrder, this.pairChunkCount);
    const dispatches = this.random() < 0.5 ? [one, ...rounds] : [...rounds, one];
    const uniforms = new ArrayBuffer(dispatches.length * UNIFORM_ALIGNMENT);
    const u32 = new Uint32Array(uniforms),
      f32 = new Float32Array(uniforms);
    dispatches.forEach((dispatch, i) => {
      const at = (i * UNIFORM_ALIGNMENT) / 4;
      u32.set(
        [this.n, dispatch.start, dispatch.count, this.iteration, 0, this.seed, this.h, 0],
        at,
      );
      f32[at + 4] = eta;
    });
    this.device.queue.writeBuffer(this.uniform, 0, uniforms);
    this.device.queue.writeBuffer(this.pivotPermutation, 0, this.pivotData);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatches.forEach((dispatch, i) => {
      if (!dispatch.count) return;
      pass.setPipeline(dispatch.one ? this.onePipeline : this.twoPipeline);
      pass.setBindGroup(0, this.binds[dispatch.chunk], [i * UNIFORM_ALIGNMENT]);
      pass.dispatchWorkgroups(Math.ceil((dispatch.one ? this.n : dispatch.count) / WORKGROUP_SIZE));
    });
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.iteration++;
    return this.readPositions();
  }
}
