import { rng, shuffle } from './random.js';
import { schedule, selfRounds } from './full-schedule.js';
import { initialPositions } from './initialization.js';
import { GpuBuffers, shaderModule } from './context.js';

/** Two-level tile RR. Every real pair is processed once per epoch. */
export class FullEngine extends GpuBuffers {
  static async create(context, prepared, seed) {
    const module = await shaderModule(context.device, new URL('./full.wgsl', import.meta.url));
    const pipeline = await context.device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const engine = new FullEngine(context.device);
    engine.pipeline = pipeline;
    try {
      engine.setup(prepared, seed);
      return engine;
    } catch (error) {
      engine.dispose();
      throw error;
    }
  }
  setup(prepared, seed) {
    const graph = { n: prepared.n },
      dist = prepared.dist;
    const d = this.device;
    this.n = graph.n;
    this.random = rng(seed);
    this.rounds = schedule(Math.ceil(this.n / 128));
    this.partial = this.n % 128 || 128;
    this.permutationLength = 128 + 127 + selfRounds(this.partial);
    if (
      dist.byteLength > d.limits.maxStorageBufferBindingSize ||
      dist.byteLength > d.limits.maxBufferSize
    )
      throw Error('距離行列がGPUのバッファ容量を超えています。より小さいグラフを選んでください。');
    this.positions = this.buffer(
      initialPositions(this.n, seed),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    this.distance = this.buffer(dist, GPUBufferUsage.STORAGE);
    this.uniform = this.buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.tiles = this.buffer(
      Math.max(...this.rounds.map((r) => r.length)) * 8,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    this.perm = this.buffer(
      this.permutationLength * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    this.positionBytes = this.n * 8;
    this.readback = this.buffer(
      this.positionBytes,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    this.bind = d.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [this.uniform, this.positions, this.distance, this.tiles, this.perm].map(
        (buffer, binding) => ({ binding, resource: { buffer } }),
      ),
    });
  }
  async epoch(eta) {
    const d = this.device;
    const uniform = new ArrayBuffer(16);
    new Uint32Array(uniform).set([this.n, 0, 255, 0]);
    new Float32Array(uniform)[1] = eta;
    d.queue.writeBuffer(this.uniform, 0, uniform);
    const order = shuffle(
      Array.from({ length: this.rounds.length }, (_, i) => i),
      this.random,
    );
    for (const index of order) {
      const round = this.rounds[index];
      const perm = new Uint32Array([
        ...shuffle(
          Array.from({ length: 128 }, (_, i) => i),
          this.random,
        ),
        ...shuffle(
          Array.from({ length: 127 }, (_, i) => i),
          this.random,
        ),
        ...shuffle(
          Array.from({ length: selfRounds(this.partial) }, (_, i) => i),
          this.random,
        ),
      ]);
      d.queue.writeBuffer(this.tiles, 0, new Uint32Array(round.flat()));
      d.queue.writeBuffer(this.perm, 0, perm);
      const encoder = d.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bind);
      pass.dispatchWorkgroups(round.length);
      pass.end();
      d.queue.submit([encoder.finish()]);
    }
    return this.readPositions();
  }
}
