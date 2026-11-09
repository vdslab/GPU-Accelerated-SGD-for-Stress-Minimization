import { createGpuContext, GpuBuffers, shaderModule } from '../sgd/context.js';

const EDGE_BATCH = 131072;
const NODE_BATCH = 262144;

/** Graph geometry stays on the GPU. Camera changes upload only a 48-byte uniform. */
export class GpuGraphRenderer extends GpuBuffers {
  static async create(graph, distances) {
    const context = await createGpuContext();
    const renderer = new GpuGraphRenderer(context.device);
    renderer.adapterInfo = {
      vendor: context.info?.vendor,
      architecture: context.info?.architecture,
      isFallbackAdapter: context.isFallbackAdapter,
    };
    try {
      await renderer.initialize(graph, distances);
      return renderer;
    } catch (error) {
      renderer.dispose();
      context.device.destroy();
      throw error;
    }
  }
  async initialize(graph, distances) {
    this.n = graph.n;
    this.maxDistance = 0;
    for (const distance of distances) this.maxDistance = Math.max(this.maxDistance, distance);
    this.distanceBuffer = this.buffer(distances, GPUBufferUsage.STORAGE, 'render topology colors');
    this.positionSlots = new Map();
    this.uniform = this.buffer(
      48,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      'render transform',
    );
    this.uniformData = new ArrayBuffer(48);
    this.floats = new Float32Array(this.uniformData);
    this.integers = new Uint32Array(this.uniformData);
    const endpoints = graph.edgeData ?? Uint32Array.from(graph.edges.flat());
    this.m = endpoints.length / 2;
    const chunkPairs = Math.max(
      1,
      Math.floor(
        Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize) /
          8,
      ),
    );
    this.edgeChunks = [];
    for (let start = 0; start < this.m; start += chunkPairs) {
      const end = Math.min(this.m, start + chunkPairs);
      const buffer = this.buffer(
        endpoints.subarray(start * 2, end * 2),
        GPUBufferUsage.STORAGE,
        'render edge endpoints',
      );
      this.edgeChunks.push({ start, count: end - start, buffer });
    }
    this.canvas = new OffscreenCanvas(4096, 4096);
    this.context = this.canvas.getContext('webgpu');
    if (!this.context) throw Error('OffscreenCanvasのWebGPU描画を利用できません。');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    const module = await shaderModule(this.device, new URL('./graph.wgsl', import.meta.url));
    this.bindLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', minBindingSize: 48 },
        },
        ...[1, 2, 3].map((binding) => ({
          binding,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage' },
        })),
      ],
    });
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.bindLayout] });
    const blend = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    };
    const pipeline = (entry) =>
      this.device.createRenderPipelineAsync({
        layout,
        vertex: { module, entryPoint: entry + '_vertex' },
        fragment: {
          module,
          entryPoint: entry + '_fragment',
          targets: [{ format: this.format, blend }],
        },
        primitive: { topology: 'triangle-list' },
      });
    [this.edgePipeline, this.nodePipeline] = await Promise.all([
      pipeline('edge'),
      pipeline('node'),
    ]);
    this.device.lost.then((info) => {
      this.failure = Error(info.message || '描画GPUの接続が失われました。');
    });
    this.device.addEventListener('uncapturederror', (event) => {
      this.failure = event.error;
    });
  }
  slot(job) {
    for (const slot of this.positionSlots.values()) if (slot.id === job.positionId) return slot;
    if (!job.positions) throw Error('描画座標のキャッシュが失われました。');
    const key = job.viewId === 'before' ? 'before' : 'after';
    let slot = this.positionSlots.get(key);
    if (!slot) {
      slot = {
        buffer: this.buffer(
          this.n * 8,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          'render positions',
        ),
      };
      this.positionSlots.set(key, slot);
    }
    this.device.queue.writeBuffer(slot.buffer, 0, job.positions);
    slot.id = job.positionId;
    return slot;
  }
  uniforms(job, stride, base = 0, count = 0) {
    const scale = job.box.scale * job.view.zoom;
    this.floats.set([
      scale,
      512 + job.view.x - job.box.x * scale,
      512 + job.view.y - job.box.y * scale,
      job.style.radius,
      job.style.edgeWidth,
      this.maxDistance,
      job.style.edgeOpacity,
      1024 / job.resolution,
    ]);
    this.integers.set([this.n, stride, base, count], 8);
    this.device.queue.writeBuffer(this.uniform, 0, this.uniformData);
  }
  bind(positions, edges) {
    return this.device.createBindGroup({
      layout: this.bindLayout,
      entries: [this.uniform, positions, edges, this.distanceBuffer].map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });
  }
  async render(job) {
    if (this.failure) throw this.failure;
    const started = performance.now();
    let bitmap;
    this.device.pushErrorScope('validation');
    try {
      const uploadStarted = performance.now(),
        slot = this.slot(job);
      const uploadMs = performance.now() - uploadStarted;
      if (this.resolution !== job.resolution) {
        this.canvas.width = this.canvas.height = job.resolution;
        this.context.configure({
          device: this.device,
          format: this.format,
          alphaMode: 'opaque',
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        this.resolution = job.resolution;
      }
      const texture = this.context.getCurrentTexture().createView();
      const stride =
        job.edgeLimit && this.m > 500000 ? Math.max(1, Math.ceil(this.m / job.edgeLimit)) : 1;
      let first = true,
        drawCalls = 0,
        drawnEdges = 0;
      const submit = async (pipeline, bind, count, start) => {
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: texture,
              loadOp: first ? 'clear' : 'load',
              storeOp: 'store',
              clearValue: { r: 1, g: 1, b: 1, a: 1 },
            },
          ],
        });
        first = false;
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.draw(6, count, 0, start);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        drawCalls++;
        // Bounded submissions avoid one long draw monopolizing the device on dense graphs.
        await this.device.queue.onSubmittedWorkDone();
        if (this.failure) throw this.failure;
      };
      for (const chunk of this.edgeChunks) {
        const local = Math.ceil(chunk.start / stride) * stride - chunk.start;
        const count = Math.max(0, Math.ceil((chunk.count - local) / stride));
        this.uniforms(job, stride, local, chunk.count);
        const bind = this.bind(slot.buffer, chunk.buffer);
        for (let start = 0; start < count; start += EDGE_BATCH) {
          const size = Math.min(EDGE_BATCH, count - start);
          await submit(this.edgePipeline, bind, size, start);
          drawnEdges += size;
        }
      }
      this.uniforms(job, stride);
      const bind = this.bind(slot.buffer, this.edgeChunks[0].buffer);
      for (let start = 0; start < this.n; start += NODE_BATCH)
        await submit(this.nodePipeline, bind, Math.min(NODE_BATCH, this.n - start), start);
      bitmap = this.canvas.transferToImageBitmap();
      return {
        bitmap,
        backend: 'webgpu',
        retainedPositionIds: [...this.positionSlots.values()].map((slot) => slot.id),
        stats: {
          renderMs: performance.now() - started,
          uploadMs,
          drawCalls,
          drawnEdges,
          drawnNodes: this.n,
          adapterInfo: this.adapterInfo,
        },
      };
    } finally {
      const error = await this.device.popErrorScope();
      if (error) {
        bitmap?.close();
        throw Error(error.message);
      }
    }
  }
}
