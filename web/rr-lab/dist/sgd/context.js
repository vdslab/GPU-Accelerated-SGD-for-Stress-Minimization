const BUFFER_BUDGET = 1024 * 2 ** 20;

/** Native browser WebGPU API initialization. */
export async function createGpuContext() {
  if (!navigator.gpu) throw new Error('WebGPUを利用できません。対応ブラウザで開いてください。');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPUアダプターが見つかりません。');
  const maxStorageBufferBindingSize = Math.min(
    adapter.limits.maxStorageBufferBindingSize,
    BUFFER_BUDGET,
  );
  const maxBufferSize = Math.min(adapter.limits.maxBufferSize, BUFFER_BUDGET);
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize, maxBufferSize },
  });
  return {
    device,
    info: adapter.info,
    isFallbackAdapter: adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null,
    limits: {
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxBufferSize: device.limits.maxBufferSize,
    },
  };
}
export async function shaderModule(device, url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('WGSLシェーダーを読み込めませんでした。');
  const module = device.createShaderModule({ code: await response.text(), label: url.pathname });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length)
    throw new Error(errors.map((error) => `${error.lineNum}: ${error.message}`).join('\n'));
  return module;
}

/** Buffer ownership and readback are shared; scheduling stays in each solver. */
export class GpuBuffers {
  constructor(device) {
    this.device = device;
    this.buffers = [];
  }
  buffer(data, usage, label) {
    const size = typeof data === 'number' ? data : data.byteLength;
    const limit =
      usage & GPUBufferUsage.STORAGE
        ? Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize)
        : this.device.limits.maxBufferSize;
    if (size > limit) {
      throw new Error(
        `${label ?? 'GPU buffer'} requires ${Math.ceil(size / 2 ** 20)} MiB; GPU limit: ${Math.floor(limit / 2 ** 20)} MiB.`,
      );
    }
    const buffer = this.device.createBuffer({
      label,
      size: Math.max(4, size),
      usage,
      mappedAtCreation: typeof data !== 'number',
    });
    if (typeof data !== 'number') {
      new Uint8Array(buffer.getMappedRange()).set(
        new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
      );
      buffer.unmap();
    }
    this.buffers.push(buffer);
    return buffer;
  }
  setupPositions(positions) {
    this.positionBytes = positions.byteLength;
    this.positions = this.buffer(
      positions,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      'positions',
    );
    this.readback = this.buffer(
      positions.byteLength,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      'position readback',
    );
  }
  async readPositions() {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.positions, 0, this.readback, 0, this.positionBytes);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const positions = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    return positions;
  }
  dispose() {
    for (const buffer of this.buffers) buffer.destroy();
    this.buffers = [];
  }
}
