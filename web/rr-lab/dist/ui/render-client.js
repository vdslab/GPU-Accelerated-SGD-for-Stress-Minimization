/** Keep rasterization off the UI thread and coalesce frames while SGD advances. */
export class RenderClient {
  constructor(graph, colorDistances, onFrame) {
    this.graph = graph;
    this.onFrame = onFrame;
    this.positionIds = new WeakMap();
    this.nextPositionId = 0;
    this.retainedPositions = new Set();
    this.colorDistances = colorDistances;
    this.queue = new Map();
    this.active = null;
    this.waiters = [];
    this.spawn();
  }
  spawn() {
    const worker = (this.worker = new Worker(
      new URL('../workers/render-worker.js', import.meta.url),
      {
        type: 'module',
      },
    ));
    this.worker.postMessage({
      kind: 'graph',
      graph: { n: this.graph.n, edgeData: this.graph.edgeData, edges: this.graph.edges },
      colorDistances: this.colorDistances,
    });
    this.worker.onmessage = ({ data }) => {
      if (this.worker !== worker) {
        data.bitmap?.close();
        return;
      }
      if (!this.active) {
        this.fail(new Error(data.error || 'Failed to initialize renderer.'));
        return;
      }
      const { target, job } = this.active;
      this.retainedPositions = new Set(data.retainedPositionIds ?? []);
      if (data.error) {
        this.fail(new Error(data.error));
        return;
      } else {
        // Show completed frames during continuous dragging, while keeping the pending indicator.
        const next = this.queue.get(target);
        if (!next || next.resolution === job.resolution)
          this.paint(target, job, data, Boolean(next));
        // Before and After start with the same geometry; one GPU image serves both.
        for (const [other, next] of this.queue) {
          if (this.sameFrame(job, next)) {
            this.queue.delete(other);
            this.paint(other, next, data);
          }
        }
        data.bitmap.close();
      }
      this.active = null;
      this.pump();
    };
    this.worker.onerror = (event) => {
      if (this.worker !== worker) return;
      this.fail(new Error(event.message || 'An error occurred in the render worker.'));
    };
  }
  fail(error) {
    this.failure = error;
    const targets = [...this.queue.keys()];
    if (this.active) targets.push(this.active.target);
    for (const target of targets) {
      target.dataset.rendering = 'false';
      target.setAttribute('aria-busy', 'false');
    }
    this.queue.clear();
    this.active = null;
    this.pump();
  }
  sameFrame(a, b) {
    if (
      a.positionId !== b.positionId ||
      a.resolution !== b.resolution ||
      a.edgeLimit !== b.edgeLimit
    )
      return false;
    return ['box', 'view', 'style'].every(
      (key) => JSON.stringify(a[key]) === JSON.stringify(b[key]),
    );
  }
  paint(target, job, data, pending = false) {
    target.width = job.resolution;
    target.height = job.resolution;
    target.getContext('2d').drawImage(data.bitmap, 0, 0);
    target.dataset.rendering = String(pending);
    target.setAttribute('aria-busy', String(pending));
    target.dataset.renderer = data.backend ?? '';
    target.dataset.renderMs = String(data.stats?.renderMs ?? 0);
    target.dataset.drawCalls = String(data.stats?.drawCalls ?? 0);
    target.dataset.drawnEdges = String(data.stats?.drawnEdges ?? 0);
    target.dataset.drawnNodes = String(data.stats?.drawnNodes ?? 0);
    target.dataset.gpuVendor = data.stats?.adapterInfo?.vendor ?? '';
    target.dataset.gpuFallback = String(data.stats?.adapterInfo?.isFallbackAdapter ?? 'unknown');
    this.onFrame?.(data.backend, data.stats);
  }
  restart() {
    this.worker.terminate();
    this.queue.clear();
    this.active = null;
    this.failure = null;
    this.retainedPositions.clear();
    this.spawn();
  }
  draw(target, job) {
    target.dataset.rendering = 'true';
    target.setAttribute('aria-busy', 'true');
    let id = this.positionIds.get(job.positions);
    if (!id) {
      id = ++this.nextPositionId;
      this.positionIds.set(job.positions, id);
    }
    this.queue.set(target, { ...job, positionId: id, viewId: target.id });
    this.pump();
  }
  pump() {
    if (this.active) return;
    if (this.queue.size && !this.failure) {
      const target =
        [...this.queue.keys()].find((target) => target.id === 'graph') ??
        this.queue.keys().next().value;
      const job = this.queue.get(target);
      this.queue.delete(target);
      this.active = { target, job };
      this.worker.postMessage({
        kind: 'render',
        ...job,
        positions: this.retainedPositions.has(job.positionId) ? undefined : job.positions,
      });
      return;
    }
    for (const { resolve, reject } of this.waiters.splice(0))
      this.failure ? reject(this.failure) : resolve();
  }
  whenIdle() {
    if (this.failure) return Promise.reject(this.failure);
    if (!this.active && !this.queue.size) return Promise.resolve();
    return new Promise((resolve, reject) => this.waiters.push({ resolve, reject }));
  }
  dispose() {
    this.worker.terminate();
    this.worker = null;
    this.queue.clear();
    this.active = null;
    for (const { resolve } of this.waiters.splice(0)) resolve();
  }
}
