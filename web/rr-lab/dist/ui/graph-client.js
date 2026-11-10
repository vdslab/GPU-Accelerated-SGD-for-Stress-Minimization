/** Typed request boundaries keep Web Worker messaging out of SGD/UI logic. */
export class GraphClient {
  constructor(onProgress) {
    this.worker = new Worker(new URL('../workers/graph-worker.js', import.meta.url), {
      type: 'module',
    });
    this.pending = new Map();
    this.nextId = 0;
    this.onProgress = onProgress;
    this.worker.onmessage = ({ data }) => {
      const request = this.pending.get(data.id);
      if (!request) return;
      if (data.type === 'progress') {
        this.onProgress(data.value, data.label);
        return;
      }
      this.pending.delete(data.id);
      if (data.type === 'error') request.reject(new Error(data.message));
      else request.resolve(data.value);
    };
    this.worker.onerror = (event) => {
      for (const request of this.pending.values())
        request.reject(
          new Error(event.message || 'An error occurred in the preprocessing worker.'),
        );
      this.pending.clear();
    };
  }
  request(kind, payload = {}) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, kind, ...payload });
    });
  }
  load(source) {
    return this.request('load', source);
  }
  prepare(config, limits) {
    return this.request('prepare', { config, limits });
  }
  evaluate(positions) {
    return this.request('evaluate', { positions });
  }
  dispose() {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error('Preprocessing cancelled.'));
    this.pending.clear();
  }
}
