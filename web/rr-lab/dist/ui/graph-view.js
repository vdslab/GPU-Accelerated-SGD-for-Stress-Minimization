import { edgeCount } from '../graph/edges.js';
import { renderGraph } from './graph-render.js';
import { RenderClient } from './render-client.js';
import { graphStyle } from './graph-style.js';

const VIEW_SIZE = 1024;
const RENDER_RESOLUTIONS = [512, 1024, 2048, 4096];

/** Canvas-only presentation. It receives graph/coordinates, never solver buffers. */
export class GraphView {
  constructor() {
    this.resolution = 4096;
    this.edgeLimit = Number(document.getElementById('edge-detail').value);
    document.getElementById('edge-detail').onchange = (event) => {
      this.edgeLimit = Number(event.target.value);
      this.renderer?.restart();
      this.updateCaption();
      this.drawBefore();
      this.drawAfter();
    };
    this.before = document.getElementById('before');
    this.after = document.getElementById('graph');
    this.camera = { x: 0, y: 0, zoom: 1 };
    this.bounds = { x: 0, y: 0, scale: 1 };
    const resize = new ResizeObserver(() => {
      this.drawBefore();
      this.fit();
    });
    resize.observe(document.getElementById('before-wrap'));
    resize.observe(document.getElementById('canvas-wrap'));
    this.bindCamera(
      this.after,
      () => this.camera,
      () => this.drawAfter(),
      (factor) => this.zoom(factor),
    );
    this.expandedCanvas = document.getElementById('expanded-image');
    this.bindCamera(
      this.expandedCanvas,
      () => this.expanded.camera,
      () => this.drawExpanded(),
      (factor) => this.zoomExpanded(factor),
    );
    document.getElementById('image-fit').onclick = () => {
      this.expanded.camera = { x: 0, y: 0, zoom: 1 };
      this.drawExpanded();
    };
    document.getElementById('image-zoom-in').onclick = () => this.zoomExpanded(1.25);
    document.getElementById('image-zoom-out').onclick = () => this.zoomExpanded(0.8);
    document.getElementById('zoom-in').onclick = () => this.zoom(1.25);
    document.getElementById('zoom-out').onclick = () => this.zoom(0.8);
    document.getElementById('fit').onclick = () => this.fit();
    for (const [id, target, title] of [
      ['expand-before', this.before, 'Before（初期配置）'],
      ['expand-after', this.after, 'After（実行後）'],
    ]) {
      document.getElementById(id).onclick = () => {
        document.getElementById('image-title').textContent = title;
        const positions = target === this.before ? this.initial : this.positions;
        this.expanded = {
          positions,
          bounds: this.box(positions),
          camera: target === this.before ? { x: 0, y: 0, zoom: 1 } : { ...this.camera },
        };
        this.expandedCanvas.setAttribute('aria-label', title);
        document.getElementById('image-dialog').showModal();
        this.drawExpanded();
      };
    }
    const dialog = document.getElementById('image-dialog');
    document.getElementById('close-image').onclick = () => dialog.close();
    dialog.onclick = (e) => {
      if (e.target === dialog) dialog.close();
    };
  }
  /** Pointer movement uses the displayed square, so both canvas sizes pan identically. */
  bindCamera(canvas, getCamera, draw, zoom) {
    let drag = null;
    canvas.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      drag = [event.clientX, event.clientY];
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!drag) return;
      const rect = canvas.getBoundingClientRect();
      const conversion = VIEW_SIZE / Math.max(1, Math.min(rect.width, rect.height));
      const camera = getCamera();
      camera.x += (event.clientX - drag[0]) * conversion;
      camera.y += (event.clientY - drag[1]) * conversion;
      drag = [event.clientX, event.clientY];
      draw();
    });
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture'])
      canvas.addEventListener(name, () => {
        drag = null;
      });
    canvas.addEventListener(
      'wheel',
      (event) => {
        event.preventDefault();
        zoom(Math.exp(-event.deltaY * 0.001));
      },
      { passive: false },
    );
  }
  drawExpanded() {
    if (!this.expanded) return;
    const { positions, bounds, camera } = this.expanded;
    this.render(this.expandedCanvas, positions, bounds, camera);
  }
  zoomExpanded(factor) {
    this.expanded.camera.zoom = Math.max(0.1, Math.min(15, this.expanded.camera.zoom * factor));
    this.drawExpanded();
  }
  setGraph(graph, colorDistances) {
    this.renderer?.dispose();
    this.renderer = null;
    this.graph = graph;
    this.colorDistances = colorDistances;
    this.initial = null;
    this.positions = null;
    this.initialSeed = null;
    this.renderBackend = null;
    this.lastRenderStats = null;
    this.palette = null;
    this.drawBefore();
    this.drawAfter();
    this.updateCaption();
  }
  ensureRenderer() {
    if (!this.renderer)
      this.renderer = new RenderClient(this.graph, this.colorDistances, (backend, stats) => {
        this.renderBackend = backend;
        this.lastRenderStats = stats;
        this.updateCaption();
      });
  }

  clear() {
    this.renderer?.dispose();
    this.renderer = null;
    this.graph = null;
    this.initial = null;
    this.positions = null;
    this.drawBefore();
    this.drawAfter();
  }
  showInitial(positions, seed) {
    const changed = !this.initial || seed === undefined || this.initialSeed !== seed;
    if (changed) {
      // Every solver readback returns a new array, so initial coordinates remain immutable.
      this.initial = positions;
      this.initialSeed = seed;
      this.drawBefore();
    }
    this.positions = positions;
    this.fit();
  }
  update(positions) {
    this.positions = positions;
    this.fit();
  }
  setResolution(resolution) {
    if (!RENDER_RESOLUTIONS.includes(resolution)) throw new Error('未対応の描画解像度です。');
    this.resolution = resolution;
    this.drawBefore();
    this.drawAfter();
    this.updateCaption();
  }
  box(positions) {
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (let i = 0; i < positions.length; i += 2) {
      minX = Math.min(minX, positions[i]);
      maxX = Math.max(maxX, positions[i]);
      minY = Math.min(minY, positions[i + 1]);
      maxY = Math.max(maxY, positions[i + 1]);
    }
    return {
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
      scale: Math.max(
        0.001,
        Math.min(
          (VIEW_SIZE - 48) / Math.max(maxX - minX, 0.001),
          (VIEW_SIZE - 48) / Math.max(maxY - minY, 0.001),
        ),
      ),
    };
  }
  fit() {
    this.camera = { x: 0, y: 0, zoom: 1 };
    if (this.positions) this.bounds = this.box(this.positions);
    this.drawAfter();
  }
  zoom(factor) {
    this.camera.zoom = Math.max(0.1, Math.min(15, this.camera.zoom * factor));
    this.drawAfter();
  }
  drawBefore() {
    this.render(this.before, this.initial, this.initial ? this.box(this.initial) : this.bounds);
  }
  drawAfter() {
    this.render(this.after, this.positions, this.bounds, this.camera);
  }
  render(target, positions, box, view = { x: 0, y: 0, zoom: 1 }) {
    const displayedSide = Math.min(
      ...(target === this.expandedCanvas ? [target] : [this.before, this.after]).map((canvas) => {
        const rect = canvas.getBoundingClientRect();
        return Math.min(rect.width, rect.height) || VIEW_SIZE;
      }),
    );
    const style = graphStyle(this.graph?.n ?? 0, displayedSide);
    const placeholder = document.getElementById(
      target === this.before
        ? 'before-placeholder'
        : target === this.after
          ? 'after-placeholder'
          : '',
    );
    if (placeholder) placeholder.hidden = Boolean(positions);
    if (this.graph && positions) {
      this.ensureRenderer();
      this.renderer.draw(target, {
        positions,
        box,
        view,
        resolution: this.resolution,
        style,
        edgeLimit: this.edgeLimit,
      });
    } else {
      target.dataset.rendering = 'false';
      target.setAttribute('aria-busy', 'false');
      delete target.dataset.renderer;
      renderGraph(
        target,
        this.graph,
        positions,
        box,
        view,
        this.resolution,
        style,
        this.palette,
        this.edgeLimit,
      );
    }
  }
  whenIdle() {
    return this.renderer?.whenIdle() ?? Promise.resolve();
  }
  updateCaption() {
    const count = this.graph ? edgeCount(this.graph) : 0;
    const shown =
      this.edgeLimit && count > 500000
        ? Math.ceil(count / Math.max(1, Math.ceil(count / this.edgeLimit)))
        : count;
    const caption = document.getElementById('view-caption');
    const adapter = this.lastRenderStats?.adapterInfo;
    caption.title = adapter
      ? `描画GPU: ${adapter.vendor ?? ''} / ${adapter.architecture ?? ''}`
      : '';
    caption.textContent = `${this.resolution} × ${this.resolution} px · 描画: ${this.renderBackend === 'webgpu' ? `WebGPU${adapter?.isFallbackAdapter ? '（ソフトウェア）' : ''}` : this.renderBackend === 'canvas2d' ? 'Canvas 2D' : '未実行'} · ${shown < count ? `表示 ${shown.toLocaleString()} / 全 ${count.toLocaleString()} 辺（計算は全辺）` : 'ドラッグで移動・スクロールでズーム'}${this.renderBackend === 'canvas2d' && this.graph?.n > 1000000 ? ' · 全頂点を画素単位で集約' : ''}`;
  }
}
