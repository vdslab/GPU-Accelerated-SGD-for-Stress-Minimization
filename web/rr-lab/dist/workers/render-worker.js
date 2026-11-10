import { GpuGraphRenderer } from '../render/gpu-renderer.js';
import { createPalette, renderGraph } from '../ui/graph-render.js';
let renderer, graph, palette;
const cpuSlots = new Map();
let tasks = Promise.resolve();
self.onmessage = ({ data }) => {
  tasks = tasks
    .then(async () => {
      if (data.kind === 'graph') {
        graph = data.graph;
        if (navigator.gpu) renderer = await GpuGraphRenderer.create(graph, data.colorDistances);
        else palette = createPalette(graph, data.colorDistances);
        return;
      }
      let result;
      if (renderer) result = await renderer.render(data);
      else {
        const key = data.viewId === 'before' ? 'before' : 'after';
        if (data.positions) cpuSlots.set(key, { id: data.positionId, positions: data.positions });
        const slot = [...cpuSlots.values()].find((slot) => slot.id === data.positionId);
        if (!slot) throw Error('Missing render coordinates.');
        const canvas = new OffscreenCanvas(data.resolution, data.resolution);
        renderGraph(
          canvas,
          graph,
          slot.positions,
          data.box,
          data.view,
          data.resolution,
          data.style,
          palette,
          data.edgeLimit,
        );
        result = {
          bitmap: canvas.transferToImageBitmap(),
          backend: 'canvas2d',
          retainedPositionIds: [...cpuSlots.values()].map((slot) => slot.id),
        };
      }
      self.postMessage(result, [result.bitmap]);
    })
    .catch((error) => self.postMessage({ error: error.message }));
};
