import { edgeCount, forEachEdge } from '../graph/edges.js';
const VIEW_SIZE = 1024;
export function createPalette(graph, colorDistances) {
  const counts = new Uint32Array(32),
    labels = new Uint8Array(graph.n);
  let max = 0;
  for (const d of colorDistances) max = Math.max(max, d);
  for (let v = 0; v < graph.n; v++) {
    const label = Math.min(31, Math.floor((colorDistances[v] / Math.max(1, max)) * 31));
    labels[v] = label;
    counts[label]++;
  }
  const offsets = new Uint32Array(33);
  for (let i = 0; i < 32; i++) offsets[i + 1] = offsets[i] + counts[i];
  const vertices = new Uint32Array(graph.n),
    cursor = offsets.slice();
  for (let v = 0; v < graph.n; v++) vertices[cursor[labels[v]]++] = v;
  const buckets = Array.from({ length: 32 }, (_, i) =>
    vertices.subarray(offsets[i], offsets[i + 1]),
  );
  const colors = buckets.map((_, index) => {
    const t = (index / 31) * 2,
      k = Math.min(1, Math.floor(t)),
      f = t - k,
      stops = [
        [86, 42, 129],
        [220, 76, 60],
        [252, 231, 149],
      ];
    return `rgb(${stops[k].map((c, j) => Math.round(c + (stops[k + 1][j] - c) * f)).join(',')})`;
  });
  return { buckets, colors };
}
export function renderGraph(
  target,
  graph,
  positions,
  box,
  view,
  resolution,
  style,
  palette,
  edgeLimit = 100000,
) {
  // A square bitmap has a stable world transform, independent of CSS size or device pixel ratio.
  // CSS object-fit contains this bitmap on screen; PNG and the enlarged image keep all pixels.
  if (target.width !== resolution || target.height !== resolution) {
    target.width = resolution;
    target.height = resolution;
  }
  const ctx = target.getContext('2d', {
    willReadFrequently: typeof OffscreenCanvas !== 'undefined' && target instanceof OffscreenCanvas,
  });
  const ratio = resolution / VIEW_SIZE;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, VIEW_SIZE, VIEW_SIZE);
  if (!positions || !graph) return;
  const scale = box.scale * view.zoom,
    xOffset = VIEW_SIZE / 2 + view.x - box.x * scale,
    yOffset = VIEW_SIZE / 2 + view.y - box.y * scale;
  const x = (v) => positions[v * 2] * scale + xOffset,
    y = (v) => positions[v * 2 + 1] * scale + yOffset;
  ctx.lineWidth = style.edgeWidth;
  ctx.strokeStyle = style.edgeColor;
  ctx.beginPath();
  let edgeBatch = 0;
  const stride =
    edgeLimit && edgeCount(graph) > 500000
      ? Math.max(1, Math.ceil(edgeCount(graph) / edgeLimit))
      : 1;
  forEachEdge(graph, (u, v, index) => {
    if (index % stride) return;
    ctx.moveTo(x(u), y(u));
    ctx.lineTo(x(v), y(v));
    if (++edgeBatch === 4096) {
      ctx.stroke();
      ctx.beginPath();
      edgeBatch = 0;
    }
  });
  ctx.stroke();
  const radius = style.radius;
  if (graph.n > 1000000) {
    // Aggregate vertices occupying the same display pixel; every vertex contributes.
    // This bounds raster work without dropping vertices from the graph or solver.
    const side = Math.ceil(VIEW_SIZE / Math.max(1, radius)),
      count = new Uint32Array(side * side),
      sums = new Uint32Array(side * side);
    palette.buckets.forEach((bucket, color) => {
      for (const v of bucket) {
        const px = Math.floor((x(v) / VIEW_SIZE) * side),
          py = Math.floor((y(v) / VIEW_SIZE) * side);
        if (px < 0 || py < 0 || px >= side || py >= side) continue;
        const at = py * side + px;
        count[at]++;
        sums[at] += color;
      }
    });
    const step = VIEW_SIZE / side;
    for (let color = 0; color < 32; color++) {
      ctx.fillStyle = palette.colors[color];
      ctx.beginPath();
      let batch = 0;
      for (let i = 0; i < count.length; i++) {
        if (!count[i] || Math.round(sums[i] / count[i]) !== color) continue;
        const px = ((i % side) + 0.5) * step,
          py = (Math.floor(i / side) + 0.5) * step;
        ctx.rect(px - radius, py - radius, radius * 2, radius * 2);
        if (++batch === 4096) {
          ctx.fill();
          ctx.beginPath();
          batch = 0;
        }
      }
      ctx.fill();
    }
    return;
  }
  // Batch by color: 32 fills instead of one fill per vertex on large graphs.
  palette.buckets.forEach((bucket, index) => {
    ctx.fillStyle = palette.colors[index];
    ctx.beginPath();
    let nodeBatch = 0;
    for (const v of bucket) {
      const px = x(v),
        py = y(v);
      ctx.moveTo(px + radius, py);
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      if (++nodeBatch === 4096) {
        ctx.fill();
        ctx.beginPath();
        nodeBatch = 0;
      }
    }
    ctx.fill();
  });
}
