import { edgeCount, edgeAt, edges } from '../graph/edges.js';
import { METHOD_LABELS } from '../sgd/learning-rate.js';
function resultMetadata(run) {
  return {
    algorithm: run.config.method === 'sparse' ? 'rr_sparse_sgd' : 'rr_sgd',
    backend: 'WebGPU',
    source: run.source,
    sourceKind: run.sourceMetadata?.kind,
    sampleId: run.sourceMetadata?.sampleId,
    settings: run.config,
    renderResolution: { width: run.view.resolution, height: run.view.resolution },
    epoch: run.epoch,
    iterationElapsedMs: run.elapsed,
    renderWaitMs: run.renderWaitMs,
    renderBackend: run.view.renderBackend,
    renderAdapter: run.view.lastRenderStats?.adapterInfo,
    preprocessMs: run.preprocessMs,
    gpuPrepareMs: run.gpuPrepareMs,
    evaluation: run.evaluation,
    initialStress: run.initialStress,
    history: run.history,
    statistics: run.prepared.stats,
    renderOptions: {
      edgeLimit: run.view.edgeLimit,
      vertexAggregation: run.view.renderBackend === 'canvas2d' && run.graph.n > 1000000,
    },
  };
}
export function resultData(run) {
  return {
    ...resultMetadata(run),
    vertexIds: Array.from(run.graph.ids),
    positions: Array.from({ length: run.graph.n }, (_, i) => [
      run.positions[i * 2],
      run.positions[i * 2 + 1],
    ]),
    edges: Array.from(edges(run.graph)),
  };
}
/** Blob parts avoid constructing a single huge JSON string or millions of JS objects. */
export async function layoutBlob(run) {
  const graph = run.graph,
    positions = run.positions;
  const parts = [new Blob([JSON.stringify(resultMetadata(run)).slice(0, -1)])];
  async function array(name, count, serialize) {
    parts.push(new Blob([`,"${name}":[`]));
    for (let start = 0; start < count; start += 16384) {
      const rows = [];
      for (let i = start; i < Math.min(start + 16384, count); i++) rows.push(serialize(i));
      parts.push(new Blob([(start ? ',' : '') + rows.join(',')]));
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    parts.push(new Blob([']']));
  }
  await array('vertexIds', graph.n, (i) => String(graph.ids[i]));
  await array('positions', graph.n, (i) => `[${positions[2 * i]},${positions[2 * i + 1]}]`);
  await array('edges', edgeCount(graph), (i) => `[${edgeAt(graph, i).join(',')}]`);
  parts.push(new Blob(['}']));
  return new Blob(parts, { type: 'application/json' });
}

export function resultSummary(run) {
  return [
    `Graph: ${run.source}`,
    `Source: ${run.sourceMetadata?.kind === 'sample' ? 'Sample' : 'Upload'}`,
    `Method: ${METHOD_LABELS[run.config.method]} / WebGPU`,
    `Vertices: ${run.graph.n} / Edges: ${edgeCount(run.graph)}`,
    `Iterations: ${run.epoch} / ${run.config.iterations}`,
    `seed: ${run.config.seed} / epsilon: ${run.config.epsilon}`,
    ...(run.config.method === 'sparse'
      ? [`pivots: ${run.prepared.stats.pivots} / matching rounds: ${run.prepared.stats.rounds}`]
      : ['tile: 128 / workgroup: 64 / shared']),
    `Preprocess: ${run.preprocessMs.toFixed(2)} ms`,
    `GPU setup: ${run.gpuPrepareMs.toFixed(2)} ms`,
    `Iteration time: ${run.elapsed.toFixed(2)} ms`,
    `Renderer: ${run.view.renderBackend === 'webgpu' ? `WebGPU${run.view.lastRenderStats?.adapterInfo?.isFallbackAdapter ? ' (Software)' : ''}` : 'Canvas 2D'}`,
    `Render wait: ${(run.renderWaitMs ?? 0).toFixed(2)} ms`,
    `Evaluation: ${evaluationLabel(run.evaluation)}`,
    `Initial stress: ${run.initialStress}`,
    `Current stress: ${run.history.at(-1)?.stress}`,
  ].join('\n');
}
export function evaluationLabel(evaluation) {
  if (!evaluation) return '—';
  return evaluation.exact
    ? 'exact full stress'
    : `sampled full stress (estimated from ${evaluation.sources} fixed sources)`;
}
export function download(blob, name) {
  const url = URL.createObjectURL(blob),
    link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    document.getElementById('copy-status').textContent = 'Copied';
  } catch {
    const field = document.getElementById('result-text');
    field.value = text;
    field.select();
    document.getElementById('copy-status').textContent = 'Copy the selected text';
  }
}
