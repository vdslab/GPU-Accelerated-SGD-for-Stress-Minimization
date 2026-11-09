import { parseMTX, parseMTXStream } from '../graph/mtx.js';
import { largestComponent } from '../graph/components.js';
import { adjacency, bfs } from '../graph/traversal.js';
import { SAMPLES, demoGraph } from '../graph/samples.js';
import { prepareFull } from '../preprocess/full.js';
import { prepareSparse } from '../preprocess/sparse.js';
import { prepareEvaluation, evaluateStress } from '../graph/evaluation.js';

let graph = null,
  evaluation = null;

/** Graph loading, preprocessing, and evaluation never run on the UI thread. */
self.onmessage = async ({ data }) => {
  const { id, kind } = data;
  const report = (value, label) => self.postMessage({ id, type: 'progress', value, label });
  try {
    if (kind === 'load') {
      let raw;
      if (data.sample === 'community') raw = demoGraph();
      else {
        let stream = data.file?.stream(),
          total = data.file?.size;
        if (data.sample) {
          const sample = SAMPLES.find((s) => s.id === data.sample);
          if (!sample?.file) throw new Error('サンプルが見つかりません。');
          report(0, 'サンプルを読み込み中');
          const response = await fetch(new URL('../samples/' + sample.file, import.meta.url));
          if (!response.ok) throw new Error('サンプルを取得できませんでした。');
          stream = response.body;
          total = Number(response.headers.get('content-length'));
        }
        raw = stream
          ? await parseMTXStream(stream, total, (v, label) => report(v * 0.4, label))
          : parseMTX(data.text);
      }
      report(0.5, '最大連結成分を抽出中');
      const candidate = largestComponent(raw);
      const colorDistances = bfs(adjacency(candidate), 0);
      // An invalid candidate must not destroy the active graph or its evaluation.
      graph = candidate;
      evaluation = null;
      self.postMessage({ id, type: 'result', value: { graph, colorDistances } }, [
        colorDistances.buffer,
      ]);
    } else if (kind === 'prepare') {
      if (!graph) throw new Error('グラフを読み込んでください。');
      const prepared =
        data.config.method === 'sparse'
          ? prepareSparse(graph, data.config, data.limits, report)
          : prepareFull(graph, data.config, data.limits, report);
      report(0.99, 'stress評価用の距離を準備中');
      evaluation = prepareEvaluation(graph);
      const transfer = Object.values(prepared)
        .filter((value) => ArrayBuffer.isView(value))
        .map((value) => value.buffer);
      self.postMessage({ id, type: 'result', value: prepared }, transfer);
    } else if (kind === 'evaluate') {
      if (!evaluation) throw new Error('評価距離が未準備です。');
      self.postMessage({ id, type: 'result', value: evaluateStress(data.positions, evaluation) });
    } else throw new Error('未知の前処理要求です。');
  } catch (error) {
    self.postMessage({ id, type: 'error', message: error.message });
  }
};
