import { edgeCount } from '../graph/edges.js';
import { SAMPLES } from '../graph/samples.js';
import { element as $ } from './settings.js';

/** Candidate inputs never overwrite the identity of the successfully loaded graph. */
export class GraphSourceView {
  constructor() {
    this.active = null;
    this.pending = null;
    for (const kind of ['sample', 'upload']) {
      $(`source-${kind}`).onchange = () => this.setPickerMode(kind);
    }
    $('sample').onchange = () => this.renderHint();
    this.setPickerMode('sample');
  }
  setPickerMode(kind) {
    $('source-sample').checked = kind === 'sample';
    $('source-upload').checked = kind === 'upload';
    $('sample-source').hidden = kind !== 'sample';
    $('upload-source').hidden = kind !== 'upload';
    this.renderHint();
  }
  begin(metadata) {
    this.pending = metadata;
    this.render();
  }
  commit(metadata, graph) {
    this.active = { ...metadata, graph };
    this.pending = null;
    this.render();
  }
  failed() {
    this.pending = null;
    this.render();
    $('source-hint').hidden = false;
    $('source-hint').textContent = this.active
      ? '読み込みに失敗しました。現在のグラフはそのままです。'
      : '読み込みに失敗しました。別のグラフを選択してください。';
  }
  clear() {
    this.active = null;
    this.pending = null;
    this.render();
  }
  render() {
    const active = this.active;
    $('active-graph-kind').textContent = active
      ? active.kind === 'sample'
        ? 'サンプル'
        : 'アップロード'
      : '未読み込み';
    $('active-graph-kind').dataset.kind = active?.kind ?? 'none';
    $('active-graph-name').textContent = active?.name ?? 'グラフを読み込んでください';
    $('active-graph-stats').textContent = active
      ? `${active.graph.n.toLocaleString()} 頂点 · ${edgeCount(active.graph).toLocaleString()} 辺 · 最大連結成分${active.graph.n !== active.graph.originalN ? `（入力 ${active.graph.originalN.toLocaleString()} 頂点）` : ''}`
      : '読み込みが完了すると、ここに表示されます。';
    $('active-graph-state').textContent = this.pending
      ? '読み込み中'
      : active
        ? '読み込み済み'
        : '未読み込み';
    $('active-graph-pending').hidden = !this.pending;
    $('active-graph-pending').textContent = this.pending
      ? `${this.pending.kind === 'sample' ? 'サンプル' : 'ファイル'}「${this.pending.name}」を読み込み中…${active ? ' 完了すると現在のグラフが切り替わります。' : ''}`
      : '';
    this.renderHint();
  }
  renderHint() {
    const hint = $('source-hint');
    if (this.pending) {
      hint.hidden = true;
      return;
    } // The pending graph is already named above.
    if ($('source-sample').checked) {
      const selected = SAMPLES.find((sample) => sample.id === $('sample').value);
      hint.hidden = this.active?.kind === 'sample' && this.active.sampleId === selected?.id;
      hint.textContent = '読み込むまで切り替わりません。';
    } else {
      hint.hidden = this.active?.kind === 'upload';
      hint.textContent = 'ファイルを選ぶと切り替わります。';
    }
  }
}
