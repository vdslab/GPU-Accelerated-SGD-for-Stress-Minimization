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
      ? 'Failed to load. Keeping the current graph intact.'
      : 'Failed to load. Please select another graph.';
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
        ? 'Sample'
        : 'Upload'
      : 'Not loaded';
    $('active-graph-kind').dataset.kind = active?.kind ?? 'none';
    $('active-graph-name').textContent = active?.name ?? 'Load a graph';
    $('active-graph-stats').textContent = active
      ? `${active.graph.n.toLocaleString()} vertices · ${edgeCount(active.graph).toLocaleString()} edges · LCC${active.graph.n !== active.graph.originalN ? ` (input ${active.graph.originalN.toLocaleString()} vertices)` : ''}`
      : 'Graph details will appear here once loaded.';
    $('active-graph-state').textContent = this.pending
      ? 'Loading'
      : active
        ? 'Loaded'
        : 'Not loaded';
    $('active-graph-pending').hidden = !this.pending;
    $('active-graph-pending').textContent = this.pending
      ? `Loading ${this.pending.kind === 'sample' ? 'sample' : 'file'} "${this.pending.name}"…${active ? ' Current graph will switch when finished.' : ''}`
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
      hint.textContent = 'Does not switch until you click Load.';
    } else {
      hint.hidden = this.active?.kind === 'upload';
      hint.textContent = 'Selecting a file will switch the graph.';
    }
  }
}
