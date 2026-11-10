import { bindRunTrigger } from './ui/run-trigger.js';
import { graphMethodSupport } from './graph/method-support.js';
import { edgeCount } from './graph/edges.js';
import { createGpuContext } from './sgd/context.js';
import { AtomicFullEngine, FullEngine } from './sgd/full-engine.js';
import { AtomicSparseEngine, SparseEngine } from './sgd/sparse-engine.js';
import { initialPositions } from './sgd/initialization.js';
import { isSparseMethod, learningRate, METHOD_LABELS } from './sgd/learning-rate.js';
import { SAMPLES } from './graph/samples.js';
import { GraphClient } from './ui/graph-client.js';
import { GraphView } from './ui/graph-view.js';
import { GraphSourceView } from './ui/graph-source.js';
import { runIntent } from './ui/run-state.js';
import {
  element as $,
  readSettings,
  setStatus,
  setProgress,
  updateControls,
} from './ui/settings.js';
import {
  resultData,
  resultSummary,
  evaluationLabel,
  download,
  copyText,
  layoutBlob,
} from './ui/results.js';

/** Only application orchestration lives here. Graph algorithms and GPU kernels are separate. */
class GraphLayoutApp {
  constructor() {
    this.phase = 'idle';
    this.epoch = 0;
    this.history = [];
    this.context = null;
    this.graph = null;
    this.positions = null;
    this.config = null;
    this.engine = null;
    this.failed = false;
    this.pauseRequested = false;
    this.view = new GraphView();
    this.client = this.newClient();
    this.bindEvents();
    this.sourceView = new GraphSourceView();
  }
  newClient() {
    return new GraphClient((value, label) => {
      setProgress(value);
      setStatus(label);
    });
  }
  controls() {
    updateControls(this);
  }
  bindEvents() {
    for (const sample of SAMPLES) $('sample').add(new Option(sample.name, sample.id));
    $('sample').value = 'powergrid';
    $('load-sample').onclick = () => this.loadSample();
    bindRunTrigger(
      $('run'),
      () => this,
      () => this.run(),
    );
    $('pause').onclick = () => {
      this.pauseRequested = true;
      this.controls();
      setStatus('Waiting for current iteration to finish…');
    };
    $('reset').onclick = () => this.reset();
    $('cancel').onclick = () => this.cancel();
    $('file').onchange = (e) => {
      this.loadFile(e.target.files[0]);
      e.target.value = '';
    };
    for (const name of ['dragover', 'dragleave', 'drop'])
      $('drop').addEventListener(name, (e) => {
        e.preventDefault();
        $('drop').classList.toggle('over', name === 'dragover');
        if (name === 'drop') this.loadFile(e.dataTransfer.files[0]);
      });
    for (const id of ['method', 'iterations', 'epsilon', 'seed', 'pivots']) {
      $(id).addEventListener('input', () => this.parametersChanged(id));
      $(id).addEventListener('change', () => this.parametersChanged(id));
    }
    $('json').onclick = async () => {
      this.exporting = true;
      this.controls();
      try {
        download(await layoutBlob(this), 'rr-layout.json');
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        this.exporting = false;
        this.controls();
      }
    };
    this.view.setResolution(Number($('resolution').value));
    $('resolution').onchange = () => this.view.setResolution(Number($('resolution').value));
    $('png').onclick = async () => {
      await this.view.whenIdle();
      $('graph').toBlob((blob) => blob && download(blob, 'rr-layout.png'));
    };
    $('copy-result').onclick = () => copyText(resultSummary(this));
    $('copy-json').onclick = () => copyText(JSON.stringify(resultData(this), null, 2));
  }
  parametersChanged(field) {
    if (this.phase !== 'idle') return;
    // Draft edits do not generate coordinates or start rendering.
    this.controls();
    if (field === 'method' && this.context && this.graph && isSparseMethod($('method').value)) {
      const maxPivots = Number($('pivots').max);
      if (maxPivots >= 1 && Number($('pivots').value) > maxPivots) {
        $('pivots').value = String(maxPivots);
        this.controls();
      }
    }
    if (this.graph) this.metrics();
  }
  async initialize() {
    try {
      this.context = await createGpuContext();
      $('gpu-state').textContent = 'WebGPU: Available';
      $('gpu-state').title = [this.context.info?.vendor, this.context.info?.architecture]
        .filter(Boolean)
        .join(' / ');
      this.context.device.addEventListener('uncapturederror', (e) => {
        this.failed = true;
        this.pauseRequested = true;
        setStatus(e.error.message, true);
        this.controls();
      });
      this.context.device.lost.then((info) => {
        this.context = null;
        this.failed = true;
        this.pauseRequested = true;
        $('gpu-state').textContent = 'WebGPU: Reload required';
        setStatus(info.message || 'GPU connection was lost.', true);
        this.controls();
      });
    } catch (error) {
      $('gpu-state').textContent = 'WebGPU: Unavailable';
      $('gpu-state').title = error.message;
    }
    if (!this.graph && this.phase === 'idle') await this.loadSample();
    else this.controls();
  }
  async loadSample() {
    if (this.phase !== 'idle') return;
    const sample = SAMPLES.find((sample) => sample.id === $('sample').value);
    if (!sample) return;
    this.sourceView.setPickerMode('sample');
    await this.load(
      { sample: sample.id },
      {
        kind: 'sample',
        sampleId: sample.id,
        name: sample.name.replace(/\s*[（(].*[）)]$/, ''),
        preferredMethod: sample.method,
      },
    );
  }
  async loadFile(file) {
    if (!file || this.phase !== 'idle') return;
    if (!file.name.toLowerCase().endsWith('.mtx')) {
      setStatus('Please select a .mtx file.', true);
      return;
    }
    this.sourceView.setPickerMode('upload');
    await this.load({ file }, { kind: 'upload', name: file.name, fileBytes: file.size });
  }
  async load(source, metadata) {
    if (this.phase !== 'idle') return;
    const client = this.client;
    this.phase = 'loading';
    this.sourceView.begin(metadata);
    this.controls();
    setStatus('Loading graph');
    try {
      const { graph, colorDistances } = await this.client.load(source);
      // Replace graph identity, parameters, and results together, only after a successful load.
      this.clearRun();
      this.graph = graph;
      this.source = metadata.name;
      this.sourceMetadata = metadata;
      if (metadata.preferredMethod) $('method').value = metadata.preferredMethod;
      if (this.context) {
        const support = graphMethodSupport(graph.n, this.context.limits);
        const isAtomic =
          $('method').value === 'atomic_sparse' ||
          ($('method').value === 'atomic_full' && !support.atomic_full.available);
        const maxPivots = isAtomic ? support.atomic_sparse.maxPivots : support.sparse.maxPivots;
        const bytesPerEntry = isAtomic ? 20 : 8;
        if (Number($('pivots').value) > maxPivots)
          $('pivots').value = String(
            Math.max(
              1,
              Math.min(maxPivots, Math.floor((512 * 2 ** 20) / (bytesPerEntry * graph.n))),
            ),
          );
      }
      this.sourceView.commit(metadata, graph);
      $('graph-title').textContent = metadata.name;
      this.view.setGraph(graph, colorDistances);
      this.positions = null; // Selection loads graph data only; Run creates the layout.
      if (client !== this.client) return;
      setStatus(this.context ? 'Ready' : 'Please open in a WebGPU-enabled browser.', !this.context);
      this.metrics();
    } catch (error) {
      if (client !== this.client) return;
      this.sourceView.failed();
      setStatus(error.message, true);
    } finally {
      if (client !== this.client) return;
      this.phase = 'idle';
      setProgress(this.config ? this.epoch / this.config.iterations : 0);
      this.controls();
    }
  }
  clearRun() {
    this.disposeEngine();
    this.config = null;
    this.prepared = null;
    this.epoch = 0;
    this.elapsed = 0;
    this.renderWaitMs = 0;
    this.preprocessMs = 0;
    this.gpuPrepareMs = 0;
    this.initialStress = null;
    this.evaluation = null;
    this.history = [];
    this.failed = false;
    this.pauseRequested = false;
    for (const id of ['stress', 'initial-stress', 'elapsed', 'preprocess-time'])
      $(id).textContent = '—';
    $('stress-change').textContent = '';
    $('evaluation-kind').textContent = '—';
    $('result-text').value = '';
    $('copy-status').textContent = '';
    setProgress(0);
  }
  reset() {
    if (this.phase !== 'idle' || !this.graph) return;
    this.clearRun();
    this.positions = initialPositions(this.graph.n, Number($('seed').value));
    this.view.showInitial(this.positions, Number($('seed').value));
    this.metrics();
    this.controls();
    setStatus('Reset to initial layout');
  }
  cancel() {
    if (!['loading', 'preparing'].includes(this.phase)) return;
    this.client.dispose();
    this.client = this.newClient();
    this.clearRun();
    this.graph = null;
    this.positions = null;
    $('graph-title').textContent = $('nodes').textContent = $('edges').textContent = '—';
    $('epoch').textContent = `0 / ${$('iterations').value}`;
    this.sourceMetadata = null;
    this.sourceView.clear();
    this.view.clear();
    this.phase = 'idle';
    this.controls();
    setStatus('Preprocessing cancelled. Please reload a graph.');
  }
  disposeEngine() {
    this.engine?.dispose();
    this.engine = null;
  }
  metrics() {
    $('nodes').textContent = this.graph.n.toLocaleString();
    $('edges').textContent = edgeCount(this.graph).toLocaleString();
    $('epoch').textContent = `${this.epoch} / ${this.config?.iterations ?? $('iterations').value}`;
    $('elapsed').textContent = `${this.elapsed.toFixed(2)} ms`;
    $('preprocess-time').textContent = `${this.preprocessMs.toFixed(2)} ms`;
  }
  async prepare(config) {
    this.phase = 'preparing';
    this.controls();
    this.config = config;
    this.positions = initialPositions(this.graph.n, config.seed);
    this.view.showInitial(this.positions, config.seed);
    setStatus('Preparing preprocessing');
    const start = performance.now();
    let prepared = await this.client.prepare(this.config, this.context.limits);
    this.preprocessMs = performance.now() - start;
    this.prepared = {
      method: prepared.method,
      weightMin: prepared.weightMin,
      weightMax: prepared.weightMax,
      stats: prepared.stats,
    };
    const prepareStart = performance.now();
    const ENGINE_BY_METHOD = {
      full: FullEngine,
      sparse: SparseEngine,
      atomic_full: AtomicFullEngine,
      atomic_sparse: AtomicSparseEngine,
    };
    const Engine = ENGINE_BY_METHOD[this.config.method] ?? FullEngine;
    this.context.device.pushErrorScope('validation');
    try {
      this.engine = await Engine.create(this.context, prepared, this.config.seed);
    } finally {
      const error = await this.context.device.popErrorScope();
      if (error) throw new Error(error.message);
    }
    prepared = null; // Release CPU copies of the large GPU tables.
    this.gpuPrepareMs = performance.now() - prepareStart;

    this.evaluation = await this.client.evaluate(this.positions);
    this.initialStress = this.evaluation.value;
    this.history = [{ epoch: 0, stress: this.initialStress }];
    this.displayEvaluation();
  }
  displayEvaluation() {
    $('evaluation-kind').textContent = evaluationLabel(this.evaluation);
    $('initial-stress').textContent = this.initialStress.toLocaleString(undefined, {
      maximumFractionDigits: 4,
    });
    $('stress').textContent = this.evaluation.value.toLocaleString(undefined, {
      maximumFractionDigits: 4,
    });
    $('stress-change').textContent = this.initialStress
      ? `(${((1 - this.evaluation.value / this.initialStress) * 100).toFixed(1)}% reduction)`
      : '';
  }
  async run() {
    if (this.phase !== 'idle' || !this.graph || !this.context) return;
    let draft;
    try {
      draft = readSettings();
    } catch (error) {
      setStatus(error.message, true);
      this.controls();
      return;
    }
    try {
      if (!runIntent(this, draft).resume) {
        this.clearRun();
        await this.prepare(draft);
      }
      this.phase = 'running';
      this.pauseRequested = false;
      this.controls();
      while (!this.pauseRequested && this.epoch < this.config.iterations) {
        const start = performance.now();
        const eta = learningRate(this.prepared, this.config, this.epoch);
        this.positions = await this.engine.epoch(eta);
        if (!this.positions.every(Number.isFinite))
          throw new Error('GPU returned non-finite coordinates.');
        this.view.update(this.positions);
        this.elapsed += performance.now() - start;
        this.epoch++;
        this.metrics();
        setProgress(this.epoch / this.config.iterations);
        setStatus(
          `${METHOD_LABELS[this.config.method]}: ${this.epoch} / ${this.config.iterations}`,
        );
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
      this.phase = 'rendering';
      this.controls();
      setStatus('SGD complete · Rendering');
      const renderStart = performance.now();
      await this.view.whenIdle();
      this.renderWaitMs += performance.now() - renderStart;
      this.phase = 'evaluating';
      this.controls();
      setStatus('Evaluating stress');
      this.evaluation = await this.client.evaluate(this.positions);
      this.history.push({ epoch: this.epoch, stress: this.evaluation.value });
      this.displayEvaluation();
      $('result-text').value = resultSummary(this);
      setStatus(
        this.epoch === this.config.iterations ? 'Completed' : `Paused (epoch ${this.epoch})`,
      );
    } catch (error) {
      this.disposeEngine();
      this.failed = true;
      setStatus(error.message, true);
    } finally {
      this.phase = 'idle';
      this.controls();
    }
  }
}
const app = new GraphLayoutApp();
app.initialize();
