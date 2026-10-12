"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

// Track destructive DOM operations without requiring browser dependencies.
class Element {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.hidden = false;
    this.disabled = false;
    this.scrollTop = 150;
    this.clientHeight = 50;
    this.scrollHeight = 200;
    this.valueWrites = 0;
    this.textWrites = 0;
    this.srcWrites = 0;
    this.replacements = 0;
    this._value = "";
    this._text = "";
    this._src = "";
  }
  get value() { return this._value; }
  set value(value) {
    this._value = value;
    this.valueWrites++;
    this.selectionStart = this.selectionEnd = value.length;
  }
  get textContent() { return this._text || this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this._text = value; this.children = []; this.textWrites++; }
  get src() { return this._src; }
  set src(value) { this._src = value; this.srcWrites++; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; this.replacements++; }
  removeAttribute(name) { if (name === "src") this._src = ""; }
  addEventListener() {}
  close() { this.open = false; }
}

const source = fs.readFileSync(path.join(__dirname, "..", "app.js"), "utf8");
const clone = (value) => JSON.parse(JSON.stringify(value));

function harness() {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const timers = [];
  let response;
  const context = vm.createContext({
    document: { getElementById: element, createElement: (tag) => new Element(tag), addEventListener() {} },
    location: new URL("http://localhost:5500/tools/single_run/index.html"),
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    navigator: { clipboard: { writeText: async () => {} } },
    URL,
    URLSearchParams,
    AbortSignal: { timeout: () => undefined },
    // Leave startup connection pending; individual tests supply poll responses.
    fetch: () => response === undefined ? new Promise(() => {}) : Promise.resolve({
      ok: true,
      headers: { get: () => "application/json" },
      json: async () => clone(response),
    }),
    setTimeout: (callback, delay) => { timers.push({ callback, delay }); return timers.length; },
    clearTimeout() {},
  });
  vm.runInContext(source, context, { filename: "app.js" });
  vm.runInContext("apiUrl = DEFAULT_API; connecting = false; connected = true;", context);
  context.testJob = null;
  return {
    element,
    timers,
    render(job) {
      context.testJob = clone(job);
      vm.runInContext("renderJob(testJob)", context);
    },
    async poll(job) {
      response = job;
      context.testJob = clone(job);
      await vm.runInContext("activeId = testJob.id; poll(pollGeneration)", context);
    },
    setApi(url) { context.testApi = url; vm.runInContext("apiUrl = testApi", context); },
    clear() { vm.runInContext("clearJob()", context); },
  };
}

function runningJob() {
  return {
    id: "experiment-one",
    state: "running",
    log: "Started\n",
    run_directory: "output/one",
    summary: "実行中\nグラフ: graph\n",
    result: { record: {
      nodes: 419, edges: 1572, iteration_time_ms: 1.5,
      algorithm_time_cold_ms: 2, algorithm_time_warm_ms: 1.7,
      cli_total_time_cold_ms: 2.3, stress_value: 0.125, stress_kind: "full",
    } },
    artifacts: {
      info: "/output/one/info.txt", result: "/output/one/result.json",
      before_image: "/output/one/before.svg", after_image: "/output/one/after.svg",
    },
    error: null,
  };
}

function selectedResult(ui) {
  const result = ui.element("result-text");
  result.selectionStart = 2;
  result.selectionEnd = 8;
  ui.element("copy-status").textContent = "コピーしました。";
  return {
    result,
    valueWrites: result.valueWrites,
    statusWrites: ui.element("copy-status").textWrites,
    metric: ui.element("metric-rows").children[0],
    artifact: ui.element("artifact-links").children[0],
  };
}

function assertResultPreserved(ui, previous) {
  assert.equal(previous.result.valueWrites, previous.valueWrites, "unchanged result must not be reassigned");
  assert.equal(previous.result.selectionStart, 2);
  assert.equal(previous.result.selectionEnd, 8);
  assert.equal(ui.element("copy-status").textContent, "コピーしました。");
  assert.equal(ui.element("copy-status").textWrites, previous.statusWrites);
  assert.equal(ui.element("metric-rows").children[0], previous.metric);
  assert.equal(ui.element("artifact-links").children[0], previous.artifact);
}

test("identical polling responses preserve result and log selection and existing nodes", async () => {
  const ui = harness();
  const job = runningJob();
  ui.render(job);
  const previous = selectedResult(ui);
  const log = ui.element("log");
  log.selectionStart = 0;
  log.selectionEnd = 4;
  const logWrites = log.valueWrites;
  const imageWrites = ui.element("before-image").srcWrites;
  const statusWrites = ui.element("run-status").textWrites;
  const buttonWrites = ui.element("run").textWrites;
  await ui.poll(job);
  assertResultPreserved(ui, previous);
  assert.equal(log.valueWrites, logWrites);
  assert.equal(log.selectionStart, 0);
  assert.equal(log.selectionEnd, 4);
  assert.equal(ui.element("before-image").srcWrites, imageWrites);
  assert.equal(ui.element("run-status").textWrites, statusWrites);
  assert.equal(ui.element("run").textWrites, buttonWrites);
  assert.equal(ui.timers.at(-1).delay, 1000, "running jobs continue polling");
});

test("log-only progress updates preserve copied results and a manually scrolled log position", async () => {
  const ui = harness();
  const job = runningJob();
  ui.render(job);
  ui.element("result-text").value = '{"result": true}'; // JSON clipboard fallback can select alternate text.
  const previous = selectedResult(ui);
  ui.element("log").scrollTop = 15;
  job.log += "Rendering images\n";
  await ui.poll(job);
  assertResultPreserved(ui, previous);
  assert.equal(ui.element("result-text").value, '{"result": true}');
  assert.equal(ui.element("log").value, job.log);
  assert.equal(ui.element("log").scrollTop, 15);
});

test("changed summaries and measurements update, while image-only progress keeps artifact links", async () => {
  const ui = harness();
  const job = runningJob();
  ui.render(job);
  const oldMetric = ui.element("metric-rows").children[0];
  const oldArtifact = ui.element("artifact-links").children[0];
  selectedResult(ui);
  job.summary = "完了\nStress: 0.1\n";
  job.result.record.stress_value = 0.1;
  job.artifacts.after_image = "/output/one/finished.svg";
  job.state = "success";
  await ui.poll(job);
  assert.equal(ui.element("result-text").value, job.summary);
  assert.equal(ui.element("copy-status").textContent, "");
  assert.notEqual(ui.element("metric-rows").children[0], oldMetric);
  assert.equal(ui.element("metric-rows").children.at(-1).children[1].textContent, "0.1 (full)");
  assert.equal(ui.element("artifact-links").children[0], oldArtifact);
  assert.equal(ui.element("after-image").src, "http://127.0.0.1:8765/output/one/finished.svg");
  assert.equal(ui.timers.length, 0, "completed jobs stop polling");
  job.artifacts.info = "/output/one/final-info.txt";
  ui.render(job);
  assert.equal(ui.element("artifact-links").children[0].href, "http://127.0.0.1:8765/output/one/final-info.txt");
});

test("a new run clears previous copy feedback, measurements, links and images", () => {
  const ui = harness();
  ui.render(runningJob());
  selectedResult(ui);
  ui.render({ id: "experiment-two", state: "running", log: "", summary: "", result: null, artifacts: {}, error: null });
  assert.equal(ui.element("result-text").value, "実行中…");
  assert.equal(ui.element("copy-status").textContent, "");
  assert.equal(ui.element("metrics").hidden, true);
  assert.equal(ui.element("metric-rows").children.length, 0);
  assert.equal(ui.element("artifact-links").children.length, 0);
  assert.equal(ui.element("copy-result").disabled, true);
  assert.equal(ui.element("copy-json").disabled, true);
  for (const stage of ["before", "after"]) {
    assert.equal(ui.element(`${stage}-link`).hidden, true);
    assert.equal(ui.element(`${stage}-image`).src, "");
    assert.equal(ui.element(`${stage}-placeholder`).hidden, false);
  }
});

test("switching API origins or clearing a job refreshes otherwise identical artifacts", () => {
  const ui = harness();
  const job = runningJob();
  ui.render(job);
  const firstArtifact = ui.element("artifact-links").children[0];
  ui.setApi("http://localhost:9876");
  ui.render(job);
  assert.notEqual(ui.element("artifact-links").children[0], firstArtifact);
  assert.equal(ui.element("artifact-links").children[0].href, "http://localhost:9876/output/one/info.txt");
  assert.equal(ui.element("before-image").src, "http://localhost:9876/output/one/before.svg");
  ui.clear();
  ui.render(job);
  assert.equal(ui.element("result-text").value, job.summary);
  assert.equal(ui.element("metric-rows").children.length, 6);
  assert.equal(ui.element("before-link").hidden, false);
});

test("restoring an unchanged failed job shows its error after reconnection cleared the message", () => {
  const ui = harness();
  const job = runningJob();
  job.state = "failure";
  job.error = "画像生成に失敗しました";
  ui.render(job);
  ui.element("error").textContent = "";
  ui.element("error").hidden = true;
  ui.render(job);
  assert.equal(ui.element("error").textContent, job.error);
  assert.equal(ui.element("error").hidden, false);
  assert.equal(ui.element("log-details").open, true);
});
