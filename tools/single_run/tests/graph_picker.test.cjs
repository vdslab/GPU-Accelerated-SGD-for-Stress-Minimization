"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

// Exercise the actual event handlers without adding browser dependencies.
class Element {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.attributes = new Map();
    this.listeners = new Map();
    this.hidden = false;
    this.disabled = false;
    this._value = "";
    this._text = "";
    this.valueWrites = 0;
    this.replacements = 0;
    this.classes = new Set();
    this.classList = {
      toggle: (name, enabled) => {
        if (enabled) this.classes.add(name);
        else this.classes.delete(name);
      },
      contains: (name) => this.classes.has(name),
    };
    this.scrollTop = 0;
    this.clientHeight = 50;
    this.scrollHeight = 50;
  }
  get value() { return this._value; }
  set value(value) { this._value = String(value); this.valueWrites++; }
  get options() { return this.children; }
  get textContent() { return this._text || this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this._text = String(value); this.children = []; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) {
    this.children = children;
    this.replacements++;
    if (this.tagName === "select") this.value = children[0]?.value || "";
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  removeAttribute(name) {
    this.attributes.delete(name);
    if (name === "src") this.src = "";
  }
  addEventListener(name, callback) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(callback);
  }
  dispatch(name, options = {}) {
    const event = {
      currentTarget: this,
      target: this,
      defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; },
      ...options,
    };
    for (const callback of this.listeners.get(name) || []) callback(event);
    return event;
  }
  contains(target) { return this === target || this.children.some((child) => child.contains(target)); }
  focus() { this.focused = true; this.dispatch("focus"); }
  scrollIntoView() { this.scrolledIntoView = true; }
  close() { this.open = false; }
}

const source = fs.readFileSync(path.join(__dirname, "..", "app.js"), "utf8");
const clone = (value) => JSON.parse(JSON.stringify(value));

function sampleConfig() {
  return {
    graphs: [
      { name: "Alpha", path: "data/Alpha.mtx", nodes: 1200, edges: 2300, versions: [] },
      { name: "Alpha", path: "data/v1_lcc/Alpha.mtx", nodes: 1100, edges: 2200, versions: ["v1"] },
      { name: "Beta", path: "data/Beta.mtx", nodes: 40, edges: 50, versions: ["v4", "v10"] },
      { name: "Gamma", path: "data/Gamma.mtx", nodes: 60, edges: 70, versions: [] },
    ],
    versions: ["v10", "v4", "v1"],
    methods: [{ id: "cpu", label: "CPU", sparse: false }],
    defaults: {
      graph: "data/Alpha.mtx", method: "cpu", iterations: 15,
      epsilon: 0.1, pivots: 200, seed: 0, image_size: "2048x2048",
    },
  };
}

function harness(data = sampleConfig()) {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const document = new Element("document");
  document.getElementById = element;
  document.createElement = (tag) => new Element(tag);
  for (const id of ["graph", "method", "iterations", "epsilon", "pivots", "seed", "image-size"]) {
    const control = new Element("select");
    elements.set(id, control);
    const values = { iterations: ["1", "15"], epsilon: ["0.1"], pivots: ["200"], seed: ["0"], "image-size": ["2048x2048"] }[id] || [];
    control.replaceChildren(...values.map((value) => {
      const option = new Element("option");
      option.value = value;
      return option;
    }));
  }
  element("experiment-form").elements = { namedItem: (name) => element(name === "image_size" ? "image-size" : name) };
  element("graph-picker").append(element("graph-search"), element("graph-dropdown"), element("graph-dropdown-toggle"));
  element("graph-dropdown").append(element("graph-choices"));
  let response;
  const context = vm.createContext({
    document,
    location: new URL("http://localhost:5500/tools/single_run/index.html"),
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    navigator: { clipboard: { writeText: async () => {} } },
    URL,
    URLSearchParams,
    AbortSignal: { timeout: () => undefined },
    Option: function Option(label, value) {
      const option = new Element("option");
      option.textContent = label;
      option.value = value;
      return option;
    },
    // The unrelated startup connection is left pending.
    fetch: () => response === undefined ? new Promise(() => {}) : Promise.resolve({
      ok: true,
      headers: { get: () => "application/json" },
      json: async () => clone(response),
    }),
    setTimeout: () => 1,
    clearTimeout() {},
  });
  vm.runInContext(source, context, { filename: "app.js" });
  context.testConfig = clone(data);
  vm.runInContext("apiUrl = DEFAULT_API; connected = true; connecting = false; populateConfig(testConfig); updateControls();", context);
  return {
    element,
    document,
    options: () => element("graph-choices").children,
    rows: () => element("graph-table-body").children,
    graphs: () => clone(vm.runInContext("config.graphs", context)),
    sourceConfig: () => clone(context.testConfig),
    restore(job) {
      context.testJob = clone(job);
      vm.runInContext("restoreSettings(testJob)", context);
    },
    search(value) {
      element("graph-search").focus();
      element("graph-search").value = value;
      element("graph-search").dispatch("input");
    },
    key(key) { return element("graph-search").dispatch("keydown", { key }); },
    render(job) {
      context.testJob = clone(job);
      vm.runInContext("renderJob(testJob)", context);
    },
    async poll(job) {
      response = job;
      context.testJob = clone(job);
      await vm.runInContext("activeId = testJob.id; poll(pollGeneration)", context);
    },
    reconnect(data) {
      context.testConfig = clone(data);
      vm.runInContext("populateConfig(testConfig); updateControls();", context);
    },
  };
}

function completedJob() {
  return {
    id: "run-one", state: "success", log: "Done\n", run_directory: "output/one",
    summary: "完了\n", result: null, artifacts: {}, error: null,
  };
}

test("search matches name and version tokens regardless of case, whitespace, or full-width input", () => {
  const ui = harness();
  ui.search("  ALPHA   V1  ");
  assert.deepEqual(ui.options().map((option) => option.textContent), ["Alpha（1,100 / 2,200） v1"]);
  ui.search(" ＡＬＰＨＡ  Ｖ１ ");
  assert.deepEqual(ui.options().map((option) => option.textContent), ["Alpha（1,100 / 2,200） v1"]);
  ui.search("  alpha  ");
  assert.equal(ui.options().length, 1);
  assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx", "filtering must not change the experiment input");
});

test("duplicate graph names produce one native option, searchable option, and table row", () => {
  const ui = harness();
  assert.equal(ui.element("graph").options.length, 3);
  assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx", "the original default must map to the canonical graph");
  ui.element("graph-list-toggle").dispatch("click");
  assert.equal(ui.rows().length, 3);
  assert.deepEqual(ui.rows().map((row) => row.children[0].textContent), ["Alpha", "Beta", "Gamma"]);
  ui.search("alpha");
  assert.equal(ui.options().length, 1);
  assert.equal(ui.rows().length, 1);
  ui.options()[0].dispatch("click");
  assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx");
  assert.equal(ui.element("graph-dropdown").hidden, true);
  ui.element("graph-list-close").dispatch("click");
  assert.equal(ui.element("graph-search").value, "Alpha（1,100 / 2,200） v1");
  assert.doesNotMatch(ui.element("graph-search").value, /data\/|\.mtx/);
});

test("v1 inputs take precedence regardless of source order and retain every input alias", () => {
  const data = sampleConfig();
  const variants = data.graphs.filter((graph) => graph.name === "Alpha");
  for (const graphs of [variants, [...variants].reverse()]) {
    const ui = harness({ ...data, graphs });
    assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx");
    assert.equal(ui.graphs().length, 1);
    assert.equal(ui.graphs()[0].nodes, 1100);
    assert.equal(ui.graphs()[0].edges, 2200);
    assert.deepEqual([...ui.graphs()[0].input_paths].sort(), ["data/Alpha.mtx", "data/v1_lcc/Alpha.mtx"]);
    assert.deepEqual(ui.sourceConfig(), { ...data, graphs }, "normalization must not mutate the API configuration");
  }
});

test("version memberships merge onto one graph with numerically sorted badges and columns", () => {
  const data = sampleConfig();
  data.graphs[0].versions = ["v2", "v10"];
  data.graphs.push({ name: "Alpha", path: "data/v10_lcc/Alpha.mtx", nodes: 1000, edges: 2000, versions: ["v10"] });
  const ui = harness(data);
  ui.element("graph-list-toggle").dispatch("click");
  ui.search("alpha v2 v10");
  assert.deepEqual(ui.options().map((option) => option.textContent), ["Alpha（1,100 / 2,200） v1 v2 v10"]);
  assert.equal(ui.rows().length, 1);
  assert.deepEqual(ui.rows()[0].children.map((cell) => cell.textContent), ["Alpha", "1,100", "2,200", "◯", "◯", "", "", "◯"]);
  assert.deepEqual(ui.graphs()[0].versions, ["v1", "v2", "v10"]);
  assert.deepEqual([...ui.graphs()[0].input_paths].sort(), ["data/Alpha.mtx", "data/v10_lcc/Alpha.mtx", "data/v1_lcc/Alpha.mtx"]);
});

test("a version-marked input takes precedence over an unmarked original when v1 is absent", () => {
  const data = sampleConfig();
  data.graphs = [
    { name: "Alpha", path: "data/Alpha.mtx", nodes: 1200, edges: 2300, versions: [] },
    { name: "Alpha", path: "data/v2_lcc/Alpha.mtx", nodes: 900, edges: 1800, versions: ["v2"] },
  ];
  const ui = harness(data);
  assert.equal(ui.element("graph").value, "data/v2_lcc/Alpha.mtx");
  assert.equal(ui.graphs()[0].nodes, 900);
});

test("restoring a job using an original input selects its canonical graph", () => {
  const ui = harness();
  ui.search("beta");
  ui.key("Enter");
  assert.equal(ui.element("graph").value, "data/Beta.mtx");
  ui.restore({ result: { config: { ...sampleConfig().defaults, input: "/repository/data/Alpha.mtx" } } });
  assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx");
  assert.equal(ui.element("graph-search").value, "Alpha（1,100 / 2,200） v1");
});

test("the non-modal list remains searchable and selectable and closing it restores results", () => {
  const ui = harness();
  ui.element("graph-list-toggle").dispatch("click");
  assert.equal(ui.element("graph-list-panel").hidden, false);
  assert.equal(ui.element("results-panel").hidden, true);
  assert.equal(ui.element("settings").disabled, false);
  assert.equal(ui.element("graph-list-toggle").getAttribute("aria-expanded"), "true");
  ui.search("alpha v1");
  assert.equal(ui.rows().length, 1);
  assert.equal(ui.element("graph-list-count").textContent, "1 / 3件");
  ui.rows()[0].children[0].children[0].dispatch("click");
  assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx");
  assert.equal(ui.element("graph-list-panel").hidden, false, "choosing a table row must keep the list open");
  assert.equal(ui.rows()[0].classList.contains("is-selected"), true);
  ui.element("graph-list-close").dispatch("click");
  assert.equal(ui.element("graph-list-panel").hidden, true);
  assert.equal(ui.element("results-panel").hidden, false);
  assert.equal(ui.element("graph-list-toggle").getAttribute("aria-expanded"), "false");
  assert.equal(ui.element("graph-list-toggle").focused, true);
});

test("version columns extend numerically and mark merged graph memberships", () => {
  const ui = harness();
  ui.element("graph-list-toggle").dispatch("click");
  assert.deepEqual(ui.element("graph-table-head").children.map((heading) => heading.textContent), ["グラフ", "ノード", "エッジ", "v1", "v2", "v3", "v4", "v10"]);
  const contents = ui.rows().map((row) => row.children.map((cell) => cell.textContent));
  assert.deepEqual(contents[0], ["Alpha", "1,100", "2,200", "◯", "", "", "", ""]);
  assert.deepEqual(contents[1], ["Beta", "40", "50", "", "", "", "◯", "◯"]);
  assert.equal(ui.rows()[0].children[0].children[0].getAttribute("aria-label"), "Alpha（1,100 / 2,200） v1");
});

test("no matches show an empty state while leaving the current graph selected", () => {
  const ui = harness();
  ui.element("graph-list-toggle").dispatch("click");
  ui.search("no such graph");
  assert.equal(ui.options().length, 0);
  assert.equal(ui.rows().length, 0);
  assert.equal(ui.element("graph-no-match").hidden, false);
  assert.equal(ui.element("graph-table-empty").hidden, false);
  assert.equal(ui.element("graph-list-count").textContent, "0 / 3件");
  assert.equal(ui.key("Enter").defaultPrevented, true);
  assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx");
});

test("an open list keeps its filter visible after blur or setting changes, and an empty query restores all rows", () => {
  const ui = harness();
  ui.element("graph-list-toggle").dispatch("click");
  ui.search("no such graph");
  ui.element("graph-search").dispatch("blur");
  assert.equal(ui.element("graph-search").value, "no such graph");
  assert.equal(ui.rows().length, 0);
  ui.element("method").dispatch("change");
  assert.equal(ui.element("graph-search").value, "no such graph");
  assert.equal(ui.rows().length, 0);
  assert.equal(ui.element("settings").disabled, false);
  ui.search("");
  ui.element("graph-search").dispatch("blur");
  assert.equal(ui.element("graph-search").value, "");
  assert.equal(ui.rows().length, 3);
  assert.equal(ui.element("graph-table-empty").hidden, true);
  assert.equal(ui.element("graph-list-count").textContent, "3 / 3件");
});

test("selecting a table row preserves the filter and table nodes for continued keyboard navigation", () => {
  const ui = harness();
  ui.element("graph-list-toggle").dispatch("click");
  ui.search("a");
  ui.element("graph-search").dispatch("blur");
  const rows = [...ui.rows()];
  const button = rows[1].children[0].children[0];
  const tableReplacements = ui.element("graph-table-body").replacements;
  button.focus();
  button.dispatch("click");
  assert.equal(ui.element("graph").value, "data/Beta.mtx");
  assert.equal(ui.element("graph-search").value, "a");
  assert.equal(ui.element("graph-table-body").replacements, tableReplacements);
  assert.equal(ui.rows()[0], rows[0]);
  assert.equal(ui.rows()[1], rows[1]);
  assert.equal(ui.rows()[1].children[0].children[0], button);
  assert.equal(button.focused, true);
  assert.equal(rows[0].classList.contains("is-selected"), false);
  assert.equal(rows[1].classList.contains("is-selected"), true);
  assert.equal(ui.element("graph-list-panel").hidden, false);
});

test("Arrow keys wrap among search matches and Enter chooses without submitting the experiment", () => {
  const data = sampleConfig();
  data.graphs[2].name = "Alpine";
  const ui = harness(data);
  ui.search("al");
  assert.equal(ui.key("ArrowUp").defaultPrevented, true);
  assert.equal(ui.options()[1].classList.contains("is-active"), true);
  assert.equal(ui.element("graph-search").getAttribute("aria-activedescendant"), ui.options()[1].id);
  ui.key("ArrowDown");
  assert.equal(ui.options()[0].classList.contains("is-active"), true);
  ui.key("ArrowDown");
  assert.equal(ui.options()[1].classList.contains("is-active"), true);
  assert.equal(ui.key("Enter").defaultPrevented, true);
  assert.equal(ui.element("graph").value, "data/Beta.mtx");
  assert.equal(ui.element("graph-dropdown").hidden, true);
  assert.equal(ui.element("graph-search").getAttribute("aria-activedescendant"), null);
  assert.equal(ui.key("Enter").defaultPrevented, true, "Enter in a closed combobox must also not submit");
});

test("Enter selects a unique match and leaves multiple unhighlighted matches unchanged", () => {
  const ui = harness();
  ui.search("a");
  assert.equal(ui.key("Enter").defaultPrevented, true);
  assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx");
  assert.equal(ui.element("graph-dropdown").hidden, false);
  ui.search("beta");
  assert.equal(ui.key("Enter").defaultPrevented, true);
  assert.equal(ui.element("graph").value, "data/Beta.mtx");
});

test("Escape and Tab close the dropdown without choosing a candidate or discarding the search", () => {
  const ui = harness();
  ui.search("beta");
  ui.key("ArrowDown");
  assert.equal(ui.key("Escape").defaultPrevented, true);
  assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx");
  assert.equal(ui.element("graph-dropdown").hidden, true);
  assert.equal(ui.element("graph-search").value, "Alpha（1,100 / 2,200） v1");
  ui.element("graph-search").focus();
  assert.equal(ui.element("graph-search").value, "beta");
  assert.equal(ui.options().length, 1);
  assert.equal(ui.key("Tab").defaultPrevented, false, "Tab must retain ordinary keyboard navigation");
  assert.equal(ui.element("graph-dropdown").hidden, true);
  assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx");
});

test("polling completed results preserves the open list, search text, selection, and table nodes", async () => {
  const ui = harness();
  const job = completedJob();
  ui.render(job);
  ui.element("graph-list-toggle").dispatch("click");
  ui.search("beta");
  ui.key("Enter");
  ui.element("graph-search").focus();
  const tableRow = ui.rows()[0];
  const option = ui.options()[0];
  const searchWrites = ui.element("graph-search").valueWrites;
  const tableReplacements = ui.element("graph-table-body").replacements;
  job.summary = "結果更新\n";
  job.log += "Another line\n";
  await ui.poll(job);
  assert.equal(ui.element("graph-search").value, "beta");
  assert.equal(ui.element("graph-search").valueWrites, searchWrites);
  assert.equal(ui.element("graph-dropdown").hidden, false);
  assert.equal(ui.element("graph").value, "data/Beta.mtx");
  assert.equal(ui.element("graph-list-panel").hidden, false);
  assert.equal(ui.element("results-panel").hidden, true);
  assert.equal(ui.rows()[0], tableRow);
  assert.equal(ui.options()[0], option);
  assert.equal(ui.element("graph-table-body").replacements, tableReplacements);
  assert.equal(ui.element("result-text").value, job.summary);
});

test("config refresh retains an exact version input and discovers additional version columns", () => {
  const ui = harness();
  ui.search("alpha v1");
  ui.key("Enter");
  ui.element("graph-list-toggle").dispatch("click");
  const data = sampleConfig();
  data.graphs[1].versions.push("v20");
  ui.reconnect(data);
  assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx");
  assert.equal(ui.element("graph-search").value, "alpha v1", "the visible table filter must survive config refresh");
  assert.equal(ui.element("graph-table-head").children.at(-1).textContent, "v20");
  assert.equal(ui.rows()[0].children.at(-1).textContent, "◯");
  ui.element("graph-list-close").dispatch("click");
  assert.equal(ui.element("graph-search").value, "Alpha（1,100 / 2,200） v1 v20");
});


test("config refresh maps an original selection to a newly preferred v1 input and preserves the open filter", () => {
  const data = sampleConfig();
  data.graphs.splice(1, 1);
  const ui = harness(data);
  assert.equal(ui.element("graph").value, "data/Alpha.mtx");
  ui.element("graph-list-toggle").dispatch("click");
  ui.search("alpha");
  ui.element("graph-search").dispatch("blur");
  ui.reconnect(sampleConfig());
  assert.equal(ui.element("graph").value, "data/v1_lcc/Alpha.mtx");
  assert.equal(ui.element("graph-list-panel").hidden, false);
  assert.equal(ui.element("graph-search").value, "alpha");
  assert.equal(ui.rows().length, 1);
  assert.equal(ui.rows()[0].classList.contains("is-selected"), true);
  assert.equal(ui.rows()[0].children[1].textContent, "1,100");
  ui.element("graph-list-close").dispatch("click");
  assert.equal(ui.element("graph-search").value, "Alpha（1,100 / 2,200） v1");
});
