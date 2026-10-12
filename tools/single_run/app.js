"use strict";

const $ = (id) => document.getElementById(id);
const DEFAULT_API = "http://127.0.0.1:8765";
let apiUrl = DEFAULT_API;
let config = null;
let connected = false;
let connecting = false;
let busy = false;
let submitting = false;
let activeId = null;
let lastJob = null;
let jobApiUrl = null;
let pollTimer = null;
let pollGeneration = 0;
let graphQuery = "";
let graphChoicesOpen = false;
let graphListOpen = false;
let activeGraphChoice = -1;
let graphOptionButtons = [];
let graphRowButtons = [];
let graphTableRows = [];

function stored(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function remember(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch { /* Storage is optional; execution still works. */ }
}

function normalizeApi(value) {
  const url = new URL(value);
  if (!/^https?:$/.test(url.protocol) || !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) {
    throw new Error("API URLはlocalhost、127.0.0.1、または[::1]のHTTP URLを指定してください。");
  }
  if (url.username || url.password) throw new Error("API URLに認証情報は指定できません。");
  return url.origin;
}

async function request(path, options = {}) {
  const response = await fetch(apiUrl + path, { ...options, signal: AbortSignal.timeout(10000) });
  const type = response.headers.get("content-type") || "";
  if (!type.includes("application/json")) throw new Error("実行APIの応答ではありません。API URLと起動コマンドを確認してください。");
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function setError(message = "") {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function updateControls() {
  $("settings").disabled = !connected || connecting || busy || submitting || !config?.graphs.length;
  $("graph-list-toggle").disabled = !config?.graphs.length;
  for (const button of graphRowButtons) button.disabled = $("settings").disabled;
  if ($("settings").disabled && graphChoicesOpen) closeGraphChoices();
  $("connect").disabled = connecting || submitting;
  $("api-url").disabled = connecting || busy || submitting;
  const label = submitting ? "実行を開始中…" : busy ? "実行中…" : "実験を実行";
  if ($("run").textContent !== label) $("run").textContent = label;
  $("pivots").disabled = !config?.methods.find((method) => method.id === $("method").value)?.sparse;
}

function describeGraph() {
  const graph = config?.graphs.find((item) => item.path === $("graph").value);
  $("graph-info").textContent = graph ? `選択: ${graph.name}${graph.versions?.length ? ` (${graph.versions.join(", ")})` : ""} / ノード: ${graph.nodes?.toLocaleString() ?? "?"} / エッジ: ${graph.edges?.toLocaleString() ?? "?"}` : "";
}

function graphLabel(graph) {
  return `${graph.name}（${graph.nodes?.toLocaleString() ?? "?"} / ${graph.edges?.toLocaleString() ?? "?"}）${graph.versions?.length ? ` ${graph.versions.join(" ")}` : ""}`;
}

function uniqueGraphs(graphs) {
  const groups = new Map();
  for (const graph of graphs) {
    if (!groups.has(graph.name)) groups.set(graph.name, []);
    groups.get(graph.name).push(graph);
  }
  const priority = (graph) => graph.versions?.includes("v1") ? 0 : graph.versions?.length ? 1 : 2;
  return [...groups.values()].map((members) => {
    // Keep the experiment input and its counts together when hiding source copies.
    const selected = [...members].sort((a, b) => priority(a) - priority(b) || a.path.localeCompare(b.path))[0];
    return {
      ...selected,
      versions: [...new Set(members.flatMap((graph) => graph.versions || []))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true })),
      input_paths: [...new Set(members.flatMap((graph) => graph.input_paths || [graph.path]))],
    };
  });
}

function matchingGraphs() {
  const words = graphQuery.normalize("NFKC").toLowerCase().trim().split(/\s+/).filter(Boolean);
  return (config?.graphs || []).filter((graph) => {
    const text = `${graph.name} ${(graph.versions || []).join(" ")}`.normalize("NFKC").toLowerCase();
    return words.every((word) => text.includes(word));
  });
}

function graphVersions() {
  return [...new Set(["v1", "v2", "v3", ...(config?.versions || []), ...(config?.graphs || []).flatMap((graph) => graph.versions || [])])]
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function closeGraphChoices() {
  graphChoicesOpen = false;
  $("graph-dropdown").hidden = true;
  $("graph-search").setAttribute("aria-expanded", "false");
  $("graph-search").removeAttribute("aria-activedescendant");
  const selected = config?.graphs.find((graph) => graph.path === $("graph").value);
  const label = graphListOpen ? graphQuery : selected ? graphLabel(selected) : "";
  if ($("graph-search").value !== label) $("graph-search").value = label;
}

function highlightGraphChoice() {
  graphOptionButtons.forEach((button, index) => button.classList.toggle("is-active", index === activeGraphChoice));
  const button = graphOptionButtons[activeGraphChoice];
  if (button) {
    $("graph-search").setAttribute("aria-activedescendant", button.id);
    button.scrollIntoView({ block: "nearest" });
  } else $("graph-search").removeAttribute("aria-activedescendant");
}

function renderGraphChoices() {
  graphOptionButtons = matchingGraphs().map((graph, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "graph-option";
    button.id = `graph-option-${index}`;
    button.tabIndex = -1;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(graph.path === $("graph").value));
    button.textContent = graphLabel(graph);
    button.addEventListener("click", () => chooseGraph(graph.path));
    return button;
  });
  $("graph-choices").replaceChildren(...graphOptionButtons);
  $("graph-no-match").hidden = graphOptionButtons.length > 0;
  activeGraphChoice = -1;
  highlightGraphChoice();
}

function openGraphChoices() {
  if ($("settings").disabled) return;
  graphChoicesOpen = true;
  $("graph-search").value = graphQuery;
  $("graph-dropdown").hidden = false;
  $("graph-search").setAttribute("aria-expanded", "true");
  renderGraphChoices();
}

function renderGraphTable() {
  if (!graphListOpen) return;
  const versions = graphVersions();
  $("graph-table-head").replaceChildren(...["グラフ", "ノード", "エッジ", ...versions].map((label, index) => {
    const heading = document.createElement("th");
    heading.scope = "col";
    heading.textContent = label;
    if (index >= 3) heading.className = "version-cell";
    return heading;
  }));
  graphRowButtons = [];
  graphTableRows = [];
  const graphs = matchingGraphs();
  $("graph-table-body").replaceChildren(...graphs.map((graph) => {
    const row = document.createElement("tr");
    row.classList.toggle("is-selected", graph.path === $("graph").value);
    graphTableRows.push({ path: graph.path, row });
    const name = document.createElement("td");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "graph-table-name";
    button.textContent = graph.name;
    button.setAttribute("aria-label", graphLabel(graph));
    button.disabled = $("settings").disabled;
    button.addEventListener("click", () => chooseGraph(graph.path));
    graphRowButtons.push(button);
    name.append(button);
    row.append(name);
    for (const value of [graph.nodes, graph.edges]) {
      const cell = document.createElement("td");
      cell.textContent = value?.toLocaleString() ?? "?";
      row.append(cell);
    }
    for (const version of versions) {
      const cell = document.createElement("td");
      cell.className = "version-cell";
      cell.textContent = graph.versions?.includes(version) ? "◯" : "";
      row.append(cell);
    }
    return row;
  }));
  $("graph-list-count").textContent = `${graphs.length} / ${config?.graphs.length || 0}件`;
  $("graph-table-empty").hidden = graphs.length > 0;
}

function setGraphListOpen(open) {
  graphListOpen = open;
  $("graph-list-panel").hidden = !open;
  $("results-panel").hidden = open;
  $("graph-list-toggle").setAttribute("aria-expanded", String(open));
  closeGraphChoices();
  renderGraphTable();
}

function chooseGraph(path) {
  if ($("settings").disabled || !config?.graphs.some((graph) => graph.path === path)) return;
  $("graph").value = path;
  describeGraph();
  closeGraphChoices();
  for (const item of graphTableRows) item.row.classList.toggle("is-selected", item.path === path);
}

function populateConfig(data) {
  if (!Array.isArray(data.graphs) || !Array.isArray(data.methods) || !data.defaults) {
    throw new Error("実行APIの設定データが不正です。");
  }
  const names = ["graph", "method", "iterations", "epsilon", "pivots", "seed", "image_size"];
  const previous = config ? Object.fromEntries(names.map((name) => [name, $("experiment-form").elements.namedItem(name).value])) : null;
  config = { ...data, graphs: uniqueGraphs(data.graphs) };
  $("graph").replaceChildren(...config.graphs.map((graph) => new Option(graphLabel(graph), graph.path)));
  $("method").replaceChildren(...data.methods.map((method) => new Option(method.label, method.id)));
  const settings = previous || data.defaults;
  for (const name of names) {
    const control = $("experiment-form").elements.namedItem(name);
    const value = name === "graph" ? config.graphs.find((graph) => graph.input_paths.includes(String(settings[name])))?.path || String(settings[name]) : String(settings[name]);
    if ([...control.options].some((option) => option.value === value)) control.value = value;
  }
  describeGraph();
  closeGraphChoices();
  renderGraphTable();
}

function savedJobKey() { return `sgd-single-last:${apiUrl}`; }

function restoreSettings(job) {
  const settings = job.result?.config;
  if (!settings) return;
  const graph = config.graphs.find((item) => item.input_paths.some((path) => settings.input === path || settings.input?.endsWith(`/${path}`)));
  if (graph) $("graph").value = graph.path;
  for (const name of ["method", "iterations", "epsilon", "pivots", "seed", "image_size"]) {
    const control = $("experiment-form").elements.namedItem(name);
    const value = String(settings[name]);
    if ([...control.options].some((option) => option.value === value)) control.value = value;
  }
  describeGraph();
  closeGraphChoices();
  renderGraphTable();
}

function clearJob() {
  lastJob = null;
  jobApiUrl = null;
  activeId = null;
  busy = false;
  $("results").hidden = true;
  $("log-details").hidden = true;
  $("log").value = "";
  $("result-text").value = "";
  $("copy-status").textContent = "";
  $("run-status").textContent = "未実行";
  for (const stage of ["before", "after"]) {
    $(`${stage}-link`).hidden = true;
    $(`${stage}-image`).removeAttribute("src");
    $(`${stage}-placeholder`).hidden = false;
    $(`${stage}-placeholder`).textContent = "未実行";
  }
  $("image-dialog").close();
}

async function connect(automatic = false) {
  clearTimeout(pollTimer);
  const generation = ++pollGeneration;
  connecting = true;
  updateControls();
  $("connection-status").textContent = "接続中…";
  const requested = $("api-url").value;
  const candidates = [requested];
  if (automatic && !new URLSearchParams(location.search).has("api") && !stored("sgd-single-api") && /^https?:$/.test(location.protocol)) {
    candidates.unshift(location.origin);
  }
  let failure = null;
  let data = null;
  const previousApi = apiUrl;
  for (const candidate of [...new Set(candidates)]) {
    try {
      apiUrl = normalizeApi(candidate);
      data = await request("/api/config");
      if (generation !== pollGeneration) return;
      populateConfig(data);
      break;
    } catch (error) { failure = error; data = null; }
  }
  if (generation !== pollGeneration) return;
  if (apiUrl !== previousApi) clearJob();
  if (!data) {
    connected = false;
    connecting = false;
    $("connection-status").textContent = "未接続";
    $("connection-settings").open = true;
    setError(`接続できませんでした: ${failure?.message}\n接続設定の起動コマンドをターミナルで実行し、再接続してください。`);
    updateControls();
    return;
  }
  connected = true;
  connecting = false;
  $("api-url").value = apiUrl;
  if (!automatic) remember("sgd-single-api", apiUrl);
  $("connection-status").textContent = "接続済み";
  $("connection-status").title = apiUrl;
  setError();
  activeId = data.active_run || stored(savedJobKey());
  busy = Boolean(data.active_run);
  if (!activeId) clearJob();
  if (!config.graphs.length) setError("グラフがありません。data/ に .mtx ファイルを置いて再読み込みしてください。");
  updateControls();
  if (activeId) await poll(generation, true);
}

function artifactUrl(path) { return new URL(path, jobApiUrl || apiUrl).href; }

function renderJob(job) {
  const previous = lastJob;
  const newRun = !previous || previous.id !== job.id || jobApiUrl !== apiUrl;
  lastJob = job;
  jobApiUrl = apiUrl;
  busy = job.state === "running";
  const labels = { running: "実行中…", success: "完了", failure: "失敗" };
  const status = labels[job.state] || job.state;
  if ($("run-status").textContent !== status) $("run-status").textContent = status;
  $("log-details").hidden = false;
  if (newRun || previous.log !== job.log) {
    const log = $("log");
    const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
    log.value = job.log || "";
    if (atEnd) log.scrollTop = log.scrollHeight;
  }
  if (job.state === "failure") $("log-details").open = true;
  $("results").hidden = !job.run_directory && !job.summary;
  if (newRun || previous.summary !== job.summary) {
    $("result-text").value = job.summary || "実行中…";
    $("copy-status").textContent = "";
  }
  $("copy-result").disabled = !job.summary;
  $("copy-json").disabled = !job.artifacts?.result;

  const record = job.result?.record;
  if (newRun || JSON.stringify(previous.result?.record) !== JSON.stringify(record)) {
    $("metric-rows").replaceChildren();
    $("metrics").hidden = !record;
    if (record) {
      const time = (value) => Number.isFinite(value) ? `${value.toFixed(3)} ms` : "n/a";
      const rows = [
        ["頂点数 / 辺数", `${record.nodes.toLocaleString()} / ${record.edges.toLocaleString()}`],
        ["SGD反復（全反復の合計）", time(record.iteration_time_ms)],
        ["処理全体（読込除外・初期化込み）", time(record.algorithm_time_cold_ms)],
        ["処理全体（初期化を除く・差引値）", time(record.algorithm_time_warm_ms)],
        ["入力読込＋処理全体（初期化込み）", time(record.cli_total_time_cold_ms)],
        ["Stress", record.stress_value == null ? "n/a" : `${record.stress_value} (${record.stress_kind})`],
      ];
      for (const [label, value] of rows) {
        const row = document.createElement("tr");
        const heading = document.createElement("th");
        heading.scope = "row";
        heading.textContent = label;
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(heading, cell);
        $("metric-rows").append(row);
      }
    }
  }

  if (newRun || ["info", "result"].some((key) => previous.artifacts?.[key] !== job.artifacts?.[key])) {
    $("artifact-links").replaceChildren();
    for (const [key, label] of [["info", "TXT"], ["result", "JSON"]]) {
      if (!job.artifacts?.[key]) continue;
      const link = document.createElement("a");
      link.href = artifactUrl(job.artifacts[key]);
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = label;
      $("artifact-links").append(link);
    }
  }
  for (const stage of ["before", "after"]) {
    const path = job.artifacts?.[`${stage}_image`];
    const link = $(`${stage}-link`);
    link.hidden = !path;
    $(`${stage}-placeholder`).hidden = Boolean(path);
    const placeholder = job.state === "failure" ? "画像は生成されませんでした" : "画像の生成待ち";
    if ($(`${stage}-placeholder`).textContent !== placeholder) $(`${stage}-placeholder`).textContent = placeholder;
    if (path) {
      const url = artifactUrl(path);
      if ($(`${stage}-image`).src !== url) $(`${stage}-image`).src = url;
    } else $(`${stage}-image`).removeAttribute("src");
  }
  const error = job.error || "";
  if ($("error").textContent !== error || $("error").hidden !== !error) setError(error);
  updateControls();
}

async function poll(generation, restoring = false) {
  if (!activeId || generation !== pollGeneration) return;
  try {
    const job = await request(`/api/runs/${encodeURIComponent(activeId)}`);
    if (generation !== pollGeneration) return;
    if (restoring) restoreSettings(job);
    renderJob(job);
    remember(savedJobKey(), job.id);
    if (busy) pollTimer = setTimeout(() => poll(generation), 1000);
  } catch (error) {
    if (generation !== pollGeneration) return;
    if (restoring && !busy && error.status === 404) {
      remember(savedJobKey(), null);
      clearJob();
    } else {
      connected = false;
      $("connection-status").textContent = "接続切断";
      $("connection-settings").open = true;
      setError(`実行状況を取得できません: ${error.message}\n「接続・再読み込み」で同じ実験の状態を確認できます。`);
    }
    updateControls();
  }
}

$("experiment-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!connected || busy || submitting) return;
  const fields = Object.fromEntries(new FormData(event.currentTarget));
  fields.pivots = Number($("pivots").value);
  fields.iterations = Number(fields.iterations);
  fields.epsilon = Number(fields.epsilon);
  setGraphListOpen(false);
  submitting = true;
  updateControls();
  setError();
  try {
    const job = await request("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(fields) });
    activeId = job.id;
    remember(savedJobKey(), job.id);
    clearTimeout(pollTimer);
    const generation = ++pollGeneration;
    renderJob(job);
    await poll(generation);
  } catch (error) {
    setError(`実行を開始できません: ${error.message}\n接続・再読み込みで、API側の実行状況を確認できます。`);
  } finally {
    submitting = false;
    updateControls();
  }
});

async function copyText(text) {
  $("copy-status").textContent = "コピー中…";
  try {
    await navigator.clipboard.writeText(text);
    $("copy-status").textContent = "コピーしました。";
  } catch {
    $("result-text").value = text;
    $("result-text").focus();
    $("result-text").select();
    $("copy-status").textContent = document.execCommand("copy") ? "コピーしました。" : "テキストを選択しました。⌘C / Ctrl+Cでコピーしてください。";
  }
}

$("copy-result").addEventListener("click", () => copyText(lastJob?.summary || ""));
$("copy-json").addEventListener("click", async () => {
  if (!lastJob?.artifacts?.result) return;
  $("copy-status").textContent = "JSONを取得中…";
  try {
    const response = await fetch(artifactUrl(lastJob.artifacts.result), { signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await copyText(await response.text());
  } catch (error) { $("copy-status").textContent = `JSONをコピーできません: ${error.message}`; }
});
$("connect").addEventListener("click", () => connect());
$("method").addEventListener("change", updateControls);
$("graph").addEventListener("change", describeGraph);
$("graph-list-toggle").addEventListener("click", () => setGraphListOpen(!graphListOpen));
$("graph-list-close").addEventListener("click", () => {
  setGraphListOpen(false);
  $("graph-list-toggle").focus();
});
$("graph-search").addEventListener("focus", openGraphChoices);
$("graph-search").addEventListener("click", () => {
  if (!graphChoicesOpen) openGraphChoices();
});
$("graph-search").addEventListener("input", () => {
  graphQuery = $("graph-search").value;
  graphChoicesOpen = true;
  $("graph-dropdown").hidden = false;
  $("graph-search").setAttribute("aria-expanded", "true");
  renderGraphChoices();
  renderGraphTable();
});
$("graph-search").addEventListener("keydown", (event) => {
  if (["ArrowDown", "ArrowUp"].includes(event.key)) {
    event.preventDefault();
    if (!graphChoicesOpen) openGraphChoices();
    const count = graphOptionButtons.length;
    if (count) {
      activeGraphChoice = activeGraphChoice < 0 ? (event.key === "ArrowDown" ? 0 : count - 1) : (activeGraphChoice + (event.key === "ArrowDown" ? 1 : -1) + count) % count;
      highlightGraphChoice();
    }
  } else if (event.key === "Enter") {
    event.preventDefault();
    if (graphChoicesOpen) {
      const index = activeGraphChoice >= 0 ? activeGraphChoice : graphOptionButtons.length === 1 ? 0 : -1;
      if (index >= 0) chooseGraph(matchingGraphs()[index].path);
    }
  } else if (event.key === "Escape" || event.key === "Tab") {
    if (event.key === "Escape" && graphChoicesOpen) event.preventDefault();
    closeGraphChoices();
  }
});
$("graph-dropdown-toggle").addEventListener("mousedown", (event) => event.preventDefault());
$("graph-dropdown-toggle").addEventListener("click", () => {
  if (graphChoicesOpen) closeGraphChoices();
  else {
    $("graph-search").focus();
    openGraphChoices();
  }
});
$("graph-choices").addEventListener("mousedown", (event) => event.preventDefault());
$("graph-search").addEventListener("blur", closeGraphChoices);
document.addEventListener("pointerdown", (event) => {
  if (graphChoicesOpen && !$("graph-picker").contains(event.target)) closeGraphChoices();
});

$("copy-command").addEventListener("click", async () => {
  const command = $("server-command").textContent;
  const button = $("copy-command");
  button.disabled = true;
  let copied = false;
  try {
    await navigator.clipboard.writeText(command);
    copied = true;
  } catch {
    const temporary = document.createElement("textarea");
    temporary.value = command;
    temporary.style.cssText = "position:fixed;left:-9999px;top:0";
    document.body.append(temporary);
    try {
      temporary.focus();
      temporary.select();
      copied = document.execCommand("copy");
    } catch { /* Keep the command available for manual copying. */ }
    finally { temporary.remove(); }
  }
  button.disabled = false;
  button.focus({ preventScroll: true });
  $("command-copy-status").textContent = copied ? "コピーしました。" : "コマンドを選択してコピーしてください。";
  if (!copied) {
    const selection = window.getSelection();
    selection.removeAllRanges();
    const range = document.createRange();
    range.selectNodeContents($("server-command"));
    selection.addRange(range);
  }
});

for (const stage of ["before", "after"]) {
  $(`${stage}-link`).addEventListener("click", () => {
    const source = $(`${stage}-image`);
    $("expanded-image").src = source.src;
    $("expanded-image").alt = source.alt;
    $("image-title").textContent = stage === "before" ? "Before（初期配置）" : "After（実行後）";
    $("image-dialog").showModal();
  });
}
$("close-image").addEventListener("click", () => $("image-dialog").close());
$("image-dialog").addEventListener("click", (event) => {
  if (event.target !== $("image-dialog")) return;
  const box = event.currentTarget.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) event.currentTarget.close();
});

$("api-url").value = new URLSearchParams(location.search).get("api") || stored("sgd-single-api") || DEFAULT_API;
connect(true);
