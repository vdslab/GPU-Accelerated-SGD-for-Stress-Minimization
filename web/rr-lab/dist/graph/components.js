import { forEachEdge } from './edges.js';

/** Union-find needs one typed array, rather than an array object for every vertex. */
export function largestComponent(graph) {
  const parents = new Int32Array(graph.n).fill(-1);
  const root = (vertex) => {
    let r = vertex;
    while (parents[r] >= 0) r = parents[r];
    while (vertex !== r) {
      const next = parents[vertex];
      parents[vertex] = r;
      vertex = next;
    }
    return r;
  };
  let largest = 0;
  forEachEdge(graph, (u, v) => {
    let a = root(u),
      b = root(v);
    if (a === b) return;
    if (parents[a] > parents[b]) [a, b] = [b, a];
    parents[a] += parents[b];
    parents[b] = a;
    if (parents[a] < parents[largest]) largest = a;
  });
  const n = -parents[largest];
  if (n < 2) throw Error('Graph must have at least 2 connected vertices.');
  const ids = new Uint32Array(n);
  if (n === graph.n) {
    for (let i = 0; i < n; i++) ids[i] = i + 1;
    return { ...graph, originalN: graph.n, ids };
  }
  const map = new Int32Array(graph.n).fill(-1);
  let next = 0;
  for (let v = 0; v < graph.n; v++)
    if (root(v) === largest) {
      map[v] = next;
      ids[next++] = v + 1;
    }
  let count = 0;
  forEachEdge(graph, (u, v) => {
    if (map[u] >= 0 && map[v] >= 0) count++;
  });
  const edgeData = new Uint32Array(count * 2);
  next = 0;
  forEachEdge(graph, (u, v) => {
    if (map[u] >= 0 && map[v] >= 0) {
      edgeData[next++] = map[u];
      edgeData[next++] = map[v];
    }
  });
  return { n, originalN: graph.n, ids, edgeData };
}
