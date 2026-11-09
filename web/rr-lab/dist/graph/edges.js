/** Packed endpoints cross Worker boundaries without millions of small JS arrays. */
export const edgeCount = (graph) =>
  graph.edgeData ? graph.edgeData.length / 2 : graph.edges.length;
export const edgeAt = (graph, index) =>
  graph.edgeData ? [graph.edgeData[index * 2], graph.edgeData[index * 2 + 1]] : graph.edges[index];
export function forEachEdge(graph, visit) {
  if (graph.edgeData) {
    for (let i = 0; i < graph.edgeData.length; i += 2)
      visit(graph.edgeData[i], graph.edgeData[i + 1], i / 2);
  } else graph.edges.forEach(([u, v], i) => visit(u, v, i));
}
export function* edges(graph) {
  for (let i = 0; i < edgeCount(graph); i++) yield edgeAt(graph, i);
}
