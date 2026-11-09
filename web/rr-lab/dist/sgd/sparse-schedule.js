/** Splitting a matching across buffers preserves its disjoint endpoint writes. */
export function matchingDispatches(offsets, roundOrder, chunkPairs) {
  const dispatches = [];
  for (const round of roundOrder) {
    let start = offsets[round],
      end = offsets[round + 1];
    while (start < end) {
      const chunk = Math.floor(start / chunkPairs);
      const next = Math.min(end, (chunk + 1) * chunkPairs);
      dispatches.push({
        one: false,
        chunk,
        start: start - chunk * chunkPairs,
        count: next - start,
      });
      start = next;
    }
  }
  return dispatches;
}
