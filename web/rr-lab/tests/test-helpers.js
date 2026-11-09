export function stress(p, d, n) {
  let sum = 0;
  for (let u = 0; u < n; u++)
    for (let v = u + 1; v < n; v++) {
      const target = d[u * n + v];
      const e = (Math.hypot(p[u * 2] - p[v * 2], p[u * 2 + 1] - p[v * 2 + 1]) - target) / target;
      sum += e * e;
    }
  return sum;
}
