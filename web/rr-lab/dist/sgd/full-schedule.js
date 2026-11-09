export function schedule(blocks) {
  const rounds = [];
  if (blocks > 1) {
    const effective = blocks + (blocks % 2),
      m = effective - 1;
    for (let r = 0; r < m; r++) {
      const pairs = [];
      if (m < blocks) pairs.push([r, m]);
      for (let k = 1; k < effective / 2; k++) {
        let u = (r + k) % m,
          v = (r + m - k) % m;
        if (u < blocks && v < blocks) pairs.push([Math.min(u, v), Math.max(u, v)]);
      }
      if (pairs.length) rounds.push(pairs);
    }
  }
  const chunk = Math.max(1, Math.floor(blocks / 2));
  for (let i = 0; i < blocks; i += chunk)
    rounds.push(Array.from({ length: Math.min(chunk, blocks - i) }, (_, k) => [i + k, i + k]));
  return rounds;
}
export function selfRounds(n) {
  return n + (n % 2) - 1;
}
