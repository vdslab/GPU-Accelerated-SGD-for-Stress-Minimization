/** Resume only the exact paused solver state; every other click starts a fresh run. */
export function sameSolverSettings(previous, next) {
  if (!previous || !next) return false;
  const keys = ['method', 'iterations', 'epsilon', 'seed'];
  if (previous.method === 'sparse') keys.push('pivots');
  return keys.every((key) => previous[key] === next[key]);
}

export function runIntent(run, draft) {
  const same = sameSolverSettings(run.config, draft);
  const resume = Boolean(
    run.engine && !run.failed && same && run.epoch > 0 && run.epoch < run.config.iterations,
  );
  if (resume)
    return { resume, label: '続きから再開', hint: '一時停止した位置から計算を続けます。' };
  if (run.history.length)
    return {
      resume: false,
      label: same ? 'もう一度計算' : 'この設定で再計算',
      hint: same
        ? '初期配置から、同じ設定で計算し直せます。'
        : '設定変更あり。初期配置から再計算します。',
    };
  return {
    resume: false,
    label: 'レイアウトを計算',
    hint: 'パラメータを指定して計算してください。',
  };
}
