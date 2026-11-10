import { isSparseMethod } from '../sgd/learning-rate.js';

/** Resume only the exact paused solver state; every other click starts a fresh run. */
export function sameSolverSettings(previous, next) {
  if (!previous || !next) return false;
  const keys = ['method', 'iterations', 'epsilon', 'seed'];
  if (isSparseMethod(previous.method)) keys.push('pivots');
  return keys.every((key) => previous[key] === next[key]);
}

export function runIntent(run, draft) {
  const same = sameSolverSettings(run.config, draft);
  const resume = Boolean(
    run.engine && !run.failed && same && run.epoch > 0 && run.epoch < run.config.iterations,
  );
  if (resume) return { resume, label: 'Resume', hint: 'Continue from the paused iteration.' };
  if (run.history.length)
    return {
      resume: false,
      label: same ? 'Run Again' : 'Recalculate',
      hint: same
        ? 'Run again from the initial layout with the same settings.'
        : 'Settings changed. Recalculates from the initial layout.',
    };
  return {
    resume: false,
    label: 'Compute Layout',
    hint: 'Set parameters and click Compute Layout.',
  };
}
