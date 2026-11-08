"""Marker-only runtime comparisons, shared by the v5 report generators.

Zheng et al., Graph Drawing by Stochastic Gradient Descent, Figs. 2, 7, 8:
https://arxiv.org/pdf/1710.04626
We adopt markers and min/max whiskers, retaining our hierarchical median
aggregation. The three-method symbol assignment below is specific to our study.
"""
from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiment_results/v5'
METHODS = {'full': ['sgd', 'atomic_sgd', 'rr_sgd'],
           'sparse': ['sparse_sgd', 'atomic_sparse_sgd', 'rr_sparse_sgd']}
SPEED_METRICS = {'iteration_time_ms', 'total_compute_time_ms',
                 'sgd_speedup', 'total_speedup'}


def method_style(method):
    if method.startswith('atomic'):
        return dict(marker='^', color='#d55e00')
    if method.startswith('rr'):
        return dict(marker='o', color='#0072b2')
    return dict(marker='x', color='#555555')


def runtime_points(ax, rows, metric, method, label, scale=lambda x: x, x=None):
    """One graph per marker; whiskers span the five seed medians, not a CI."""
    values = [r[metric] for r in rows]
    medians = [scale(v['median']) for v in values]
    bounds = [[scale(v['median']) - scale(v['min']) for v in values],
              [scale(v['max']) - scale(v['median']) for v in values]]
    style = method_style(method)
    return ax.errorbar(
        [r['nodes'] for r in rows] if x is None else x, medians, yerr=bounds,
        fmt=style['marker'], linestyle='none', color=style['color'],
        markersize=4, markeredgewidth=.8, markerfacecolor='none',
        elinewidth=.65, capsize=1.5, alpha=.8, label=label)


def runtime_axes(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of vertices |V|')
    ax.grid(alpha=.18, linewidth=.5)
    ax.set_axisbelow(True)


def iteration_figure(summary):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
    for ax, (family, methods) in zip(axes, METHODS.items()):
        for method, label in zip(methods, ['CPU', 'Atomic', 'RoundRobin']):
            rows = sorted([r for r in summary if r['method'] == method],
                          key=lambda r: (r['nodes'], r['graph']))
            runtime_points(ax, rows, 'iteration_time_ms', method, label,
                           scale=lambda x: x / 15)
        runtime_axes(ax)
        ax.set_ylabel('Mean time per SGD iteration (ms)')
        ax.set_title(f'{family.capitalize()} ({len(rows)} graphs)')
        ax.legend(frameon=False, fontsize=8)
    return fig


def speedup_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout='constrained')
    for i, (family, methods) in enumerate(METHODS.items()):
        for j, metric in enumerate(['sgd_speedup', 'total_speedup']):
            ax = axes[i, j]
            for method, label in zip(methods[1:], ['Atomic', 'RoundRobin']):
                rows = sorted([r for r in summary if r['method'] == method],
                              key=lambda r: (r['nodes'], r['graph']))
                runtime_points(ax, rows, metric, method, label)
            runtime_axes(ax)
            ax.axhline(1, color='#777777', linewidth=.8, linestyle='--')
            ax.set_ylabel('CPU time / GPU time')
            ax.set_title(f'{family.capitalize()}: ' +
                         ('SGD only' if j == 0 else 'Total compute'))
            ax.legend(frameon=False, fontsize=8)
    return fig


def main():
    """Redraw speed figures from existing aggregates without rerunning reports."""
    from run_v5 import VAULT
    summary = json.loads((BASE / 'tables/comparison-hierarchical.json').read_text())
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    charts = [('iteration_time_ms', 'Mean time per SGD iteration (ms)', lambda x: x/15, 'sgd-time'),
              ('total_compute_time_ms', 'Input to final CPU coordinates (s)', lambda x: x/1000, 'total-time'),
              ('sgd_speedup', 'Paired CPU / method SGD time', lambda x: x, 'sgd-speedup'),
              ('total_speedup', 'Paired CPU / method total time', lambda x: x, 'total-speedup')]
    for family, methods in METHODS.items():
        for metric, ylabel, scale, suffix in charts:
            fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
            for method, label in zip(methods, ['CPU', 'Atomic', 'RoundRobin']):
                rows = sorted([r for r in summary if r['method'] == method],
                              key=lambda r: (r['nodes'], r['graph']))
                runtime_points(ax, rows, metric, method, label, scale)
            runtime_axes(ax)
            if metric.endswith('speedup'):
                ax.axhline(1, color='#777777', linewidth=.8, linestyle='--')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{family.capitalize()}: median and min–max over 5 seeds')
            ax.legend(frameon=False)
            name = f'{family}-{suffix}'
            for ext in ['png', 'pdf']:
                fig.savefig(BASE / f'figures/{name}.{ext}', dpi=190)
            shutil.copy2(BASE / f'figures/{name}.png', VAULT / f'v5_図表/{name}.png')
            plt.close(fig)
    for name, fig in [('f00-iteration-runtime', iteration_figure(summary)),
                      ('f02-runtime', speedup_figure(summary))]:
        for ext in ['png', 'pdf']:
            fig.savefig(BASE / f'paper/figures/{name}.{ext}', dpi=190)
        shutil.copy2(BASE / f'paper/figures/{name}.png', VAULT / f'v5_図表/paper/{name}.png')
        plt.close(fig)
    data = [dict(graph=r['graph'], nodes=r['nodes'], method=r['method'],
                 median_ms=r['iteration_time_ms']['median']/15,
                 min_ms=r['iteration_time_ms']['min']/15,
                 max_ms=r['iteration_time_ms']['max']/15) for r in summary]
    (BASE / 'paper/data/f00-iteration-runtime.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    provenance = dict(
        source='tables/comparison-hierarchical.json', new_layout_runs=0,
        reference='https://arxiv.org/pdf/1710.04626', reference_figures=[2, 7, 8],
        adaptation='Markers and min/max whiskers; method symbols are assigned for this study.',
        symbols={'CPU': 'x', 'Atomic': '^', 'RoundRobin': 'o'},
        aggregation='Median within each seed over 3 repetitions, then median over 5 seeds.',
        whiskers='Minimum and maximum of the 5 seed medians; not a confidence interval.',
        x_axis='Number of vertices; log scale, different graphs, no connecting lines.',
        iteration_time='15-iteration SGD elapsed time / 15, in milliseconds.')
    (BASE / 'paper/runtime-plot-style.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print('Redrew 10 runtime/speedup figures from 282 existing summaries.')


if __name__ == '__main__':
    main()
