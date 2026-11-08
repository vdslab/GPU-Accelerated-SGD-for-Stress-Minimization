"""Create study tables, static paper figures, and a report in a result version directory."""
import argparse
import csv
import hashlib
import json
import subprocess
from functools import lru_cache
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

try:
    from artifact_paths import provenance_path, resolve_artifact_path
except ModuleNotFoundError:
    from experiments.artifact_paths import provenance_path, resolve_artifact_path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'experiment_results/v1'
METHODS = ['sgd', 'atomic_sgd', 'rr_sgd', 'sparse_sgd', 'atomic_sparse_sgd', 'rr_sparse_sgd']
LABELS = {'sgd': 'CPU SGD', 'atomic_sgd': 'Atomic SGD', 'rr_sgd': 'RR-SGD',
          'sparse_sgd': 'CPU Sparse SGD', 'atomic_sparse_sgd': 'Atomic Sparse SGD', 'rr_sparse_sgd': 'RR-Sparse SGD'}
COLORS = {'sgd': '#555555', 'atomic_sgd': '#d55e00', 'rr_sgd': '#0072b2',
          'sparse_sgd': '#555555', 'atomic_sparse_sgd': '#cc79a7', 'rr_sparse_sgd': '#009e73'}
FAMILIES = {'full': METHODS[:3], 'sparse': METHODS[3:]}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelsize': 10, 'legend.fontsize': 8, 'svg.fonttype': 'none',
                     'savefig.dpi': 240, 'svg.hashsalt': 'stress-sgd-v1'})


def dump_csv(path, records):
    if not records:
        return
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values, statistic='mean'):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return [None, None, None]
    function = np.mean if statistic == 'mean' else np.median
    center = float(function(x))
    if len(x) == 1:
        return [center, center, center]
    rng = np.random.default_rng(20261011)
    samples = function(x[rng.integers(0, len(x), (10000, len(x)))], axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return [center, float(low), float(high)]


def save_figure(fig, name, base=BASE):
    fig.savefig(base / 'figures' / (name + '.png'), bbox_inches='tight')
    fig.savefig(base / 'figures' / (name + '.svg'), bbox_inches='tight',
                metadata={'Date': None, 'Creator': 'Stress SGD ' + base.name})
    svg = base / 'figures' / (name + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def resolve_run_path(value, relocation, *, prefer_relocated=False):
    path = Path(value)
    if relocation:
        original = Path(relocation['original_root'])
        stored = Path(relocation['stored_root'])
        try:
            path = ROOT / stored / path.relative_to(original)
        except ValueError:
            # Inputs may live outside the raw directory in the original checkout.
            if original.parts[-len(stored.parts):] == stored.parts:
                original_checkout = original.parents[len(stored.parts) - 1]
                try:
                    path = ROOT / path.relative_to(original_checkout)
                except ValueError:
                    pass
    return resolve_artifact_path(path, ROOT, prefer_relocated=prefer_relocated)


def load_dataset_catalog(base=BASE):
    datasets = json.loads((base / 'config/datasets.json').read_text())
    archived = base.resolve().is_relative_to(ROOT / 'archive')
    for item in datasets:
        for field in ['source_path', 'path', 'vertex_map_path']:
            if item.get(field):
                item[field] = str(resolve_artifact_path(
                    item[field], ROOT, prefer_relocated=archived))
    return datasets


def load_data(base=BASE):
    records, sources = [], []
    expectations = {}
    conditions = {}
    for folder in sorted((base / 'raw').iterdir()) if (base / 'raw').is_dir() else []:
        if not folder.is_dir() or not (folder / 'manifest.json').is_file():
            continue
        path = folder / 'results.jsonl'
        if not path.exists():
            continue
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['repetitions'] != 1 or any(len(manifest[field]) != 1
                                               for field in ('iterations', 'epsilon', 'pivots')):
            raise SystemExit('summarize_v1 supports one iteration/epsilon/pivot condition and '
                             'repetitions=1; use aggregate_results.py for condition sweeps.')
        for field in ('iterations', 'epsilon', 'pivots'):
            value = manifest[field][0]
            if field in conditions and conditions[field] != value:
                raise SystemExit('summarize_v1 cannot combine different study conditions; '
                                 'use aggregate_results.py instead.')
            conditions[field] = value
        for dataset in manifest['datasets']:
            for method in manifest['methods']:
                key = (dataset['name'], method)
                if key in expectations:
                    raise SystemExit('Duplicate dataset/method across manifests; '
                                     'use aggregate_results.py for independent experiments.')
                expectations[key] = len(manifest['seeds'])
        sources.append({'path': provenance_path(path, ROOT), 'sha256': sha(path),
                        'artifact_relocation_sha256': sha(folder/'artifact-relocation.json')
                        if (folder/'artifact-relocation.json').exists() else None})
        # Keep retry history in raw; use only latest attempt for each run id.
        latest = {}
        relocation_path = folder / 'artifact-relocation.json'
        relocation = json.loads(relocation_path.read_text()) if relocation_path.exists() else None
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row['experiment_id'] = folder.name
            for field in ['input_path', 'final_positions_path', 'vertex_map_path', 'stderr_log_path']:
                if row.get(field):
                    row[field] = str(resolve_run_path(
                        row[field], relocation,
                        prefer_relocated=folder.resolve().is_relative_to(ROOT / 'archive')))
            latest[row['run_id']] = row
        records.extend(latest.values())
    return records, sources, expectations


def seed_count_label(expectations, names):
    values = sorted({count for (dataset, _), count in expectations.items() if dataset in names})
    return '/'.join(str(value) for value in values) if values else '0'


# Only Atomic Sparse's additive run may reuse earlier baselines across commits.
# Verify the actual Git trees; a changed existing implementation invalidates reuse.
SHARED_SOURCE_PATHS = ('baseline-sgd-non-gpu', 'vram-lock-native', 'rr_gpu',
                       'baseline-sparse-sgd-non-gpu', 'sparse-sgd-gpu', 'experiment-common')
SOURCE_PATH_ALIASES = {
    'baseline-sgd-non-gpu': 'implementations/full/baseline-sgd-non-gpu',
    'vram-lock-native': 'implementations/full/vram-lock-native',
    'rr_gpu': 'implementations/full/rr_gpu',
    'baseline-sparse-sgd-non-gpu': 'implementations/sparse/baseline-sparse-sgd-non-gpu',
    'sparse-sgd-gpu': 'implementations/sparse/sparse-sgd-gpu',
    'experiment-common': 'shared/experiment-common',
    'vram-lock-sparse': 'implementations/sparse/vram-lock-sparse',
}


@lru_cache(maxsize=None)
def source_extension_evidence(baseline_commit, added_commit):
    def tree(commit, path):
        for candidate in (path, SOURCE_PATH_ALIASES[path]):
            try:
                value = subprocess.check_output(
                    ['git', '-C', str(ROOT), 'rev-parse', '--verify', f'{commit}:{candidate}'],
                    text=True, stderr=subprocess.DEVNULL).strip()
                return candidate, value
            except subprocess.CalledProcessError:
                pass
        raise subprocess.CalledProcessError(1, ['git', 'rev-parse', f'{commit}:{path}'])
    try:
        trees = {}
        for path in SHARED_SOURCE_PATHS:
            baseline_path, baseline_tree = tree(baseline_commit, path)
            added_path, added_tree = tree(added_commit, path)
            trees[path] = {'baseline_tree': baseline_tree, 'added_tree': added_tree,
                           'baseline_path': baseline_path, 'added_path': added_path}
        if any(v['baseline_tree'] != v['added_tree'] for v in trees.values()):
            return None
        atomic_path, atomic_tree = tree(added_commit, 'vram-lock-sparse')
    except subprocess.CalledProcessError:
        return None
    return {'baseline_commit': baseline_commit, 'added_commit': added_commit,
            'shared_source_trees': trees, 'atomic_sparse_tree': atomic_tree,
            'atomic_sparse_path': atomic_path}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=BASE,
                        help='result version directory containing config/raw/tables/figures/report')
    return parser.parse_args(argv)


def main(argv=None):
    base = parse_args(argv).results_dir.resolve()
    for name in ['tables', 'figures', 'report']:
        (base / name).mkdir(parents=True, exist_ok=True)
    rows, sources, expectations = load_data(base)
    if not rows:
        raise SystemExit(f'No results in {base}; run experiments first.')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['method'])].append(row)
    available = {(r['dataset'], r['method'], r['seed']): r for r in rows if r['status'] == 'success'}
    if len(available) != sum(r['status'] == 'success' for r in rows):
        raise SystemExit('Duplicate dataset/method/seed measurements; use aggregate_results.py.')
    iterations = {r['iterations'] for r in rows}
    if len(iterations) != 1:
        raise SystemExit('Multiple iteration conditions; use aggregate_results.py.')
    iterations = next(iter(iterations))
    datasets = load_dataset_catalog(base)
    names = [item['name'] for item in datasets if any((item['name'], m) in groups for m in METHODS)]
    small = [item['name'] for item in datasets[:10] if item['name'] in names]
    large = [item['name'] for item in datasets[10:] if item['name'] in names]
    small_seeds = seed_count_label(expectations, small)
    large_seeds = seed_count_label(expectations, large)
    dataset_info = {item['name']: item for item in datasets}
    paired = defaultdict(list)
    validation = {'errors': [], 'warnings': [], 'formal_publication_ready': False,
                  'source_extensions': []}
    for r in available.values():
        baseline = 'sgd' if r['family'] == 'full' else 'sparse_sgd'
        ref = available.get((r['dataset'], baseline, r['seed']))
        if ref is None:
            continue
        for key in ['input_sha256', 'initial_positions_sha256', 'nodes', 'edges', 'constraints',
                    'iterations', 'epsilon', 'pivots', 'cpu_model']:
            if r[key] != ref[key]:
                validation['errors'].append(f"{r['run_id']}: mismatched {key}")
        if r['git_commit'] != ref['git_commit']:
            evidence = (source_extension_evidence(ref['git_commit'], r['git_commit'])
                        if r['method'] == 'atomic_sparse_sgd' else None)
            if evidence is None:
                validation['errors'].append(f"{r['run_id']}: mismatched git_commit without unchanged-source evidence")
            elif evidence not in validation['source_extensions']:
                validation['source_extensions'].append(evidence)
        if r['method'] == 'atomic_sparse_sgd':
            gpu_ref = available.get((r['dataset'], 'rr_sparse_sgd', r['seed']))
            if gpu_ref and any(r[k] != gpu_ref[k] for k in ['gpu_name', 'gpu_backend']):
                validation['errors'].append(f"{r['run_id']}: mismatched GPU environment")
        if r['family'] == 'sparse' and r['preprocess_sha256'] != ref['preprocess_sha256']:
            validation['errors'].append(f"{r['run_id']}: mismatched sparse preprocessing")
        if r['stress_kind'] != ref['stress_kind'] or r['stress_seed'] != ref['stress_seed'] or r['stress_samples'] != ref['stress_samples']:
            validation['errors'].append(f"{r['run_id']}: mismatched stress evaluator")
        if r['git_dirty']:
            validation['errors'].append(f"{r['run_id']}: dirty source tree")
        if r['method'] in ['atomic_sgd', 'atomic_sparse_sgd']:
            planned = r['constraints'] * r['iterations']
            if (r['attempted_updates'] != planned or r['completed_updates'] != planned
                    or r['retry_failures'] != 0):
                validation['errors'].append(f"{r['run_id']}: atomic update counters do not match planned constraints")
        if r.get('retry_failures', 0):
            validation['errors'].append(f"{r['run_id']}: {r['retry_failures']} incomplete atomic updates")
        if not (np.isfinite(r['stress_value']) and r['iteration_time_ms'] > 0):
            validation['errors'].append(f"{r['run_id']}: invalid measurement")
        paired[(r['dataset'], r['method'])].append({
            'seed': r['seed'], 'stress_ratio': r['stress_value'] / ref['stress_value'],
            'iteration_speedup': ref['iteration_time_ms'] / r['iteration_time_ms']})
    summaries = []
    for name in names:
        for method in METHODS:
            expected = expectations.get((name, method))
            if expected is None:
                continue
            measured = [r for r in groups[(name, method)] if r['status'] == 'success']
            failed = [r for r in groups[(name, method)] if r['status'] != 'success']
            pairs = paired[(name, method)]
            times = [r['iteration_time_ms'] for r in measured]
            stress = [r['stress_value'] for r in measured]
            quality_ci = interval([r['stress_ratio'] for r in pairs])
            speed_ci = interval([r['iteration_speedup'] for r in pairs], 'median')
            summaries.append(dict(dataset=name, method=method, family='full' if method in METHODS[:3] else 'sparse',
                nodes=dataset_info[name]['nodes'], edges=dataset_info[name]['edges'],
                constraints=measured[0]['constraints'] if measured else None,
                rounds_median=float(np.median([r['rounds'] for r in measured if r['rounds'] is not None])) if any(r['rounds'] is not None for r in measured) else None,
                dispatches_median=float(np.median([r['dispatches'] for r in measured if r['dispatches'] is not None])) if any(r['dispatches'] is not None for r in measured) else None,
                retry_failures_total=sum(r.get('retry_failures') or 0 for r in measured),
                expected=expected, success=len(measured), failures=len(failed), missing=expected-len(measured)-len(failed),
                stress_kind=measured[0]['stress_kind'] if measured else None,
                iteration_median_ms=float(np.median(times)) if times else None,
                iteration_q1_ms=float(np.quantile(times,.25)) if times else None,
                iteration_q3_ms=float(np.quantile(times,.75)) if times else None,
                stress_mean=float(np.mean(stress)) if stress else None,
                stress_min=min(stress) if stress else None, stress_max=max(stress) if stress else None,
                stress_ratio_mean=quality_ci[0], stress_ratio_ci_low=quality_ci[1], stress_ratio_ci_high=quality_ci[2],
                paired_iteration_speedup_median=speed_ci[0], speedup_ci_low=speed_ci[1], speedup_ci_high=speed_ci[2],
                paired_seeds=len(pairs)))
    lookup = {(s['dataset'],s['method']):s for s in summaries}
    for s in summaries:
        if s['success'] < s['expected']:
            validation['warnings'].append(f"{s['dataset']} {s['method']}: {s['success']}/{s['expected']} successful")
    validation['warnings'].extend([f'Small-graph planned seeds: {small_seeds}; planned final quality requires 25 seeds.',
                                   f'Large-graph planned seeds: {large_seeds}; planned final timing requires at least 10 seeds.',
                                   'Publication readiness is not certified by this exploratory study summary.'])
    if validation['source_extensions']:
        validation['warnings'].append('Atomic Sparse was measured in an added batch; '
                                      'earlier methods reuse unchanged-source baseline runs.')
    dump_csv(base/'tables/runs.csv',rows)
    dump_csv(base/'tables/summary.csv',summaries)
    flat_pairs=[dict(dataset=d,method=m,**item) for (d,m),items in paired.items() for item in items]
    dump_csv(base/'tables/paired-ratios.csv',flat_pairs)
    (base/'tables/validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    (base/'tables/summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    table=['| Graph | Method | n | Success | Stress | SGD time, ms (median [Q1, Q3]) | Speedup | Stress ratio |',
           '|---|---|---:|---:|---|---|---|---|']
    latex=[f'% {base.name} exploratory results: planned {small_seeds} small-graph seeds, {large_seeds} large-graph seeds',
           r'\begin{tabular}{llrrrr}',r'\hline',r'Graph & Method & Runs & Stress & SGD time (ms) & Speedup \\',r'\hline']
    for s in summaries:
        if s['success']:
            t=f"{s['iteration_median_ms']:.2f} [{s['iteration_q1_ms']:.2f}, {s['iteration_q3_ms']:.2f}]"
            sp=f"{s['paired_iteration_speedup_median']:.2f}" if s['paired_seeds'] else 'N/A'
            qr=f"{s['stress_ratio_mean']:.3f}" if s['paired_seeds'] else 'N/A'
            quality=f"{s['stress_mean']:.2f} ({s['stress_kind']})"
            graph_label = s['dataset'].replace('_', '\\_')
            latex.append(f"{graph_label} & {LABELS[s['method']]} & {s['success']} & {s['stress_mean']:.2f} & {s['iteration_median_ms']:.2f} & {sp}" + ' \\\\')
        else:
            t=sp=qr=quality='N/A'
        table.append(f"| {s['dataset']} | {LABELS[s['method']]} | {s['nodes']} | {s['success']}/{s['expected']} | {quality} | {t} | {sp} | {qr} |")
    latex.extend([r'\hline',r'\end{tabular}'])
    (base/'tables/results.md').write_text('\n'.join(table)+'\n')
    (base/'tables/results.tex').write_text('\n'.join(latex)+'\n')

    # Paired means/medians with seeded bootstrap confidence intervals; no pooled graph averages.
    for kind, field, lowfield, highfield, ylabel, filename in [
        ('quality','stress_ratio_mean','stress_ratio_ci_low','stress_ratio_ci_high','Full-stress ratio to family CPU baseline','01_stress_ratio'),
        ('speed','paired_iteration_speedup_median','speedup_ci_low','speedup_ci_high','Host SGD iteration speedup (CPU / GPU)','02_iteration_speedup')]:
        fig, axes=plt.subplots(2,1,figsize=(11,6.4),sharex=True)
        for ax,(family,methods) in zip(axes,FAMILIES.items()):
            ax.axhline(1,color='#666666',linestyle='--',linewidth=.8)
            if kind=='quality':ax.axhline(1.05,color='#999999',linestyle=':',linewidth=1,label='5% margin (exploratory)')
            for mi,m in enumerate(methods[1:]):
                xpos=[]; values=[]; lows=[]; highs=[]
                for j,d in enumerate(small):
                    s=lookup.get((d,m))
                    if s and s[field] is not None:
                        xpos.append(j+(mi-(len(methods[1:])-1)/2)*.18)
                        values.append(s[field]);lows.append(max(0,s[field]-s[lowfield]));highs.append(max(0,s[highfield]-s[field]))
                ax.errorbar(xpos,values,yerr=[lows,highs],fmt='o' if mi==0 else 's',color=COLORS[m],
                            label=LABELS[m],capsize=3,markersize=4,linewidth=1)
            ax.set_ylabel(ylabel);ax.set_title(('(a) Full SGD' if family=='full' else '(b) Sparse SGD')+f'; planned {small_seeds} seeds',loc='left')
            if kind=='speed':ax.set_yscale('log')
            ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,loc='best')
        axes[-1].set_xticks(range(len(small)),small,rotation=30,ha='right')
        fig.tight_layout();save_figure(fig, filename, base)

    fig,axes=plt.subplots(2,1,figsize=(11,6.4),sharex=True)
    for ax,(family,methods) in zip(axes,FAMILIES.items()):
        for mi,m in enumerate(methods):
            x=[];y=[];lo=[];hi=[]
            for j,d in enumerate(small):
                s=lookup.get((d,m))
                if s and s['success']:
                    x.append(j+(mi-(len(methods)-1)/2)*.22);y.append(s['iteration_median_ms'])
                    lo.append(y[-1]-s['iteration_q1_ms']);hi.append(s['iteration_q3_ms']-y[-1])
            ax.bar(x,y,width=.2,color=COLORS[m],label=LABELS[m],yerr=[lo,hi],capsize=2)
        ax.set_yscale('log');ax.set_ylabel(f'{iterations}-iteration host time (ms)')
        ax.set_title(f'{family.capitalize()} SGD: median and IQR',loc='left');ax.legend(frameon=False);ax.grid(axis='y',alpha=.18)
    axes[-1].set_xticks(range(len(small)),small,rotation=30,ha='right')
    fig.tight_layout();save_figure(fig, '03_iteration_time', base)

    fig,axes=plt.subplots(2,5,figsize=(13,5.6))
    for ax,d in zip(axes.flat,small):
        values=[[r['stress_value'] for r in groups[(d,m)] if r['status']=='success'] for m in METHODS]
        nonempty=[(i,v) for i,v in enumerate(values) if v]
        if nonempty:
            box=ax.boxplot([v for i,v in nonempty],positions=[i for i,v in nonempty],patch_artist=True,widths=.6,
                           showfliers=True,medianprops={'color':'black','linewidth':1})
            for artist,(i,v) in zip(box['boxes'],nonempty):artist.set_facecolor(COLORS[METHODS[i]]);artist.set_alpha(.6)
        ax.set_title(d);ax.set_xticks(range(len(METHODS)),['CPU','AT','RR','S-CPU','S-AT','S-RR'],rotation=45)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(axis='y',alpha=.15)
    axes[0,0].set_ylabel('Exact full stress');axes[1,0].set_ylabel('Exact full stress')
    fig.tight_layout();save_figure(fig, '04_final_stress_distribution', base)

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(family,methods) in zip(axes,FAMILIES.items()):
        for m in methods[1:]:
            for d in small:
                s=lookup.get((d,m))
                if s and s['paired_seeds']:
                    ax.scatter(s['paired_iteration_speedup_median'],s['stress_ratio_mean'],c=COLORS[m],s=22)
                    if d in ['dwt_419', '1138_bus', 'EVA', '3elt', 'USpowerGrid']:
                        offset = (-30, -12) if d == '3elt' else (4, -13 if m == 'atomic_sparse_sgd' else 7)
                        ax.annotate(d,(s['paired_iteration_speedup_median'],s['stress_ratio_mean']),fontsize=6,xytext=offset,textcoords='offset points')
            ax.plot([],[],color=COLORS[m],marker='o',ls='',label=LABELS[m])
        ax.axvline(1,color='#666666',linestyle='--',linewidth=.8);ax.axhline(1.05,color='#999999',linestyle=':',linewidth=1)
        ax.set_xscale('log');ax.set_xlabel('Iteration speedup (CPU / GPU)');ax.set_ylabel('Mean full-stress ratio')
        ax.set_title(family.capitalize()+' SGD');ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.tight_layout();save_figure(fig, '05_quality_speed_tradeoff', base)

    if large:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for mi,m in enumerate(METHODS[3:]):
            x=[];y=[];lo=[];hi=[]
            for j,d in enumerate(large):
                s=lookup.get((d,m))
                if s and s['success']:
                    x.append(j+(mi-(len(FAMILIES['sparse'])-1)/2)*.24);y.append(s['iteration_median_ms'])
                    lo.append(y[-1]-s['iteration_q1_ms']);hi.append(s['iteration_q3_ms']-y[-1])
            axes[0].bar(x,y,width=.22,color=COLORS[m],label=LABELS[m],yerr=[lo,hi],capsize=2)
        axes[0].set_xticks(range(len(large)),large,rotation=35,ha='right');axes[0].set_yscale('log')
        axes[0].set_ylabel(f'{iterations}-iteration host time (ms)');axes[0].legend(frameon=False);axes[0].grid(axis='y',alpha=.15)
        for m in FAMILIES['sparse'][1:]:
            for d in small+large:
                item=lookup.get((d,m))
                if item and item['paired_seeds']:
                    axes[1].scatter(item['nodes'],item['paired_iteration_speedup_median'],color=COLORS[m],marker='s' if d in large else 'o',s=25)
                    if d in large and (m == 'rr_sparse_sgd' or d in ['G65', 'web-NotreDame']):
                        offset = ((-60, -16) if d == 'web-NotreDame' else (-10, -15)) if m == 'atomic_sparse_sgd' else ((-60, 6) if d == 'web-NotreDame' else (3, 3))
                        axes[1].annotate(d,(item['nodes'],item['paired_iteration_speedup_median']),fontsize=6,xytext=offset,textcoords='offset points')
            axes[1].plot([],[],color=COLORS[m],marker='o',ls='',label=LABELS[m])
        axes[1].set_xscale('log');axes[1].set_yscale('log');axes[1].axhline(1,color='#666666',linestyle='--',linewidth=.8)
        axes[1].set_xlabel('Nodes in largest connected component');axes[1].set_ylabel('Sparse iteration speedup (CPU / GPU)');axes[1].grid(alpha=.15);axes[1].legend(frameon=False)
        fig.suptitle(f'Sparse scaling: exploratory large-graph results (planned {large_seeds} seeds)');fig.tight_layout();save_figure(fig, '06_sparse_scaling', base)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names_with_rr = [d for d in names if lookup.get((d, 'rr_sparse_sgd'), {}).get('success')]
    round_values = [lookup[(d, 'rr_sparse_sgd')]['rounds_median'] for d in names_with_rr]
    axes[0].bar(range(len(names_with_rr)), round_values, color=COLORS['rr_sparse_sgd'])
    axes[0].set_yscale('log'); axes[0].set_ylabel('RR-Sparse rounds per iteration (median)')
    axes[0].set_xticks(range(len(names_with_rr)), names_with_rr, rotation=55, ha='right', fontsize=7)
    for d in names_with_rr:
        item = lookup[(d, 'rr_sparse_sgd')]
        if item['paired_seeds']:
            axes[1].scatter(item['rounds_median'], item['paired_iteration_speedup_median'], color=COLORS['rr_sparse_sgd'], marker='s' if d in large else 'o')
            if d in ['web-NotreDame', 'venkat50', 'luxembourg_osm', 'bcsstk31', 'G65', 'EVA', 'dwt_419']:
                axes[1].annotate(d, (item['rounds_median'], item['paired_iteration_speedup_median']), fontsize=6, xytext=(3, 3), textcoords='offset points')
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('RR-Sparse rounds per iteration (median)'); axes[1].set_ylabel('Iteration speedup (CPU / GPU)')
    axes[1].axhline(1, color='#666666', ls='--', lw=.8)
    for ax in axes: ax.grid(axis='y', alpha=.15)
    fig.tight_layout(); save_figure(fig, '08_rr_sparse_rounds', base)

    # Same seed across all methods, picked using CPU full-SGD median stress only.
    layout_selections=[]
    fig,axes=plt.subplots(3,len(METHODS),figsize=(14.4,7))
    for ri,d in enumerate(['USpowerGrid','3elt','EVA']):
        refs=[r for r in groups[(d,'sgd')] if r['status']=='success']
        if not refs:continue
        median=np.median([r['stress_value'] for r in refs])
        ref=min(refs,key=lambda r:abs(r['stress_value']-median));seed=ref['seed']
        for ci,m in enumerate(METHODS):
            ax=axes[ri,ci];r=available.get((d,m,seed))
            ax.set_axis_off();ax.set_aspect('equal')
            if ri==0:ax.set_title(LABELS[m])
            if r is None:ax.text(.5,.5,'Unavailable',ha='center',transform=ax.transAxes);continue
            edges=[];positions=[];section=None
            for line in Path(r['final_positions_path']).read_text().splitlines():
                if line.startswith('# Edges'):section='edges'
                elif line.startswith('# Positions'):section='positions'
                elif line.strip() and not line.startswith('#'):
                    if section=='edges':edges.append([int(v) for v in line.split()[:2]])
                    elif section=='positions':positions.append([float(v) for v in line.split()[:2]])
            xy=np.asarray(positions);ed=np.asarray(edges)
            ax.add_collection(LineCollection(xy[ed],colors='#333333',linewidths=.18,alpha=.6,rasterized=True));ax.autoscale()
            ax.text(.02,.02,f'{d}; seed {seed}\nStress {r["stress_value"]:.0f}',transform=ax.transAxes,fontsize=7,
                    bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none', 'pad': 1})
            layout_selections.append({'dataset':d,'method':m,'seed':seed,'path':r['final_positions_path'],'sha256':sha(Path(r['final_positions_path']))})
    fig.tight_layout();save_figure(fig, '07_representative_layouts', base)
    (base/'tables/layout-selection.json').write_text(json.dumps(layout_selections,indent=2)+'\n')

    failures=[r for r in rows if r['status']!='success']
    dump_csv(base/'tables/failures.csv',failures)
    if not failures:
        (base/'tables/failures.csv').write_text('run_id,dataset,method,error_stage,error_message\n')
    report=[f'# 実験結果 {base.name}（暫定評価）','',
            f'正式実行：{len(rows)}件。成功 {len(available)}件、失敗 {len(failures)}件。',
            '', '## 比較条件',
            f'- 前処理を除いた{iterations}反復のhost実時間を主速度指標とする。同期・シャッフルは含む。',
            f'- 計画標本数：小・中規模{small_seeds} seed、大規模{large_seeds} seed。成功数は表で確認する。',
            '- exact full stressとsampled stressを混ぜず、評価条件を確認して各グラフ内の対応seedで比較。',
            '- 全手法へ同じ無向・自己ループなし・最大連結成分の入力を渡す。',
            '', '## グラフ別の結果',
            '| Graph | Method | SGD speedup | Full-stress ratio | Paired seeds |',
            '|---|---|---:|---:|---:|']
    for s in summaries:
        if s['method'] in ['atomic_sgd','rr_sgd','atomic_sparse_sgd','rr_sparse_sgd'] and s['paired_seeds']:
            report.append(f"| {s['dataset']} | {LABELS[s['method']]} | {s['paired_iteration_speedup_median']:.2f}x | {s['stress_ratio_mean']:.3f} | {s['paired_seeds']} |")
    report+=['','## 解釈と限界']
    for m in ['atomic_sgd','rr_sgd','atomic_sparse_sgd','rr_sparse_sgd']:
        subset=[s for s in summaries if s['method']==m and s['paired_seeds']]
        if subset:
            fast=sum(s['paired_iteration_speedup_median']>1 for s in subset)
            stable=sum(s['stress_ratio_ci_high']<=1.05 for s in subset)
            report.append(f'- {LABELS[m]}：対応比較可能な{len(subset)}グラフ中、中央値で高速化は{fast}グラフ。stress比の暫定95% CI上限が1.05以下は{stable}グラフ。')
    atomic_summary = [lookup[(d, 'atomic_sparse_sgd')] for d in names
                      if lookup.get((d, 'atomic_sparse_sgd'), {}).get('paired_seeds')]
    if atomic_summary:
        for cohort, description in [(small, '小・中規模'), (large, '大規模')]:
            subset = [item for item in atomic_summary if item['dataset'] in cohort]
            faster = sum(item['paired_iteration_speedup_median'] > 1 for item in subset)
            report.append(f'- Atomic Sparseの{description}：CPU基準より速い中央値は{len(subset)}グラフ中{faster}グラフ。')
        report.append('- Atomic SparseとRR-Sparseは同じ制約と方向別重みを使う。差には競合処理方式、更新順、GPU投入・同期の回数が含まれ、ロックのコストだけを独立に測った結果ではない。')
    report+=['- 全グラフをまとめた単一の速度向上率で代表させない。グラフの構造・規模に依存する。',
             '- GPU実装はf32、CPUはf64。差には精度と更新順の両方が含まれる。',
             '- 初期配置の一致はFull群・Sparse群のそれぞれの中で確認。両群は初期配置生成が異なるため、群をまたぐ対応seed比は使わない。',
             '- ラウンド数とdispatch数はグラフごとの表で確認する。同期・投入コストが性能を制限する可能性があるが、device時間の詳細測定なしには原因を断定しない。',
             '- この結果はSGD部分の性能であり、初回レイアウト生成全体の高速化率ではない。',
             '- 少数seedから信頼性の高い速度・品質の結論を出さない。',
             '- 同一seedの時間は1回ずつ。将来は固定seedの繰返しで機器の時間変動とseed差を分離する。',
             '- Atomic Sparseはロック失敗ペアを再投入するため、成功時の更新漏れは0だが、再投入・ホスト同期の時間を含む。',
             '- 欠測や失敗は図でゼロにせず、failures.csvおよび元stderrに理由を残す。',
             '', '## 検証', f'- 比較整合性エラー：{len(validation["errors"])}件。',
             '- 検証詳細：../tables/validation.json。本集計では正式掲載可否を認定しない。',
             '', '## 次に必要な実験',
             '1. 小・中規模で25 seedを確保し、品質の非劣性を判定する。',
             '2. 大規模で最低10 seedを確保し、stress比が悪化したグラフを詳しく確認する。',
             '3. Atomic Sparseを含む6手法で実行順を回し、バッチ間の時間変動を確認する。',
             f'4. 代表3グラフのiterationごとのstress収束を別診断実行で記録する（固定{iterations}反復の学習率を維持）。',
             '', '## 図の読み方',
             '- 01：CPU基準のstress比。点は平均、誤差棒は対応seedのbootstrap 95% CI。',
             '- 02：SGD部分の速度向上率。点は対応比の中央値、誤差棒はbootstrap 95% CI。',
             '- 03：SGD時間。棒は中央値、誤差棒はIQR。',
             '- 04：最終exact full stressの分布。',
             '- 05：品質と速度の両立。右側ほど速く、下側ほどstressが低い。',
             f'- 06：Sparseの規模拡大。丸は小・中規模（計画{small_seeds} seed）、四角は大規模（計画{large_seeds} seed）。',
             '- 07：CPU full-SGDのstress中央値に近いseedを全6手法で共有した代表配置。',
             '- 08：RR-Sparseのラウンド数と速度向上率。構造によるラウンド増加を確認する。',
             '', '## 論文の体裁との対応',
             '- SGD論文Figure 7・14の品質分布、Figure 10の品質と時間を同時に示す考え方を参考にした。',
             '- DRGraph Table 2・5のグラフ別比較とFigure 6の規模に応じた速度評価を参考にした。',
             '- DRGraphは尺度最適化したnormalized stressを使う。本結果のraw full stressと掲載数値は直接比較しない。',
             '- PNGは確認用、SVGは論文で編集・拡大するための版。収束曲線は未取得。']
    if validation['source_extensions']:
        report += ['', '## 追加バッチの出所',
                   '- Atomic Sparseは別測定バッチで追加した。既存手法と共通前処理のソースが不変であることをGit treeで検証し、過去のCPU・RR結果を再利用している。機器の時間変動や実行順の影響を完全には除けない。']
    (base/'report/RESULTS.md').write_text('\n'.join(report)+'\n')
    metadata={'sources':sources,'script_sha256':sha(Path(__file__)),
              'relocation_journal_sha256':sha(ROOT/'archive/relocations.json')
                  if (ROOT/'archive/relocations.json').is_file() else None,
              'config_sha256':sha(base/'config/datasets.json'),
              'manifest_sha256':{p.name:sha(p) for p in (base/'config').glob('*.json')},
              'source_extensions':validation['source_extensions'],'bootstrap_resamples':10000,
              'bootstrap_seed':20261011,'publication_ready':False,
              'commits':sorted({r['git_commit'] for r in rows}),
              'dirty_runs':sum(bool(r['git_dirty']) for r in rows),
              'discarded_measurements':[
                  {'path':provenance_path(p, ROOT), 'sha256':sha(p),
                   'records':len(p.read_text().splitlines())}
                  for p in sorted((base/'raw').glob('*/discarded-dirty-records.jsonl'))],
              'figures':{p.name:sha(p) for p in (base/'figures').glob('*') if p.is_file()}}
    (base/'tables/provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'{len(rows)} runs; {len(available)} success; {len(failures)} failures; {len(validation["errors"])} validation errors')
    if validation['errors']:
        raise SystemExit('Comparison validation failed; figures are drafts, inspect validation.json.')


if __name__ == '__main__':
    main()
