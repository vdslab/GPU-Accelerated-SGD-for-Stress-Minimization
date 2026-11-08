"""Summarize v2 measurements without mixing RR conditions or unpaired seeds.

Raw runs are immutable. Every numerical plot is backed by a CSV table. Bootstrap
resampling uses seeds, rather than treating repeated timings as independent runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
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
BASE = ROOT / 'experiment_results/v2'
METHODS = ['sgd', 'atomic_sgd', 'rr_sgd', 'sparse_sgd', 'atomic_sparse_sgd', 'rr_sparse_sgd']
FAMILIES = {'full': METHODS[:3], 'sparse': METHODS[3:]}
LABELS = dict(zip(METHODS, ['CPU', 'Atomic GPU', 'RR GPU', 'CPU', 'Atomic GPU', 'RR GPU']))
COLORS = dict(zip(METHODS, ['#666666', '#D55E00', '#0072B2', '#666666', '#CC79A7', '#009E73']))
BOOTSTRAP_SEED = 20261013
BOOTSTRAP_RESAMPLES = 10000
MATCH_FIELDS = ['input_sha256', 'initial_positions_sha256', 'nodes', 'edges',
                'constraints', 'iterations', 'epsilon', 'pivots', 'cpu_model',
                'stress_kind', 'stress_seed', 'stress_samples']
PARAMETERS = ['tile_size', 'workgroup_size', 'coordinate_memory', 'outer_shuffle', 'inner_shuffle']
TIME_COMPONENTS = [('common_preprocess_time_ms', 'Common preprocess', '#999999'),
                   ('method_setup_time_ms', 'Method setup', '#E69F00'),
                   ('runtime_init_time_ms', 'GPU initialization', '#CC79A7'),
                   ('upload_time_ms', 'Upload', '#56B4E9'),
                   ('iteration_time_ms', 'SGD iterations', '#0072B2'),
                   ('readback_time_ms', 'Readback', '#009E73'),
                   ('postprocess_time_ms', 'Postprocess', '#F0E442')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelsize': 10, 'legend.fontsize': 8, 'svg.fonttype': 'none',
                     'svg.hashsalt': 'stress-sgd-v2', 'savefig.dpi': 240})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def interval(values, statistic='median'):
    """Center and percentile 95% bootstrap interval, resampling independent seeds."""
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None, None, None
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite bootstrap input')
    function = np.mean if statistic == 'mean' else np.median
    center = float(function(x))
    if len(x) == 1:
        return center, center, center
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = function(x[rng.integers(0, len(x), (BOOTSTRAP_RESAMPLES, len(x)))], axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return center, float(low), float(high)


def dump_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                         if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def save_figure(fig, name, base):
    for extension in ['png', 'svg', 'pdf']:
        metadata = ({'Date': None, 'Creator': 'Stress SGD v2'} if extension == 'svg' else
                    {'CreationDate': None, 'ModDate': None, 'Creator': 'Stress SGD v2'}
                    if extension == 'pdf' else None)
        fig.savefig(base / 'figures' / f'{name}.{extension}', bbox_inches='tight', metadata=metadata)
    plt.close(fig)


def variant(row):
    result = dict(row.get('variant') or {})
    defaults = dict(tile_size=1024, workgroup_size=1024, coordinate_memory='shared',
                    outer_shuffle=True, inner_shuffle=True) if row.get('method') == 'rr_sgd' else {}
    for key in PARAMETERS:
        if key not in result:
            actual_key = 'rr_memory_mode' if key == 'coordinate_memory' else 'rr_' + key
            result[key] = row.get(actual_key, row.get(key, defaults.get(key)))
    return result


def condition(row):
    params = variant(row)
    return (row['study_group'], row['dataset'], row['method'], row.get('variant_id', ''),
            *(params[key] for key in PARAMETERS), row.get('iterations'),
            row.get('epsilon'), row.get('pivots'))


def planned_conditions(manifest, folder):
    output = {}
    for dataset, method, iterations, epsilon in itertools.product(
            manifest.get('datasets', []), manifest.get('methods', []),
            manifest.get('iterations', [15]), manifest.get('epsilon', [.1])):
        for pivots in manifest.get('pivots', [200]) if method in METHODS[3:] else [None]:
            for item in manifest.get('variants', {}).get(method, [{'id': 'default'}]):
                row = dict(study_group=manifest.get('study_group', folder.name), dataset=dataset['name'],
                           method=method, variant=item, variant_id=item.get('id', 'default'),
                           iterations=iterations, epsilon=epsilon, pivots=pivots)
                output[condition(row)] = len(manifest.get('seeds', [])) * manifest.get('repetitions', 1)
    return output


def load_frozen_evidence(folder):
    path = folder / 'execution-integrity.json'
    if not path.exists():
        return None, False
    evidence = json.loads(path.read_text())
    files = evidence.get('source_tree', {}).get('files', {})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    verified = bool(files) and digest == evidence.get('source_tree', {}).get('sha256')
    for name, expected in files.items():
        try:
            verified = verified and sha(folder / 'source-snapshot' / name) == expected
        except OSError:
            verified = False
    return evidence, verified


def load_data(base):
    rows, sources, expectations = [], [], {}
    for folder in sorted((base / 'raw').iterdir()) if (base / 'raw').is_dir() else []:
        if not folder.is_dir() or not (folder / 'manifest.json').is_file():
            continue
        manifest = json.loads((folder / 'manifest.json').read_text())
        study_group = manifest.get('study_group', folder.name)
        if (manifest.get('run_mode', 'benchmark') != 'benchmark'
                or 'pilot' in study_group or 'pilot' in folder.name):
            continue
        for key, expected in planned_conditions(manifest, folder).items():
            if key in expectations:
                raise ValueError(f'Overlapping planned condition across experiments: {key}')
            expectations[key] = expected
        path = folder / 'results.jsonl'
        if not path.is_file():
            continue
        evidence, snapshot_verified = load_frozen_evidence(folder)
        latest = {}
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.update(experiment_id=folder.name, study_group=study_group,
                       source_line=number, source_results=provenance_path(path, ROOT))
            row.setdefault('repetition', 0)
            row.setdefault('variant_id', 'default')
            before = row.get('integrity', {}).get('before', {})
            row['frozen_source_verified'] = bool(snapshot_verified and evidence
                and before.get('source_tree_sha256') == evidence['source_tree']['sha256']
                and before.get('binary_sha256') == evidence['binaries'].get(row.get('binary'))
                and before.get('input_sha256') == row.get('input_sha256'))
            if row.get('run_mode', 'benchmark') != 'benchmark':
                continue
            for key in ['input_path', 'final_positions_path', 'vertex_map_path', 'stderr_log_path']:
                if row.get(key):
                    row[key] = str(resolve_artifact_path(row[key], ROOT))
            latest[row['run_id']] = row
        rows.extend(latest.values())
        sources.append({'path': provenance_path(path, ROOT), 'sha256': sha(path),
                        'manifest_sha256': sha(folder / 'manifest.json'),
                        'execution_integrity_sha256': sha(folder / 'execution-integrity.json')
                        if evidence else None, 'source_snapshot_verified': snapshot_verified})
    return rows, sources, expectations


def compare_conditions(row, ref, memory=False):
    fields = MATCH_FIELDS + (['preprocess_sha256'] if row['family'] == 'sparse' else [])
    if memory:
        fields += ['gpu_name', 'gpu_backend', 'git_commit']
        for key in PARAMETERS:
            if key != 'coordinate_memory' and variant(row)[key] != variant(ref)[key]:
                yield key
    for key in fields:
        if row.get(key) != ref.get(key) or (key in ['input_sha256', 'initial_positions_sha256']
                                         and not row.get(key)):
            yield key
    if not memory and row.get('git_commit') != ref.get('git_commit'):
        yield 'git_commit'
    a = row.get('integrity', {}).get('before', {}).get('source_tree_sha256')
    b = ref.get('integrity', {}).get('before', {}).get('source_tree_sha256')
    if a != b:
        yield 'source_tree_sha256'


def validate_rows(rows):
    errors, warnings, invalid = [], [], set()
    for row in rows:
        if row['status'] != 'success':
            continue
        reasons = []
        if not finite(row.get('iteration_time_ms'), positive=True):
            reasons.append('invalid iteration_time_ms')
        if not finite(row.get('stress_value')):
            reasons.append('invalid stress_value')
        integrity = row.get('integrity')
        if integrity is not None and (not integrity.get('unchanged') or integrity.get('before') != integrity.get('after')):
            reasons.append('execution integrity changed')
        if integrity is not None and not row.get('frozen_source_verified'):
            reasons.append('frozen source/binary evidence not verified')
        if row.get('git_dirty'):
            if row.get('frozen_source_verified') and integrity and integrity.get('unchanged'):
                warnings.append('Uncommitted execution source: frozen snapshot and per-run input/binary/source hashes verified')
            else:
                reasons.append('dirty source tree without verified frozen snapshot')
        for field, digest in row.get('artifacts_sha256', {}).items():
            try:
                if sha(row[field]) != digest:
                    reasons.append(f'{field} artifact checksum changed')
            except (OSError, KeyError, TypeError):
                reasons.append(f'{field} artifact missing')
        if row['method'].startswith('atomic_'):
            planned = (row.get('constraints') or 0) * row.get('iterations', 0)
            if (row.get('attempted_updates') != planned or row.get('completed_updates') != planned
                    or row.get('retry_failures') != 0):
                reasons.append('atomic update coverage is incomplete')
        for reason in reasons:
            errors.append(f"{row['run_id']}: {reason}")
        if reasons:
            invalid.add((row['experiment_id'], row['run_id']))
        if row.get('stress_kind') == 'sampled':
            warnings.append(f"{row['dataset']}: sampled full stress; not an exact all-pairs score")
    return errors, sorted(set(warnings)), invalid


def is_main(row):
    return row['study_group'] in ['main', 'large-sparse', 'quality-timing']


def make_cpu_pairs(rows, invalid, errors):
    refs = defaultdict(list)
    for row in rows:
        if is_main(row) and row['status'] == 'success' and row['method'] in ['sgd', 'sparse_sgd']:
            refs[(row['dataset'], row['family'], row['seed'], row['repetition'])].append(row)
    pairs = []
    for row in rows:
        if row['status'] != 'success' or (row['experiment_id'], row['run_id']) in invalid:
            continue
        candidates = refs[(row['dataset'], row['family'], row['seed'], row['repetition'])]
        candidates = [r for r in candidates if all(r.get(key) == row.get(key)
                      for key in ['iterations', 'epsilon', 'pivots'])]
        if len(candidates) != 1:
            if is_main(row):
                errors.append(f"{row['run_id']}: {len(candidates)} CPU baselines; paired ratio excluded")
            continue
        ref = candidates[0]
        if (ref['experiment_id'], ref['run_id']) in invalid:
            continue
        mismatch = list(compare_conditions(row, ref))
        if mismatch or not finite(ref.get('stress_value'), positive=True):
            errors.append(f"{row['run_id']}: CPU pair mismatch {', '.join(mismatch) or 'zero baseline stress'}")
            continue
        pairs.append(dict(study_group=row['study_group'], dataset=row['dataset'], family=row['family'],
                          method=row['method'], variant_id=row.get('variant_id', ''), **variant(row),
                          iterations=row['iterations'], epsilon=row['epsilon'], pivots=row['pivots'],
                          seed=row['seed'], repetition=row['repetition'], run_id=row['run_id'],
                          cpu_run_id=ref['run_id'], iteration_speedup=ref['iteration_time_ms']/row['iteration_time_ms'],
                          stress_ratio=row['stress_value']/ref['stress_value']))
    return pairs


def seed_values(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if finite(row.get(field)):
            groups[row['seed']].append(row[field])
    return [float(np.median(groups[seed])) for seed in sorted(groups)]


def component_mean(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if finite(row.get(field)):
            groups[row['seed']].append(row[field])
    return float(np.mean([np.mean(value) for value in groups.values()])) if groups else None


def summarize(rows, expectations, pairs, invalid):
    groups, paired = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[condition(row)].append(row)
    for pair in pairs:
        paired[condition(pair)].append(pair)
    result = []
    for key in sorted(set(groups) | set(expectations), key=str):
        all_rows = groups[key]
        valid = [r for r in all_rows if r['status'] == 'success'
                 and (r['experiment_id'], r['run_id']) not in invalid]
        row = dict(zip(['study_group', 'dataset', 'method', 'variant_id', *PARAMETERS,
                        'iterations', 'epsilon', 'pivots'], key))
        row['family'] = 'full' if row['method'] in METHODS[:3] else 'sparse'
        failed = sum(r['status'] != 'success' for r in all_rows)
        expected = expectations.get(key, len(all_rows))
        times, stresses = seed_values(valid, 'iteration_time_ms'), seed_values(valid, 'stress_value')
        speed, quality = seed_values(paired[key], 'iteration_speedup'), seed_values(paired[key], 'stress_ratio')
        row.update(expected=expected, success=sum(r['status'] == 'success' for r in all_rows),
                   usable=len(valid), failures=failed, missing=max(0, expected-len(all_rows)),
                   seeds=len(times), paired_seeds=len(speed), paired_runs=len(paired[key]),
                   nodes=valid[0].get('nodes') if valid else None,
                   stress_kind=valid[0].get('stress_kind') if valid else None,
                   iteration_median_ms=float(np.median(times)) if times else None,
                   iteration_q1_ms=float(np.quantile(times, .25)) if times else None,
                   iteration_q3_ms=float(np.quantile(times, .75)) if times else None,
                   stress_median=float(np.median(stresses)) if stresses else None,
                   stress_min=min(stresses) if stresses else None,
                   stress_max=max(stresses) if stresses else None)
        for prefix, values, stat in [('iteration_speedup', speed, 'median'), ('stress_ratio', quality, 'mean')]:
            center, low, high = interval(values, stat)
            row.update({prefix + '_center': center, prefix + '_ci_low': low, prefix + '_ci_high': high})
        for field, _, _ in TIME_COMPONENTS:
            row[field + '_mean'] = component_mean(valid, field)
        for field in ['algorithm_time_cold_ms', 'algorithm_time_warm_ms', 'rounds', 'dispatches']:
            values = seed_values(valid, field)
            row[field + '_median'] = float(np.median(values)) if values else None
        result.append(row)
    return result


def read_layout(path):
    edges, positions, section = [], [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith('# Edges'):
            section = 'edges'
        elif line.startswith('# Positions'):
            section = 'positions'
        elif line.strip() and not line.startswith('#'):
            if section == 'edges':
                edges.append([int(v) for v in line.split()[:2]])
            elif section == 'positions':
                positions.append([float(v) for v in line.split()[:2]])
    xy, ed = np.asarray(positions, dtype=float), np.asarray(edges, dtype=int).reshape(-1, 2)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Invalid or missing positions')
    if len(ed) and (ed.min() < 0 or ed.max() >= len(xy)):
        raise ValueError('Edge index outside position array')
    return xy, ed


def memory_pairs(rows, invalid, errors):
    groups = defaultdict(dict)
    for row in rows:
        if row['study_group'] != 'rr-memory-tiling' or row['status'] != 'success':
            continue
        if (row['experiment_id'], row['run_id']) in invalid:
            continue
        params = variant(row)
        key = (row['dataset'], row['method'], row['seed'], row['repetition'],
               params['tile_size'], params['workgroup_size'], row['iterations'], row['epsilon'], row['pivots'])
        mode = params['coordinate_memory']
        if mode in groups[key]:
            errors.append(f'{key}: duplicate {mode} ablation record')
            continue
        groups[key][mode] = row
    result = []
    for key, modes in sorted(groups.items(), key=str):
        if 'shared' not in modes or 'global' not in modes:
            continue
        shared, global_ = modes['shared'], modes['global']
        mismatch = list(compare_conditions(shared, global_, memory=True))
        if mismatch:
            errors.append(f"{shared['run_id']}: shared/global mismatch {', '.join(mismatch)}")
            continue
        parity, max_diff, exact = 'unavailable', None, None
        try:
            a, aedges = read_layout(shared['final_positions_path'])
            b, bedges = read_layout(global_['final_positions_path'])
            if a.shape != b.shape or not np.array_equal(aedges, bedges):
                raise ValueError('coordinate count or edge set differs')
            max_diff = float(np.max(np.abs(a-b)))
            exact = bool(np.array_equal(a, b))
            parity = 'pass' if np.allclose(a, b, rtol=1e-6, atol=1e-6) else 'fail'
            if parity == 'fail':
                errors.append(f"{shared['run_id']}: shared/global coordinates differ (max abs {max_diff})")
        except (OSError, ValueError, KeyError, TypeError) as error:
            parity = f'unavailable: {error}'
        result.append(dict(dataset=key[0], method=key[1], seed=key[2], repetition=key[3],
                           tile_size=key[4], workgroup_size=key[5], iterations=key[6],
                           epsilon=key[7], pivots=key[8], shared_run_id=shared['run_id'],
                           global_run_id=global_['run_id'],
                           shared_speedup=global_['iteration_time_ms']/shared['iteration_time_ms'],
                           shared_stress_ratio=shared['stress_value']/global_['stress_value']
                           if finite(global_['stress_value'], positive=True) else None,
                           coordinate_parity=parity, coordinates_exactly_equal=exact,
                           max_coordinate_abs_difference=max_diff))
    return result


def plot_main(summary, pairs, base):
    main = [r for r in summary if is_main(r)]
    names = list(dict.fromkeys(r['dataset'] for r in sorted(main, key=lambda r: (r['nodes'] or 0, r['dataset']))))
    sampled = {r['dataset'] for r in main if r.get('stress_kind') == 'sampled'}
    display_names = [name + '*' if name in sampled else name for name in names]
    if not main:
        return []
    made = []
    for field, ylabel, filename, log in [
        ('iteration_speedup', 'Iteration speedup (CPU / method)', '01_iteration_speedup', True),
        ('stress_ratio', 'Final stress / family CPU stress', '02_final_stress_ratio', False)]:
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2))
        for ax, (family, methods) in zip(axes, FAMILIES.items()):
            family_names = [name for name in names if any(r['family'] == family and r['dataset'] == name for r in main)]
            family_sampled = sampled.intersection(family_names)
            ax.axhline(1, color='#666666', ls='--', lw=.8)
            for mi, method in enumerate(methods):
                subset = [r for r in main if r['method'] == method and r[field + '_center'] is not None]
                x = [family_names.index(r['dataset']) + (mi-1)*.2 for r in subset]
                y = [r[field + '_center'] for r in subset]
                lo = [max(0, r[field + '_center']-r[field + '_ci_low']) for r in subset]
                hi = [max(0, r[field + '_ci_high']-r[field + '_center']) for r in subset]
                ax.errorbar(x, y, yerr=[lo, hi], fmt=['o', '^', 's'][mi], color=COLORS[method],
                            label=LABELS[method], capsize=3, ms=4, lw=1)
            statistic = 'median' if field == 'iteration_speedup' else 'mean'
            ax.set_title(f'{family.capitalize()} SGD; seed-paired {statistic} ratios, bootstrap 95% intervals'
                         + ('; * sampled stress' if family_sampled and field == 'stress_ratio' else ''), loc='left')
            ax.set_ylabel(ylabel)
            if log:
                ax.set_yscale('log')
            ax.grid(axis='y', alpha=.2)
            ax.legend(frameon=False, ncol=3)
            labels = [name + '*' if name in family_sampled and field == 'stress_ratio' else name for name in family_names]
            ax.set_xticks(range(len(family_names)), labels, rotation=30, ha='right')
        fig.tight_layout()
        save_figure(fig, filename, base)
        made.append(filename)
    columns = min(7, len(names))
    family_rows = math.ceil(len(names)/columns)
    fig, axes = plt.subplots(2*family_rows, columns, figsize=(12.3, 2.45*2*family_rows), squeeze=False)
    for fi, (family, methods) in enumerate(FAMILIES.items()):
        for di, dataset in enumerate(names):
            row_index = fi*family_rows + di//columns
            ax = axes[row_index, di%columns]
            values = [seed_values([r for r in pairs if is_main(r) and r['dataset'] == dataset
                                   and r['method'] == method], 'stress_ratio') for method in methods]
            valid = [(mi, value) for mi, value in enumerate(values) if value]
            if not valid:
                ax.set_axis_off()
                ax.text(.5, .5, f'{dataset}\n{family.capitalize()} not measured', ha='center', va='center',
                        transform=ax.transAxes, color='#777777', fontsize=8)
                continue
            if valid:
                boxes = ax.boxplot([v for _, v in valid], positions=[mi for mi, _ in valid],
                                   widths=.6, patch_artist=True, medianprops={'color': 'black'})
                for box, (mi, _) in zip(boxes['boxes'], valid):
                    box.set_facecolor(COLORS[methods[mi]])
                    box.set_alpha(.55)
                for mi, value in valid:
                    offsets = np.linspace(-.12, .12, len(value))
                    ax.scatter(mi+offsets, value, color=COLORS[methods[mi]], s=6, alpha=.65)
            ax.axhline(1, color='#666666', ls='--', lw=.7)
            ax.set_xticks(range(3), ['CPU', 'AT', 'RR'], rotation=40)
            ax.set_title(display_names[di], fontsize=8)
            if di%columns == 0:
                ax.set_ylabel(f'{family.capitalize()}\nstress ratio')
            ax.grid(axis='y', alpha=.15)
        for di in range(len(names), family_rows*columns):
            axes[fi*family_rows + di//columns, di%columns].set_axis_off()
    fig.suptitle('Seed-paired final stress ratios; * sampled stress; panel y-axis ranges differ', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .975], h_pad=1.4)
    save_figure(fig, '03_stress_ratio_distributions', base)
    made.append('03_stress_ratio_distributions')
    # Use seed-balanced means so stacked components retain additive timing totals.
    for fi, (family, methods) in enumerate(FAMILIES.items()):
        subset = [r for r in main if r['family'] == family and r['usable']]
        if not subset:
            continue
        family_names = [name for name in names if any(r['dataset'] == name for r in subset)]
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True)
        for ax, cold in zip(axes, [True, False]):
            for mi, method in enumerate(methods):
                selected = [r for r in subset if r['method'] == method]
                x = np.array([family_names.index(r['dataset'])+(mi-1)*.23 for r in selected])
                bottom = np.zeros(len(selected))
                for field, label, color in TIME_COMPONENTS:
                    if not cold and field == 'runtime_init_time_ms':
                        continue
                    values = np.array([r[field+'_mean'] or 0 for r in selected])
                    ax.bar(x, values, bottom=bottom, width=.21, color=color,
                           label=label if mi == 0 else None)
                    bottom += values
                for xpos, value in zip(x, bottom):
                    ax.text(xpos, value*1.05, LABELS[method].replace(' GPU', ''), rotation=90,
                            va='bottom', ha='center', fontsize=6)
            ax.set_yscale('log')
            ax.set_ylabel('Algorithm time (ms)')
            ax.set_title(f'{family.capitalize()}: {"cold" if cold else "warm (init subtracted)"}; component means', loc='left')
            ax.grid(axis='y', alpha=.15)
            ax.margins(y=.3)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper center', frameon=False, ncol=4, fontsize=7)
        axes[-1].set_xticks(range(len(family_names)), family_names, rotation=35, ha='right')
        fig.tight_layout(rect=[0, 0, 1, .91])
        filename = f'04_time_breakdown_{family}'
        save_figure(fig, filename, base)
        made.append(filename)
    return made


def plot_ablation(summary, memories, base):
    selected = [r for r in summary if r['study_group'] == 'rr-memory-tiling' and r['usable']]
    if not selected:
        return []
    datasets = list(dict.fromkeys(r['dataset'] for r in selected))
    fig, axes = plt.subplots(2, len(datasets), figsize=(max(8, 3.8*len(datasets)), 6.1), squeeze=False)
    for column, dataset in enumerate(datasets):
        subset = [r for r in selected if r['dataset'] == dataset]
        for method, workgroup, memory in sorted({(r['method'], r['workgroup_size'], r['coordinate_memory'])
                                                for r in subset}, key=str):
            group = sorted([r for r in subset if (r['method'], r['workgroup_size'], r['coordinate_memory'])
                            == (method, workgroup, memory)], key=lambda r: r['tile_size'])
            x = [r['tile_size'] for r in group]
            label = f'{method.replace("_sgd", "")} W={workgroup} {memory}'
            color = COLORS[method]
            for ax, field in [(axes[0, column], 'iteration_median_ms'), (axes[1, column], 'stress_median')]:
                y = [r[field] for r in group]
                ax.plot(x, y, marker='o' if memory == 'shared' else '^', ms=4,
                        ls='-' if memory == 'shared' else '--', color=color, label=label)
                if field == 'iteration_median_ms':
                    ax.fill_between(x, [r['iteration_q1_ms'] for r in group],
                                    [r['iteration_q3_ms'] for r in group], color=color, alpha=.12)
                else:
                    ax.fill_between(x, [r['stress_min'] for r in group], [r['stress_max'] for r in group],
                                    color=color, alpha=.08)
                ax.set_xscale('log', base=2)
                ax.set_xticks(sorted(set(x)), [str(v) for v in sorted(set(x))])
                ax.grid(alpha=.18)
        axes[0, column].set_title(dataset)
        axes[0, column].set_yscale('log')
        axes[0, column].set_ylabel('SGD time (ms), median / IQR')
        axes[0, column].legend(frameon=False, fontsize=7)
        axes[1, column].set_ylabel('Final full stress, median / range')
        axes[1, column].set_xlabel('Block size T (vertices per block)')
    fig.tight_layout()
    save_figure(fig, '05_rr_tile_sensitivity', base)
    if not memories:
        return ['05_rr_tile_sensitivity']
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(7, 3.8*len(datasets)), 3.6), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        rows = [r for r in memories if r['dataset'] == dataset]
        for method, workgroup in sorted({(r['method'], r['workgroup_size']) for r in rows}, key=str):
            tiles = sorted({r['tile_size'] for r in rows if r['method'] == method and r['workgroup_size'] == workgroup})
            centers, lows, highs = [], [], []
            for tile in tiles:
                values = seed_values([r for r in rows if (r['method'], r['workgroup_size'], r['tile_size'])
                                      == (method, workgroup, tile)], 'shared_speedup')
                center, low, high = interval(values)
                centers.append(center)
                lows.append(max(0, center-low))
                highs.append(max(0, high-center))
            ax.errorbar(tiles, centers, yerr=[lows, highs], marker='o', color=COLORS[method],
                        capsize=3, label=f'{method}; W={workgroup}')
        ax.axhline(1, color='#666666', ls='--', lw=.8)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted({r['tile_size'] for r in rows}), [str(v) for v in sorted({r['tile_size'] for r in rows})])
        ax.set_xlabel('Block size T')
        ax.set_ylabel('Shared speedup (global time / shared time)')
        ax.set_title(dataset)
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, '06_shared_memory_ablation', base)
    return ['05_rr_tile_sensitivity', '06_shared_memory_ablation']


def workgroup_pairs(rows, invalid, errors):
    groups = defaultdict(dict)
    for row in rows:
        if row['study_group'] != 'rr-workgroup' or row['status'] != 'success':
            continue
        if (row['experiment_id'], row['run_id']) in invalid:
            continue
        params = variant(row)
        key = (row['dataset'], row['method'], row['seed'], row['repetition'],
               params['tile_size'], params['coordinate_memory'], row['iterations'], row['epsilon'], row['pivots'])
        if params['workgroup_size'] in groups[key]:
            errors.append(f'{key}: duplicate WG condition')
        groups[key][params['workgroup_size']] = row
    result = []
    for key, conditions in sorted(groups.items(), key=str):
        ref = conditions.get(256)
        if ref is None:
            continue
        for workgroup, row in sorted(conditions.items()):
            if workgroup == 256:
                continue
            mismatch = list(compare_conditions(row, ref))
            mismatch += [field for field in ['tile_size', 'coordinate_memory', 'outer_shuffle', 'inner_shuffle']
                         if variant(row)[field] != variant(ref)[field]]
            mismatch += [field for field in ['gpu_name', 'gpu_backend'] if row.get(field) != ref.get(field)]
            if mismatch:
                errors.append(f'{row["run_id"]}: WG pair mismatch {", ".join(mismatch)}')
                continue
            parity, exact, diff = 'unavailable', None, None
            try:
                a, ae = read_layout(row['final_positions_path'])
                b, be = read_layout(ref['final_positions_path'])
                if a.shape != b.shape or not np.array_equal(ae, be):
                    raise ValueError('coordinate count or edge set differs')
                diff = float(np.max(np.abs(a-b)))
                exact = bool(np.array_equal(a, b))
                parity = 'pass' if np.allclose(a, b, atol=1e-6, rtol=1e-6) else 'fail'
                if parity == 'fail':
                    errors.append(f'{row["run_id"]}: WG coordinates differ (max abs {diff})')
            except (OSError, ValueError, KeyError, TypeError) as error:
                parity = f'unavailable: {error}'
            result.append(dict(dataset=key[0], method=key[1], seed=key[2], repetition=key[3],
                               tile_size=key[4], coordinate_memory=key[5], workgroup_size=workgroup,
                               reference_workgroup_size=256, run_id=row['run_id'], reference_run_id=ref['run_id'],
                               iteration_speedup=ref['iteration_time_ms']/row['iteration_time_ms'],
                               coordinate_parity=parity, coordinates_exactly_equal=exact,
                               max_coordinate_abs_difference=diff))
    return result


def rr_geometry(summary):
    """Logical RR allocation derived from N and T, not hardware occupancy profiling.

    build_schedule() pads an odd block count with a dummy for cross-block RR,
    skips dummy pairs, and chunks self-tiles into floor(B/2) workgroups per round.
    """
    result = []
    for row in summary:
        if row['method'] != 'rr_sgd' or not row['usable']:
            continue
        nodes, tile, workgroup = row['nodes'], row['tile_size'], row['workgroup_size']
        if not all(isinstance(value, int) and value > 0 for value in [nodes, tile, workgroup]):
            continue
        blocks = (nodes + tile - 1) // tile
        last = nodes - (blocks - 1) * tile
        maximum = max(1, blocks // 2)
        cross_rounds = 0 if blocks == 1 else blocks + (blocks % 2) - 1
        self_rounds = 1 if blocks == 1 else (blocks + maximum - 1) // maximum
        rounds = cross_rounds + self_rounds
        result.append(dict(study_group=row['study_group'], dataset=row['dataset'],
                           variant_id=row['variant_id'], N=nodes, T=tile, W=workgroup,
                           coordinate_memory=row['coordinate_memory'], block_count=blocks,
                           last_block_vertices=last, tail_fill_fraction=last/tile,
                           all_blocks_fill_fraction=nodes/(blocks*tile),
                           rr_shared_bytes=16*tile if row['coordinate_memory'] == 'shared' else 0,
                           max_workgroups_per_round=maximum, cross_rounds_per_iteration=cross_rounds,
                           self_rounds_per_iteration=self_rounds, derived_rounds_per_iteration=rounds,
                           rounds=row['rounds_median'], dispatches=row['dispatches_median'],
                           derived_dispatches=rounds*row['iterations'], iterations=row['iterations'],
                           evidence_kind='logical RR geometry, not hardware profiler'))
    return result


def plot_workgroups(summary, base):
    rows = [r for r in summary if r['study_group'] == 'rr-workgroup' and r['usable']]
    if not rows:
        return []
    names = list(dict.fromkeys(r['dataset'] for r in rows))
    fig, axes = plt.subplots(2, len(names), figsize=(max(8, 3.8*len(names)), 5.8), squeeze=False)
    for column, dataset in enumerate(names):
        subset = sorted([r for r in rows if r['dataset'] == dataset], key=lambda r: r['workgroup_size'])
        x = [r['workgroup_size'] for r in subset]
        color = COLORS[subset[0]['method']]
        axes[0, column].plot(x, [r['iteration_median_ms'] for r in subset], marker='o', color=color)
        axes[0, column].fill_between(x, [r['iteration_q1_ms'] for r in subset],
                                    [r['iteration_q3_ms'] for r in subset], color=color, alpha=.15)
        axes[1, column].plot(x, [r['stress_median'] for r in subset], marker='o', color=color)
        axes[1, column].fill_between(x, [r['stress_min'] for r in subset],
                                    [r['stress_max'] for r in subset], color=color, alpha=.15)
        axes[0, column].set_title(f'{dataset}; T={subset[0]["tile_size"]} {subset[0]["coordinate_memory"]}')
        axes[0, column].set_ylabel('SGD time (ms), median / IQR')
        axes[1, column].set_ylabel('Full stress, median / range')
        axes[1, column].set_xlabel('Workgroup size W (threads)')
        for ax in axes[:, column]:
            ax.set_xscale('log', base=2)
            ax.set_xticks(x, [str(value) for value in x])
            ax.grid(alpha=.18)
    fig.tight_layout()
    save_figure(fig, '08_workgroup_sensitivity', base)
    return ['08_workgroup_sensitivity']


def plot_layouts(rows, invalid, base, warnings):
    usable = [r for r in rows if is_main(r) and r['status'] == 'success'
              and (r['experiment_id'], r['run_id']) not in invalid]
    choices, selected = [], []
    available_names = {r['dataset'] for r in usable if r['method'] == 'sgd'}
    names = [name for name in ['3elt', 'EVA', 'USpowerGrid'] if name in available_names]
    names += sorted(available_names-set(names))[:max(0, 3-len(names))]
    if not names:
        return [], []
    fig, axes = plt.subplots(len(names), 6, figsize=(13.5, 2.7*len(names)), squeeze=False)
    for ri, dataset in enumerate(names):
        refs = [r for r in usable if r['dataset'] == dataset and r['method'] == 'sgd']
        target = np.median([r['stress_value'] for r in refs])
        ref = min(refs, key=lambda r: (abs(r['stress_value']-target), r['seed'], r['repetition']))
        for ci, method in enumerate(METHODS):
            ax = axes[ri, ci]
            ax.set_axis_off()
            ax.set_aspect('equal', adjustable='datalim')
            if ri == 0:
                ax.set_title(f'{"Full" if ci < 3 else "Sparse"} {LABELS[method]}', fontsize=9)
            candidates = [r for r in usable if (r['dataset'], r['method'], r['seed'], r['repetition'])
                          == (dataset, method, ref['seed'], ref['repetition'])]
            if len(candidates) != 1:
                ax.text(.5, .5, 'Unavailable', ha='center', transform=ax.transAxes)
                continue
            row = candidates[0]
            try:
                xy, edges = read_layout(row['final_positions_path'])
                ax.add_collection(LineCollection(xy[edges], colors='#333333', linewidths=.2,
                                                 alpha=.6, rasterized=True))
                ax.autoscale()
                ax.margins(.08)
                ax.text(.02, -.04, f'{dataset}; seed {ref["seed"]}\nStress {row["stress_value"]:.1f}',
                        transform=ax.transAxes, fontsize=7, va='top', clip_on=False)
                choices.append(dict(dataset=dataset, method=method, seed=ref['seed'],
                                    repetition=ref['repetition'], run_id=row['run_id'],
                                    path=provenance_path(Path(row['final_positions_path']), ROOT),
                                    sha256=sha(row['final_positions_path'])))
            except (OSError, ValueError, KeyError, TypeError) as error:
                warnings.append(f'{row["run_id"]}: layout unavailable: {error}')
                ax.text(.5, .5, 'Unavailable', ha='center', transform=ax.transAxes)
    if choices:
        fig.tight_layout(h_pad=2.5, w_pad=.5)
        save_figure(fig, '07_representative_layouts', base)
        selected.append('07_representative_layouts')
    else:
        plt.close(fig)
    return selected, choices


def number(value, digits=2):
    return f'{value:.{digits}f}' if value is not None else '—'


def write_report(base, rows, summary, memories, workgroups, geometry, validation, figures):
    main = [r for r in summary if is_main(r)]
    failed = sum(r['status'] != 'success' for r in rows)
    gpu_names = sorted({r['gpu_name'] for r in rows if r.get('gpu_name')})
    cpu_names = sorted({r['cpu_model'] for r in rows if r.get('cpu_model')})
    report = ['# v2 実験結果', '',
              f'記録 {len(rows)} 件、成功 {len(rows)-failed} 件、失敗 {failed} 件。'
              f'未実行 {sum(r["missing"] for r in summary)} 件。比較整合性エラー {len(validation["errors"])} 件。', '',
              f'機器：CPU {", ".join(cpu_names)}、GPU {", ".join(gpu_names)}。'
              f'主比較は {len({r["dataset"] for r in main})} 個の代表グラフ。'
              'SGD論文の243グラフによる網羅評価と同じ範囲を検証した結果ではない。', '',
              '速度は前処理を除くSGD反復のhost実時間。同期・シャッフル・投入を含む。'
              '速度比は同一グラフ・同一seedのCPU時間÷各手法時間、stress比は各手法÷同じfamilyのCPU。'
              '1より大きい速度比は高速、1より小さいstress比は良好。', '', '## 今回わかったこと']
    full_rr = [r for r in main if r['method'] == 'rr_sgd' and r['paired_seeds']]
    sparse_rr = [r for r in main if r['method'] == 'rr_sparse_sgd' and r['paired_seeds']]
    if full_rr and sparse_rr:
        full_best = max(full_rr, key=lambda r: r['iteration_speedup_center'])
        sparse_best = max(sparse_rr, key=lambda r: r['iteration_speedup_center'])
        report.append(f'- Full RRは最大 {full_best["iteration_speedup_center"]:.2f}×（{full_best["dataset"]}）、'
                      f'Sparse RRは最大 {sparse_best["iteration_speedup_center"]:.2f}×（{sparse_best["dataset"]}）。'
                      '小グラフでCPUより遅い例もあり、常に速いわけではない。')
        notable = [r for r in main if r['method'] in ['rr_sgd', 'rr_sparse_sgd'] and r['dataset'] in ['EVA', 'web-NotreDame']
                   and r['stress_ratio_center'] is not None and r['stress_ratio_center'] > 1.1]
        if notable:
            report.append('- 品質も同時に確認する。' + '、'.join(
                f'{r["dataset"]} / {r["family"]} RRの平均stress比 {r["stress_ratio_center"]:.3f}' for r in notable)
                + 'と悪化した例があり、高速化だけで有効性を結論づけない。')
    eva = {r['tile_size']: r for r in summary if r['study_group'] == 'rr-memory-tiling' and r['dataset'] == 'EVA'
           and r['coordinate_memory'] == 'shared' and r['usable']}
    if 32 in eva and 1024 in eva:
        a, b = eva[32], eva[1024]
        report.append(f'- EVAではT=32→1024でSGD時間中央値 {a["iteration_median_ms"]/1000:.3f}→'
                      f'{b["iteration_median_ms"]/1000:.3f} 秒、平均CPU stress比 '
                      f'{a["stress_ratio_center"]:.3f}→{b["stress_ratio_center"]:.3f}。'
                      'Tは速度と更新順の両方を変える。傾向はグラフ依存で、単調に変わるとは限らない。')
    if memories:
        groups = defaultdict(list)
        for row in memories:
            groups[(row['dataset'], row['method'], row['tile_size'], row['workgroup_size'])].append(row)
        overlapping = sum(low <= 1 <= high for _, low, high in [interval(seed_values(group, 'shared_speedup')) for group in groups.values()])
        report.append(f'- sharedの一貫した高速化は見られず、{len(groups)}条件中{overlapping}条件で95%区間が1を含む。'
                      f'shared/global座標は{sum(bool(r["coordinates_exactly_equal"]) for r in memories)}/{len(memories)}完全一致。'
                      + (f'WG変更でも{sum(bool(r["coordinates_exactly_equal"]) for r in workgroups)}/{len(workgroups)}完全一致。' if workgroups else ''))
    report += ['', '## 実験条件', '| 実験 | グラフ数 | 比較 | seed / グラフ | 記録数 |',
               '|---|---:|---|---:|---:|']
    descriptions = {'quality-timing': 'Full 3手法 + Sparse 3手法', 'main': 'Full 3手法 + Sparse 3手法',
                    'large-sparse': 'Sparse 3手法', 'rr-memory-tiling': 'Full RR; T=32,64,128,256,512,1024; W=256; shared/global',
                    'rr-workgroup': 'Full RR; T=1024; shared; W=128,256,512,1024'}
    for group in ['quality-timing', 'main', 'large-sparse', 'rr-memory-tiling', 'rr-workgroup']:
        subset = [r for r in summary if r['study_group'] == group]
        if not subset:
            continue
        report.append(f'| {group} | {len({r["dataset"] for r in subset})} | {descriptions[group]} | '
                      f'{"/".join(str(n) for n in sorted({r["seeds"] for r in subset}))} | {sum(r["success"] for r in subset)} |')
    iterations = sorted({r['iterations'] for r in rows if r['status'] == 'success'})
    epsilons = sorted({r['epsilon'] for r in rows if r['status'] == 'success'})
    pivots = sorted({r['pivots'] for r in rows if r['status'] == 'success' and r['pivots'] is not None})
    samples = sorted({r['stress_samples'] for r in rows if r['status'] == 'success' and r['stress_kind'] == 'sampled'})
    report += ['', f'反復数 {", ".join(map(str, iterations))}、ε {", ".join(map(str, epsilons))}、Sparse pivot数 {", ".join(map(str, pivots))}。'
               f'今回の評価はn≤8,000でexact full stress、それを超える4グラフは{", ".join(map(str, samples))}起点のsampled full stress。'
               '各seedの時間測定は1回。入力は無向・単純グラフの最大連結成分。',
               'Tはブロック当たりの頂点数（CLIのtile-size）。異なる2ブロックの処理タイルは最大2T頂点を扱う。'
               'Wは1WGのスレッド数。',
               '入力の規模・由来・hash：[datasets.json](../config/datasets.json)。', '',
               '## すぐ見る図', '| 図 | 示すこと | ファイル |', '|---|---|---|']
    captions = {'01_iteration_speedup': 'Full/Sparse各3手法のSGD高速化',
                '02_final_stress_ratio': '速度比較と同じseedでの最終品質',
                '03_stress_ratio_distributions': '品質のseed差と外れた例',
                '04_time_breakdown_full': 'Fullの前処理・初期化・SGDの時間内訳',
                '04_time_breakdown_sparse': 'Sparseの時間内訳と全体の速度',
                '05_rr_tile_sensitivity': 'タイルTによる速度・品質の変化',
                '06_shared_memory_ablation': '同一RR順序でsharedが効くか',
                '07_representative_layouts': '数値差が最終描画にどう現れるか',
                '08_workgroup_sensitivity': 'T固定でWGスレッド数Wの効果'}
    for name in sorted(figures):
        report.append(f'| {name[:2]} | {captions[name]} | [PNG](../figures/{name}.png) · [PDF](../figures/{name}.pdf) |')
    table = ['| Graph | Family | Method | 成功 / 計画 | 評価 | 対応seed | 時間 ms（中央値） | 高速化 | stress比（平均） |',
             '|---|---|---|---:|---|---:|---:|---:|---:|']
    for row in sorted(main, key=lambda r: (r['nodes'] or 0, r['dataset'], METHODS.index(r['method']))):
        table.append(f'| {row["dataset"]} | {row["family"]} | {LABELS[row["method"]]} | '
                      f'{row["success"]}/{row["expected"]} | {row["stress_kind"] or "—"} | {row["paired_seeds"]} | '
                      f'{number(row["iteration_median_ms"])} | {number(row["iteration_speedup_center"])}× | '
                      f'{number(row["stress_ratio_center"], 3)} |')
    report += ['', '## この結果で示せること']
    for method in METHODS:
        if method in ['sgd', 'sparse_sgd']:
            continue
        subset = [r for r in main if r['method'] == method and r['paired_seeds']]
        if subset:
            speeds = [r['iteration_speedup_center'] for r in subset]
            ratios = [r['stress_ratio_center'] for r in subset]
            report.append(f'- {method}：{len(subset)}グラフ中{sum(v>1 for v in speeds)}グラフでCPUより高速。'
                          f'速度比 {min(speeds):.2f}–{max(speeds):.2f}×、平均stress比 {min(ratios):.3f}–{max(ratios):.3f}。'
                          'グラフ別の差を主結果とする。')
    if memories:
        report += ['', '## Shared memory とタイル']
        for dataset, method, workgroup in sorted({(r['dataset'], r['method'], r['workgroup_size']) for r in memories}, key=str):
            subset = [r for r in memories if (r['dataset'], r['method'], r['workgroup_size']) == (dataset, method, workgroup)]
            tiles = sorted({r['tile_size'] for r in subset})
            stats = []
            for tile in tiles:
                values = seed_values([r for r in subset if r['tile_size'] == tile], 'shared_speedup')
                stats.append(f'T={tile}: {np.median(values):.2f}×')
            report.append(f'- {dataset} / {method} / W={workgroup}：global時間÷shared時間は ' + '、'.join(stats) + '。')
        parity = [r for r in memories if r['coordinate_parity'] == 'pass']
        exact = sum(bool(r['coordinates_exactly_equal']) for r in memories)
        report.append(f'- 同一schedule・f32の座標一致：許容誤差内 {len(parity)}/{len(memories)} ペア、完全一致 {exact} ペア。')
        report.append('- Tを変える実験はWGサイズWを固定。Tが大きいほど同時WG数は減る一方、座標再利用が増え、dispatchが減る。'
                      '速度の優劣とstressへの影響はFigure 05で確認する。T=1でも完全な全ペアshuffleにはならない。')
    if workgroups:
        report += ['', '## Workgroup の割り当て']
        for dataset in sorted({r['dataset'] for r in workgroups}):
            subset = [r for r in summary if r['study_group'] == 'rr-workgroup' and r['dataset'] == dataset and r['usable']]
            best = min(subset, key=lambda r: r['iteration_median_ms'])
            report.append(f'- {dataset}：今回の範囲で最短の中央値は W={best["workgroup_size"]}、'
                          f'{best["iteration_median_ms"]:.2f} ms（T={best["tile_size"]}固定）。')
        report.append(f'- W=256との座標比較：許容誤差内 {sum(r["coordinate_parity"]=="pass" for r in workgroups)}'
                      f'/{len(workgroups)} ペア。shared容量は16T bytesでWに依存しない。'
                      'Wを増やしても常に速くなるとは限らず、担当仕事量と同期の効率で変わる。')
    if geometry:
        report += ['', '## タイル割り当ての数値',
                   '- [rr-geometry.csv](../tables/rr-geometry.csv)に各T・Wのブロック数、'
                   '端数ブロックの頂点数・充足率、shared bytes、ラウンド数、dispatch数、'
                   '1ラウンド当たりの最大WG数を保存。Nとスケジュールからの計算値で、GPUの実測occupancyではない。',
                   '- B=ceil(N/T)、最大WG数=max(1, floor(B/2))。Tを増やすとBと並列WG数が減り、'
                   'Wだけを変えてもB・shared容量は変わらない。端数充足率が低いと最後のタイルに空きが増える。'
                   'これらは時間差の説明に使えるが、原因の断定にはハードウェア測定が必要。']
    report += ['', '## 図の読み方',
               '- 01：full/sparse各3手法の高速化。点は対応比の中央値、棒はseedを再標本化した95% bootstrap区間。',
               '- 02・03：最終stressのCPU比。02は平均と95%区間、03は各seedの分布。',
               '- 03の各panelは縦軸範囲が異なるため、箱の見かけの高さをグラフ間で直接比較しない。未測定Full条件は明示する。',
               '- 04：coldアルゴリズム時間の内訳は前処理・初期化も含む。下段warmはGPU初期化時間を差し引いた値。積み上げ棒は各成分の平均。',
               '- 05：WGサイズ固定でタイルTを変更。上は時間（中央値・IQR）、下はstress（中央値・最小〜最大）。',
               '- 06：同一T・W・更新順でshared/globalを比較。1を超えるとsharedが速い。',
               '- 07：CPU fullの中央値に近いseedを選び、同じseedの6手法を並べた描画。最良例の選抜ではない。',
               '- 08：T=1024・shared固定でWGサイズWを変更。時間とstressを並べ、スレッド割り当ての影響を示す。',
               f'- 生成済み：{", ".join(figures)}。存在しない条件の図は作らない。', '', '## 解釈の範囲',
               '- 同一family内で入力・初期配置・制約・評価条件を照合し、不一致の対応比は除外。Full/Sparseをまたぐ対応比は計算しない。',
               '- 実行ソースに未commitの変更を含む場合は、ソースsnapshot・binary hash・各実行前後のhash一致を検証。Git commitだけで実行版を表さない。',
               '- CPUはf64、GPUはf32。6手法比較には精度差と更新順の差が含まれる。shared/global比較は同じf32と同じRR順序。',
               '- v2のFull RRはペア順列と端数タイルを修正した実装。v1との差をタイル調整だけによる高速化と説明しない。',
               '- sampled stressのグラフはexactと混ぜて絶対値を比較しない。同じ評価ペア・seedでCPU比を取る。',
               '- 各seed内の繰返しを中央値でまとめ、bootstrapはseed単位。反復時間1回だけのseedは機器の時間変動とseed差を分離できない。',
               '- 少数seed、欠測、失敗はCSVとvalidation.jsonに残す。性能・品質の保証や正式掲載可否をこの集計だけで認定しない。',
               '- Full RRのGPU timestamp区間は保存済み。最初〜最後のdispatch間には投入間の待ちも含むためkernel単独時間とは扱わない。'
               'cache miss・occupancy・物理globalアクセス回数は未計測。sharedの効果にはアクセスと同期方法の違いも含む。',
               '- 固定反復後の結果であり、同じstressまで到達する時間や途中の収束曲線は未取得。']
    if validation['errors'] or validation['warnings']:
        report += ['', '## 確認事項']
        report.extend('- ' + item for item in validation['errors'][:12])
        report.extend('- ' + item for item in validation['warnings'][:12])
        report.append('- 全件：../tables/validation.json。')
    report += ['', '## グラフ別の数値（詳細）', '', *table]
    spaced = []
    for index, line in enumerate(report):
        spaced.append(line)
        if line.startswith('#') and index + 1 < len(report) and report[index + 1]:
            spaced.append('')
    (base / 'report/RESULTS.md').write_text('\n'.join(spaced) + '\n', encoding='utf-8')
    (base / 'README.md').write_text(
        '# v2 実験\n\n[結果と説明](report/RESULTS.md) · [集計CSV](tables/summary.csv) · '
        '[比較した各seedの比](tables/paired-ratios.csv) · [検証](tables/validation.json)\n\n'
        'raw/に実測JSONL・ログ・座標、config/に計画とデータ出所を保存。'
        'figures/は確認用PNG、編集用SVG、論文用PDF。\n\n'
        '再集計：`python3 experiments/summarize_v2.py`（リポジトリのルートから）。\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=BASE)
    args = parser.parse_args(argv)
    base = args.results_dir.resolve()
    for directory in ['tables', 'figures', 'report']:
        (base / directory).mkdir(parents=True, exist_ok=True)
    rows, sources, expectations = load_data(base)
    if not rows:
        raise SystemExit(f'No benchmark results in {base}')
    errors, warnings, invalid = validate_rows(rows)
    pairs = make_cpu_pairs(rows, invalid, errors)
    summary = summarize(rows, expectations, pairs, invalid)
    memories = memory_pairs(rows, invalid, errors)
    workgroups = workgroup_pairs(rows, invalid, errors)
    geometry = rr_geometry(summary)
    for row in summary:
        if row['usable'] < row['expected']:
            warnings.append(f'{row["dataset"]} {row["method"]} {row["variant_id"]}: '
                            f'{row["usable"]}/{row["expected"]} usable; failures {row["failures"]}, missing {row["missing"]}')
        if row['seeds'] and row['seeds'] < 10:
            warnings.append(f'{row["dataset"]} {row["method"]} {row["variant_id"]}: {row["seeds"]} seeds; exploratory estimate')
    figures = plot_main(summary, pairs, base) + plot_ablation(summary, memories, base)
    figures += plot_workgroups(summary, base)
    layout_figures, layouts = plot_layouts(rows, invalid, base, warnings)
    figures += layout_figures
    validation = dict(errors=sorted(set(errors)), warnings=sorted(set(warnings)),
                      excluded_successful_runs=len(invalid), publication_ready=False)
    dump_csv(base/'tables/runs.csv', rows)
    dump_csv(base/'tables/summary.csv', summary)
    dump_csv(base/'tables/paired-ratios.csv', pairs)
    dump_csv(base/'tables/shared-global-pairs.csv', memories)
    dump_csv(base/'tables/workgroup-pairs.csv', workgroups)
    dump_csv(base/'tables/rr-geometry.csv', geometry)
    dump_csv(base/'tables/failures.csv', [r for r in rows if r['status'] != 'success'],
             ['experiment_id', 'run_id', 'dataset', 'method', 'variant_id', 'seed', 'repetition',
              'status', 'study_group', 'source_line', 'source_results', 'error_stage', 'error_message',
              'stderr_log_path', 'exit_code'] if not any(r['status'] != 'success' for r in rows) else None)
    (base/'tables/summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (base/'tables/validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    (base/'tables/layout-selection.json').write_text(json.dumps(layouts, ensure_ascii=False, indent=2) + '\n')
    write_report(base, rows, summary, memories, workgroups, geometry, validation, figures)
    metadata = dict(sources=sources, script_sha256=sha(__file__), bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                    bootstrap_seed=BOOTSTRAP_SEED, bootstrap_unit='seed (median across repetitions)',
                    commits=sorted({r.get('git_commit', '') for r in rows}), publication_ready=False,
                    config_sha256={p.name:sha(p) for p in sorted((base/'config').glob('*.json'))},
                    figures={p.name:sha(p) for p in sorted((base/'figures').glob('*')) if p.is_file()})
    (base/'tables/provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(rows)} runs; {len(pairs)} CPU pairs; {len(memories)} memory pairs; {len(validation["errors"])} validation errors')
    if errors:
        raise SystemExit('Comparison validation failed; inspect tables/validation.json. Figures are drafts.')


if __name__ == '__main__':
    main()
