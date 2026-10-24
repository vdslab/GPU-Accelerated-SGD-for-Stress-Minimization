#!/usr/bin/env python3
"""Check 64/256/1024-source stress sensitivity on three paired large layouts.

This is postprocessing, not an SGD rerun. Sources follow the Rust SplitMix64
prefix sampler. Distances are computed in small blocks and discarded immediately.
Only successful basic seed-0 layouts shared by all three Sparse methods qualify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    from summarize_v3 import BASE, dump_csv, load, path_of, sample_indices, open_layout_text
    from v3_recovery import active_runners
except ModuleNotFoundError:
    from experiments.summarize_v3 import BASE, dump_csv, load, path_of, sample_indices, open_layout_text
    from experiments.v3_recovery import active_runners

METHODS = ('sparse_sgd', 'atomic_sparse_sgd', 'rr_sparse_sgd')
COUNTS = (64, 256, 1024)
PREFERRED = ('luxembourg_osm', 'web-Stanford', 'delaunay_n20')


def sha256(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def coordinates(path: Path, n: int) -> np.ndarray:
    """Read only positions, avoiding Python lists for potentially millions of edges."""
    xy = np.empty((n, 2), dtype=np.float64)
    section = False
    count = 0
    with open_layout_text(path) as source:
        for line in source:
            if line.startswith('#'):
                section = line.startswith('# Positions')
            elif section and line.strip():
                if count >= n:
                    raise ValueError('Too many saved positions')
                values = line.split()
                if len(values) != 2:
                    raise ValueError('Expected two saved coordinate components')
                xy[count] = (float(values[0]), float(values[1]))
                count += 1
    if count != n or not np.isfinite(xy).all():
        raise ValueError(f'Invalid saved coordinates: got {count}, expected {n}')
    return xy


def finite_signed(value):
    """Allow both signs for ratio changes; negative percentages are meaningful."""
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def add_quality_and_native_audit(measurements):
    cpu={r['stress_samples']:r['stress_value'] for r in measurements if r['method']=='sparse_sgd'}
    for row in measurements:
        reference=cpu.get(row['stress_samples'])
        row['quality_ratio_vs_cpu']=(row['stress_value']/reference
            if finite_signed(reference) and reference>0 and finite_signed(row['stress_value']) else None)
        row.update(saved256_vs_native_signed_percent=None,saved256_vs_native_absolute_percent=None,saved256_native_audit_flag=None)
        if row['stress_samples']!=256: continue
        original=row.get('original_stress_value'); recalculated=row.get('stress_value')
        definition_ok=(row.get('original_stress_kind')=='sampled' and row.get('original_stress_samples')==256 and row.get('original_stress_seed')==0)
        difference=None
        if finite_signed(original) and finite_signed(recalculated):
            if original>0: difference=(recalculated-original)/original*100
            elif original==recalculated==0: difference=0.
        row['saved256_vs_native_signed_percent']=difference
        row['saved256_vs_native_absolute_percent']=abs(difference) if difference is not None else None
        row['saved256_native_audit_flag']=('definition_mismatch' if not definition_ok else 'invalid_or_zero_reference' if difference is None
                                          else 'investigate' if abs(difference)>=.1 else 'within_0_1_percent')
    return measurements


def comparison_summary(results):
    changes=[]; audits=[]
    for graph,method in sorted({(r['graph'],r['method']) for r in results}):
        records={r['stress_samples']:r for r in results if r['graph']==graph and r['method']==method}
        r256=records.get(256,{}); r1024=records.get(1024,{})
        q256=r256.get('quality_ratio_vs_cpu'); q1024=r1024.get('quality_ratio_vs_cpu')
        difference=q256-q1024 if finite_signed(q256) and finite_signed(q1024) else None
        relative=(q256/q1024-1)*100 if difference is not None and q1024>0 else None
        changes.append(dict(graph=graph,method=method,quality_ratio_256=q256,quality_ratio_1024=q1024,
                            ratio_difference_256_minus_1024=difference,signed_relative_change_percent=relative,
                            absolute_relative_change_percent=abs(relative) if relative is not None else None))
        if r256:
            audits.append(dict(graph=graph,method=method,stress_samples=256,native_stress_value=r256.get('original_stress_value'),
                               saved_coordinates_reevaluated_stress_value=r256.get('stress_value'),
                               signed_relative_difference_percent=r256.get('saved256_vs_native_signed_percent'),
                               absolute_relative_difference_percent=r256.get('saved256_vs_native_absolute_percent'),
                               flag=r256.get('saved256_native_audit_flag')))
    return changes,audits


def source_terms(xy: np.ndarray, distances: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Return each source's sum of (Euclidean distance - graph distance)^2/d^2."""
    xy = np.asarray(xy, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    sources = np.asarray(sources)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('Expected a nonempty array of finite 2D coordinates')
    if sources.ndim != 1 or not np.issubdtype(sources.dtype, np.integer):
        raise ValueError('Expected one-dimensional integer source indices')
    if np.any(sources < 0) or np.any(sources >= len(xy)):
        raise ValueError('Source index outside saved coordinates')
    if distances.shape != (len(sources), len(xy)):
        raise ValueError('Graph-distance dimensions do not match sources and coordinates')
    # Zero self-distances and +inf for unreachable vertices contribute nothing.
    # NaN, negative values and -inf cannot be valid shortest-path output.
    if np.isnan(distances).any() or np.any(distances < 0):
        raise ValueError('Graph distances must be nonnegative finite values or +inf')
    terms = []
    for index, source in enumerate(sources):
        distance = distances[index]
        valid = (distance > 0) & np.isfinite(distance)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                delta = xy - xy[source]
                actual = np.sqrt(np.einsum('ij,ij->i', delta, delta, optimize=False))
                if not np.isfinite(actual).all():
                    raise ValueError(f'Nonfinite Euclidean distance for source {source}')
                residual = np.zeros(len(xy), dtype=np.float64)
                np.subtract(actual, distance, out=residual, where=valid)
                np.divide(residual, distance, out=residual, where=valid)
                if not np.isfinite(residual).all():
                    raise ValueError(f'Nonfinite normalized residual for source {source}')
                # Accelerate ddot can leave spurious IEEE exception flags even
                # for ones(1024). Keep this identical sum of squares off BLAS.
                term = float(np.einsum('i,i->', residual, residual, optimize=False))
        except FloatingPointError as error:
            raise ValueError(f'Nonfinite stress arithmetic for source {source}') from error
        if not math.isfinite(term):
            raise ValueError(f'Nonfinite squared-residual sum for source {source}')
        terms.append(term)
    return np.asarray(terms, dtype=np.float64)


def paired_candidates(datasets, rows, base: Path):
    catalog = {row['name']: row for row in datasets}
    groups = {}
    issues = []
    for row in rows:
        if (row.get('study') != 'basic' or row.get('status') != 'success'
                or row.get('seed') != 0 or row.get('method') not in METHODS
                or row.get('pivots') != 200):
            continue
        graph = row['graph']
        info = catalog.get(graph, {})
        if info.get('family') != 'large' or not info.get('eligible'):
            continue
        groups.setdefault(graph, {})[row['method']] = row
    candidates = {}
    for graph, methods in groups.items():
        if set(methods) != set(METHODS):
            issues.append({'graph': graph, 'status': 'unavailable',
                           'reason': 'Missing successful basic seed-0 p=200 method',
                           'successful_methods': sorted(methods)})
            continue
        reference = methods[METHODS[0]]
        keys = ('input_sha256', 'initial_positions_sha256', 'iterations', 'epsilon')
        if any(reference.get(key) is None for key in keys) or any(
                row.get(key) != reference[key] for row in methods.values() for key in keys):
            issues.append({'graph': graph, 'status': 'unavailable',
                           'reason': 'Input, initialization, or algorithm parameters do not match'})
            continue
        missing = [method for method, row in methods.items()
                   if not row.get('final_positions_path')
                   or not path_of(row['final_positions_path'], base).is_file()]
        if missing:
            issues.append({'graph': graph, 'status': 'unavailable',
                           'reason': 'Saved final coordinates missing', 'methods': missing})
            continue
        candidates[graph] = methods
    names = [name for name in PREFERRED if name in candidates]
    names.extend(name for name in sorted(candidates) if name not in names)
    return catalog, [(name, candidates[name]) for name in names[:3]], issues


def evaluate_graph(name, info, methods, base: Path, block_size: int):
    from scipy.io import mmread
    from scipy.sparse.csgraph import shortest_path
    from threadpoolctl import threadpool_limits

    source = path_of(info['path'], base)
    digest = sha256(source)
    if digest != info['sha256'] or any(row['input_sha256'] != digest for row in methods.values()):
        raise ValueError('Prepared input checksum mismatch')
    n = info['nodes']
    layouts = {}
    layout_hashes = {}
    for method, row in methods.items():
        path = path_of(row['final_positions_path'], base)
        layouts[method] = coordinates(path, n)
        layout_hashes[method] = sha256(path)
    started = time.monotonic()
    with threadpool_limits(limits=4):
        graph = mmread(source).tocsr()
        graph = graph.maximum(graph.T)
        if graph.shape != (n, n):
            raise ValueError('Prepared input dimensions differ from catalog')
        sources = sample_indices(n, min(max(COUNTS), n), seed=0)
        sums = {method: 0.0 for method in METHODS}
        measurements = []
        for start in range(0, len(sources), block_size):
            stop = min(start + block_size, len(sources))
            # Split at every checkpoint even when a custom block size is used.
            stop = min([stop] + [min(k, n) for k in COUNTS if start < min(k, n) < stop])
            # The outer range requires aligned block boundaries; CLI restricts
            # block size to divisors of all three sampling checkpoints.
            batch = sources[start:stop]
            distances = shortest_path(graph, directed=False, unweighted=True, indices=batch)
            for method in METHODS:
                sums[method] += float(source_terms(layouts[method], distances, batch).sum())
            del distances
            if stop in {min(k, n) for k in COUNTS}:
                for method in METHODS:
                    row = methods[method]
                    measurements.append({'graph': name, 'method': method, 'status': 'success',
                        'seed': 0, 'pivots': 200, 'nodes': n, 'stress_seed': 0,
                        'stress_samples': stop, 'stress_value': n * sums[method] / (2 * stop),
                        'input_sha256': digest, 'coordinates_sha256': layout_hashes[method],
                        'final_positions_path': row['final_positions_path'],
                        'analysis_origin_base': row.get('analysis_origin_base'),
                        'analysis_origin_scope': row.get('analysis_origin_scope'),
                        'native_git_commit': row.get('git_commit'),
                        'original_stress_kind': row.get('stress_kind'),
                        'original_stress_seed': row.get('stress_seed'),
                        'original_stress_samples': row.get('stress_samples'),
                        'original_stress_value': row.get('stress_value')})
                print(f'Sampling sensitivity {name}: {stop} sources', flush=True)
    for method in METHODS:
        selected = [row for row in measurements if row['method'] == method]
        baseline = next(row['stress_value'] for row in selected if row['stress_samples'] == min(1024, n))
        s256 = next(row['stress_value'] for row in selected if row['stress_samples'] == min(256, n))
        for row in selected:
            row['change_from_1024_percent'] = (100 * (row['stress_value'] / baseline - 1)
                                                if baseline else None)
            row['absolute_change_256_vs_1024_percent'] = (100 * abs(s256 / baseline - 1)
                                                          if baseline else None)
            row['evaluation_wall_seconds'] = time.monotonic() - started
    return add_quality_and_native_audit(measurements)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=BASE)
    parser.add_argument('--block-size', type=int, choices=(1, 2, 4, 8, 16), default=8)
    args = parser.parse_args(argv)
    base = args.results_dir.resolve()
    runners = active_runners({base, (base / 'snap-origin-recovery').resolve()})
    if runners:
        raise SystemExit('Core or SNAP-recovery native runner is active; no sampling analysis started')
    datasets, raw, cohort = load(base, with_cohort=True)
    recovery_status = cohort.get('status')
    if recovery_status in ('incomplete', 'rejected'):
        raise SystemExit(f'SNAP recovery is {recovery_status}; complete and validate it before sampling analysis')
    if recovery_status == 'adopted':
        if (cohort.get('core_raw_records') != 3210 or cohort.get('recovery_raw_records') != 90
                or cohort.get('effective_conditions') != 3210 or cohort.get('errors')):
            raise SystemExit('Validated recovery cohort is not complete; no sampling analysis started')
    elif recovery_status in (None, 'absent'):
        # Without an adopted overlay, the loader preserves a partial core for
        # lightweight reports. Heavy BFS analysis needs all unique planned IDs.
        plan = json.loads((base / 'config/plan.json').read_text())
        jobs = plan.get('jobs', plan.get('tasks', [])) if isinstance(plan, dict) else plan
        records = [json.loads(line) for line in (base / 'raw/results.jsonl').read_text().splitlines()
                   if line.strip()]
        planned_ids = [row['run_id'] for row in jobs]
        recorded_ids = [row['run_id'] for row in records]
        if (len(planned_ids) != 3210 or len(set(planned_ids)) != 3210
                or len(recorded_ids) != 3210 or len(set(recorded_ids)) != 3210
                or set(recorded_ids) != set(planned_ids)):
            raise SystemExit('Core 3210-condition plan is incomplete or nonunique; no sampling analysis started')
    else:
        raise SystemExit(f'Unknown recovery status {recovery_status!r}; no sampling analysis started')
    catalog, chosen, issues = paired_candidates(datasets, raw, base)
    tables = base / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    results = []
    for name, methods in chosen:
        try:
            results.extend(evaluate_graph(name, catalog[name], methods, base, args.block_size))
        except Exception as error:
            issues.append({'graph': name, 'status': 'evaluation_error',
                           'reason': f'{type(error).__name__}: {error}'})
        dump_csv(tables / 'sampling-sensitivity.csv', results or [{'status': 'unavailable'}])
    checked = sorted({row['graph'] for row in results})
    complete = len(checked) == 3
    if not complete:
        issues.append({'status': 'incomplete', 'reason': f'Only {len(checked)}/3 graphs evaluated'})
    changes = [row['absolute_change_256_vs_1024_percent'] for row in results
               if row['stress_samples'] == 256 and row['absolute_change_256_vs_1024_percent'] is not None]
    ratio_changes,native_audits=comparison_summary(results)
    ratio_absolute=[r['absolute_relative_change_percent'] for r in ratio_changes if finite_signed(r.get('absolute_relative_change_percent'))]
    report = {'status': 'success' if complete else 'incomplete',
              'cohort': {key:value for key,value in cohort.items() if key not in ('record_provenance','superseded_records')},
              'description': 'Identical saved layouts; nested Rust-compatible SplitMix64 sources; n/(2k) estimator',
              'requested_samples': list(COUNTS), 'stress_seed': 0, 'layout_seed': 0,
              'preferred_graphs': list(PREFERRED), 'selected_graphs': [name for name, _ in chosen],
              'evaluated_graphs': checked, 'max_absolute_change_256_vs_1024_percent': max(changes) if changes else None,
              'quality_ratio_sensitivity_256_vs_1024': ratio_changes,
              'max_absolute_quality_ratio_relative_change_percent': max(ratio_absolute) if ratio_absolute else None,
              'saved256_vs_native_audit': native_audits,
              'saved256_audit_description': 'Final saved coordinates reevaluated with 256 sources versus native original_stress_value; separate from pre-centering checkpoint audit. Positive and negative percentage differences preserved; abs difference >=0.1% flags investigation without excluding runs.',
              'records': results, 'issues': issues}
    if not results:
        dump_csv(tables / 'sampling-sensitivity.csv', [{'status': 'unavailable', 'reason': 'No eligible paired layouts'}])
    (tables / 'sampling-sensitivity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('records', 'issues')}, indent=2))
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
