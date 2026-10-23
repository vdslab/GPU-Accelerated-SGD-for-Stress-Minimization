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
import time
from pathlib import Path

import numpy as np

try:
    from summarize_v3 import BASE, dump_csv, load, path_of, sample_indices
except ModuleNotFoundError:
    from experiments.summarize_v3 import BASE, dump_csv, load, path_of, sample_indices

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
    with path.open() as source:
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


def source_terms(xy: np.ndarray, distances: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Return each source's sum of (Euclidean distance - graph distance)^2/d^2."""
    terms = []
    for index, source in enumerate(sources):
        distance = distances[index]
        valid = (distance > 0) & np.isfinite(distance)
        delta = xy - xy[source]
        actual = np.sqrt(np.einsum('ij,ij->i', delta, delta))
        residual = np.zeros(len(xy), dtype=np.float64)
        np.subtract(actual, distance, out=residual, where=valid)
        np.divide(residual, distance, out=residual, where=valid)
        terms.append(float(np.dot(residual, residual)))
    return np.asarray(terms)


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
    return measurements


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=BASE)
    parser.add_argument('--block-size', type=int, choices=(1, 2, 4, 8, 16), default=8)
    args = parser.parse_args(argv)
    base = args.results_dir.resolve()
    datasets, raw = load(base)
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
    report = {'status': 'success' if complete else 'incomplete',
              'description': 'Identical saved layouts; nested Rust-compatible SplitMix64 sources; n/(2k) estimator',
              'requested_samples': list(COUNTS), 'stress_seed': 0, 'layout_seed': 0,
              'preferred_graphs': list(PREFERRED), 'selected_graphs': [name for name, _ in chosen],
              'evaluated_graphs': checked, 'max_absolute_change_256_vs_1024_percent': max(changes) if changes else None,
              'records': results, 'issues': issues}
    if not results:
        dump_csv(tables / 'sampling-sensitivity.csv', [{'status': 'unavailable', 'reason': 'No eligible paired layouts'}])
    (tables / 'sampling-sensitivity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('records', 'issues')}, indent=2))
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
