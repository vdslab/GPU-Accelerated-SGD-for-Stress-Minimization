#!/usr/bin/env python3
"""Prepare the canonical 100-graph v3 selection with bounded worker processes.

The graph is the nonzero pattern: discard explicit zeros, numerical weights,
directions, loops and duplicate entries, then retain the largest component.
No experiment is run by this script. Existing successful checkpoints are reused
only after source, prepared input and vertex-map checksums have been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DATA = Path('/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/data')
CANONICAL = {'existing_verified', 'downloaded_verified'}
MISSING_NODES = {'com-Youtube': 1134890, 'com-LiveJournal': 3997962,
                 'soc-Pokec': 1632803, 'power197k': 197156}
POLICY = 'v3-nonzero-simple-undirected-largest-component-1'


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def base_row(source: dict, source_data: Path) -> dict:
    name = source['graph'].split('/')[-1]
    nodes = source.get('nrows') or MISSING_NODES.get(name)
    if nodes is None:
        raise ValueError(f'Unknown original dimension for {name}')
    return {'name': name, 'family': 'small' if nodes < 100000 else 'large',
            'source_graph': source['graph'], 'category': source['graph'].split('/')[0],
            'source_path': str((source_data / source['path']).resolve()),
            'source_sha256': source.get('sha256'), 'source_nodes': nodes,
            'source_distribution_status': source['status'],
            'source_page': source.get('source_page', source.get('suitesparse_page')),
            'download_url': source.get('download_url'),
            'preparation_policy': POLICY, 'eligible': False}


def prepare_worker(request: dict) -> dict:
    # Imports occur only in each disposable worker so native allocations are
    # released by process exit before the next source is loaded.
    import numpy as np
    from scipy.io import mmread
    from scipy.sparse import csr_matrix, triu
    from scipy.sparse.csgraph import connected_components
    from threadpoolctl import threadpool_limits

    row = request['row']
    source = Path(row['source_path'])
    destination = Path(request['data_dir'])
    started = time.monotonic()
    if sha256(source) != row['source_sha256']:
        raise ValueError('Source checksum differs from acquisition catalog')
    with threadpool_limits(limits=4):
        matrix = mmread(source).tocoo()
        if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != row['source_nodes']:
            raise ValueError('Source must be square and match catalog dimensions')
        # Keep stored nonzero entries before duplicate coalescing: opposite
        # numerical values do not cancel an edge in a pattern graph.
        keep = (matrix.data != 0) & (matrix.row != matrix.col)
        discarded_zeros = int(np.count_nonzero(matrix.data == 0))
        discarded_loops = int(np.count_nonzero(matrix.row == matrix.col))
        rows, cols = matrix.row[keep], matrix.col[keep]
        shape = matrix.shape
        del keep, matrix
        # int32 prevents uint8 duplicate counts overflowing to zero.
        graph = csr_matrix((np.ones(rows.size, dtype=np.int32), (rows, cols)), shape=shape)
        del rows, cols
        graph.data[:] = 1
        graph = graph.maximum(graph.T)
        graph.sort_indices()
        components, labels = connected_components(graph, directed=False)
        counts = np.bincount(labels)
        # argmax deterministically chooses the first component in a size tie.
        vertices = np.flatnonzero(labels == int(np.argmax(counts)))
        del labels, counts
        source_nodes = graph.shape[0]
        if vertices.size != source_nodes:
            graph = graph[vertices][:, vertices]
        degrees = np.diff(graph.indptr).astype(np.float64)
        mean_degree = float(degrees.mean())
        nodes = graph.shape[0]
        edges = int(graph.nnz // 2)
        stats = {'nodes': nodes, 'edges': edges, 'source_components': int(components),
                 'mean_degree': mean_degree, 'max_degree': int(degrees.max(initial=0)),
                 'degree_cv': float(degrees.std() / mean_degree) if mean_degree else 0.0,
                 'discarded_explicit_zeros': discarded_zeros,
                 'discarded_self_loop_entries': discarded_loops}
        del degrees
        upper = triu(graph, k=1, format='csr')
        del graph
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / (row['name'] + '.mtx')
        temporary = target.with_suffix('.mtx.tmp')
        with temporary.open('w') as output:
            output.write('%%MatrixMarket matrix coordinate pattern general\n')
            output.write('% Nonzero simple undirected LCC; each edge listed once.\n')
            output.write(f'{nodes} {nodes} {edges}\n')
            for vertex in range(nodes):
                neighbours = upper.indices[upper.indptr[vertex]:upper.indptr[vertex + 1]]
                output.writelines(f'{vertex + 1} {int(other) + 1}\n' for other in neighbours)
        temporary.replace(target)
        mapping = destination / (row['name'] + '.vertex-map.npy')
        map_temporary = mapping.with_suffix('.npy.tmp')
        with map_temporary.open('wb') as output:
            np.save(output, vertices.astype(np.int64), allow_pickle=False)
        map_temporary.replace(mapping)
    row.update(stats)
    below_threshold = row['family'] == 'large' and nodes < 100000
    row.update({'status': 'ready', 'eligible': not below_threshold,
                'path': relative(target), 'sha256': sha256(target),
                'vertex_map_path': relative(mapping), 'vertex_map_sha256': sha256(mapping),
                'prepared_bytes': target.stat().st_size,
                'preprocess_time_seconds': time.monotonic() - started})
    if below_threshold:
        row['reason'] = 'Original large graph has LCC below 100000 vertices; excluded from large cohort'
    return row


def reusable(row: dict, checkpoint: dict) -> bool:
    if checkpoint.get('status') != 'ready':
        return False
    if any(checkpoint.get(key) != row.get(key) for key in
           ('source_sha256', 'source_graph', 'preparation_policy')):
        return False
    try:
        return (sha256(Path(row['source_path'])) == row['source_sha256'] and
                sha256(resolve(checkpoint['path'])) == checkpoint['sha256'] and
                sha256(resolve(checkpoint['vertex_map_path'])) == checkpoint['vertex_map_sha256'])
    except (OSError, KeyError):
        return False


def bounded_worker(row: dict, args, checkpoint: Path) -> dict:
    import psutil
    request = checkpoint.with_suffix('.request.json')
    result = checkpoint.with_suffix('.worker.json')
    atomic_json(request, {'row': row, 'data_dir': str(args.data_dir)})
    environment = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4')
    started = time.monotonic()
    peak = 0
    with checkpoint.with_suffix('.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                    '--worker', str(request), '--worker-result', str(result)],
                                   stdout=log, stderr=log, env=environment)
        reason = None
        while process.poll() is None:
            try:
                peak = max(peak, psutil.Process(process.pid).memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if time.monotonic() - started > args.timeout_seconds:
                reason = f'Preprocessing exceeded {args.timeout_seconds} seconds'
            elif peak > args.max_rss_gib * 1024 ** 3:
                reason = f'Preprocessing exceeded RSS limit {args.max_rss_gib} GiB'
            if reason:
                process.kill()
                break
            time.sleep(0.2)
        process.wait()
    if not reason and process.returncode == 0 and result.is_file():
        row = json.loads(result.read_text())
    else:
        row.update({'status': 'preprocess_error', 'eligible': False,
                    'reason': reason or f'Preprocessing worker exited {process.returncode}',
                    'preprocess_log_path': relative(checkpoint.with_suffix('.log'))})
    row['preprocess_peak_rss_bytes'] = peak
    row['preprocess_wall_seconds'] = time.monotonic() - started
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data', type=Path, default=ORIGINAL_DATA)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'experiment_results/v3')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/v3_lcc')
    parser.add_argument('--timeout-seconds', type=float, default=180)
    parser.add_argument('--max-rss-gib', type=float, default=8)
    parser.add_argument('--dataset', action='append', help='prepare only these names; keep all catalog rows')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0 or args.max_rss_gib <= 0:
        parser.error('Resource limits must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        atomic_json(args.worker_result, prepare_worker(json.loads(args.worker.read_text())))
        return 0
    args.source_data = args.source_data.resolve()
    args.data_dir = args.data_dir.resolve()
    config = args.results_dir.resolve() / 'config'
    source_catalog = args.source_data / 'suitesparse-100-manifest.json'
    source_bytes = source_catalog.read_bytes()
    sources = json.loads(source_bytes)['graphs']
    frozen = config / 'source-catalog.json'
    config.mkdir(parents=True, exist_ok=True)
    if frozen.exists() and frozen.read_bytes() != source_bytes:
        raise SystemExit('Source catalog changed; use another result directory')
    frozen.write_bytes(source_bytes)
    catalog = []
    for source in sources:
        row = base_row(source, args.source_data)
        checkpoint = config / 'prepared' / (row['name'] + '.json')
        if source['status'] not in CANONICAL:
            row.update({'status': 'unavailable',
                        'reason': source.get('reason', source.get('note', 'Canonical source unavailable'))})
        elif checkpoint.is_file() and reusable(row, json.loads(checkpoint.read_text())):
            row = json.loads(checkpoint.read_text())
        elif args.dataset and row['name'] not in args.dataset:
            row.update({'status': 'unavailable', 'reason': 'Not selected in this preparation invocation'})
        else:
            print(f"Preparing {row['name']} ({row['family']})", flush=True)
            row = bounded_worker(row, args, checkpoint)
            atomic_json(checkpoint, row)
        catalog.append(row)
        atomic_json(config / 'datasets.json', catalog)
        print(f"{row['name']}: {row['status']} eligible={row['eligible']} "
              f"n={row.get('nodes', '-')} m={row.get('edges', '-')}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
