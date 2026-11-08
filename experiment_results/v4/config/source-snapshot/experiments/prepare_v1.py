"""Prepare common largest-component inputs and versioned study manifests."""
import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from artifact_paths import provenance_path
except ModuleNotFoundError:
    from experiments.artifact_paths import provenance_path

import numpy as np
from scipy.io import mmread
from scipy.sparse.csgraph import connected_components
from scipy.sparse import triu

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT / 'experiment_results/v1'
SMALL = ['dwt_419', 'qh882', '1138_bus', 'dwt_1005', 'cage8', 'poli',
         'dwt_2680', 'EVA', '3elt', 'USpowerGrid']
LARGE = ['G65', 'fe_4elt2', 'bcsstk31', 'venkat50', 'luxembourg_osm', 'web-NotreDame']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR,
                        help='result version directory containing config/raw/tables/figures/report')
    parser.add_argument('--experiment-prefix', help='experiment ID prefix; defaults to results directory name')
    parser.add_argument('--data-dir', type=Path, help='prepared inputs; defaults to data/<prefix>_lcc')
    args = parser.parse_args(argv)
    args.results_dir = args.results_dir.resolve()
    args.experiment_prefix = args.experiment_prefix or args.results_dir.name
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.experiment_prefix):
        parser.error('--experiment-prefix must use letters, digits, -, _, or .')
    args.data_dir = (args.data_dir or ROOT / 'data' / (args.experiment_prefix + '_lcc')).resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    raw = args.results_dir / 'raw'
    if raw.exists() and any(raw.iterdir()):
        raise SystemExit('Existing raw runs must keep their original config; choose a new --results-dir.')
    config = args.results_dir / 'config'
    config.mkdir(parents=True, exist_ok=True)
    destination = args.data_dir
    destination.mkdir(parents=True, exist_ok=True)
    catalog = []
    for name in SMALL + LARGE:
        source = ROOT / 'data' / (name + '.mtx')
        matrix = mmread(source).tocsr()
        matrix.data[:] = 1
        matrix = matrix.maximum(matrix.T)
        matrix.setdiag(0)
        matrix.eliminate_zeros()
        components, labels = connected_components(matrix, directed=False)
        label = int(np.argmax(np.bincount(labels)))
        vertices = np.flatnonzero(labels == label)
        graph = matrix[vertices][:, vertices]
        edges = triu(graph, k=1).tocoo()
        target = destination / (name + '.mtx')
        with target.open('w') as out:
            out.write('%%MatrixMarket matrix coordinate pattern general\n')
            out.write('% Undirected simple largest component; each edge listed once.\n')
            out.write(f'{len(vertices)} {len(vertices)} {edges.nnz}\n')
            for u, v in zip(edges.row, edges.col):
                out.write(f'{u + 1} {v + 1}\n')
        mapping = destination / (name + '.vertex-map.txt')
        np.savetxt(mapping, vertices, fmt='%d', header='new index -> original zero-based index')
        catalog.append(dict(name=name, source_path=provenance_path(source, ROOT),
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            path=provenance_path(target, ROOT),
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            source_nodes=matrix.shape[0], nodes=len(vertices), edges=edges.nnz,
                            source_components=components,
                            vertex_map_path=provenance_path(mapping, ROOT)))
        print(name, len(vertices), edges.nnz, 'components:', components, flush=True)
    (config / 'datasets.json').write_text(json.dumps(catalog, indent=2) + '\n')
    # Preserve v1's original 5/2-method recipe. New result versions include the
    # subsequently added Atomic Sparse implementation in their initial plan.
    full_methods = ['sgd', 'atomic_sgd', 'rr_sgd', 'sparse_sgd', 'rr_sparse_sgd']
    sparse_methods = ['sparse_sgd', 'rr_sparse_sgd']
    if args.results_dir != DEFAULT_RESULTS_DIR.resolve():
        full_methods.insert(-1, 'atomic_sparse_sgd')
        sparse_methods.insert(-1, 'atomic_sparse_sgd')
    for group, names, seeds, methods, timeout in [
        ('quality-timing', SMALL, 10,
         full_methods, 120),
        ('large-sparse', LARGE, 3, sparse_methods, 180),
    ]:
        manifest = dict(experiment_id=args.experiment_prefix + '-' + group, run_mode='benchmark', methods=methods,
                        datasets=[{key: item[key] for key in ['name', 'path', 'sha256']}
                                  for item in catalog if item['name'] in names],
                        seeds=list(range(seeds)), iterations=[15], epsilon=[0.1],
                        pivots=[200], repetitions=1, warmups=1, timeout_seconds=timeout,
                        fail_fast=False)
        (config / (group + '.json')).write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
