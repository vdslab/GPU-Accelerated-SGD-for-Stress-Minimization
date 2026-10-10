"""Prepare identical unweighted, undirected largest components for the v1 study."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse.csgraph import connected_components
from scipy.sparse import triu

ROOT = Path(__file__).resolve().parent.parent
SMALL = ['dwt_419', 'qh882', '1138_bus', 'dwt_1005', 'cage8', 'poli',
         'dwt_2680', 'EVA', '3elt', 'USpowerGrid']
LARGE = ['G65', 'fe_4elt2', 'bcsstk31', 'venkat50', 'luxembourg_osm', 'web-NotreDame']


def main():
    config = ROOT / 'experiment_results/v1/config'
    destination = ROOT / 'prod_data/v1_lcc'
    destination.mkdir(parents=True, exist_ok=True)
    catalog = []
    for name in SMALL + LARGE:
        source = ROOT / 'prod_data' / (name + '.mtx')
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
        catalog.append(dict(name=name, source_path=str(source.relative_to(ROOT)),
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            path=str(target.relative_to(ROOT)),
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            source_nodes=matrix.shape[0], nodes=len(vertices), edges=edges.nnz,
                            source_components=components,
                            vertex_map_path=str(mapping.relative_to(ROOT))))
        print(name, len(vertices), edges.nnz, 'components:', components, flush=True)
    (config / 'datasets.json').write_text(json.dumps(catalog, indent=2) + '\n')
    for group, names, seeds, methods, timeout in [
        ('quality-timing', SMALL, 10,
         ['sgd', 'atomic_sgd', 'rr_sgd', 'sparse_sgd', 'rr_sparse_sgd'], 120),
        ('large-sparse', LARGE, 3, ['sparse_sgd', 'rr_sparse_sgd'], 180),
    ]:
        manifest = dict(experiment_id='v1-' + group, run_mode='benchmark', methods=methods,
                        datasets=[{key: item[key] for key in ['name', 'path', 'sha256']}
                                  for item in catalog if item['name'] in names],
                        seeds=list(range(seeds)), iterations=[15], epsilon=[0.1],
                        pivots=[200], repetitions=1, warmups=1, timeout_seconds=timeout,
                        fail_fast=False)
        (config / (group + '.json')).write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
