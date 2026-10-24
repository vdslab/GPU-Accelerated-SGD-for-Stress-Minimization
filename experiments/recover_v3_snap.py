#!/usr/bin/env python3
"""Recover six v3 inputs from official SNAP, after the frozen main run finishes.

This prepares a separate 90-condition supplement; it never launches SGD and never
changes the original v3 configuration, inputs, plan, or measurement records.
"""
from __future__ import annotations

import argparse
import array
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from urllib.request import Request, urlopen

try:
    from v3_recovery import IDENTITY_CODE
except ModuleNotFoundError:
    from experiments.v3_recovery import IDENTITY_CODE

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'experiment_results/v3'
BASE = MAIN / 'snap-origin-recovery'
DATA = ROOT / 'data/v3_snap_origin'
POLICY = 'v3-nonzero-simple-undirected-largest-component-1'
IDENTITY = 'SNAP公式原配布から再構成（SuiteSparse配布物との辺・行番号直接照合未実施）'
COMMUNITIES = 'https://snap.stanford.edu/data/bigdata/communities/'
SPECS = [
    dict(name='com-Amazon', nodes=334863, edges=925872, nnz=1851744, minimum_id=1, maximum_id=548551,
         directed=False, origin_url=COMMUNITIES + 'com-amazon.ungraph.txt.gz'),
    dict(name='com-DBLP', nodes=317080, edges=1049866, nnz=2099732, minimum_id=0, maximum_id=425956,
         directed=False, origin_url=COMMUNITIES + 'com-dblp.ungraph.txt.gz'),
    dict(name='com-Youtube', nodes=1134890, edges=2987624, nnz=5975248, minimum_id=1, maximum_id=1157827,
         directed=False, origin_url=COMMUNITIES + 'com-youtube.ungraph.txt.gz'),
    dict(name='com-LiveJournal', nodes=3997962, edges=34681189, nnz=69362378, minimum_id=0, maximum_id=4036537,
         directed=False, origin_url=COMMUNITIES + 'com-lj.ungraph.txt.gz'),
    dict(name='soc-Pokec', nodes=1632803, edges=30622564, nnz=30622564, minimum_id=1, maximum_id=1632803,
         directed=True, origin_url='https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz'),
    dict(name='loc-Gowalla', nodes=196591, edges=950327, nnz=1900654, minimum_id=0, maximum_id=196590,
         directed=False, origin_url='https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz'),
]
METADATA_VERIFIED_JST = '2026-10-24'
CHUNK = 1024 * 1024


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def json_read(path):
    return json.loads(path.read_text())


def guard():
    """No main or supplemental native runner may overlap preparation."""
    import psutil
    running = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline', 'status'], ad_value=None):
        command = process.info.get('cmdline') or []
        if process.info.get('status') == psutil.STATUS_ZOMBIE:
            continue
        if (str(process.info.get('name', '')).lower().startswith('python')
                and any(Path(value).name == 'run_v3.py' for value in command)):
            running.append(process.pid)
        if (process.pid != os.getpid() and str(process.info.get('name', '')).lower().startswith('python')
                and any(Path(value).name == 'recover_v3_snap.py' for value in command)):
            raise RuntimeError(f'Another recovery preparation is already running: {process.pid}')
    if running:
        raise RuntimeError(f'run_v3.py is still alive: {running}; preparation is forbidden')
    plan = json_read(MAIN / 'config/plan.json')
    results = [json.loads(line) for line in (MAIN / 'raw/results.jsonl').read_text().splitlines() if line.strip()]
    plan_ids = [row['run_id'] for row in plan]
    actual = [row['run_id'] for row in results]
    if (len(plan_ids) != 3210 or len(actual) != 3210 or len(set(actual)) != 3210
            or set(actual) != set(plan_ids)):
        raise RuntimeError('The original 3210-condition main plan is not complete and unique')
    execution = json_read(MAIN / 'config/execution.json')
    if sha(ROOT / 'experiments/run_v3.py') != execution['runner_sha256']:
        raise RuntimeError('Runner differs from frozen v3 runner')
    frozen_sources = json_read(MAIN / 'config/source-hashes.json')
    if sha(ROOT / 'experiments/prepare_v3.py') != frozen_sources.get('experiments/prepare_v3.py'):
        raise RuntimeError('Existing v3 LCC preparation script differs from the frozen source')
    for name, digest in frozen_sources.items():
        if name.startswith(('implementations/', 'shared/')) and sha(ROOT / name) != digest:
            raise RuntimeError(f'Native/shared source changed: {name}')
    from run_v3 import METHODS, plan as make_plan
    for method, binary in METHODS.items():
        if sha(ROOT / 'target/v3-release/release' / binary) != execution['binary_sha256'][method]:
            raise RuntimeError(f'Frozen binary changed: {method}')
    names = {spec['name'] for spec in SPECS}
    existing = {row['name']: row for row in json_read(MAIN / 'config/datasets.json')}
    source_catalog = {row['graph'].split('/')[-1]: row
                      for row in json_read(MAIN / 'config/source-catalog.json')['graphs']}
    for spec in SPECS:
        previous = existing[spec['name']]
        if previous.get('source_nodes') != spec['nodes'] or previous.get('status') != 'unavailable':
            raise RuntimeError(f'Main cohort state or node metadata differs: {spec["name"]}')
        catalog = source_catalog[spec['name']]
        for key in ('nrows', 'ncols'):
            if catalog.get(key) is not None and catalog[key] != spec['nodes']:
                raise RuntimeError(f'Catalog {key} differs: {spec["name"]}')
        if catalog.get('expanded_nonzeros') is not None and catalog['expanded_nonzeros'] != spec['nnz']:
            raise RuntimeError(f'Catalog nonzeros differ: {spec["name"]}')
    original_jobs = [row for row in plan if row['graph'] in names]
    selected_results = [row for row in results if row['graph'] in names]
    if len(original_jobs) != 90 or len(selected_results) != 90:
        raise RuntimeError('Six recovered graphs must have exactly 90 original basic conditions')
    if any(row['status'] != 'unavailable' or row.get('process_wall_seconds') is not None
           or row.get('reused_from') for row in selected_results):
        raise RuntimeError('Recovery is restricted to the originally unavailable, unexecuted conditions')
    if len(make_plan([dict(name=name, family='large') for name in names])) != 90:
        raise RuntimeError('Recovered inputs unexpectedly enter convergence, tiling, or pivot studies')
    return execution, original_jobs


def free_space(args):
    if shutil.disk_usage(ROOT).free < args.min_free_gib * 1024 ** 3:
        raise RuntimeError(f'Free disk below preparation reserve {args.min_free_gib:g} GiB')


def download(spec, args):
    directory = DATA / 'originals'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec['origin_url'].rsplit('/', 1)[-1]
    receipt_path = target.with_suffix(target.suffix + '.download.json')
    if target.exists():
        if not receipt_path.is_file():
            raise RuntimeError(f'Existing unverified source retained; inspect or move it before retry: {target}')
        receipt = json_read(receipt_path)
        if (receipt.get('origin_url') != spec['origin_url'] or target.stat().st_size != receipt.get('bytes')
                or sha(target) != receipt.get('sha256')):
            raise RuntimeError(f'Existing source failed SHA/size verification; retained: {target}')
        return target, receipt
    if receipt_path.is_file():
        # A crash after receipt commit but before promotion is recoverable only
        # from a complete, SHA-matching partial. Other partials remain untouched.
        receipt = json_read(receipt_path)
        candidates = []
        if receipt.get('origin_url') == spec['origin_url']:
            for candidate in directory.glob(target.name + '.partial-*'):
                if candidate.stat().st_size == receipt.get('bytes') and sha(candidate) == receipt.get('sha256'):
                    candidates.append(candidate)
        if len(candidates) == 1:
            candidates[0].replace(target)
            return target, receipt
        receipt_path.replace(receipt_path.with_name(receipt_path.name + '.orphan-' + stamp()))
    partial = target.with_name(target.name + '.partial-' + stamp())
    digest, count, started = hashlib.sha256(), 0, time.monotonic()
    with urlopen(Request(spec['origin_url'], headers={'User-Agent': 'v3-snap-origin-recovery'}),
                 timeout=args.socket_timeout_seconds) as response:
        if response.status != 200 or response.url != spec['origin_url']:
            raise RuntimeError(f'Unexpected HTTP response or redirected origin: {response.status} {response.url}')
        expected = response.headers.get('Content-Length')
        with partial.open('xb') as stream:
            while block := response.read(CHUNK):
                free_space(args)
                if time.monotonic() - started > args.download_timeout_seconds:
                    raise TimeoutError('Source download exceeded wall-time bound; partial file retained')
                stream.write(block)
                digest.update(block)
                count += len(block)
            stream.flush()
            os.fsync(stream.fileno())
    if expected is not None and count != int(expected):
        raise RuntimeError('Content-Length mismatch; partial source retained')
    receipt = dict(origin_url=spec['origin_url'], sha256=digest.hexdigest(), bytes=count,
                   content_length=int(expected) if expected is not None else None,
                   downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                   checksum_scope='locally computed SHA-256, not an upstream published checksum')
    # Receipt first: a crash between these operations leaves an untrusted partial,
    # never a silently reusable completed source without a matching SHA receipt.
    save(receipt_path, receipt)
    partial.replace(target)
    return target, receipt


def sorted_unique(source, target):
    """External-backed sort, then bounded-memory adjacent deduplication."""
    import numpy as np
    count = source.stat().st_size // 8
    if count == 0:
        raise ValueError('Empty edge list')
    values = np.memmap(source, dtype='<u8', mode='r+', shape=(count,))
    values.sort(kind='quicksort')
    values.flush()
    previous, unique = None, 0
    with target.open('xb') as stream:
        for start in range(0, count, CHUNK):
            block = np.asarray(values[start:start + CHUNK])
            keep = np.empty(block.size, dtype=bool)
            keep[0] = previous is None or int(block[0]) != previous
            keep[1:] = block[1:] != block[:-1]
            block[keep].astype('<u8', copy=False).tofile(stream)
            unique += int(keep.sum())
            previous = int(block[-1])
    del values
    return unique


def convert(spec, source, receipt, directory, args):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory / 'raw-ordered-pairs.u64le'
    seen = bytearray(spec['maximum_id'] + 1)
    header_nodes = header_edges = None
    headers, records, loop_lines = [], 0, 0
    buffer = array.array('Q')
    if buffer.itemsize != 8:
        raise RuntimeError('64-bit unsigned array type required')
    with source.open('rb') as compressed:
        if compressed.read(2) != b'\x1f\x8b':
            raise ValueError('Official response is not gzip')
    # Reading to EOF validates gzip CRC and truncated-stream handling. No text
    # decompression result or Python collection of millions of edges is kept.
    with gzip.open(source, 'rt', encoding='ascii') as text, raw.open('xb') as output:
        for line_number, line in enumerate(text, 1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('#'):
                if len(headers) < 50:
                    headers.append(stripped)
                match = re.search(r'Nodes:\s*(\d+)\s+Edges:\s*(\d+)', stripped, re.IGNORECASE)
                if match:
                    values = tuple(map(int, match.groups()))
                    if header_nodes is not None and values != (header_nodes, header_edges):
                        raise ValueError('Conflicting SNAP header counts')
                    header_nodes, header_edges = values
                continue
            fields = stripped.split()
            if len(fields) != 2:
                raise ValueError(f'Expected exactly two node IDs, line {line_number}')
            u, v = map(int, fields)
            if not (spec['minimum_id'] <= u <= spec['maximum_id']
                    and spec['minimum_id'] <= v <= spec['maximum_id']):
                raise ValueError(f'Node ID outside official range on line {line_number}')
            seen[u] = seen[v] = 1
            loop_lines += u == v
            records += 1
            buffer.append((u << 32) | v)
            if len(buffer) >= CHUNK:
                free_space(args)
                if sys.byteorder != 'little':
                    buffer.byteswap()
                buffer.tofile(output)
                buffer = array.array('Q')
        if sys.byteorder != 'little':
            buffer.byteswap()
        buffer.tofile(output)
    node_ids = np.flatnonzero(np.frombuffer(seen, dtype=np.uint8)).astype(np.int64)
    if (node_ids.size != spec['nodes'] or int(node_ids[0]) != spec['minimum_id']
            or int(node_ids[-1]) != spec['maximum_id']):
        raise ValueError('Unique ID count/range differs from official metadata; isolated vertices or ID mapping must be investigated')
    if header_nodes is not None and header_nodes != spec['nodes']:
        raise ValueError('SNAP header node count differs from official metadata; no silent isolated-node deletion')
    mapping = directory / (spec['name'] + '.matrix-node-ids.npy')
    np.save(mapping, node_ids, allow_pickle=False)
    directed_path = directory / 'unique-ordered-pairs.u64le'
    directed_count = sorted_unique(raw, directed_path)
    canonical = directory / 'unordered-pairs.u64le'
    values = np.memmap(directed_path, dtype='<u8', mode='r', shape=(directed_count,))
    with canonical.open('xb') as stream:
        for start in range(0, directed_count, CHUNK):
            block = np.asarray(values[start:start + CHUNK])
            u, v = block >> np.uint64(32), block & np.uint64(0xffffffff)
            ((np.minimum(u, v) << np.uint64(32)) | np.maximum(u, v)).astype('<u8', copy=False).tofile(stream)
    del values
    unique_path = directory / 'unique-unordered-pairs.u64le'
    undirected_count = sorted_unique(canonical, unique_path)
    unique = np.memmap(unique_path, dtype='<u8', mode='r', shape=(undirected_count,))
    unique_loops = 0
    for start in range(0, undirected_count, CHUNK):
        block = unique[start:start + CHUNK]
        unique_loops += int(np.count_nonzero((block >> np.uint64(32)) == (block & np.uint64(0xffffffff))))
    expanded = directed_count if spec['directed'] else 2 * (undirected_count - unique_loops) + unique_loops
    observed_edges = directed_count if spec['directed'] else undirected_count
    if expanded != spec['nnz'] or observed_edges != spec['edges']:
        raise ValueError(f'Official node/edge/nnz validation failed: directed={directed_count}, unordered={undirected_count}, nnz={expanded}')
    header_interpretations = []
    if header_edges is not None:
        header_interpretations = [label for label, value in [
            ('physical_edge_lines', records), ('unique_ordered_pairs', directed_count),
            ('unique_unordered_pairs', undirected_count), ('official_published_edges', spec['edges'])]
            if header_edges == value]
        if not header_interpretations:
            raise ValueError('SNAP header edge count matches neither physical records nor verified graph statistics')
    lookup = np.full(spec['maximum_id'] + 1, -1, dtype=np.int32)
    lookup[node_ids] = np.arange(node_ids.size, dtype=np.int32)
    converted = directory / (spec['name'] + '.mtx')
    with converted.open('x', buffering=CHUNK) as output:
        output.write('%%MatrixMarket matrix coordinate pattern general\n')
        output.write('% Official SNAP origin, sorted compact IDs; simple undirected pairs once.\n')
        output.write(f'{node_ids.size} {node_ids.size} {undirected_count}\n')
        for start in range(0, undirected_count, CHUNK):
            block = unique[start:start + CHUNK]
            u = lookup[(block >> np.uint64(32)).astype(np.int64)]
            v = lookup[(block & np.uint64(0xffffffff)).astype(np.int64)]
            output.writelines(f'{int(a) + 1} {int(b) + 1}\n' for a, b in zip(u, v))
    del unique
    validation = dict(schema_version=1, kind='official_snap_origin_validation', name=spec['name'],
        identity_code=IDENTITY_CODE, identity_claim=IDENTITY,
        source_page='https://sparse.tamu.edu/SNAP/' + spec['name'],
        origin_url=spec['origin_url'], origin_sha256=receipt['sha256'], origin_bytes=receipt['bytes'],
        gzip_crc_validated=True, official_nodes=spec['nodes'], official_edges=spec['edges'],
        official_header_nodes=spec['nodes'], official_header_edges=spec['edges'],
        official_header_field_definition='Published official N/E statistics, not an assertion that an edge-file header exists',
        official_expanded_nnz=spec['nnz'], official_edges_are_directed=spec['directed'],
        header_present=bool(headers), headers=headers, parsed_header_nodes=header_nodes, parsed_header_edges=header_edges,
        parsed_header_edge_interpretations=header_interpretations, observed_unique_nodes=int(node_ids.size),
        observed_unique_edges=undirected_count, observed_unique_directed_edges=directed_count,
        observed_unique_undirected_edges=undirected_count, observed_expanded_nnz=expanded,
        observed_undirected_expanded_nnz=2 * (undirected_count - unique_loops) + unique_loops,
        raw_edge_lines=records, raw_self_loop_lines=loop_lines, unique_self_loops=unique_loops,
        directed_duplicate_lines=records - directed_count, reciprocal_pair_count=directed_count - undirected_count,
        observed_minimum_id=int(node_ids[0]), observed_maximum_id=int(node_ids[-1]),
        zero_indexed=spec['minimum_id'] == 0,
        gaps_within_observed_id_range=int(node_ids[-1] - node_ids[0] + 1 - node_ids.size),
        vertex_mapping='matrix row i (zero-based) = matrix-node-ids.npy[i], ascending observed original SNAP ID',
        metadata_matches_official=True, metadata_verified_date_jst=METADATA_VERIFIED_JST,
        converted_mtx_sha256=sha(converted), matrix_node_ids_sha256=sha(mapping),
        matrix_node_ids_path=relative(mapping), preparation_policy=POLICY)
    return converted, mapping, validation


def verify_checkpoint(checkpoint, spec):
    row = json_read(checkpoint)
    manifest = row.get('recovery_source', {})
    if row.get('status') != 'ready' or row.get('origin_url') != spec['origin_url']:
        raise RuntimeError('Preparation checkpoint is incomplete or belongs to another source')
    for path_key, hash_key in [('origin_path', 'origin_sha256'), ('converted_mtx_path', 'converted_mtx_sha256'),
            ('matrix_node_ids_path', 'matrix_node_ids_sha256'), ('prepared_path', 'prepared_sha256'),
            ('vertex_map_path', 'vertex_map_sha256'), ('lcc_origin_node_ids_path', 'lcc_origin_node_ids_sha256'),
            ('validation_path', 'validation_sha256')]:
        if not manifest.get(path_key) or sha(resolve(manifest[path_key])) != manifest.get(hash_key):
            raise RuntimeError(f'Prepared checkpoint SHA verification failed for {path_key}; evidence retained')
    if (row['sha256'] != manifest['prepared_sha256'] or row['preparation_policy'] != POLICY
            or row.get('source_distribution_status') != 'official_snap_origin_recovery'
            or row.get('preparation_script_sha256') != sha(Path(__file__))
            or row.get('source_sha256') != manifest.get('converted_mtx_sha256')
            or row.get('origin_sha256') != manifest.get('origin_sha256')
            or row.get('path') != manifest.get('prepared_path')):
        raise RuntimeError('Preparation checkpoint metadata differs from recovery policy')
    validation = json_read(resolve(manifest['validation_path']))
    expected_edges = validation.get('observed_unique_directed_edges' if spec['directed'] else 'observed_unique_edges')
    if (validation.get('schema_version') != 1 or validation.get('kind') != 'official_snap_origin_validation'
            or validation.get('name') != spec['name'] or validation.get('origin_url') != spec['origin_url']
            or validation.get('metadata_matches_official') is not True
            or validation.get('observed_unique_nodes') != spec['nodes'] or expected_edges != spec['edges']
            or validation.get('observed_expanded_nnz') != spec['nnz']
            or validation.get('prepared_nodes') != row.get('nodes')
            or validation.get('prepared_edges') != row.get('edges')
            or validation.get('prepared_sha256') != row.get('sha256')):
        raise RuntimeError('Verified checkpoint validation disagrees with official metadata or prepared dataset')
    return row


def prepare_one(spec, args):
    checkpoint = BASE / 'config/prepared' / (spec['name'] + '.json')
    if checkpoint.is_file():
        return verify_checkpoint(checkpoint, spec)
    free_space(args)
    origin, receipt = download(spec, args)
    directory = DATA / 'graphs' / spec['name'] / stamp()
    converted, mapping, validation = convert(spec, origin, receipt, directory, args)
    from prepare_v3 import bounded_worker
    row = dict(name=spec['name'], family='large', source_graph='SNAP/' + spec['name'], category='SNAP',
        source_path=str(converted), source_sha256=validation['converted_mtx_sha256'], source_nodes=spec['nodes'],
        source_distribution_status='official_snap_origin_recovery', source_page=validation['source_page'],
        origin_url=spec['origin_url'], download_url=spec['origin_url'], origin_path=relative(origin),
        origin_sha256=receipt['sha256'], identity_claim=IDENTITY, preparation_policy=POLICY, eligible=False)
    prep_args = argparse.Namespace(data_dir=directory / 'prepared', timeout_seconds=args.preprocess_timeout_seconds,
                                   max_rss_gib=args.max_preprocess_rss_gib)
    row = bounded_worker(row, prep_args, directory / 'lcc-preparation.json')
    if row.get('status') != 'ready' or not row.get('eligible'):
        save(directory / 'failed-preparation.json', row)
        raise RuntimeError(row.get('reason', 'LCC preparation was not ready/eligible'))
    if (row['nodes'] != spec['nodes'] or row['source_components'] != 1
            or row['edges'] != validation['observed_unique_undirected_edges'] - validation['unique_self_loops']):
        save(directory / 'failed-preparation.json', row)
        raise RuntimeError('LCC N/E/component count differs from the official connected graph and verified unordered pairs')
    import numpy as np
    ids = np.load(mapping, allow_pickle=False)
    vertices = np.load(resolve(row['vertex_map_path']), allow_pickle=False)
    lcc_ids_path = directory / (spec['name'] + '.lcc-origin-node-ids.npy')
    np.save(lcc_ids_path, ids[vertices], allow_pickle=False)
    validation.update(prepared_nodes=row['nodes'], prepared_edges=row['edges'], prepared_sha256=row['sha256'],
        source_components=row['source_components'], vertex_map_path=row['vertex_map_path'],
        vertex_map_sha256=row['vertex_map_sha256'], lcc_origin_node_ids_path=relative(lcc_ids_path),
        lcc_origin_node_ids_sha256=sha(lcc_ids_path),
        lcc_mapping='LCC row i -> converted zero-based row vertex-map.npy[i] -> original SNAP ID lcc-origin-node-ids.npy[i]')
    validation_path = directory / 'validation.json'
    save(validation_path, validation)
    manifest = dict(name=spec['name'], origin_url=spec['origin_url'], origin_path=relative(origin),
        origin_sha256=receipt['sha256'], origin_bytes=receipt['bytes'], converted_mtx_path=relative(converted),
        converted_mtx_sha256=sha(converted), matrix_node_ids_path=relative(mapping), matrix_node_ids_sha256=sha(mapping),
        prepared_path=row['path'], prepared_sha256=row['sha256'], vertex_map_path=row['vertex_map_path'],
        vertex_map_sha256=row['vertex_map_sha256'], lcc_origin_node_ids_path=relative(lcc_ids_path),
        lcc_origin_node_ids_sha256=sha(lcc_ids_path), validation_path=relative(validation_path),
        validation_sha256=sha(validation_path))
    row.update(matrix_node_ids_path=relative(mapping), matrix_node_ids_sha256=sha(mapping),
               recovery_source=manifest, preparation_script_sha256=sha(Path(__file__)))
    save(checkpoint, row)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket-timeout-seconds', type=float, default=60)
    parser.add_argument('--download-timeout-seconds', type=float, default=900)
    parser.add_argument('--preprocess-timeout-seconds', type=float, default=1800)
    parser.add_argument('--max-preprocess-rss-gib', type=float, default=8)
    parser.add_argument('--min-free-gib', type=float, default=8)
    args = parser.parse_args(argv)
    if any(value <= 0 for value in vars(args).values()):
        parser.error('All preparation limits must be positive')
    execution, original_jobs = guard()
    if (ROOT / 'data').is_symlink() or DATA.is_symlink():
        raise RuntimeError('data parent is now a symlink; inspect destination before recovery')
    if (BASE / 'config/execution.json').exists():
        raise RuntimeError('Supplemental measurements already started; recovery config must stay frozen')
    BASE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rows, outcomes = [], []
    current = None
    try:
        for spec in SPECS:
            current = spec['name']
            print('Recovering', current, flush=True)
            row = prepare_one(spec, args)
            rows.append(row)
            outcomes.append(dict(name=current, status='ready', nodes=row['nodes'], edges=row['edges']))
            save(BASE / 'preparation-outcomes.json', outcomes)
        # Never publish a runnable partial cohort. All six must pass validation.
        from run_v3 import plan as make_plan
        recovered_jobs = make_plan(rows)
        if len(recovered_jobs) != 90 or {r['run_id']: r for r in recovered_jobs} != {r['run_id']: r for r in original_jobs}:
            raise RuntimeError('Recovered 90-condition plan does not exactly match originally missing plan entries')
        manifest = dict(schema_version=1, kind='official_snap_origin_recovery', target_graphs=[s['name'] for s in SPECS],
            main_base=str(MAIN), main_execution_sha256=sha(MAIN / 'config/execution.json'),
            main_results_sha256=sha(MAIN / 'raw/results.jsonl'), main_datasets_sha256=sha(MAIN / 'config/datasets.json'),
            main_plan_sha256=sha(MAIN / 'config/plan.json'), main_source_hashes_sha256=sha(MAIN / 'config/source-hashes.json'),
        preparation_policy=POLICY, identity_code=IDENTITY_CODE, identity_claim=IDENTITY,
        sources=[r['recovery_source'] for r in rows],
            preparation_script_sha256=sha(Path(__file__)), existing_prepare_script_sha256=sha(ROOT / 'experiments/prepare_v3.py'),
            data_parent_is_symlink=False, data_parent_realpath=str((ROOT / 'data').resolve()),
            preparation_limits=vars(args), native_limits_unchanged=True, expected_conditions=90,
            native_binary_sha256=execution['binary_sha256'])
        config = BASE / 'config'
        save(config / 'recovery.json', manifest)
        save(config / 'datasets.json', rows)
        (config / 'limits.json').write_bytes((MAIN / 'config/limits.json').read_bytes())
        save(config / 'expected-official-metadata.json', dict(verified_date_jst=METADATA_VERIFIED_JST, graphs=SPECS))
        save(BASE / 'preparation-status.json', dict(status='ready', prepared_graphs=6, expected_conditions=90))
        prose = ['# v3 SNAP 原配布からの補完準備', '', IDENTITY, '',
            '正式一巡の欠測90条件を独立baseで補完する。元の3210条件・欠測statusは変更しない。', '',
            '| Graph | Prepared N | Prepared E | Validation |', '|---|---:|---:|---|']
        for row in rows:
            evidence = row['recovery_source']['validation_path']
            prose.append(f"| {row['name']} | {row['nodes']} | {row['edges']} | `{evidence}` |")
        prose += ['', '取得済みgzipをSHA確認し、EOFまで読んでCRCを検証した。元ID・有向unique・無向unique・selfloop・重複を別計数し、公式N/E/nnzと照合した。',
            '昇順の元IDから行番号を決定し、既存prepare_v3の同じLCC処理を再利用した。SuiteSparse配布物と行番号・辺・同seedの初期配置割当が同一とは主張しない。',
            'raw gzip、全体のmatrix→元ID対応、LCC→matrix行→元ID対応と各SHAを保持する。失敗したpartial原本・中間ファイルは自動削除しない。', '',
            '次の実行：`python3 experiments/run_v3.py --results-dir experiment_results/v3/snap-origin-recovery`。同じ実行ファイル・native制限のSparse3手法×5seed×6graph=90条件、各設定のwarmupを別途実行する。']
        (BASE / 'preparation-report.md').write_text('\n'.join(prose) + '\n')
        print('Prepared six verified official-origin inputs for 90 supplemental conditions; no experiment launched.', flush=True)
        return 0
    except BaseException as error:
        failure = dict(status='error', graph=current, error=repr(error), traceback=traceback.format_exc(),
            evidence_retained=True, completed_graphs=[r['name'] for r in rows],
            reason='Preparation stopped; no runnable datasets.json is generated until all six sources pass')
        save(BASE / ('preparation-failure-' + stamp() + '.json'), failure)
        # Human-readable failure evidence is separate from frozen main results.
        (BASE / 'preparation-failure.md').write_text('# SNAP 補完準備の停止\n\n'
            f'対象：{current}。理由：`{error!r}`。\n\n'
            '途中原本・検証結果・中間ファイルを保持した。元v3の計測記録と設定は変更していない。'
            '正常完了checkpointは再開時に全成果物SHAを照合し、未検証partialは再利用しない。\n')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
