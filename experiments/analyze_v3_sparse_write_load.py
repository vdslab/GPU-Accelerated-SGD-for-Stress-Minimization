#!/usr/bin/env python3
"""Offline Sparse endpoint-load diagnostics; refuses to run before v3 finishes.

Never imports or changes the measurement runner. No GPU or SGD is executed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPHS = ['luxembourg_osm', 'web-Stanford', 'coAuthorsCiteseer',
                  'preferentialAttachment', 'kron_g500-logn17']
GRAPH_SOURCE = 'implementations/sparse/baseline-sparse-sgd-non-gpu/src/graph.rs'
METHOD_ORDER = ['sparse_sgd', 'atomic_sparse_sgd', 'rr_sparse_sgd']
METRICS = ['positive_weight_endpoint_updates', 'atomic_lock_target_touches']
STAT_FIELDS = ['vertices', 'total_per_iteration', 'mean', 'max', 'coefficient_of_variation',
               'p50', 'p90', 'p99', 'top_1pct_vertex_count', 'top_1pct_total',
               'top_1pct_share', 'share_of_all_vertex_total']
SUMMARY_FIELDS = ['graph', 'seed', 'pivots', 'metric', 'vertex_group', *STAT_FIELDS]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def write_summary(path, rows):
    """Always replace the CSV, including a header-only failure/empty result."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def reusable_result(result_path, provenance_path, expected_provenance):
    try:
        sidecar = json.loads(provenance_path.read_text())
        return (sidecar['provenance'] == expected_provenance
                and sidecar['result_sha256'] == sha(result_path))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_result(result, selection):
    request = selection['request']
    if not isinstance(result, dict):
        raise ValueError('Diagnostic result must be an object')
    if result.get('status') != 'success' or result.get('hashes_match_measured_record') is not True:
        raise ValueError('Diagnostic did not report verified success')
    for actual, expected in [('input_sha256', 'expected_input_sha256'),
                             ('graph_source_sha256', 'expected_graph_source_sha256'),
                             ('preprocess_sha256', 'expected_preprocess_sha256'),
                             ('initial_positions_sha256', 'expected_initial_positions_sha256'),
                             ('nodes', 'expected_nodes'), ('constraints', 'expected_constraints')]:
        if result.get(actual) != request[expected]:
            raise ValueError(f'Diagnostic identity mismatch: {actual}')
    for field in ['seed', 'pivots', 'iterations', 'epsilon']:
        if result.get(field) != request[field]:
            raise ValueError(f'Diagnostic parameter mismatch: {field}')
    if result.get('gpu_executed') is not False or result.get('sgd_executed') is not False:
        raise ValueError('Expected a preprocessing-only diagnostic')
    local_rows = []
    for metric in METRICS:
        for group in ['all', 'pivot', 'nonpivot']:
            statistics = result[metric][group]
            if not isinstance(statistics, dict):
                raise ValueError(f'Invalid statistics object: {metric}/{group}')
            if not isinstance(statistics.get('histogram_count_to_vertices'), dict):
                raise ValueError(f'Invalid histogram: {metric}/{group}')
            row = {field: statistics[field] for field in STAT_FIELDS}
            for field in ['vertices', 'total_per_iteration', 'top_1pct_vertex_count', 'top_1pct_total']:
                if not isinstance(row[field], int) or isinstance(row[field], bool) or row[field] < 0:
                    raise ValueError(f'Invalid integer count: {metric}/{group}/{field}')
            for field, value in row.items():
                if value is not None and (not isinstance(value, (int, float))
                                          or isinstance(value, bool) or not math.isfinite(value) or value < 0):
                    raise ValueError(f'Invalid statistic: {metric}/{group}/{field}')
            for field in ['top_1pct_share', 'share_of_all_vertex_total']:
                if row[field] is not None and row[field] > 1:
                    raise ValueError(f'Invalid fraction: {metric}/{group}/{field}')
            local_rows.append(dict(graph=selection['graph'], seed=request['seed'], pivots=request['pivots'],
                                   metric=metric, vertex_group=group, **row))
    # Return only after all six rows and identities validate successfully.
    return local_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'experiment_results/v3')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--graphs', nargs='+', default=DEFAULT_GRAPHS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=180)
    args = parser.parse_args()
    base = args.results_dir.resolve()
    out = (args.output_dir or base / 'analysis/sparse-write-load').resolve()

    # Mandatory guard before compilation, graph parsing or any native work.
    plan = json.loads((base / 'config/plan.json').read_text())
    jobs = plan['jobs'] if isinstance(plan, dict) else plan
    rows = [json.loads(line) for line in (base / 'raw/results.jsonl').read_text().splitlines()]
    completed = {row['run_id'] for row in rows}
    pending = {job['run_id'] for job in jobs} - completed
    if pending:
        raise SystemExit(f'Official measurements incomplete: {len(pending)} planned records remain; no build/diagnostic started')
    # Also prevent work while a completed run's runner is still shutting down.
    import psutil
    for process in psutil.process_iter(['pid', 'cmdline'], ad_value=None):
        if process.pid == os.getpid():
            continue
        command = process.info.get('cmdline') or []
        if any(Path(argument).name == 'run_v3.py' for argument in command):
            raise SystemExit(f'Official measurement runner still present (PID {process.pid}); no diagnostic started')

    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / 'summary.csv'
    write_summary(summary_path, [])
    save(out / 'outcomes.json', dict(analysis_status='running', results=[], skipped=[]))
    execution = json.loads((base / 'config/execution.json').read_text())
    source_hashes = json.loads((base / 'config/source-hashes.json').read_text())
    expected_source = source_hashes[GRAPH_SOURCE]
    if sha(ROOT / GRAPH_SOURCE) != expected_source:
        raise SystemExit('Shared preprocessing source differs from the measured source snapshot')
    manifest = ROOT / 'tools/diagnostics/sparse-write-load/Cargo.toml'
    target = ROOT / 'target/v3-diagnostics'
    binary = (args.binary or target / 'release/sparse-write-load').resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.binary is None:
        with (out / 'build.log').open('w') as log:
            subprocess.run(['cargo', 'build', '--release', '--manifest-path', str(manifest),
                            '--target-dir', str(target)], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    if not binary.is_file():
        raise SystemExit(f'Diagnostic binary missing: {binary}')

    tool_provenance = dict(schema_version=1, diagnostic_binary_sha256=sha(binary),
                           diagnostic_source_sha256=sha(manifest.parent / 'src/main.rs'),
                           diagnostic_manifest_sha256=sha(manifest),
                           diagnostic_lock_sha256=sha(manifest.parent / 'Cargo.lock') if (manifest.parent / 'Cargo.lock').exists() else None,
                           shared_graph_source_sha256=expected_source,
                           orchestration_sha256=sha(__file__))
    selected = []
    skipped = []
    for graph in args.graphs:
        candidates = [row for row in rows if row.get('status') == 'success'
                      and row.get('study') == 'basic' and row.get('family') == 'sparse'
                      and row.get('graph') == graph and row.get('seed') == args.seed
                      and row.get('pivots') == 200 and row.get('iterations') == 15
                      and row.get('epsilon') == 0.1]
        candidates = [row for row in candidates if row.get('method') in METHOD_ORDER]
        if not candidates:
            skipped.append(dict(graph=graph, seed=args.seed, reason='No successful measured basic Sparse/P200/seed record'))
            continue
        candidates.sort(key=lambda row: METHOD_ORDER.index(row['method']))
        reference = candidates[0]
        identity = ['input_sha256', 'initial_positions_sha256', 'preprocess_sha256', 'nodes', 'constraints']
        if any(any(row.get(key) != reference.get(key) for key in identity) for row in candidates):
            raise SystemExit(f'Same-seed Sparse records disagree on preprocessing identity: {graph}')
        input_path = Path(reference['input_path'])
        if not input_path.is_absolute():
            input_path = ROOT / input_path
        request = dict(input_path=str(input_path), seed=args.seed, pivots=200, iterations=15,
                       epsilon=0.1, expected_input_sha256=reference['input_sha256'],
                       expected_graph_source_sha256=expected_source,
                       expected_preprocess_sha256=reference['preprocess_sha256'],
                       expected_initial_positions_sha256=reference['initial_positions_sha256'],
                       expected_nodes=reference['nodes'], expected_constraints=reference['constraints'])
        selected.append(dict(graph=graph, seed=args.seed, reference_run_id=reference['run_id'],
                             matched_successful_methods=[row['method'] for row in candidates], request=request))
    save(out / 'selection.json', dict(measured_execution=execution, selected=selected, skipped=skipped,
                                      diagnostic_binary=str(binary), diagnostic_provenance=tool_provenance))
    outcomes = []
    table = []
    for selection in selected:
        prefix = out / f"{selection['graph']}-s{args.seed}-p200"
        request_path = prefix.with_suffix('.request.json')
        result_path = prefix.with_suffix('.result.json')
        provenance_path = prefix.with_suffix('.provenance.json')
        save(request_path, selection['request'])
        expected_provenance = dict(tool=tool_provenance, request=selection['request'])
        entry = dict(graph=selection['graph'], seed=args.seed, pivots=200,
                     reference_run_id=selection['reference_run_id'], result_path=str(result_path),
                     provenance_path=str(provenance_path))
        try:
            reuse = reusable_result(result_path, provenance_path, expected_provenance)
            candidate_path = result_path if reuse else prefix.with_suffix(f'.pending-{uuid.uuid4().hex}.json')
            if not reuse:
                with prefix.with_suffix('.log').open('w') as log:
                    subprocess.run([str(binary), '--request', str(request_path), '--output', str(candidate_path)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=args.timeout_seconds)
            result = json.loads(candidate_path.read_text())
            local_rows = validate_result(result, selection)
            if not reuse:
                # Retain superseded evidence; the native diagnostic refuses overwrite.
                if result_path.exists():
                    result_path.replace(prefix.with_suffix(f'.retired-{uuid.uuid4().hex}.json'))
                candidate_path.replace(result_path)
                save(provenance_path, dict(provenance=expected_provenance, result_sha256=sha(result_path)))
            result_hash = sha(result_path)
            entry.update(status='success', hashes_match_measured_record=True,
                         result_sha256=result_hash, reused=reuse)
            table.extend(local_rows)
        except (OSError, ValueError, KeyError, TypeError, AssertionError, subprocess.SubprocessError) as error:
            entry.update(status='error', error=repr(error))
        outcomes.append(entry)
        save(out / 'outcomes.json', dict(analysis_status='running', results=outcomes, skipped=skipped))
    write_summary(summary_path, table)
    save(out / 'outcomes.json', dict(analysis_status='complete', summary_sha256=sha(summary_path),
                                     results=outcomes, skipped=skipped))
    print(json.dumps(dict(diagnosed=sum(row['status'] == 'success' for row in outcomes),
                          errors=sum(row['status'] != 'success' for row in outcomes),
                          skipped=len(skipped), output_dir=str(out)), ensure_ascii=False))
    return int(any(row['status'] != 'success' for row in outcomes))


if __name__ == '__main__':
    sys.exit(main())
