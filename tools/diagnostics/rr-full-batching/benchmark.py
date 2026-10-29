#!/usr/bin/env python3
"""Sequential same-order RR regression and CPU comparison; run at repository root."""
import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import time

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--before', type=pathlib.Path, required=True)
parser.add_argument('--after', type=pathlib.Path, default=pathlib.Path('implementations/full/rr_gpu/target/release/rr-gpu'))
parser.add_argument('--output', type=pathlib.Path, required=True)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)
after = args.after
cpu = pathlib.Path('implementations/full/baseline-sgd-non-gpu/target/release/baseline-sgd-non-gpu')


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def run(tag, binary, graph, seed, kernel=None):
    record_path = out / (tag + '.json')
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record['measured_binary_sha256'] == sha(binary), tag
        return record
    cmd = [str(binary), '--input', f'data/{graph}.mtx', '--iterations', '15',
           '--seed', str(seed), '--output-format', 'json', '--output-dir',
           str(out / 'artifacts'), '--run-id', tag]
    if kernel:
        tile, workgroup, memory, shuffled = kernel
        cmd += ['--tile-size', str(tile), '--workgroup-size', str(workgroup),
                '--coordinate-memory', memory]
        if not shuffled:
            cmd += ['--no-outer-shuffle', '--no-inner-shuffle']
    print('START', tag, flush=True)
    started = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    (out / (tag + '.stderr')).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError((tag, result.returncode, result.stderr[-1500:]))
    record = json.loads(result.stdout)
    record.update(measured_command=cmd, measured_binary_sha256=sha(binary),
                  measured_process_wall_seconds=time.monotonic() - started)
    data = pathlib.Path(record['final_positions_path']).read_bytes().split(b'# Positions', 1)[1]
    record['coordinate_text_sha256'] = hashlib.sha256(data).hexdigest()
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print('DONE', tag, 'SGD ms', record['iteration_time_ms'], flush=True)
    return record


def compare(old, new):
    for key in ['coordinate_text_sha256', 'initial_positions_sha256', 'input_sha256',
                'stress_value', 'dispatches', 'iterations', 'seed', 'constraints',
                'stress_kind', 'stress_samples', 'stress_seed']:
        assert old.get(key) == new.get(key), (key, old.get(key), new.get(key))


# GPU/shader warmup is not included in the comparison.
for label, binary in [('before', args.before), ('after', after)]:
    run(f'warmup-{label}', binary, '1138_bus', 99, (1024, 1024, 'shared', True))

pairs = []
for graph in ['fe_4elt2', 'delaunay_n14']:
    for seed in [0, 1, 3]:
        variants = [('before', args.before), ('after', after), ('cpu', cpu)]
        if seed == 1:
            variants.reverse()
        records = {}
        for label, binary in variants:
            records[label] = run(f'{graph}-s{seed}-{label}', binary, graph, seed,
                                 None if label == 'cpu' else (1024, 1024, 'shared', True))
        old, new, reference = (records[k] for k in ['before', 'after', 'cpu'])
        compare(old, new)
        assert new['input_sha256'] == reference['input_sha256']
        assert new['initial_positions_sha256'] == reference['initial_positions_sha256']
        pairs.append(dict(graph=graph, seed=seed, exact_coordinates_and_stress=True,
                          before_ms=old['iteration_time_ms'], after_ms=new['iteration_time_ms'],
                          cpu_ms=reference['iteration_time_ms'],
                          before_after_speedup=old['iteration_time_ms'] / new['iteration_time_ms'],
                          cpu_after_speedup=reference['iteration_time_ms'] / new['iteration_time_ms'],
                          rr_cpu_stress_ratio=new['stress_value'] / reference['stress_value']))
        (out / 'pairs.json').write_text(json.dumps(pairs, indent=2) + '\n')

# Odd/partial tiles, W<T, shared/global, disabled shuffling, and >256 rounds.
regressions = []
cases = [('USpowerGrid', 127, 64, 'shared', True), ('EVA', 32, 32, 'global', True),
         ('1138_bus', 4, 32, 'shared', True), ('1138_bus', 127, 64, 'global', False)]
for graph, tile, workgroup, memory, shuffled in cases:
    for seed in [0, 3]:
        records = [run(f'regression-{graph}-t{tile}-{memory}-s{seed}-{label}', binary,
                       graph, seed, (tile, workgroup, memory, shuffled))
                   for label, binary in [('before', args.before), ('after', after)]]
        compare(*records)
        regressions.append(dict(graph=graph, tile=tile, workgroup=workgroup,
                                memory=memory, shuffled=shuffled, seed=seed,
                                exact_coordinates_and_stress=True))
        (out / 'regressions.json').write_text(json.dumps(regressions, indent=2) + '\n')

repeat_pairs = []
for graph in ['fe_4elt2', 'delaunay_n14']:
    for repeat in range(3):
        variants = [('before', args.before), ('after', after)]
        if repeat % 2:
            variants.reverse()
        records = {label: run(f'repeat{repeat}-{graph}-s0-{label}', binary, graph, 0,
                               (1024, 1024, 'shared', True))
                   for label, binary in variants}
        old, new = (records[k] for k in ['before', 'after'])
        compare(old, new)
        repeat_pairs.append(dict(graph=graph, seed=0, repeat=repeat,
                                 exact_coordinates_and_stress=True,
                                 before_ms=old['iteration_time_ms'],
                                 after_ms=new['iteration_time_ms'],
                                 speedup=old['iteration_time_ms'] / new['iteration_time_ms']))
        (out / 'repeat-pairs.json').write_text(json.dumps(repeat_pairs, indent=2) + '\n')

summary = []
for graph in ['fe_4elt2', 'delaunay_n14']:
    group = [r for r in pairs if r['graph'] == graph]
    summary.append(dict(graph=graph, seeds=len(group), **{
        key: statistics.median(r[key] for r in group)
        for key in ['before_ms', 'after_ms', 'cpu_ms', 'before_after_speedup',
                    'cpu_after_speedup', 'rr_cpu_stress_ratio']}))
    summary[-1]['repeat_seed0_speedup_median'] = statistics.median(
        r['speedup'] for r in repeat_pairs if r['graph'] == graph)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
