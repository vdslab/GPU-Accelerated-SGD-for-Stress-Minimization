#!/usr/bin/env python3
"""Fail-closed correctness/sanitizer gate, full verification, warmup, >=10 CPU/CUDA CLI runs."""
import argparse, csv, hashlib, io, json, pathlib, platform, statistics, subprocess, time

def run(command):
    begin = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return result.stdout + result.stderr, time.perf_counter() - begin

def measure(binary, options, backend):
    output, elapsed = run([str(binary), '--backend', backend, '--mode', 'run', *options])
    lines = output.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('backend,n,h,'))
    row = next(csv.DictReader(io.StringIO('\n'.join(lines[index:]))))
    row['cli_s'] = elapsed
    return row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=pathlib.Path)
    parser.add_argument('--binary', type=pathlib.Path, default=pathlib.Path('baseline-sparse-sgd-non-gpu/target/release/pivot-preprocess'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('results/cuda-pivots'))
    args = parser.parse_args()
    if args.repeats < 10: parser.error('--repeats must be >=10')
    binary = args.binary.resolve()
    options = ['--input', str(args.input.resolve()), '--seed', str(args.seed), '--pivots', '200', '--iterations', '15', '--epsilon', '0.1']
    # Gate cannot be bypassed. A nonzero exit, missing GPU or missing sanitizer stops measurement.
    validation = []
    for tool in ['memcheck', 'initcheck']:
        output, _ = run(['compute-sanitizer', '--tool', tool, '--error-exitcode', '1', str(binary), '--mode', 'validate', '--backend', 'cuda'])
        validation.append(output)
    output, _ = run([str(binary), '--mode', 'verify', *options])
    validation.append(output)
    # One discarded whole-CLI warmup for EACH backend.
    for backend in ['cpu', 'cuda']: measure(binary, options, backend)
    rows = []
    fingerprint = None
    keys = ['n','h','seed','iterations','epsilon','constraint_count','pivot_hash','distance_hash','constraint_hash','initial_hash','eta_hash','final_hash']
    for trial in range(args.repeats):
        for backend in (['cpu','cuda'] if trial % 2 == 0 else ['cuda','cpu']):
            row = measure(binary, options, backend)
            current = tuple(row[k] for k in keys)
            if fingerprint is not None and current != fingerprint: raise RuntimeError('fairness mismatch: benchmark aborted')
            fingerprint = current
            row['trial'] = trial
            rows.append(row)
            print(f'{backend} trial={trial}: preprocess={row["preprocess_s"]} s', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'trials.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {'input': str(args.input.resolve()), 'seed': args.seed, 'repeats': args.repeats,
               'definition': 'ratio of median CPU / median CUDA preprocessing including input+CSR+setup+transfer. Each exact context warms up before pivot timing; true external CLI time includes warmup and output.',
               'dataset_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'platform': platform.platform(),
               'cpu': {}, 'cuda': {}}
    summary['hardware'] = {}
    for name, command in [('gpu', ['nvidia-smi']), ('cuda_toolkit', ['nvcc','--version']), ('rust', ['rustc','--version']), ('commit', ['git','rev-parse','HEAD']), ('cpu_model', ['lscpu'])]:
        try: summary['hardware'][name] = run(command)[0]
        except (OSError, subprocess.CalledProcessError) as error: summary['hardware'][name] = str(error)
    for backend in ['cpu','cuda']:
        summary[backend] = {}
        for key in rows[0]:
            if key.endswith('_s'):
                values = [float(r[key]) for r in rows if r['backend'] == backend]
                summary[backend][key] = {'median':statistics.median(values), 'mean':statistics.mean(values), 'stdev':statistics.stdev(values), 'min':min(values), 'max':max(values)}
    summary['preprocess_speedup'] = summary['cpu']['preprocess_s']['median'] / summary['cuda']['preprocess_s']['median']
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.output / 'verification.txt').write_text('\n'.join(validation))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__': main()
