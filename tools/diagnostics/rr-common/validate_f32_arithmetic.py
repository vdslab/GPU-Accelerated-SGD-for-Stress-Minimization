#!/usr/bin/env python3
"""Compare an explicitly selected f32 candidate with CPU and archived original RR."""
import argparse
import csv
import hashlib
import json
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--mode', choices=['standard', 'shuffle'], default='shuffle')
    parser.add_argument('--binary', type=pathlib.Path, default=pathlib.Path('implementations/sparse/sparse-sgd-gpu/target/release/examples/quality_search'))
    parser.add_argument('--cpu-reference', type=pathlib.Path)
    parser.add_argument('--baseline-reference', type=pathlib.Path)
    parser.add_argument('--variants', default='4@0')
    parser.add_argument('--graphs', default='EVA,USpowerGrid,luxembourg_osm,preferentialAttachment,smallworld,coAuthorsCiteseer,web-NotreDame,web-Stanford')
    parser.add_argument('--seeds', default='0,1,3,7')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    rows = []
    for graph in args.graphs.split(','):
        for seed in map(int, args.seeds.split(',')):
            name = f'{graph}-s{seed}'
            out = args.output / (name + '.jsonl')
            meta = args.output / (name + '.command.json')
            reference_cpu = None
            if args.cpu_reference:
                reference = args.cpu_reference / (name + '.jsonl')
                reference_cpu = json.loads(reference.read_text().splitlines()[-1])
                assert reference_cpu['method'] == 'cpu' and reference_cpu['iterations'] == 15
            command = [str(binary), f'data/{graph}.mtx', str(seed), args.variants,
                       '256', '200', args.mode, 'no-cpu' if reference_cpu else 'cpu']
            provenance = dict(command=command, binary_sha256=digest)
            if args.cpu_reference:
                provenance['cpu_reference'] = str(args.cpu_reference)
            data = []
            if out.exists() and meta.exists() and json.loads(meta.read_text()) == provenance:
                try:
                    data = [json.loads(line) for line in out.read_text().splitlines()]
                except ValueError:
                    pass
            if len(data) != len(args.variants.split(',')) + (0 if reference_cpu else 1):
                print('START', name, args.mode, flush=True)
                meta.write_text(json.dumps(provenance, indent=2) + '\n')
                with out.open('w') as stdout, (args.output / (name + '.stderr')).open('w') as stderr:
                    subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
                data = [json.loads(line) for line in out.read_text().splitlines()]
            if reference_cpu:
                data.append(reference_cpu)
            cpu = data[-1]
            assert cpu['method'] == 'cpu'
            baseline_path = pathlib.Path('experiment_results/v2/validation/rr-f32-quality-20261030/validation') / f'validate-{graph}-h200-s{seed}.jsonl'
            if args.baseline_reference:
                baseline_path = args.baseline_reference / (name + '.jsonl')
            baseline = json.loads(baseline_path.read_text().splitlines()[0])
            for result in data[:-1]:
                for key in ['initial_sha256', 'preprocess_sha256', 'samples', 'seed', 'pivots']:
                    assert result[key] == cpu[key] == baseline[key], (name, key)
                row = dict(graph=graph, seed=seed, mode=args.mode, variant=result['variant'],
                           cpu_stress=cpu['stress'], original_stress=baseline['stress'],
                           candidate_stress=result['stress'], ratio_cpu=result['stress']/cpu['stress'],
                           ratio_original=result['stress']/baseline['stress'],
                           cpu_seconds=cpu['iteration_seconds'], candidate_seconds=result['iteration_seconds'],
                           speedup=cpu['iteration_seconds']/result['iteration_seconds'])
                rows.append(row)
                print(name, row['variant'], 'CPU ratio', round(row['ratio_cpu'], 4),
                      'original ratio', round(row['ratio_original'], 4), flush=True)
            with (args.output / 'comparison.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = dict(cases=len(rows), max_ratio_cpu=max(r['ratio_cpu'] for r in rows),
                   max_ratio_original=max(r['ratio_original'] for r in rows),
                   cpu_regressions_over_5_percent=[r for r in rows if r['ratio_cpu'] > 1.05])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
