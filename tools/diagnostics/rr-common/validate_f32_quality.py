#!/usr/bin/env python3
"""Compare fixed RR schedules with identical inputs and evaluation sources.

Times are single, ordered observations, not warmed/repeated benchmarks.
A failed quality gate is reported in summary.json; it is never hidden by averaging.
"""
import argparse
import csv
import json
import pathlib
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--seeds', default='0,1,3,7')
    parser.add_argument('--graphs', default=(
        'EVA,USpowerGrid,luxembourg_osm,preferentialAttachment,smallworld,'
        'coAuthorsCiteseer,web-NotreDame,web-Stanford'))
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--variants', default='1@0,200@5')
    parser.add_argument('--pivots', type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = args.variants.split(',')
    if len(variants) < 2 or args.samples < 1 or args.pivots < 1:
        parser.error('need baseline and candidate variants, positive samples and pivots')
    rows = []
    for graph in args.graphs.split(','):
        for seed in map(int, args.seeds.split(',')):
            tag = f'validate-{graph}-h{args.pivots}-s{seed}'
            output = args.output / (tag + '.jsonl')
            command = [
                'implementations/sparse/sparse-sgd-gpu/target/release/examples/quality_search',
                f'data/{graph}.mtx', str(seed), args.variants,
                str(args.samples), str(args.pivots)]
            existing = []
            if output.exists():
                try:
                    existing = [json.loads(line) for line in output.read_text().splitlines()]
                except ValueError:
                    pass
            reusable = len(existing) == len(variants) and all(
                entry.get('variant') == variant
                and entry.get('graph') == f'data/{graph}.mtx'
                and entry.get('seed') == seed
                and entry.get('samples') == args.samples
                and entry.get('pivots') == args.pivots
                for entry, variant in zip(existing, variants))
            if not reusable:
                print('START', tag, flush=True)
                with output.open('w') as stdout, (args.output / (tag + '.stderr')).open('w') as stderr:
                    subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
            data = [json.loads(line) for line in output.read_text().splitlines()]
            baseline = data[0]
            for result in data[1:]:
                for key in ['initial_sha256', 'preprocess_sha256', 'samples', 'seed', 'pivots']:
                    if result[key] != baseline[key]:
                        raise ValueError(f'{tag}: input mismatch for {key}')
                row = dict(
                    graph=graph, seed=seed, nodes=result['nodes'], pivots=result['pivots'],
                    variant=result['variant'], before_stress=baseline['stress'],
                    after_stress=result['stress'], stress_ratio=result['stress'] / baseline['stress'],
                    before_seconds=baseline['iteration_seconds'],
                    after_seconds=result['iteration_seconds'],
                    time_ratio=result['iteration_seconds'] / baseline['iteration_seconds'])
                rows.append(row)
                print(tag, result['variant'], 'stress ratio', round(row['stress_ratio'], 6),
                      'time ratio', round(row['time_ratio'], 3), flush=True)
            with (args.output / 'comparison.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = dict(
        runs=len(rows), max_stress_ratio=max(row['stress_ratio'] for row in rows),
        median_stress_ratio=statistics.median(row['stress_ratio'] for row in rows),
        regressions_over_2_percent=[row for row in rows if row['stress_ratio'] > 1.02])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
