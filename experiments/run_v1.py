"""Run v1 sequentially with rotating method order and a warmup per dataset/method."""
import argparse
import json
from pathlib import Path

try:
    from experiment_plan import REPO_ROOT, expand_runs, load_manifest
    from run_experiments import (execute_run, load_existing_records, prepare_output,
                                verify_inputs_and_binaries)
except ModuleNotFoundError:
    from experiments.experiment_plan import REPO_ROOT, expand_runs, load_manifest
    from experiments.run_experiments import (execute_run, load_existing_records, prepare_output,
                                            verify_inputs_and_binaries)

DEFAULT_RESULTS_DIR = REPO_ROOT / 'experiment_results/v1'


def ordered_runs(runs, methods, datasets):
    def key(run):
        # Rotate the first method across seeds to reduce fixed-order temperature bias.
        return (datasets.index(run.dataset.name), run.seed, run.repetition,
                (methods.index(run.method) - run.seed) % len(methods))
    return sorted(runs, key=key)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR,
                        help='result version directory containing config/raw/tables/figures/report')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = args.manifest.resolve()
    manifest = load_manifest(path)
    runs = ordered_runs(expand_runs(manifest, path), manifest['methods'],
                        [dataset['name'] for dataset in manifest['datasets']])
    verify_inputs_and_binaries(runs)
    folder = prepare_output(manifest, path, args.results_dir.resolve() / 'raw', args.resume)
    results = folder / 'results.jsonl'
    done = {row['run_id'] for row in load_existing_records(results) if row['status'] == 'success'}
    warmed = set()
    failures = 0
    for index, run in enumerate(runs, 1):
        if run.run_id in done:
            continue
        key = (run.dataset.name, run.method)
        configuration = dict(manifest, warmups=manifest['warmups'] if key not in warmed else 0)
        print(f'[{index}/{len(runs)}] {run.dataset.name} {run.method} seed={run.seed}', flush=True)
        successful = execute_run(run, folder, configuration, results)
        warmed.add(key)
        failures += not successful
        print('success' if successful else 'FAILURE (see raw log)', flush=True)
    (folder / 'execution-policy.json').write_text(json.dumps({
        'method_order': 'dataset, seed, rotating method order',
        'warmups': 'one per dataset/method at the start of this invocation',
        'formal_runs': len(runs), 'failed_in_this_invocation': failures,
    }, indent=2) + '\n')
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
