import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import summarize_v2 as summary


def record(method='rr_sgd', seed=0, memory='shared', tile=1024, workgroup=256, **extra):
    sparse = method in summary.METHODS[3:]
    params = ({'tile_size': tile, 'workgroup_size': workgroup, 'coordinate_memory': memory,
               'outer_shuffle': True, 'inner_shuffle': True} if method == 'rr_sgd' else {})
    value = dict(run_id=f'{method}-{seed}-{tile}-{workgroup}-{memory}', experiment_id='fixture', dataset='fixture',
                 study_group='quality-timing', status='success', run_mode='benchmark',
                 method=method, family='sparse' if sparse else 'full', seed=seed, repetition=0,
                 variant_id=f'T{tile}-W{workgroup}-{memory}' if params else 'default', variant=params,
                 iterations=15, epsilon=.1, pivots=200 if sparse else None,
                 input_sha256='input', initial_positions_sha256='initial', preprocess_sha256='preprocess',
                 nodes=4, edges=3, constraints=6, cpu_model='fixture-cpu', gpu_name='fixture-gpu',
                 gpu_backend='fixture-backend', git_commit='fixture-commit', git_dirty=False,
                 iteration_time_ms=2., stress_value=10., stress_kind='exact', stress_seed=0,
                 stress_samples=None, attempted_updates=90, completed_updates=90, retry_failures=0)
    value.update(extra)
    return value


class PairingTests(unittest.TestCase):
    def test_speedup_and_quality_pair_only_with_same_family_seed(self):
        rows = [record('sgd', iteration_time_ms=8., stress_value=20.),
                record('rr_sgd'), record('sparse_sgd', iteration_time_ms=15., stress_value=25.),
                record('rr_sparse_sgd', iteration_time_ms=3., stress_value=30.)]
        errors = []
        pairs = summary.make_cpu_pairs(rows, set(), errors)
        rr = next(r for r in pairs if r['method'] == 'rr_sgd')
        sparse = next(r for r in pairs if r['method'] == 'rr_sparse_sgd')
        self.assertEqual(rr['iteration_speedup'], 4.)
        self.assertEqual(rr['stress_ratio'], .5)
        self.assertEqual(sparse['iteration_speedup'], 5.)
        self.assertEqual(sparse['stress_ratio'], 1.2)
        self.assertEqual(errors, [])

    def test_mismatched_initial_coordinates_or_evaluator_never_get_a_ratio(self):
        for override in [{'initial_positions_sha256': 'different'}, {'stress_kind': 'sampled'},
                         {'stress_seed': 9}, {'stress_samples': 3}, {'git_commit': 'different'}]:
            with self.subTest(override=override):
                errors = []
                pairs = summary.make_cpu_pairs([record('sgd'), record(**override)], set(), errors)
                self.assertFalse(any(r['method'] == 'rr_sgd' for r in pairs))
                self.assertTrue(errors)

    def test_source_hash_is_compared_even_when_git_commit_matches(self):
        a = record('sgd', integrity={'before': {'source_tree_sha256': 'a'}})
        b = record(integrity={'before': {'source_tree_sha256': 'b'}})
        self.assertIn('source_tree_sha256', list(summary.compare_conditions(a, b)))

    def test_invalid_baseline_is_not_used(self):
        cpu = record('sgd')
        pairs = summary.make_cpu_pairs([cpu, record()], {(cpu['experiment_id'], cpu['run_id'])}, [])
        self.assertEqual(pairs, [])

    def test_repetitions_are_not_counted_as_independent_bootstrap_seeds(self):
        rows = [record(seed=0, repetition=0, iteration_time_ms=1.),
                record(seed=0, repetition=1, run_id='rep-1', iteration_time_ms=99.),
                record(seed=1, iteration_time_ms=3.)]
        self.assertEqual(summary.seed_values(rows, 'iteration_time_ms'), [50., 3.])
        output = summary.summarize(rows, {}, [], set())[0]
        self.assertEqual(output['seeds'], 2)
        self.assertEqual(output['iteration_median_ms'], 26.5)

    def test_bootstrap_interval_is_deterministic_and_single_seed_has_no_invented_width(self):
        self.assertEqual(summary.interval([1., 2., 5.]), summary.interval([1., 2., 5.]))
        self.assertEqual(summary.interval([7.]), (7., 7., 7.))
        with self.assertRaises(ValueError):
            summary.interval([float('nan')])


class ValidationTests(unittest.TestCase):
    def test_atomic_incomplete_updates_are_excluded(self):
        row = record('atomic_sgd', completed_updates=89, retry_failures=1)
        errors, _, invalid = summary.validate_rows([row])
        self.assertTrue(any('coverage' in item for item in errors))
        self.assertIn(('fixture', row['run_id']), invalid)

    def test_dirty_source_requires_frozen_verified_snapshot(self):
        identity = {'source_tree_sha256': 'source', 'input_sha256': 'input', 'binary_sha256': 'binary'}
        row = record(git_dirty=True, frozen_source_verified=True,
                     integrity={'before': identity, 'after': dict(identity), 'unchanged': True})
        errors, warnings, invalid = summary.validate_rows([row])
        self.assertEqual(errors, [])
        self.assertEqual(invalid, set())
        self.assertTrue(any('Uncommitted' in item for item in warnings))
        row['frozen_source_verified'] = False
        errors, _, invalid = summary.validate_rows([row])
        self.assertTrue(errors)
        self.assertTrue(invalid)

    def test_changed_saved_coordinates_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'coordinates.txt'
            path.write_text('original')
            row = record(final_positions_path=str(path), artifacts_sha256={'final_positions_path': summary.sha(path)})
            path.write_text('changed')
            errors, _, invalid = summary.validate_rows([row])
        self.assertTrue(any('checksum' in item for item in errors))
        self.assertTrue(invalid)


class LoadTests(unittest.TestCase):
    def test_latest_attempt_preserves_failure_and_counts_missing_planned_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'raw/v2-quality-timing'
            folder.mkdir(parents=True)
            manifest = dict(study_group='quality-timing', datasets=[{'name': 'fixture'}], methods=['sgd'],
                            seeds=[0, 1, 2], iterations=[15], epsilon=[.1], pivots=[200], repetitions=1)
            (folder/'manifest.json').write_text(json.dumps(manifest))
            older = record('sgd', dataset='fixture')
            newer = dict(older, status='failure', error_stage='timeout')
            (folder/'results.jsonl').write_text(json.dumps(older)+'\n'+json.dumps(newer)+'\n')
            rows, _, planned = summary.load_data(Path(temporary))
            result = summary.summarize(rows, planned, [], set())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'failure')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['expected'], 3)
        self.assertEqual(result[0]['failures'], 1)
        self.assertEqual(result[0]['missing'], 2)
        self.assertIsNone(result[0]['iteration_median_ms'])

    def test_planned_full_has_no_pivots_and_non_rr_uses_default_variant(self):
        manifest = dict(study_group='quality-timing', datasets=[{'name': 'fixture'}], methods=['sgd', 'sparse_sgd'],
                        seeds=[0], iterations=[15], epsilon=[.1], pivots=[200], repetitions=1)
        planned = summary.planned_conditions(manifest, Path('folder'))
        for method, pivots in [('sgd', None), ('sparse_sgd', 200)]:
            row = record(method, dataset='fixture')
            self.assertEqual(planned[summary.condition(row)], 1)
            self.assertEqual(row['pivots'], pivots)

    def test_frozen_snapshot_is_checked_against_file_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            target = folder/'source-snapshot/implementations/lib.rs'
            target.parent.mkdir(parents=True)
            target.write_text('execution source')
            files = {'implementations/lib.rs': summary.sha(target)}
            digest = summary.hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            (folder/'execution-integrity.json').write_text(json.dumps({'source_tree': {'files': files, 'sha256': digest}}))
            self.assertTrue(summary.load_frozen_evidence(folder)[1])
            target.write_text('changed source')
            self.assertFalse(summary.load_frozen_evidence(folder)[1])


class MemoryTests(unittest.TestCase):
    def test_geometry_matches_odd_even_and_single_block_rr_schedules(self):
        for nodes, tile, expected in [(1, 8, 1), (4, 2, 3), (5, 2, 6), (8, 2, 5), (9, 2, 8)]:
            with self.subTest(nodes=nodes, tile=tile):
                row = record(nodes=nodes, tile=tile, rounds=expected, dispatches=expected*15)
                table = summary.rr_geometry(summary.summarize([row], {}, [], set()))[0]
                self.assertEqual(table['derived_rounds_per_iteration'], expected)
                self.assertEqual(table['rounds'], expected)
                self.assertEqual(table['derived_dispatches'], expected*15)
                self.assertEqual(table['last_block_vertices'], (nodes-1)%tile+1)
                self.assertEqual(table['max_workgroups_per_round'], max(1, ((nodes+tile-1)//tile)//2))
                self.assertEqual(table['rr_shared_bytes'], 16*tile)
        rows = [record(nodes=5, tile=2, workgroup=w, memory='global') for w in [32, 256]]
        tables = summary.rr_geometry(summary.summarize(rows, {}, [], set()))
        self.assertTrue(all(r['rr_shared_bytes'] == 0 for r in tables))
        self.assertTrue(all(r['block_count'] == 3 and r['tail_fill_fraction'] == .5 for r in tables))

    def test_shared_global_pair_checks_same_schedule_and_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for name in ['shared', 'global']:
                path = Path(temporary)/f'{name}.txt'
                path.write_text('# Edges\n0 1\n# Positions\n0 0\n1 1\n')
                paths.append(str(path))
            rows = [record(study_group='rr-memory-tiling', final_positions_path=paths[0], iteration_time_ms=2.),
                    record(memory='global', study_group='rr-memory-tiling', final_positions_path=paths[1], iteration_time_ms=8.)]
            errors = []
            pairs = summary.memory_pairs(rows, set(), errors)
            self.assertEqual(pairs[0]['shared_speedup'], 4.)
            self.assertTrue(pairs[0]['coordinates_exactly_equal'])
            self.assertEqual(pairs[0]['coordinate_parity'], 'pass')
            self.assertEqual(errors, [])
            rows[1]['variant']['outer_shuffle'] = False
            self.assertEqual(summary.memory_pairs(rows, set(), errors), [])
            self.assertTrue(any('outer_shuffle' in item for item in errors))

    def test_changed_workgroup_can_preserve_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'coordinates.txt'
            path.write_text('# Edges\n0 1\n# Positions\n0 0\n1 1\n')
            rows = [record(workgroup=w, study_group='rr-workgroup', final_positions_path=str(path),
                           iteration_time_ms=1024/w) for w in [128, 256, 512, 1024]]
            errors = []
            pairs = summary.workgroup_pairs(rows, set(), errors)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(r['coordinate_parity'] == 'pass' for r in pairs))
        self.assertEqual(next(r for r in pairs if r['workgroup_size'] == 1024)['iteration_speedup'], 4.)
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
