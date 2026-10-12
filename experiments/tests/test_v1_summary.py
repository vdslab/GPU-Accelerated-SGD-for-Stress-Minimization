import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments import summarize_v1 as summary


class SourceExtensionTests(unittest.TestCase):
    def setUp(self):
        summary.source_extension_evidence.cache_clear()

    def test_unchanged_shared_trees_allow_an_additive_atomic_method(self):
        with patch.object(summary.subprocess, 'check_output', return_value='tree-id\n'):
            evidence = summary.source_extension_evidence('baseline', 'addition')
        self.assertEqual(evidence['baseline_commit'], 'baseline')
        self.assertEqual(set(evidence['shared_source_trees']), set(summary.SHARED_SOURCE_PATHS))

    def test_any_changed_shared_implementation_prevents_baseline_reuse(self):
        def git_tree(command, **kwargs):
            return 'changed\n' if command[-1] == 'addition:experiment-common' else 'same\n'
        with patch.object(summary.subprocess, 'check_output', side_effect=git_tree):
            self.assertIsNone(summary.source_extension_evidence('baseline', 'addition'))

    def test_git_history_follows_directory_moves_but_rejects_changed_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*args):
                return subprocess.check_output(['git', '-C', str(root), *args],
                                               text=True, stderr=subprocess.DEVNULL).strip()

            def commit(message):
                git('add', '.')
                git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-m', message)
                return git('rev-parse', 'HEAD')

            git('init')
            for path in summary.SHARED_SOURCE_PATHS:
                (root / path).mkdir()
                (root / path / 'lib.rs').write_text('unchanged source\n')
            baseline = commit('baseline')
            for old in summary.SHARED_SOURCE_PATHS:
                new = root / summary.SOURCE_PATH_ALIASES[old]
                new.parent.mkdir(parents=True, exist_ok=True)
                (root / old).rename(new)
            atomic = root / summary.SOURCE_PATH_ALIASES['vram-lock-sparse']
            atomic.mkdir()
            (atomic / 'lib.rs').write_text('additive method\n')
            addition = commit('move and add atomic sparse')
            with patch.object(summary, 'ROOT', root):
                evidence = summary.source_extension_evidence(baseline, addition)
            self.assertIsNotNone(evidence)
            for old, values in evidence['shared_source_trees'].items():
                self.assertEqual(values['baseline_path'], old)
                self.assertEqual(values['added_path'], summary.SOURCE_PATH_ALIASES[old])
                self.assertEqual(values['baseline_tree'], values['added_tree'])
            (root / summary.SOURCE_PATH_ALIASES['experiment-common'] / 'lib.rs').write_text('changed source\n')
            changed = commit('change shared source')
            with patch.object(summary, 'ROOT', root):
                self.assertIsNone(summary.source_extension_evidence(baseline, changed))


class VersionedLoadTests(unittest.TestCase):
    def test_preserved_v1_input_paths_follow_data_move_and_external_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = root / 'experiment_results/v1'
            folder = base / 'raw/v1-atomic-sparse-quality-timing'
            folder.mkdir(parents=True)
            (folder / 'manifest.json').write_text(json.dumps({
                'datasets': [{'name': 'fixture'}], 'methods': ['atomic_sparse_sgd'], 'seeds': [0],
                'iterations': [15], 'epsilon': [0.1], 'pivots': [200], 'repetitions': 1,
            }))
            original_checkout = Path('/previous/checkout')
            stored = folder.relative_to(root)
            (folder / 'artifact-relocation.json').write_text(json.dumps({
                'original_root': str(original_checkout / stored), 'stored_root': str(stored),
            }))
            data = root / 'data/v1_lcc/fixture.mtx'
            mapping = root / 'data/v1_lcc/fixture.vertex-map.txt'
            data.parent.mkdir(parents=True)
            data.write_text('input')
            mapping.write_text('vertex map')
            old_data = original_checkout / 'prod_data/v1_lcc/fixture.mtx'
            record = json.dumps({'run_id': 'fixture', 'input_path': str(old_data)}) + '\n'
            (folder / 'results.jsonl').write_text(record)
            config = base / 'config'
            config.mkdir()
            catalog = json.dumps([{
                'name': 'fixture', 'source_path': str(root / 'prod_data/v1_lcc/fixture.mtx'),
                'path': 'prod_data/v1_lcc/fixture.mtx',
                'vertex_map_path': 'prod_data/v1_lcc/fixture.vertex-map.txt',
            }])
            (config / 'datasets.json').write_text(catalog)
            (root / 'archive').mkdir()
            (root / 'archive/relocations.json').write_text(json.dumps({
                'schema_version': 1, 'repository_root': str(root),
                'moves': [{'from': 'prod_data', 'to': 'data'}],
            }))
            with patch.object(summary, 'ROOT', root):
                records, _, _ = summary.load_data(base)
                loaded_catalog = summary.load_dataset_catalog(base)
            self.assertEqual(records[0]['input_path'], str(data))
            self.assertEqual(loaded_catalog[0]['source_path'], str(data))
            self.assertEqual(loaded_catalog[0]['path'], str(data))
            self.assertEqual(loaded_catalog[0]['vertex_map_path'], str(mapping))
            self.assertEqual((folder / 'results.jsonl').read_text(), record)
            self.assertEqual((config / 'datasets.json').read_text(), catalog)

    def test_new_experiment_ids_load_and_coordinates_follow_relocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = root / 'experiment_results/v2'
            folder = base / 'raw/v2-quality-timing'
            folder.mkdir(parents=True)
            (folder / 'manifest.json').write_text(json.dumps({
                'datasets': [{'name': 'fixture'}], 'methods': ['sgd'], 'seeds': [0],
                'iterations': [15], 'epsilon': [0.1], 'pivots': [200], 'repetitions': 1,
            }))
            archived = root / 'archive/experiments/legacy-output/run'
            archived.mkdir(parents=True)
            final = archived / 'final.txt'
            final.write_text('coordinates\n')
            record_path = root / 'output/run/final.txt'
            original = json.dumps({'run_id': 'fixture', 'final_positions_path': str(record_path)}) + '\n'
            (folder / 'results.jsonl').write_text(original)
            (root / 'archive/relocations.json').write_text(json.dumps({
                'schema_version': 1, 'repository_root': str(root),
                'moves': [{'from': 'output', 'to': 'archive/experiments/legacy-output'}],
            }))
            with patch.object(summary, 'ROOT', root):
                records, sources, expectations = summary.load_data(base)
            self.assertEqual(records[0]['final_positions_path'], str(final))
            self.assertEqual(Path(records[0]['final_positions_path']).read_text(), 'coordinates\n')
            self.assertEqual(records[0]['experiment_id'], 'v2-quality-timing')
            self.assertEqual(expectations[('fixture', 'sgd')], 1)
            self.assertEqual(sources[0]['path'], 'experiment_results/v2/raw/v2-quality-timing/results.jsonl')
            self.assertEqual((folder / 'results.jsonl').read_text(), original)

    def test_unsupported_sweeps_and_overlapping_experiments_are_rejected(self):
        for override in ({'iterations': [15, 30]}, {'repetitions': 2}, {}):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                manifest = {'datasets': [{'name': 'fixture'}], 'methods': ['sgd'], 'seeds': [0],
                            'iterations': [15], 'epsilon': [0.1], 'pivots': [200],
                            'repetitions': 1, **override}
                folders = ['experiment-a', 'experiment-b'] if not override else ['experiment-a']
                for name in folders:
                    folder = base / 'raw' / name
                    folder.mkdir(parents=True)
                    (folder / 'manifest.json').write_text(json.dumps(manifest))
                    (folder / 'results.jsonl').write_text('')
                with self.assertRaisesRegex(SystemExit, 'aggregate_results.py'):
                    summary.load_data(base)
