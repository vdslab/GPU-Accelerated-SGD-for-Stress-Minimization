import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.artifact_paths import provenance_path, resolve_artifact_path
from experiments import experiment_plan


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        (self.root / 'archive').mkdir()
        self.original = Path('/previous/checkout')
        (self.root / 'archive/relocations.json').write_text(json.dumps({
            'schema_version': 1, 'repository_root': str(self.original),
            'moves': [{'from': 'output', 'to': 'archive/experiments/legacy-output'},
                      {'from': 'output/formal', 'to': 'experiment_results/v1/raw'},
                      {'from': 'prod_data', 'to': 'data'},
                      {'from': 'archive/datasets/data', 'to': 'data'}],
        }))

    def tearDown(self):
        self.temp.cleanup()

    def test_old_absolute_and_relative_paths_read_moved_artifacts(self):
        destination = self.root / 'archive/experiments/legacy-output/run/final.txt'
        destination.parent.mkdir(parents=True)
        destination.write_text('coordinates')
        for old in ('output/run/final.txt', self.original / 'output/run/final.txt',
                    self.root / 'output/run/final.txt'):
            self.assertEqual(resolve_artifact_path(old, self.root), destination)

    def test_existing_files_take_precedence_and_prefixes_are_exact(self):
        old = self.root / 'output/run/final.txt'
        old.parent.mkdir(parents=True)
        old.write_text('original')
        self.assertEqual(resolve_artifact_path(old, self.root), old)
        missing = self.root / 'output-other/run/final.txt'
        self.assertEqual(resolve_artifact_path(missing, self.root), missing)

    def test_specific_move_takes_precedence_over_parent_move(self):
        destination = self.root / 'experiment_results/v1/raw/run/final.txt'
        destination.parent.mkdir(parents=True)
        destination.write_text('coordinates')
        self.assertEqual(resolve_artifact_path('output/formal/run/final.txt', self.root), destination)

    def test_external_result_provenance_remains_absolute(self):
        self.assertEqual(provenance_path(self.root / 'archive/file', self.root), 'archive/file')
        external = Path('/external/run/results.jsonl')
        self.assertEqual(provenance_path(external, self.root), str(external))

    def test_archived_reads_prefer_saved_artifacts_when_old_path_is_reused(self):
        old = self.root / 'output/run/final.txt'
        saved = self.root / 'archive/experiments/legacy-output/run/final.txt'
        for path, contents in ((old, 'new run'), (saved, 'historical run')):
            path.parent.mkdir(parents=True)
            path.write_text(contents)
        self.assertEqual(resolve_artifact_path(old, self.root).read_text(), 'new run')
        self.assertEqual(resolve_artifact_path(old, self.root, prefer_relocated=True).read_text(),
                         'historical run')

    def test_unmoved_input_from_another_checkout_rebases_to_current_repository(self):
        data = self.root / 'data/fixture.mtx'
        data.parent.mkdir()
        data.write_text('input')
        self.assertEqual(resolve_artifact_path(self.original / 'data/fixture.mtx', self.root), data)

    def test_old_input_paths_resolve_to_data_for_records_and_manifests(self):
        data = self.root / 'data/v1_lcc/fixture.mtx'
        data.parent.mkdir(parents=True)
        data.write_text('input')
        old_paths = ('prod_data/v1_lcc/fixture.mtx',
                     self.original / 'prod_data/v1_lcc/fixture.mtx',
                     'archive/datasets/data/v1_lcc/fixture.mtx')
        manifest = self.root / 'experiment_results/v1/config/manifest.json'
        with patch.object(experiment_plan, 'REPO_ROOT', self.root):
            for old in old_paths:
                self.assertEqual(resolve_artifact_path(old, self.root), data)
                self.assertEqual(experiment_plan.resolve_path(str(old), manifest), data)
            self.assertEqual(experiment_plan.resolve_path('data/v1_lcc/fixture.mtx', manifest), data)

    def test_archived_manifest_reads_prefer_saved_paths(self):
        current = self.root / 'output/fixture.mtx'
        saved = self.root / 'archive/experiments/legacy-output/fixture.mtx'
        for path in (current, saved):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('input')
        with patch.object(experiment_plan, 'REPO_ROOT', self.root):
            manifest = self.root / 'archive/experiments/legacy-output/manifest.json'
            self.assertEqual(experiment_plan.resolve_path('output/fixture.mtx', manifest), saved)
