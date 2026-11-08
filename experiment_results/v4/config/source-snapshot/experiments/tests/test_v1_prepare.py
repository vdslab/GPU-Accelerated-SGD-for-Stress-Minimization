import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import prepare_v1 as preparation
from experiments import run_v1, summarize_v1


class VersionedPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        (self.root / 'data').mkdir()
        (self.root / 'data/fixture.mtx').write_text(
            '%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2\n2 3\n')

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, base):
        with patch.object(preparation, 'ROOT', self.root), \
             patch.object(preparation, 'DEFAULT_RESULTS_DIR', self.root / 'experiment_results/v1'), \
             patch.object(preparation, 'SMALL', ['fixture']), \
             patch.object(preparation, 'LARGE', []), contextlib.redirect_stdout(io.StringIO()):
            preparation.main(['--results-dir', str(base)])

    def test_new_version_manifest_and_dataset_paths_are_separate_from_v1(self):
        base = self.root / 'experiment_results/v2'
        self.prepare(base)
        manifest = json.loads((base / 'config/quality-timing.json').read_text())
        self.assertEqual(manifest['experiment_id'], 'v2-quality-timing')
        self.assertEqual(len(manifest['methods']), 6)
        self.assertEqual(manifest['datasets'][0]['path'], 'data/v2_lcc/fixture.mtx')
        self.assertTrue((self.root / manifest['datasets'][0]['path']).is_file())
        self.assertFalse((self.root / 'experiment_results/v1').exists())

    def test_v1_recipe_is_preserved_and_existing_raw_blocks_repreparation(self):
        base = self.root / 'experiment_results/v1'
        self.prepare(base)
        manifest_path = base / 'config/quality-timing.json'
        original = manifest_path.read_bytes()
        self.assertEqual(len(json.loads(original)['methods']), 5)
        (base / 'raw/existing-run').mkdir(parents=True)
        with self.assertRaisesRegex(SystemExit, 'Existing raw runs'):
            self.prepare(base)
        self.assertEqual(manifest_path.read_bytes(), original)

    def test_all_version_clis_keep_v1_as_the_default(self):
        expected = preparation.DEFAULT_RESULTS_DIR
        self.assertEqual(preparation.parse_args([]).results_dir, expected)
        self.assertEqual(run_v1.parse_args(['manifest.json']).results_dir, expected)
        self.assertEqual(summarize_v1.parse_args([]).results_dir, expected)
