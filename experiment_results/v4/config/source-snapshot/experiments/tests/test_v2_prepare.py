import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import prepare_v2, run_v2


class V2PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data/v1_lcc").mkdir(parents=True)
        graph = "%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2\n2 3\n"
        for path in (self.root / "data/fixture.mtx", self.root / "data/v1_lcc/fixture.mtx"):
            path.write_text(graph)
        mapping = self.root / "data/v1_lcc/fixture.vertex-map.txt"
        mapping.write_text("# new index -> original zero-based index\n0\n1\n2\n")
        digest = hashlib.sha256(graph.encode()).hexdigest()
        entry = {"name": "fixture", "source_path": "data/fixture.mtx", "source_sha256": digest,
                 "path": "data/v1_lcc/fixture.mtx", "sha256": digest,
                 "source_nodes": 3, "nodes": 3, "edges": 2, "source_components": 1,
                 "vertex_map_path": "data/v1_lcc/fixture.vertex-map.txt"}
        (self.root / "experiment_results/v1/config").mkdir(parents=True)
        self.catalog = self.root / "experiment_results/v1/config/datasets.json"
        self.catalog.write_text(json.dumps([entry]))
        self.results = self.root / "experiment_results/v2"

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        with patch.object(prepare_v2, "ROOT", self.root), \
             patch.object(prepare_v2, "SMALL", ["fixture"]), \
             patch.object(prepare_v2, "LARGE", ["fixture"]), \
             patch.object(prepare_v2, "ABLATION", ["fixture"]), contextlib.redirect_stdout(io.StringIO()):
            return prepare_v2.main(["--results-dir", str(self.results)])

    def test_versions_are_separate_and_all_four_studies_have_reproducible_parameters(self):
        prior = self.catalog.read_bytes()
        self.assertEqual(self.prepare(), 0)
        self.assertEqual(self.catalog.read_bytes(), prior)
        catalog = json.loads((self.results / "config/datasets.json").read_text())
        self.assertEqual(catalog[0]["path"], "data/v1_lcc/fixture.mtx")
        self.assertEqual(catalog[0]["full_pairs"], 3)
        self.assertEqual(len(catalog[0]["vertex_map_sha256"]), 64)
        expected = {"quality-timing": 150, "large-sparse": 30,
                    "rr-memory-tiling": 120, "rr-workgroup": 40}
        for group, count in expected.items():
            path = self.results / "config" / (group + ".json")
            manifest = json.loads(path.read_text())
            self.assertEqual(len(run_v2.expand_v2_runs(manifest, path)), count)
            self.assertEqual(manifest["iterations"], [15])
            self.assertEqual(manifest["epsilon"], [0.1])
        main = json.loads((self.results / "config/quality-timing.json").read_text())
        self.assertEqual(main["variants"]["rr_sgd"][0]["workgroup_size"], 256)

    def test_input_hash_drift_and_existing_raw_prevent_repreparation(self):
        source = self.root / "data/fixture.mtx"
        previous = source.read_text()
        source.write_text(previous + "% changed\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.prepare()
        source.write_text(previous)
        self.assertEqual(self.prepare(), 0)
        original = (self.results / "config/datasets.json").read_bytes()
        (self.results / "raw/existing").mkdir(parents=True)
        with self.assertRaisesRegex(SystemExit, "Existing raw"):
            self.prepare()
        self.assertEqual((self.results / "config/datasets.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
