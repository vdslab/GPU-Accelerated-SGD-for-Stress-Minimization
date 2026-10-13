from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from experiments import run_v2
from experiments.tests.test_runner import FAKE_METHOD

FAKE_V2_METHOD = FAKE_METHOD.replace(
    'args = parser.parse_args()',
    '''parser.add_argument("--tile-size", type=int, default=1024)
parser.add_argument("--workgroup-size", type=int, default=1024)
parser.add_argument("--coordinate-memory", default="shared")
parser.add_argument("--no-outer-shuffle", action="store_true")
parser.add_argument("--no-inner-shuffle", action="store_true")
args = parser.parse_args()''').replace(
    'print(json.dumps(record, separators=(",", ":")))',
    '''if method == "rr_sgd":
    record.update(rr_tile_size=args.tile_size, rr_workgroup_size=args.workgroup_size,
                  rr_memory_mode=args.coordinate_memory,
                  rr_outer_shuffle=not args.no_outer_shuffle,
                  rr_inner_shuffle=not args.no_inner_shuffle,
                  rr_shared_bytes=16*args.tile_size if args.coordinate_memory == "shared" else 0)
if method in {"atomic_sgd", "atomic_sparse_sgd"}:
    record.update(attempted_updates=record["constraints"]*args.iterations,
                  completed_updates=record["constraints"]*args.iterations, retry_failures=0)
print(json.dumps(record, separators=(",", ":")))''')


class V2RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.graph = self.root / "fixture.mtx"
        self.graph.write_text("%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 2\n")
        self.fake = self.root / "fake.py"
        self.fake.write_text(FAKE_V2_METHOD)
        self.fake.chmod(0o755)
        self.results = self.root / "results"

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, methods=None, seeds=None, variants=None, warmups=0):
        methods = methods or ["rr_sgd"]
        data = {"experiment_id": "v2-fixture", "study_group": "rr-memory-tiling",
                "run_mode": "benchmark", "methods": methods,
                "datasets": [{"name": "fixture", "path": str(self.graph),
                              "sha256": hashlib.sha256(self.graph.read_bytes()).hexdigest()}],
                "seeds": seeds or [0], "iterations": [15], "epsilon": [0.1],
                "pivots": [200], "repetitions": 1, "warmups": warmups,
                "timeout_seconds": 10, "fail_fast": False,
                "binaries": {method: str(self.fake) for method in methods}}
        if "rr_sgd" in methods:
            data["variants"] = {"rr_sgd": variants or [
                {"id": "shared", "tile_size": 32, "workgroup_size": 256, "coordinate_memory": "shared"},
                {"id": "global", "tile_size": 32, "workgroup_size": 256, "coordinate_memory": "global"}]}
        path = self.root / "manifest.json"
        path.write_text(json.dumps(data))
        return path

    def call(self, path, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_v2.main([str(path), "--results-dir", str(self.results), *extra])

    def raw(self):
        return self.results / "raw/v2-fixture"

    def test_variant_identity_includes_parameters_and_command_is_independent_of_workgroup(self):
        path = self.manifest()
        config = run_v2.load_manifest(path)
        items = run_v2.expand_v2_runs(config, path)
        self.assertEqual(len(items), 2)
        self.assertEqual(len({item.run_id for item in items}), 2)
        changed = json.loads(path.read_text())
        changed["variants"]["rr_sgd"][0]["workgroup_size"] = 128
        self.assertNotEqual(items[0].run_id, run_v2.expand_v2_runs(changed, path)[0].run_id)
        command = run_v2.command_for(items[0], self.root)
        self.assertEqual(command[command.index("--tile-size") + 1], "32")
        self.assertEqual(command[command.index("--workgroup-size") + 1], "256")

    def test_warmups_raw_stdout_and_coordinates_are_retained_without_counting_as_formal(self):
        path = self.manifest(warmups=2)
        self.assertEqual(self.call(path), 0)
        records = run_v2.common.load_existing_records(self.raw() / "results.jsonl")
        warmups = run_v2.common.load_existing_records(self.raw() / "warmup-results.jsonl")
        self.assertEqual(len(records), 2)
        self.assertEqual(len(warmups), 4)
        for record in records:
            self.assertTrue(Path(record["stdout_log_path"]).is_file())
            self.assertTrue(Path(record["stderr_log_path"]).is_file())
            self.assertTrue(record["integrity"]["unchanged"])
            stdout = json.loads(Path(record["stdout_log_path"]).read_text())
            self.assertNotIn("variant", stdout)
            self.assertEqual(record["rr_shared_bytes"], 512 if record["variant_id"] == "shared" else 0)
            self.assertIn("final_positions_path", record["artifacts_sha256"])
        checkpoint = json.loads((self.raw() / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["success"], 2)
        self.assertEqual(checkpoint["pending"], [])
        self.assertTrue((self.raw() / "source-snapshot/experiments/run_v2.py").is_file())

    def test_limit_resume_skips_success_and_rejects_changed_binary(self):
        path = self.manifest(seeds=[0, 1])
        self.assertEqual(self.call(path, "--limit", "1"), 0)
        self.assertEqual(len(run_v2.common.load_existing_records(self.raw() / "results.jsonl")), 1)
        self.assertEqual(self.call(path, "--resume"), 0)
        records = run_v2.common.load_existing_records(self.raw() / "results.jsonl")
        self.assertEqual(len(records), 4)
        self.assertEqual(len({row["run_id"] for row in records}), 4)
        self.fake.write_text(FAKE_V2_METHOD + "\n# changed\n")
        self.assertEqual(self.call(path, "--resume"), 2)
        self.assertEqual(len(run_v2.common.load_existing_records(self.raw() / "results.jsonl")), 4)

    def test_resume_rejects_artifact_tampering_and_manifest_drift(self):
        path = self.manifest(methods=["sgd"])
        self.assertEqual(self.call(path), 0)
        record = run_v2.common.load_existing_records(self.raw() / "results.jsonl")[0]
        Path(record["final_positions_path"]).write_text("changed")
        self.assertEqual(self.call(path, "--resume"), 2)
        modified = json.loads(path.read_text())
        modified["seeds"] = [1]
        path.write_text(json.dumps(modified))
        self.assertEqual(self.call(path, "--resume"), 2)

    def test_dry_run_does_not_write_or_require_built_binary(self):
        path = self.manifest()
        self.fake.unlink()
        self.assertEqual(self.call(path, "--dry-run"), 0)
        self.assertFalse(self.results.exists())

    def test_invalid_variant_and_actual_metadata_drift_are_rejected(self):
        path = self.manifest(variants=[{"id": "bad", "workgroup_size": 0}])
        self.assertEqual(self.call(path, "--dry-run"), 2)
        path = self.manifest()
        self.fake.write_text(FAKE_V2_METHOD.replace('rr_workgroup_size=args.workgroup_size', 'rr_workgroup_size=999'))
        self.assertEqual(self.call(path), 1)
        records = run_v2.common.load_existing_records(self.raw() / "results.jsonl")
        self.assertTrue(all(row["status"] == "failure" for row in records))
        self.assertTrue(all(row["error_stage"] == "validate" for row in records))
        self.assertTrue(all(Path(row["stdout_log_path"]).is_file() for row in records))

    def test_atomic_incomplete_updates_are_failure_with_raw_metadata_retained(self):
        path = self.manifest(methods=["atomic_sgd", "atomic_sparse_sgd"])
        self.fake.write_text(FAKE_V2_METHOD.replace(
            'completed_updates=record["constraints"]*args.iterations',
            'completed_updates=record["constraints"]*args.iterations-1'))
        self.assertEqual(self.call(path), 1)
        records = run_v2.common.load_existing_records(self.raw() / "results.jsonl")
        self.assertEqual(len(records), 2)
        for row in records:
            self.assertEqual(row["status"], "failure")
            self.assertEqual(row["error_stage"], "validate")
            self.assertIn("Atomic update coverage incomplete", row["error_message"])
            raw = json.loads(Path(row["stdout_log_path"]).read_text())
            self.assertEqual(raw["status"], "success")
            self.assertLess(raw["completed_updates"], raw["attempted_updates"])

    def test_failed_run_resume_keeps_each_attempt(self):
        path = self.manifest(methods=["sgd"], seeds=[0, 1])
        old = os.environ.get("FAKE_FAIL_SEED")
        os.environ["FAKE_FAIL_SEED"] = "1"
        try:
            self.assertEqual(self.call(path), 1)
            os.environ.pop("FAKE_FAIL_SEED", None)
            self.assertEqual(self.call(path, "--resume"), 0)
        finally:
            if old is None:
                os.environ.pop("FAKE_FAIL_SEED", None)
            else:
                os.environ["FAKE_FAIL_SEED"] = old
        records = run_v2.common.load_existing_records(self.raw() / "results.jsonl")
        failed = [row for row in records if row["seed"] == 1]
        self.assertEqual([row["status"] for row in failed], ["failure", "success"])
        self.assertEqual([row["attempt"] for row in failed], [1, 2])
        self.assertNotEqual(failed[0]["stdout_log_path"], failed[1]["stdout_log_path"])


if __name__ == "__main__":
    unittest.main()
