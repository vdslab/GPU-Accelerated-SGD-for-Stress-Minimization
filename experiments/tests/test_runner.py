from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "run_experiments.py"
SPEC = importlib.util.spec_from_file_location("run_experiments", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


FAKE_METHOD = r"""#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run-id")
parser.add_argument("--input")
parser.add_argument("--iterations", type=int)
parser.add_argument("--epsilon", type=float)
parser.add_argument("--seed", type=int)
parser.add_argument("--output-format")
parser.add_argument("--output-dir")
parser.add_argument("--run-mode")
parser.add_argument("--pivots", type=int)
args = parser.parse_args()
method = os.environ["EXPERIMENT_METHOD"]
if log := os.environ.get("FAKE_CALL_LOG"):
    with open(log, "a", encoding="utf-8") as target:
        target.write(f"{args.run_id}\n")
if os.environ.get("FAKE_FAIL_SEED") == str(args.seed):
    raise SystemExit("intentional failure")
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
final_path = output_dir / "final.txt"
final_path.write_text("positions\n", encoding="utf-8")
vertex_map = output_dir / "vertex-map.txt"
if args.pivots is not None:
    vertex_map.write_text("0 0\n", encoding="utf-8")
family = "sparse" if args.pivots is not None else "full"
gpu = method not in {"sgd", "sparse_sgd"}
record = {
    "schema_version": 1, "run_id": args.run_id, "run_mode": args.run_mode,
    "status": "success", "method": method, "family": family,
    "binary": __file__, "git_commit": "fake", "git_dirty": False,
    "dataset": Path(args.input).stem,
    "input_path": args.input,
    "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
    "seed": args.seed, "initial_positions_sha256": "initial",
    "preprocess_sha256": "preprocess" if family == "sparse" else None,
    "nodes": 2, "edges": 1, "constraints": 1, "pivots": args.pivots,
    "iterations": args.iterations, "epsilon": args.epsilon,
    "cpu_model": "fake-cpu", "gpu_name": "fake-gpu" if gpu else None,
    "gpu_backend": "fake" if gpu else None,
    "input_time_ms": 1.0, "common_preprocess_time_ms": 2.0,
    "method_setup_time_ms": 3.0 if gpu else None,
    "runtime_init_time_ms": 4.0 if gpu else None,
    "upload_time_ms": 5.0 if gpu else None, "iteration_time_ms": 6.0,
    "gpu_device_time_ms": 2.0 if gpu else None,
    "readback_time_ms": 7.0 if gpu else None, "postprocess_time_ms": 8.0,
    "algorithm_time_cold_ms": 35.0 if gpu else 16.0,
    "algorithm_time_warm_ms": 31.0 if gpu else 16.0,
    "cli_total_time_cold_ms": 36.0 if gpu else 17.0,
    "stress_kind": "exact", "stress_value": 1.0, "stress_eval_time_ms": 0.5,
    "stress_samples": None, "stress_seed": None,
    "attempted_updates": None, "completed_updates": None,
    "retry_failures": None, "rounds": None, "dispatches": None,
    "final_positions_path": str(final_path),
    "vertex_map_path": str(vertex_map) if family == "sparse" else None,
    "error_stage": None, "error_message": None, "exit_code": None,
    "stderr_log_path": None
}
print(json.dumps(record, separators=(",", ":")))
"""


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.graph = self.root / "fixture.mtx"
        self.graph.write_text(
            "%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 2\n",
            encoding="utf-8",
        )
        self.fake = self.root / "fake_method.py"
        self.fake.write_text(FAKE_METHOD, encoding="utf-8")
        self.fake.chmod(0o755)
        self.output_root = self.root / "output"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def manifest(
        self,
        *,
        methods: list[str] | None = None,
        seeds: list[int] | None = None,
        warmups: int = 0,
        repetitions: int = 1,
    ) -> Path:
        selected = methods or ["sgd", "rr_sparse_sgd"]
        value = {
            "experiment_id": "fixture",
            "run_mode": "benchmark",
            "methods": selected,
            "datasets": [
                {
                    "name": "fixture",
                    "path": str(self.graph),
                    "sha256": hashlib.sha256(self.graph.read_bytes()).hexdigest(),
                }
            ],
            "seeds": seeds or [0],
            "iterations": [2],
            "epsilon": [0.1],
            "pivots": [1],
            "repetitions": repetitions,
            "warmups": warmups,
            "timeout_seconds": 10,
            "fail_fast": False,
            "binaries": {method: str(self.fake) for method in selected},
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_expansion_is_deterministic_and_sparse_only_uses_pivots(self) -> None:
        path = self.manifest(methods=["sgd", "rr_sparse_sgd"], seeds=[0, 1])
        manifest = runner.load_manifest(path)
        first = runner.expand_runs(manifest, path)
        second = runner.expand_runs(manifest, path)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(run.pivots is None for run in first if run.method == "sgd"))
        self.assertTrue(
            all(run.pivots == 1 for run in first if run.method == "rr_sparse_sgd")
        )

    def test_e0_run_ids_and_commands_are_stable(self) -> None:
        path = MODULE_PATH.parent / "manifests/e0-uspowergrid.json"
        manifest = runner.load_manifest(path)
        runs = runner.expand_runs(manifest, path)
        self.assertEqual(
            [run.run_id for run in runs],
            [
                "e0-uspowergrid-USpowerGrid-sgd-7dd9767ead4f",
                "e0-uspowergrid-USpowerGrid-sgd-078f8c31de3f",
                "e0-uspowergrid-USpowerGrid-sgd-925ec82a7ca2",
                "e0-uspowergrid-USpowerGrid-atomic_sgd-c03a325b0fa2",
                "e0-uspowergrid-USpowerGrid-atomic_sgd-40a9a26f1ba2",
                "e0-uspowergrid-USpowerGrid-atomic_sgd-6188e124906e",
                "e0-uspowergrid-USpowerGrid-rr_sgd-f8ba8b0629d3",
                "e0-uspowergrid-USpowerGrid-rr_sgd-c60c475167ac",
                "e0-uspowergrid-USpowerGrid-rr_sgd-4442af2f0d2a",
                "e0-uspowergrid-USpowerGrid-sparse_sgd-a6b781876e70",
                "e0-uspowergrid-USpowerGrid-sparse_sgd-cf94ddbc6db6",
                "e0-uspowergrid-USpowerGrid-sparse_sgd-ceb4c0101559",
                "e0-uspowergrid-USpowerGrid-rr_sparse_sgd-c2ee7f883926",
                "e0-uspowergrid-USpowerGrid-rr_sparse_sgd-332ad7c707f5",
                "e0-uspowergrid-USpowerGrid-rr_sparse_sgd-080dfb88888a",
            ],
        )
        full_command = runner.command_for(runs[0], Path("/tmp/artifacts"))
        sparse_command = runner.command_for(runs[-1], Path("/tmp/artifacts"))
        self.assertNotIn("--pivots", full_command)
        self.assertEqual(sparse_command[-2:], ["--pivots", "200"])

    def test_repetition_and_paired_key_are_exposed(self) -> None:
        path = self.manifest(
            methods=["sgd", "rr_sgd", "sparse_sgd", "rr_sparse_sgd"],
            repetitions=2,
        )
        runs = runner.expand_runs(runner.load_manifest(path), path)
        self.assertEqual([run.repetition for run in runs[:2]], [0, 1])
        by_method = {
            (run.method, run.repetition): run
            for run in runs
        }
        self.assertEqual(
            by_method[("sgd", 1)].paired_key,
            by_method[("rr_sgd", 1)].paired_key,
        )
        self.assertEqual(
            by_method[("sparse_sgd", 0)].paired_key,
            by_method[("rr_sparse_sgd", 0)].paired_key,
        )
        self.assertNotEqual(
            by_method[("sgd", 0)].paired_key,
            by_method[("sparse_sgd", 0)].paired_key,
        )

    def test_dry_run_does_not_create_experiment_output(self) -> None:
        path = self.manifest()
        result = runner.main(
            [str(path), "--output-root", str(self.output_root), "--dry-run"]
        )
        self.assertEqual(result, 0)
        self.assertFalse((self.output_root / "fixture").exists())

    def test_warmup_is_not_written_to_results(self) -> None:
        call_log = self.root / "calls.txt"
        previous = os.environ.get("FAKE_CALL_LOG")
        os.environ["FAKE_CALL_LOG"] = str(call_log)
        try:
            path = self.manifest(methods=["sgd"], warmups=1)
            result = runner.main([str(path), "--output-root", str(self.output_root)])
        finally:
            if previous is None:
                os.environ.pop("FAKE_CALL_LOG", None)
            else:
                os.environ["FAKE_CALL_LOG"] = previous
        self.assertEqual(result, 0)
        records = [
            json.loads(line)
            for line in (
                self.output_root / "fixture/results.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "success")
        self.assertEqual(len(call_log.read_text(encoding="utf-8").splitlines()), 2)

    def test_failure_continues_and_resume_skips_existing_success(self) -> None:
        call_log = self.root / "calls.txt"
        old_log = os.environ.get("FAKE_CALL_LOG")
        old_failure = os.environ.get("FAKE_FAIL_SEED")
        os.environ["FAKE_CALL_LOG"] = str(call_log)
        os.environ["FAKE_FAIL_SEED"] = "1"
        path = self.manifest(methods=["sgd"], seeds=[0, 1])
        try:
            first = runner.main([str(path), "--output-root", str(self.output_root)])
            self.assertEqual(first, 1)
            os.environ.pop("FAKE_FAIL_SEED", None)
            second = runner.main(
                [
                    str(path),
                    "--output-root",
                    str(self.output_root),
                    "--resume",
                ]
            )
            self.assertEqual(second, 0)
        finally:
            if old_log is None:
                os.environ.pop("FAKE_CALL_LOG", None)
            else:
                os.environ["FAKE_CALL_LOG"] = old_log
            if old_failure is None:
                os.environ.pop("FAKE_FAIL_SEED", None)
            else:
                os.environ["FAKE_FAIL_SEED"] = old_failure
        records = [
            json.loads(line)
            for line in (
                self.output_root / "fixture/results.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [record["status"] for record in records],
            ["success", "failure", "success"],
        )
        calls = call_log.read_text(encoding="utf-8").splitlines()
        seed_zero_id = records[0]["run_id"]
        self.assertEqual(calls.count(seed_zero_id), 1)

    def test_incomplete_final_line_is_isolated(self) -> None:
        results = self.root / "results.jsonl"
        results.write_bytes(b'{"status":"success","run_id":"ok"}\n{"status":')
        records = runner.load_existing_records(results)
        self.assertEqual(records, [{"status": "success", "run_id": "ok"}])
        self.assertEqual(
            json.loads(results.read_text(encoding="utf-8")),
            {"status": "success", "run_id": "ok"},
        )
        self.assertEqual(len(list(self.root.glob("results.corrupt-*.jsonl"))), 1)

    def test_paired_preprocessing_rejects_hash_drift(self) -> None:
        base = {
            "status": "success",
            "family": "sparse",
            "dataset": "graph",
            "input_sha256": "input",
            "seed": 7,
            "iterations": 15,
            "epsilon": 0.1,
            "pivots": 200,
            "method": "sparse_sgd",
            "initial_positions_sha256": "initial",
            "preprocess_sha256": "preprocess",
        }
        matching = dict(base, method="rr_sparse_sgd")
        runner.validate_paired_preprocessing(matching, [base])
        with self.assertRaises(runner.RunnerError):
            runner.validate_paired_preprocessing(
                dict(matching, initial_positions_sha256="different"), [base]
            )
        with self.assertRaises(runner.RunnerError):
            runner.validate_paired_preprocessing(
                dict(matching, preprocess_sha256="different"), [base]
            )


if __name__ == "__main__":
    unittest.main()
