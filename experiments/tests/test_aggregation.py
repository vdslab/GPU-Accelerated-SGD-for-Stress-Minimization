from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments import aggregate_results
from experiments import aggregation
from experiments import experiment_plan


class AggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.graph = self.root / "fixture.mtx"
        self.graph.write_text(
            "%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 2\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_experiment(
        self,
        *,
        experiment_id: str = "fixture",
        dataset: str = "fixture",
        methods: list[str] | None = None,
        seeds: list[int] | None = None,
        repetitions: int = 1,
        dirty: bool = False,
        run_mode: str = "benchmark",
        platform_name: str = "test-platform",
    ) -> tuple[Path, list[dict[str, object]]]:
        selected_methods = methods or ["sgd", "atomic_sgd"]
        selected_seeds = [0, 1] if seeds is None else seeds
        directory = self.root / experiment_id
        directory.mkdir()
        manifest = {
            "experiment_id": experiment_id,
            "run_mode": run_mode,
            "methods": selected_methods,
            "datasets": [
                {
                    "name": dataset,
                    "path": str(self.graph),
                    "sha256": hashlib.sha256(self.graph.read_bytes()).hexdigest(),
                }
            ],
            "seeds": selected_seeds,
            "iterations": [15],
            "epsilon": [0.1],
            "pivots": [1],
            "repetitions": repetitions,
            "warmups": 0,
            "timeout_seconds": 10,
            "fail_fast": False,
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        (directory / "environment.json").write_text(
            json.dumps(
                {
                    "captured_at_utc": "ignored",
                    "git_commit": "a" * 40,
                    "git_dirty": dirty,
                    "platform": platform_name,
                    "machine": "test-machine",
                    "python": "3.11-test",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        runs = experiment_plan.expand_runs(manifest, manifest_path)
        records = [self.record_for(run, directory, dirty=dirty) for run in runs]
        self.write_records(directory, records)
        return directory, records

    def record_for(
        self,
        run: experiment_plan.RunDefinition,
        directory: Path,
        *,
        dirty: bool,
    ) -> dict[str, object]:
        iteration_by_method = {
            "sgd": 10.0,
            "atomic_sgd": 5.0,
            "rr_sgd": 4.0,
            "sparse_sgd": 8.0,
            "rr_sparse_sgd": 4.0,
        }
        stress_factor = {
            "sgd": 1.0,
            "atomic_sgd": 1.1,
            "rr_sgd": 0.9,
            "sparse_sgd": 1.0,
            "rr_sparse_sgd": 1.05,
        }
        gpu = run.method not in {"sgd", "sparse_sgd"}
        sparse = run.family == "sparse"
        iteration = iteration_by_method[run.method]
        method_setup = 1.0 if gpu else None
        runtime_init = 1.0 if gpu else None
        upload = 1.0 if gpu else None
        readback = 1.0 if gpu else None
        algorithm = (
            2.0
            + float(method_setup or 0)
            + float(runtime_init or 0)
            + float(upload or 0)
            + iteration
            + float(readback or 0)
            + 1.0
        )
        artifact_dir = directory / "artifacts" / run.run_id
        artifact_dir.mkdir(parents=True)
        final_path = artifact_dir / "final.txt"
        final_path.write_text("positions\n", encoding="utf-8")
        vertex_map = artifact_dir / "vertex-map.txt"
        if sparse:
            vertex_map.write_text("0 0\n", encoding="utf-8")
        baseline_stress = (200.0 if sparse else 100.0) + run.seed
        return {
            "schema_version": 1,
            "run_id": run.run_id,
            "run_mode": run.run_mode,
            "status": "success",
            "method": run.method,
            "family": run.family,
            "binary": str(run.binary),
            "git_commit": "a" * 40,
            "git_dirty": dirty,
            "dataset": run.dataset.name,
            "input_path": str(run.dataset.path),
            "input_sha256": run.dataset.sha256,
            "seed": run.seed,
            "initial_positions_sha256": f"initial-{run.seed}-{run.family}",
            "preprocess_sha256": f"preprocess-{run.seed}" if sparse else None,
            "nodes": 2,
            "edges": 1,
            "constraints": 1,
            "pivots": run.pivots,
            "iterations": run.iterations,
            "epsilon": run.epsilon,
            "cpu_model": "test-cpu",
            "gpu_name": "test-gpu" if gpu else None,
            "gpu_backend": "test-backend" if gpu else None,
            "input_time_ms": 1.0,
            "common_preprocess_time_ms": 2.0,
            "method_setup_time_ms": method_setup,
            "runtime_init_time_ms": runtime_init,
            "upload_time_ms": upload,
            "iteration_time_ms": iteration,
            "gpu_device_time_ms": iteration / 2 if gpu else None,
            "readback_time_ms": readback,
            "postprocess_time_ms": 1.0,
            "algorithm_time_cold_ms": algorithm,
            "algorithm_time_warm_ms": algorithm - float(runtime_init or 0),
            "cli_total_time_cold_ms": algorithm + 1.0,
            "stress_kind": "exact",
            "stress_value": baseline_stress * stress_factor[run.method],
            "stress_eval_time_ms": 0.5,
            "stress_samples": None,
            "stress_seed": None,
            "attempted_updates": 100 if run.method == "atomic_sgd" else None,
            "completed_updates": 100 if run.method == "atomic_sgd" else None,
            "retry_failures": 0 if run.method == "atomic_sgd" else None,
            "rounds": 3 if run.method in {"rr_sgd", "rr_sparse_sgd"} else None,
            "dispatches": 45
            if run.method in {"rr_sgd", "rr_sparse_sgd"}
            else None,
            "final_positions_path": str(final_path),
            "vertex_map_path": str(vertex_map) if sparse else None,
            "error_stage": None,
            "error_message": None,
            "exit_code": None,
            "stderr_log_path": None,
        }

    def write_records(
        self, directory: Path, records: list[dict[str, object]]
    ) -> None:
        (directory / "results.jsonl").write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def test_load_normalizes_repetition_and_failure_history(self) -> None:
        directory, records = self.make_experiment(repetitions=2)
        failure = dict(
            records[0],
            status="failure",
            error_stage="execute",
            error_message="first attempt",
            exit_code=1,
        )
        self.write_records(directory, [failure, *records])
        loaded = aggregation.load_experiment(directory)
        first = next(
            row for row in loaded.normalized_rows if row["run_id"] == records[0]["run_id"]
        )
        self.assertEqual(first["status"], "success")
        self.assertEqual(first["history_count"], 2)
        self.assertEqual(first["prior_failure_count"], 1)
        self.assertEqual({row["repetition"] for row in loaded.normalized_rows}, {0, 1})

    def test_latest_failure_is_not_hidden_by_older_success(self) -> None:
        directory, records = self.make_experiment(methods=["sgd"], seeds=[0])
        failure = dict(
            records[0],
            status="failure",
            error_stage="execute",
            error_message="latest failure",
            exit_code=1,
        )
        self.write_records(directory, [records[0], failure])
        loaded = aggregation.load_experiment(directory)
        self.assertEqual(loaded.normalized_rows[0]["status"], "failure")
        report = aggregation.build_validation_report([loaded], "validation")
        self.assertEqual(report["successful_runs"], 0)
        self.assertEqual(report["failed_runs"], 1)
        self.assertEqual(
            report["retry_history"][0]["statuses"], ["success", "failure"]
        )

    def test_malformed_unknown_and_mismatched_records_are_rejected(self) -> None:
        directory, records = self.make_experiment(methods=["sgd"], seeds=[0])
        (directory / "results.jsonl").write_text("{not-json}\n", encoding="utf-8")
        with self.assertRaises(aggregation.AggregationError):
            aggregation.load_experiment(directory)
        self.write_records(directory, [dict(records[0], run_id="unknown")])
        with self.assertRaises(aggregation.AggregationError):
            aggregation.load_experiment(directory)
        self.write_records(directory, [dict(records[0], seed=999)])
        with self.assertRaises(aggregation.AggregationError):
            aggregation.load_experiment(directory)

    def test_validation_profile_separates_groups_and_never_publishes(self) -> None:
        directory, _ = self.make_experiment(
            methods=[
                "sgd",
                "atomic_sgd",
                "rr_sgd",
                "sparse_sgd",
                "rr_sparse_sgd",
            ],
            seeds=[0, 1, 2],
            dirty=True,
        )
        loaded = aggregation.load_experiment(directory)
        report = aggregation.build_validation_report([loaded], "validation")
        self.assertFalse(report["publication_ready"])
        self.assertEqual(report["group_count"], 2)
        codes = {warning["code"] for warning in report["warnings"]}
        self.assertIn("dirty_commit", codes)
        self.assertIn("insufficient_timing_samples", codes)
        self.assertIn("insufficient_quality_seeds", codes)
        rows = loaded.normalized_rows
        original = len({aggregation.comparison_group(row) for row in rows})
        changed = dict(rows[0], run_mode="diagnostic")
        changed_environment = dict(
            rows[1], environment_fingerprint="different-environment"
        )
        changed_stress = dict(rows[2], stress_kind="sampled")
        self.assertEqual(
            len(
                {
                    aggregation.comparison_group(row)
                    for row in [*rows, changed, changed_environment, changed_stress]
                }
            ),
            original + 3,
        )

    def test_statistics_speedup_quality_and_method_stats(self) -> None:
        self.assertEqual(aggregation.percentile([1, 2, 3, 4], 0.25), 1.75)
        stats = aggregation.summary_stats([1, 2, 3, 4])
        self.assertEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["sample_sd"], math.sqrt(5 / 3))
        directory, _ = self.make_experiment(repetitions=2)
        rows = aggregation.load_experiment(directory).normalized_rows
        speed = aggregation.aggregate_speed(rows)
        atomic_iteration = next(
            row
            for row in speed
            if row["method"] == "atomic_sgd"
            and row["metric"] == "iteration_time_ms"
        )
        self.assertEqual(atomic_iteration["median"], 5.0)
        self.assertEqual(atomic_iteration["speedup_median"], 2.0)
        quality = aggregation.aggregate_quality(rows)
        atomic_quality = next(
            row for row in quality if row["method"] == "atomic_sgd"
        )
        self.assertAlmostEqual(atomic_quality["stress_ratio_mean"], 1.1)
        method_stats = aggregation.aggregate_method_stats(rows)
        self.assertTrue(
            any(
                row["method"] == "atomic_sgd"
                and row["stat"] == "retry_failures"
                for row in method_stats
            )
        )
        with self.assertRaises(aggregation.AggregationError):
            aggregation.summary_stats([1.0, float("nan")])

    def test_publication_profiles_enforce_and_accept_thresholds(self) -> None:
        short_dir, _ = self.make_experiment(
            experiment_id="short", methods=["sgd", "atomic_sgd"], seeds=[0]
        )
        short = aggregation.load_experiment(short_dir)
        timing_report = aggregation.build_validation_report([short], "timing")
        self.assertTrue(timing_report["errors"])
        no_baseline_dir, _ = self.make_experiment(
            experiment_id="no-baseline",
            methods=["atomic_sgd"],
            seeds=[0],
        )
        no_baseline = aggregation.build_validation_report(
            [aggregation.load_experiment(no_baseline_dir)], "timing"
        )
        self.assertIn(
            "missing_baseline",
            {entry["code"] for entry in no_baseline["errors"]},
        )
        timing_dir, _ = self.make_experiment(
            experiment_id="timing",
            methods=["sgd", "atomic_sgd"],
            seeds=[0],
            repetitions=10,
        )
        timing = aggregation.load_experiment(timing_dir)
        self.assertTrue(
            aggregation.build_validation_report([timing], "timing")[
                "publication_ready"
            ]
        )
        timing_output = self.root / "timing-tables"
        timing_result = aggregation.generate_artifacts(
            [timing], "timing", timing_output
        )
        self.assertTrue(timing_result["publication_ready"])
        self.assertEqual(
            {path.name for path in timing_output.iterdir()},
            set(aggregation.ARTIFACT_NAMES),
        )
        quality_dir, _ = self.make_experiment(
            experiment_id="quality",
            methods=["sgd", "atomic_sgd"],
            seeds=list(range(25)),
        )
        quality = aggregation.load_experiment(quality_dir)
        self.assertTrue(
            aggregation.build_validation_report([quality], "quality")[
                "publication_ready"
            ]
        )
        quality_output = self.root / "quality-tables"
        quality_result = aggregation.generate_artifacts(
            [quality], "quality", quality_output
        )
        self.assertTrue(quality_result["publication_ready"])
        self.assertEqual(
            {path.name for path in quality_output.iterdir()},
            set(aggregation.ARTIFACT_NAMES),
        )

    def test_generation_is_deterministic_safe_and_matches_golden(self) -> None:
        directory, _ = self.make_experiment(
            dataset="power_grid%", methods=["sgd", "atomic_sgd"]
        )
        loaded = aggregation.load_experiment(directory)
        output_one = self.root / "tables-one"
        output_two = self.root / "tables-two"
        aggregation.generate_artifacts([loaded], "validation", output_one)
        aggregation.generate_artifacts([loaded], "validation", output_two)
        self.assertEqual(
            {path.name: path.read_bytes() for path in output_one.iterdir()},
            {path.name: path.read_bytes() for path in output_two.iterdir()},
        )
        golden = Path(__file__).parent / "golden"
        self.assertEqual(
            (output_one / "speed-table.md").read_text(encoding="utf-8"),
            (golden / "speed-table.md").read_text(encoding="utf-8"),
        )
        self.assertEqual(
            (output_one / "quality-table.md").read_text(encoding="utf-8"),
            (golden / "quality-table.md").read_text(encoding="utf-8"),
        )
        latex = (output_one / "speed-table.tex").read_text(encoding="utf-8")
        self.assertIn(r"power\_grid\%", latex)
        metadata = json.loads(
            (output_one / "aggregation-metadata.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("generated_at", metadata)
        self.assertEqual(
            set(metadata["artifacts"]),
            set(aggregation.ARTIFACT_NAMES) - {"aggregation-metadata.json"},
        )
        with self.assertRaises(aggregation.AggregationError):
            aggregation.generate_artifacts([loaded], "validation", output_one)

    def test_dry_run_cli_creates_no_output(self) -> None:
        directory, _ = self.make_experiment()
        output = self.root / "not-created"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = aggregate_results.main(
                [
                    "--experiment-dir",
                    str(directory),
                    "--profile",
                    "validation",
                    "--output-dir",
                    str(output),
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0)
        self.assertFalse(output.exists())
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["successful_runs"], 4)
        self.assertEqual(summary["would_write"], list(aggregation.ARTIFACT_NAMES))


if __name__ == "__main__":
    unittest.main()
