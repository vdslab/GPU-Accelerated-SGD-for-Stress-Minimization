from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import aggregation, experiment_plan, run_experiments
from experiments import record_metadata, run_v2
from experiments.tests import test_aggregation, test_runner, test_single

METADATA = {
    "rr_tile_size": 1024, "rr_workgroup_size": 256, "rr_memory_mode": "shared",
    "rr_shared_bytes": 16384, "rr_outer_shuffle": True, "rr_inner_shuffle": True,
}


class RRMetadataCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_aggregation.AggregationTests("test_statistics_speedup_quality_and_method_stats")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.directory, records = self.fixture.make_experiment(methods=["rr_sgd"], seeds=[0])
        self.record = records[0]
        path = self.directory / "manifest.json"
        self.run = experiment_plan.expand_runs(experiment_plan.load_manifest(path), path)[0]

    def parse(self, record, method="rr_sgd"):
        completed = subprocess.CompletedProcess([], 0, json.dumps(record) + "\n", "")
        return run_experiments.parse_and_validate_success(dataclasses.replace(self.run, method=method), completed)

    def check_aggregation(self, record):
        aggregation.validate_record_shape(dict(record, _source_line=1))

    def reject_both(self, record, method="rr_sgd"):
        with self.assertRaises(run_experiments.RunnerError):
            self.parse(record, method)
        with self.assertRaises(aggregation.AggregationError):
            self.check_aggregation(record)

    def test_legacy_rr_is_accepted_without_new_required_fields(self):
        self.assertEqual(self.parse(self.record), self.record)
        self.check_aggregation(self.record)
        schema = json.loads(run_experiments.SCHEMA_PATH.read_text())
        self.assertFalse(schema["additionalProperties"])
        self.assertTrue(set(record_metadata.RR_METADATA_FIELDS).isdisjoint(schema["required"]))

    def test_new_shared_global_and_independent_workgroup_metadata_survive_both_consumers(self):
        for memory in ("shared", "global"):
            with self.subTest(memory=memory):
                metadata = dict(METADATA, rr_tile_size=1, rr_workgroup_size=1024,
                                rr_memory_mode=memory, rr_shared_bytes=16 if memory == "shared" else 0)
                record = dict(self.record, **metadata)
                self.assertEqual(self.parse(record), record)
                self.check_aggregation(record)

    def test_rr_metadata_on_another_method_is_rejected(self):
        self.reject_both(dict(self.record, method="sgd", **METADATA), "sgd")
        self.reject_both(dict(self.record, method="atomic_sgd", **METADATA), "atomic_sgd")

    def test_partial_metadata_is_rejected_for_any_present_field(self):
        for field in METADATA:
            with self.subTest(field=field):
                self.reject_both(dict(self.record, **{field: METADATA[field]}))

    def test_wrong_types_ranges_memory_modes_and_shared_sizes_are_rejected(self):
        malformed = [
            {"rr_tile_size": 0}, {"rr_tile_size": 1025}, {"rr_tile_size": True},
            {"rr_workgroup_size": 0}, {"rr_workgroup_size": 1025}, {"rr_workgroup_size": 256.0},
            {"rr_memory_mode": "unknown"}, {"rr_memory_mode": []},
            {"rr_outer_shuffle": 1}, {"rr_inner_shuffle": "true"},
            {"rr_shared_bytes": 16368}, {"rr_shared_bytes": 16384.0},
            {"rr_memory_mode": "global", "rr_shared_bytes": 16},
        ]
        for bad in malformed:
            with self.subTest(bad=bad):
                self.reject_both(dict(self.record, **dict(METADATA, **bad)))

    def test_unknown_metadata_is_still_rejected(self):
        self.reject_both(dict(self.record, **METADATA, rr_unknown_field=1))

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "optional jsonschema package unavailable")
    def test_json_schema_accepts_old_and_new_and_rejects_wrong_method_partial_or_bad_types(self):
        import jsonschema
        schema = json.loads(run_experiments.SCHEMA_PATH.read_text())
        validator = jsonschema.Draft202012Validator(schema)
        validator.validate(self.record)
        validator.validate(dict(self.record, **METADATA))
        for bad in (dict(self.record, rr_tile_size=1024),
                    dict(self.record, method="sgd", **METADATA),
                    dict(self.record, **dict(METADATA, rr_workgroup_size=True)),
                    dict(self.record, **dict(METADATA, rr_memory_mode="global"))):
            with self.subTest(bad=bad):
                self.assertTrue(list(validator.iter_errors(bad)))

    def test_generic_runner_retains_new_rr_metadata(self):
        fixture = test_runner.RunnerTests("test_warmup_is_not_written_to_results")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.fake.write_text(test_runner.FAKE_METHOD.replace(
            'print(json.dumps(record, separators=(",", ":")))',
            'record.update(' + repr(METADATA) + ')\nprint(json.dumps(record, separators=(",", ":")))'))
        manifest = fixture.manifest(methods=["rr_sgd"], seeds=[0], warmups=1)
        environment = {"git_commit": "fake", "git_dirty": True, "platform": "fake"}
        import contextlib
        import io
        with patch.object(run_experiments, "git_environment", return_value=environment), contextlib.redirect_stdout(io.StringIO()):
            code = run_experiments.main([str(manifest), "--output-root", str(fixture.output_root)])
        self.assertEqual(code, 0)
        result = run_experiments.load_existing_records(fixture.output_root / "fixture/results.jsonl")[0]
        self.assertEqual({key: result[key] for key in METADATA}, METADATA)

    def test_single_runner_retains_new_full_rr_metadata(self):
        fixture = test_single.SingleRunTests("test_single_run_preserves_config_timings_and_layout_and_renders_twice")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        binary = fixture.root / test_single.run_single.METHODS["rr_sgd"].binary
        binary.write_text(binary.read_text().replace(
            'print(json.dumps(record, separators=(",", ":")))',
            'record.update(' + repr(METADATA) + ')\nprint(json.dumps(record, separators=(",", ":")))'))
        code, _, error = fixture.execute("--method", "rr_sgd", "--iterations", "2")
        self.assertEqual(code, 0, error)
        result = fixture.results()[0]["record"]
        self.assertEqual({key: result[key] for key in METADATA}, METADATA)

    def test_single_runner_and_v2_retain_sparse_rr_metadata(self):
        fixture = test_single.SingleRunTests("test_single_run_preserves_config_timings_and_layout_and_renders_twice")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        metadata = {"rr_pivot_order": "per-vertex-feistel4-cyclewalk", "rr_phase_chunks": 1,
                    "rr_step_cap": 1.0, "rr_arithmetic": "f32"}
        binary = fixture.root / test_single.run_single.METHODS["rr_sparse_sgd"].binary
        binary.write_text(binary.read_text().replace(
            'print(json.dumps(record, separators=(",", ":")))',
            'record.update(' + repr(metadata) + ')\nprint(json.dumps(record, separators=(",", ":")))'))
        code, _, error = fixture.execute("--method", "rr_sparse_sgd", "--iterations", "2", "--pivots", "1")
        self.assertEqual(code, 0, error)
        result = fixture.results()[0]["record"]
        self.assertEqual({key: result[key] for key in metadata}, metadata)
        self.check_aggregation(dict(result, vertex_map_path=self.record["final_positions_path"]))
        completed = subprocess.CompletedProcess([], 0, json.dumps(result) + "\n", "")
        item = run_v2.VariantRun(dataclasses.replace(self.run, method="rr_sparse_sgd"), "default", {})
        with patch.object(run_v2.common, "parse_and_validate_success", return_value=result) as validate:
            self.assertEqual(run_v2.validate_success(item, completed), result)
            validate.assert_called_once_with(item.run, completed)

    def test_sparse_metadata_validation_rejects_partial_wrong_method_and_wrong_chunks(self):
        valid = {"method": "rr_sparse_sgd", "pivots": 200,
                 "rr_pivot_order": "per-vertex-feistel4-cyclewalk", "rr_phase_chunks": 4,
                 "rr_step_cap": 1.0, "rr_arithmetic": "f32"}
        record_metadata.validate_rr_metadata(valid)
        for pivots in (1, 2, 3, 4, 200):
            record_metadata.validate_rr_metadata(dict(valid, pivots=pivots, rr_phase_chunks=min(4, pivots)))
        bad_records = [dict(valid, **change) for change in (
            {"method": "rr_sgd"}, {"rr_phase_chunks": 3}, {"rr_phase_chunks": 4.0},
            {"rr_step_cap": True}, {"rr_step_cap": 0.9}, {"rr_arithmetic": "f64"},
            {"rr_pivot_order": "unknown"}, {"pivots": None}, {"pivots": True})]
        bad_records += [{k: v for k, v in valid.items() if k != field}
                        for field in record_metadata.RR_SPARSE_METADATA_FIELDS]
        for bad in bad_records:
            with self.subTest(record=bad), self.assertRaises(ValueError):
                record_metadata.validate_rr_metadata(bad)
        if importlib.util.find_spec("jsonschema"):
            import jsonschema
            validator = jsonschema.Draft202012Validator(json.loads(run_experiments.SCHEMA_PATH.read_text()))
            validator.validate(dict(self.record, **valid))
            for bad in bad_records:
                # JSON Schema treats integral floats as integers; Python consumers are stricter.
                if isinstance(bad.get("rr_phase_chunks"), float):
                    continue
                self.assertTrue(list(validator.iter_errors(dict(self.record, **bad))))

    def test_future_source_snapshot_covers_schema_and_shared_validator(self):
        files = {str(path.relative_to(run_v2.REPO_ROOT)) for path in run_v2.source_files()}
        self.assertIn("experiments/record_metadata.py", files)
        self.assertIn("experiments/schema/experiment-record-v1.json", files)


if __name__ == "__main__":
    unittest.main()
