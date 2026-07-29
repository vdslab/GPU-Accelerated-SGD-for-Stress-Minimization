"""Validation, aggregation, and deterministic table rendering for experiments."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from experiment_plan import METHODS, REPO_ROOT, RunDefinition, expand_runs, load_manifest
except ModuleNotFoundError:
    from experiments.experiment_plan import (
        METHODS,
        REPO_ROOT,
        RunDefinition,
        expand_runs,
        load_manifest,
    )

SCHEMA_PATH = REPO_ROOT / "experiments/schema/experiment-record-v1.json"
AGGREGATION_RULES_VERSION = 1
PROFILES = {"timing", "quality", "validation"}
BASELINES = {"full": "sgd", "sparse": "sparse_sgd"}
TIMING_METRICS = (
    "iteration_time_ms",
    "algorithm_time_cold_ms",
    "cli_total_time_cold_ms",
)
METHOD_STAT_FIELDS = (
    "attempted_updates",
    "completed_updates",
    "retry_failures",
    "rounds",
    "dispatches",
    "gpu_device_time_ms",
)
ARTIFACT_NAMES = (
    "runs.csv",
    "speed-summary.csv",
    "quality-summary.csv",
    "method-stats.csv",
    "speed-table.md",
    "speed-table.tex",
    "quality-table.md",
    "quality-table.tex",
    "validation-report.json",
    "aggregation-metadata.json",
)


class AggregationError(RuntimeError):
    pass


class PublicationGateError(AggregationError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("publication profileの集計前条件を満たしていません")


@dataclasses.dataclass(frozen=True)
class InputChecksums:
    manifest_sha256: str
    environment_sha256: str
    results_sha256: str


@dataclasses.dataclass
class LoadedExperiment:
    directory: Path
    manifest: dict[str, Any]
    environment: dict[str, Any]
    checksums: InputChecksums
    planned: list[RunDefinition]
    histories: dict[str, list[dict[str, Any]]]
    current: dict[str, dict[str, Any]]
    normalized_rows: list[dict[str, Any]]

    @property
    def experiment_id(self) -> str:
        return str(self.manifest["experiment_id"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reject_json_constant(value: str) -> None:
    raise ValueError(f"非有限JSON numberは使用できません: {value}")


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AggregationError(f"{label}がありません: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_json_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AggregationError(f"{label}を読めません: {path}: {error}") from error
    if not isinstance(value, dict):
        raise AggregationError(f"{label}のrootはJSON objectである必要があります")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AggregationError(f"results.jsonlがありません: {path}")
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise AggregationError(f"results.jsonlを読めません: {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise AggregationError(f"results.jsonl line {line_number} が空です")
        try:
            value = json.loads(line, parse_constant=reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise AggregationError(
                f"results.jsonl line {line_number} が不正です: {error}"
            ) from error
        if not isinstance(value, dict):
            raise AggregationError(
                f"results.jsonl line {line_number} はobjectではありません"
            )
        value["_source_line"] = line_number
        records.append(value)
    return records


def load_schema_contract() -> tuple[set[str], set[str]]:
    schema = read_json_object(SCHEMA_PATH, "experiment schema")
    return set(schema["required"]), set(schema["properties"])


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_finite_nonnegative(record: dict[str, Any], field: str) -> None:
    value = record[field]
    if value is None:
        return
    if not is_number(value) or not math.isfinite(float(value)) or value < 0:
        raise AggregationError(
            f"line {record['_source_line']} の{field}はnullまたは有限な非負値です"
        )


def assert_close(record: dict[str, Any], field: str, expected: float) -> None:
    actual = record[field]
    if not is_number(actual) or not math.isclose(
        float(actual), expected, rel_tol=1e-9, abs_tol=1e-6
    ):
        raise AggregationError(
            f"line {record['_source_line']} の{field}が内訳と一致しません: "
            f"actual={actual!r}, expected={expected}"
        )


def validate_record_shape(record: dict[str, Any]) -> None:
    required, properties = load_schema_contract()
    public_keys = set(record) - {"_source_line"}
    missing = sorted(required - public_keys)
    unknown = sorted(public_keys - properties)
    if missing:
        raise AggregationError(
            f"line {record['_source_line']} に必須fieldがありません: {', '.join(missing)}"
        )
    if unknown:
        raise AggregationError(
            f"line {record['_source_line']} に未知fieldがあります: {', '.join(unknown)}"
        )
    if record["schema_version"] != 1:
        raise AggregationError(
            f"line {record['_source_line']} のschema_versionは1だけに対応します"
        )
    if record["status"] not in {"success", "failure"}:
        raise AggregationError(f"line {record['_source_line']} のstatusが不正です")
    if record["method"] not in METHODS:
        raise AggregationError(
            f"line {record['_source_line']} のmethodは未実装または未知です"
        )
    expected_family = METHODS[record["method"]].family
    if record["family"] != expected_family:
        raise AggregationError(
            f"line {record['_source_line']} のmethodとfamilyが一致しません"
        )
    if not isinstance(record["run_id"], str) or not record["run_id"]:
        raise AggregationError(f"line {record['_source_line']} のrun_idが不正です")
    if not isinstance(record["git_dirty"], bool):
        raise AggregationError(f"line {record['_source_line']} のgit_dirtyが不正です")
    for field in (
        "run_id",
        "run_mode",
        "method",
        "family",
        "binary",
        "git_commit",
        "dataset",
        "input_path",
        "input_sha256",
        "cpu_model",
    ):
        if not isinstance(record[field], str) or not record[field]:
            raise AggregationError(
                f"line {record['_source_line']} の{field}は空でないstringです"
            )
    for field, minimum in (("seed", 0), ("iterations", 1)):
        value = record[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise AggregationError(
                f"line {record['_source_line']} の{field}が不正です"
            )
    if (
        not is_number(record["epsilon"])
        or not math.isfinite(float(record["epsilon"]))
        or record["epsilon"] <= 0
    ):
        raise AggregationError(
            f"line {record['_source_line']} のepsilonは正の有限値です"
        )
    for field in (
        "nodes",
        "edges",
        "constraints",
        "pivots",
        "stress_samples",
        "stress_seed",
        "attempted_updates",
        "completed_updates",
        "retry_failures",
        "rounds",
        "dispatches",
        "exit_code",
    ):
        value = record[field]
        if value is not None and (
            not isinstance(value, int)
            or isinstance(value, bool)
            or (field != "exit_code" and value < (1 if field in {"pivots", "stress_samples"} else 0))
        ):
            raise AggregationError(
                f"line {record['_source_line']} の{field}が不正です"
            )
    for field in (
        "initial_positions_sha256",
        "preprocess_sha256",
        "gpu_name",
        "gpu_backend",
        "stress_kind",
        "final_positions_path",
        "vertex_map_path",
        "error_stage",
        "error_message",
        "stderr_log_path",
    ):
        value = record[field]
        if value is not None and not isinstance(value, str):
            raise AggregationError(
                f"line {record['_source_line']} の{field}はstringまたはnullです"
            )
    for field in (
        "input_time_ms",
        "common_preprocess_time_ms",
        "method_setup_time_ms",
        "runtime_init_time_ms",
        "upload_time_ms",
        "iteration_time_ms",
        "gpu_device_time_ms",
        "readback_time_ms",
        "postprocess_time_ms",
        "algorithm_time_cold_ms",
        "algorithm_time_warm_ms",
        "cli_total_time_cold_ms",
        "stress_eval_time_ms",
    ):
        require_finite_nonnegative(record, field)
    if record["status"] == "success":
        validate_success_record(record)


def validate_success_record(record: dict[str, Any]) -> None:
    for field in (
        "input_time_ms",
        "common_preprocess_time_ms",
        "iteration_time_ms",
        "postprocess_time_ms",
        "algorithm_time_cold_ms",
        "algorithm_time_warm_ms",
        "cli_total_time_cold_ms",
        "stress_eval_time_ms",
    ):
        if record[field] is None:
            raise AggregationError(
                f"line {record['_source_line']} の成功recordで{field}はnullにできません"
            )
    algorithm = sum(
        float(record[field] or 0.0)
        for field in (
            "common_preprocess_time_ms",
            "method_setup_time_ms",
            "runtime_init_time_ms",
            "upload_time_ms",
            "iteration_time_ms",
            "readback_time_ms",
            "postprocess_time_ms",
        )
    )
    assert_close(record, "algorithm_time_cold_ms", algorithm)
    assert_close(
        record,
        "algorithm_time_warm_ms",
        algorithm - float(record["runtime_init_time_ms"] or 0.0),
    )
    assert_close(
        record, "cli_total_time_cold_ms", float(record["input_time_ms"]) + algorithm
    )
    stress = record["stress_value"]
    if not is_number(stress) or not math.isfinite(float(stress)) or stress <= 0:
        raise AggregationError(
            f"line {record['_source_line']} のstress_valueは正の有限値が必要です"
        )
    if record["stress_kind"] not in {"exact", "sampled"}:
        raise AggregationError(
            f"line {record['_source_line']} のstress_kindが不正です"
        )
    if record["stress_kind"] == "exact":
        if record["stress_samples"] is not None or record["stress_seed"] is not None:
            raise AggregationError(
                f"line {record['_source_line']} のexact stressにsample情報があります"
            )
    elif (
        not isinstance(record["stress_samples"], int)
        or isinstance(record["stress_samples"], bool)
        or record["stress_samples"] <= 0
        or not isinstance(record["stress_seed"], int)
        or isinstance(record["stress_seed"], bool)
        or record["stress_seed"] < 0
    ):
        raise AggregationError(
            f"line {record['_source_line']} のsampled stress情報が不正です"
        )
    final_path = resolve_record_path(record["final_positions_path"])
    if final_path is None or not final_path.is_file():
        raise AggregationError(
            f"line {record['_source_line']} のfinal_positions_pathが存在しません"
        )
    if record["family"] == "sparse":
        vertex_map = resolve_record_path(record["vertex_map_path"])
        if vertex_map is None or not vertex_map.is_file():
            raise AggregationError(
                f"line {record['_source_line']} のSparse vertex_map_pathが存在しません"
            )
        if not isinstance(record["preprocess_sha256"], str):
            raise AggregationError(
                f"line {record['_source_line']} のSparse preprocess_sha256が不正です"
            )
    elif record["preprocess_sha256"] is not None or record["pivots"] is not None:
        raise AggregationError(
            f"line {record['_source_line']} のFull recordにSparse値があります"
        )


def resolve_record_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def validate_record_against_plan(
    record: dict[str, Any], run: RunDefinition
) -> None:
    expected = {
        "run_id": run.run_id,
        "run_mode": run.run_mode,
        "method": run.method,
        "family": run.family,
        "dataset": run.dataset.name,
        "input_sha256": run.dataset.sha256,
        "seed": run.seed,
        "iterations": run.iterations,
        "pivots": run.pivots,
    }
    for field, expected_value in expected.items():
        if record[field] != expected_value:
            raise AggregationError(
                f"line {record['_source_line']} の{field}がmanifestと不一致です: "
                f"actual={record[field]!r}, expected={expected_value!r}"
            )
    if not is_number(record["epsilon"]) or not math.isclose(
        float(record["epsilon"]), run.epsilon, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise AggregationError(
            f"line {record['_source_line']} のepsilonがmanifestと不一致です"
        )


def environment_fingerprint(environment: dict[str, Any]) -> str:
    stable = {
        key: environment.get(key)
        for key in ("platform", "machine", "python")
    }
    return canonical_sha256(stable)


def normalize_row(
    experiment: LoadedExperiment,
    run: RunDefinition,
    record: dict[str, Any],
) -> dict[str, Any]:
    row = {
        key: value
        for key, value in record.items()
        if key != "_source_line"
    }
    row.update(
        {
            "experiment_id": run.experiment_id,
            "repetition": run.repetition,
            "environment_fingerprint": environment_fingerprint(
                experiment.environment
            ),
            "history_count": len(experiment.histories[run.run_id]),
            "prior_failure_count": sum(
                item["status"] == "failure"
                for item in experiment.histories[run.run_id][:-1]
            ),
            "_paired_key": run.paired_key,
        }
    )
    return row


def load_experiment(directory: Path) -> LoadedExperiment:
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    environment_path = directory / "environment.json"
    results_path = directory / "results.jsonl"
    if not directory.is_dir():
        raise AggregationError(f"experiment directoryがありません: {directory}")
    try:
        manifest = load_manifest(manifest_path)
    except Exception as error:
        raise AggregationError(f"manifestを検証できません: {error}") from error
    environment = read_json_object(environment_path, "environment.json")
    records = read_jsonl(results_path)
    planned = expand_runs(manifest, manifest_path)
    plan_by_id = {run.run_id: run for run in planned}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        validate_record_shape(record)
        run_id = record["run_id"]
        if run_id not in plan_by_id:
            raise AggregationError(
                f"line {record['_source_line']} のrun_idはmanifest計画にありません: {run_id}"
            )
        validate_record_against_plan(record, plan_by_id[run_id])
        histories[run_id].append(record)
    current = {
        run_id: history[-1]
        for run_id, history in histories.items()
    }
    experiment = LoadedExperiment(
        directory=directory,
        manifest=manifest,
        environment=environment,
        checksums=InputChecksums(
            manifest_sha256=file_sha256(manifest_path),
            environment_sha256=file_sha256(environment_path),
            results_sha256=file_sha256(results_path),
        ),
        planned=planned,
        histories=dict(histories),
        current=current,
        normalized_rows=[],
    )
    experiment.normalized_rows = [
        normalize_row(experiment, run, current[run.run_id])
        for run in planned
        if run.run_id in current
    ]
    return experiment


def comparison_group(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["experiment_id"],
        row["dataset"],
        row["input_sha256"],
        row["family"],
        row["run_mode"],
        row["iterations"],
        float(row["epsilon"]),
        row["pivots"],
        row["stress_kind"],
        row["environment_fingerprint"],
    )


def logical_group(row: dict[str, Any]) -> tuple[Any, ...]:
    """Group before commit/environment checks are applied."""
    return (
        row["experiment_id"],
        row["dataset"],
        row["family"],
        row["run_mode"],
        row["iterations"],
        float(row["epsilon"]),
        row["pivots"],
        row["stress_kind"],
    )


def group_fields(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(
        zip(
            (
                "experiment_id",
                "dataset",
                "input_sha256",
                "family",
                "run_mode",
                "iterations",
                "epsilon",
                "pivots",
                "stress_kind",
                "environment_fingerprint",
            ),
            key,
            strict=True,
        )
    )


def issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def build_validation_report(
    experiments: Sequence[LoadedExperiment], profile: str
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise AggregationError(f"未知のprofileです: {profile}")
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def add_publication_issue(
        code: str, message: str, **details: Any
    ) -> None:
        target = warnings if profile == "validation" else errors
        target.append(issue(code, message, **details))

    all_rows = [row for exp in experiments for row in exp.normalized_rows]
    for experiment in experiments:
        planned_ids = {run.run_id for run in experiment.planned}
        current_ids = set(experiment.current)
        missing = sorted(planned_ids - current_ids)
        if missing:
            add_publication_issue(
                "missing_runs",
                "manifestで計画されたrunが欠損しています",
                experiment_id=experiment.experiment_id,
                run_ids=missing,
            )
        failures = sorted(
            run_id
            for run_id, record in experiment.current.items()
            if record["status"] == "failure"
        )
        if failures:
            add_publication_issue(
                "failed_runs",
                "最新recordがfailureのrunがあります",
                experiment_id=experiment.experiment_id,
                run_ids=failures,
            )
    success_rows = [row for row in all_rows if row["status"] == "success"]
    dirty_ids = sorted(row["run_id"] for row in success_rows if row["git_dirty"])
    if dirty_ids:
        add_publication_issue(
            "dirty_commit",
            "dirty worktree由来のrunがあります",
            run_ids=dirty_ids,
        )
    by_logical: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in success_rows:
        by_logical[logical_group(row)].append(row)
    for key, rows in sorted(by_logical.items(), key=lambda item: repr(item[0])):
        for field in ("git_commit", "input_sha256", "environment_fingerprint"):
            values = sorted({str(row[field]) for row in rows})
            if len(values) > 1:
                add_publication_issue(
                    f"mixed_{field}",
                    f"比較group内で{field}が一致しません",
                    group=list(key),
                    values=values,
                )
    if profile in {"timing", "quality"}:
        non_benchmark = sorted(
            row["run_id"] for row in success_rows if row["run_mode"] != "benchmark"
        )
        if non_benchmark:
            errors.append(
                issue(
                    "non_benchmark",
                    "publication profileにはbenchmark runだけを使用できます",
                    run_ids=non_benchmark,
                )
            )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in success_rows:
        grouped[comparison_group(row)].append(row)
    for key, rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
        methods = defaultdict(list)
        for row in rows:
            methods[row["method"]].append(row)
        baseline = BASELINES[group_fields(key)["family"]]
        if baseline not in methods:
            add_publication_issue(
                "missing_baseline",
                "比較groupにCPU基準手法がありません",
                group=list(key),
                baseline=baseline,
            )
            continue
        if profile == "timing":
            for method, method_rows in sorted(methods.items()):
                if len(method_rows) < 10:
                    errors.append(
                        issue(
                            "insufficient_timing_samples",
                            "timing profileは手法ごとに10標本以上必要です",
                            group=list(key),
                            method=method,
                            expected=10,
                            actual=len(method_rows),
                        )
                    )
                missing_pairs = missing_baseline_pairs(method_rows, methods[baseline])
                if missing_pairs:
                    errors.append(
                        issue(
                            "missing_timing_pairs",
                            "paired speedup用のCPU基準runがありません",
                            group=list(key),
                            method=method,
                            pairs=missing_pairs,
                        )
                    )
        if profile == "quality":
            common_seeds = set.intersection(
                *(set(int(row["seed"]) for row in method_rows)
                  for method_rows in methods.values())
            )
            if len(common_seeds) < 25:
                errors.append(
                    issue(
                        "insufficient_quality_seeds",
                        "quality profileは25個以上の共通seedが必要です",
                        group=list(key),
                        expected=25,
                        actual=len(common_seeds),
                        common_seeds=sorted(common_seeds),
                    )
                )
    if profile == "validation":
        for key, rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
            methods = defaultdict(list)
            for row in rows:
                methods[row["method"]].append(row)
            for method, method_rows in sorted(methods.items()):
                if len(method_rows) < 10:
                    warnings.append(
                        issue(
                            "insufficient_timing_samples",
                            "正式timing表には10標本以上必要です",
                            group=list(key),
                            method=method,
                            expected=10,
                            actual=len(method_rows),
                        )
                    )
            common_seeds = (
                set.intersection(
                    *(set(int(row["seed"]) for row in method_rows)
                      for method_rows in methods.values())
                )
                if methods
                else set()
            )
            if len(common_seeds) < 25:
                warnings.append(
                    issue(
                        "insufficient_quality_seeds",
                        "正式quality表には25個以上の共通seedが必要です",
                        group=list(key),
                        expected=25,
                        actual=len(common_seeds),
                    )
                )

    retries = []
    for experiment in experiments:
        for run_id, history in sorted(experiment.histories.items()):
            if len(history) > 1 or any(item["status"] == "failure" for item in history):
                retries.append(
                    {
                        "experiment_id": experiment.experiment_id,
                        "run_id": run_id,
                        "attempts": len(history),
                        "statuses": [item["status"] for item in history],
                        "failure_stages": [
                            item["error_stage"]
                            for item in history
                            if item["status"] == "failure"
                        ],
                    }
                )
    return {
        "aggregation_rules_version": AGGREGATION_RULES_VERSION,
        "profile": profile,
        "publication_ready": profile != "validation" and not errors,
        "planned_runs": sum(len(exp.planned) for exp in experiments),
        "current_records": len(all_rows),
        "successful_runs": len(success_rows),
        "failed_runs": sum(row["status"] == "failure" for row in all_rows),
        "group_count": len(grouped),
        "warnings": warnings,
        "errors": errors,
        "retry_history": retries,
    }


def missing_baseline_pairs(
    method_rows: Sequence[dict[str, Any]], baseline_rows: Sequence[dict[str, Any]]
) -> list[list[Any]]:
    baseline_keys = {tuple(row["_paired_key"]) for row in baseline_rows}
    return [
        list(row["_paired_key"])
        for row in method_rows
        if tuple(row["_paired_key"]) not in baseline_keys
    ]


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AggregationError("空の標本からpercentileは計算できません")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summary_stats(
    values: Sequence[float],
    *,
    seed_count: int | None = None,
    repetition_count: int | None = None,
) -> dict[str, Any]:
    if not values:
        raise AggregationError("空の標本は集計できません")
    finite = [float(value) for value in values]
    if any(not math.isfinite(value) for value in finite):
        raise AggregationError("統計標本にNaNまたはInfがあります")
    q1 = percentile(finite, 0.25)
    q3 = percentile(finite, 0.75)
    return {
        "n": len(finite),
        "seed_count": seed_count,
        "repetition_count": repetition_count,
        "mean": statistics.fmean(finite),
        "sample_sd": statistics.stdev(finite) if len(finite) >= 2 else None,
        "median": statistics.median(finite),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "minimum": min(finite),
        "maximum": max(finite),
    }


def aggregate_speed(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [
        row
        for row in rows
        if row["status"] == "success" and row["run_mode"] == "benchmark"
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        groups[comparison_group(row)].append(row)
    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        fields = group_fields(key)
        baseline_method = BASELINES[fields["family"]]
        baseline_rows = [
            row for row in group_rows if row["method"] == baseline_method
        ]
        baseline_by_pair = {
            tuple(row["_paired_key"]): row for row in baseline_rows
        }
        methods = sorted({row["method"] for row in group_rows})
        for method in methods:
            method_rows = [row for row in group_rows if row["method"] == method]
            seeds = {int(row["seed"]) for row in method_rows}
            repetitions = {int(row["repetition"]) for row in method_rows}
            for metric in TIMING_METRICS:
                values = [float(row[metric]) for row in method_rows]
                ratios = [
                    float(baseline_by_pair[tuple(row["_paired_key"])][metric])
                    / float(row[metric])
                    for row in method_rows
                    if tuple(row["_paired_key"]) in baseline_by_pair
                ]
                stats = summary_stats(
                    values,
                    seed_count=len(seeds),
                    repetition_count=len(repetitions),
                )
                speedup = (
                    summary_stats(
                        ratios,
                        seed_count=len(seeds),
                        repetition_count=len(repetitions),
                    )
                    if ratios
                    else None
                )
                output.append(
                    {
                        **fields,
                        "method": method,
                        "baseline_method": baseline_method,
                        "metric": metric,
                        **stats,
                        **prefixed_stats("speedup", speedup),
                    }
                )
    return output


def aggregate_quality(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [row for row in rows if row["status"] == "success"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        groups[comparison_group(row)].append(row)
    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        fields = group_fields(key)
        baseline_method = BASELINES[fields["family"]]
        seed_values: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in group_rows:
            seed_values[(row["method"], int(row["seed"]))].append(
                float(row["stress_value"])
            )
        representatives = {
            key_value: statistics.median(values)
            for key_value, values in seed_values.items()
        }
        baseline_by_seed = {
            seed: value
            for (method, seed), value in representatives.items()
            if method == baseline_method
        }
        for method in sorted({row["method"] for row in group_rows}):
            method_values = sorted(
                (seed, value)
                for (candidate, seed), value in representatives.items()
                if candidate == method
            )
            values = [value for _, value in method_values]
            ratios = [
                value / baseline_by_seed[seed]
                for seed, value in method_values
                if seed in baseline_by_seed
            ]
            repetitions = {
                int(row["repetition"])
                for row in group_rows
                if row["method"] == method
            }
            stats = summary_stats(
                values,
                seed_count=len(method_values),
                repetition_count=len(repetitions),
            )
            ratio_stats = (
                summary_stats(
                    ratios,
                    seed_count=len(ratios),
                    repetition_count=None,
                )
                if ratios
                else None
            )
            output.append(
                {
                    **fields,
                    "method": method,
                    "baseline_method": baseline_method,
                    **stats,
                    **prefixed_stats("stress_ratio", ratio_stats),
                }
            )
    return output


def aggregate_method_stats(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [row for row in rows if row["status"] == "success"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        grouped[comparison_group(row) + (row["method"],)].append(row)
    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
        fields = group_fields(key[:-1])
        method = key[-1]
        for field in METHOD_STAT_FIELDS:
            values = [
                float(row[field])
                for row in group_rows
                if row[field] is not None
            ]
            if not values:
                continue
            output.append(
                {
                    **fields,
                    "method": method,
                    "stat": field,
                    **summary_stats(
                        values,
                        seed_count=len({int(row["seed"]) for row in group_rows}),
                        repetition_count=len(
                            {int(row["repetition"]) for row in group_rows}
                        ),
                    ),
                }
            )
    return output


def prefixed_stats(prefix: str, stats: dict[str, Any] | None) -> dict[str, Any]:
    names = (
        "n",
        "seed_count",
        "repetition_count",
        "mean",
        "sample_sd",
        "median",
        "q1",
        "q3",
        "iqr",
        "minimum",
        "maximum",
    )
    return {
        f"{prefix}_{name}": stats[name] if stats is not None else None
        for name in names
    }


RUN_COLUMNS = (
    "experiment_id",
    "run_id",
    "repetition",
    "run_mode",
    "status",
    "method",
    "family",
    "dataset",
    "input_sha256",
    "seed",
    "iterations",
    "epsilon",
    "pivots",
    "git_commit",
    "git_dirty",
    "environment_fingerprint",
    "nodes",
    "edges",
    "constraints",
    *TIMING_METRICS,
    "gpu_device_time_ms",
    "stress_kind",
    "stress_value",
    "attempted_updates",
    "completed_updates",
    "retry_failures",
    "rounds",
    "dispatches",
    "history_count",
    "prior_failure_count",
)
GROUP_COLUMNS = (
    "experiment_id",
    "dataset",
    "input_sha256",
    "family",
    "run_mode",
    "iterations",
    "epsilon",
    "pivots",
    "stress_kind",
    "environment_fingerprint",
)
STAT_COLUMNS = (
    "n",
    "seed_count",
    "repetition_count",
    "mean",
    "sample_sd",
    "median",
    "q1",
    "q3",
    "iqr",
    "minimum",
    "maximum",
)
SPEED_COLUMNS = (
    *GROUP_COLUMNS,
    "method",
    "baseline_method",
    "metric",
    *STAT_COLUMNS,
    *(f"speedup_{name}" for name in STAT_COLUMNS),
)
QUALITY_COLUMNS = (
    *GROUP_COLUMNS,
    "method",
    "baseline_method",
    *STAT_COLUMNS,
    *(f"stress_ratio_{name}" for name in STAT_COLUMNS),
)
METHOD_STAT_COLUMNS = (
    *GROUP_COLUMNS,
    "method",
    "stat",
    *STAT_COLUMNS,
)


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def write_csv(
    path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=list(columns),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in sorted(rows, key=stable_row_key):
            writer.writerow({column: csv_value(row.get(column)) for column in columns})


def stable_row_key(row: dict[str, Any]) -> tuple[str, ...]:
    preferred = (
        "experiment_id",
        "dataset",
        "family",
        "method",
        "metric",
        "stat",
        "seed",
        "repetition",
        "run_id",
    )
    return tuple(str(row.get(field, "")) for field in preferred)


METHOD_LABELS = {
    "sgd": "SGD",
    "atomic_sgd": "AtomicSGD",
    "rr_sgd": "RR-SGD",
    "sparse_sgd": "SparseSGD",
    "rr_sparse_sgd": "RR-SparseSGD",
}
METRIC_LABELS = {
    "iteration_time_ms": "Iteration (ms)",
    "algorithm_time_cold_ms": "Algorithm cold (ms)",
    "cli_total_time_cold_ms": "CLI total cold (ms)",
}


def format_interval(median: Any, q1: Any, q3: Any, digits: int = 2) -> str:
    if median is None or q1 is None or q3 is None:
        return "—"
    return (
        f"{float(median):.{digits}f} "
        f"[{float(q1):.{digits}f}, {float(q3):.{digits}f}]"
    )


def format_mean_sd(mean: Any, sample_sd: Any, digits: int = 2) -> str:
    if mean is None:
        return "—"
    if sample_sd is None:
        return f"{float(mean):.{digits}f} ± —"
    return f"{float(mean):.{digits}f} ± {float(sample_sd):.{digits}f}"


def speed_table_model(speed_rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    columns = [
        "Dataset",
        "Family",
        "Method",
        *(METRIC_LABELS[metric] for metric in TIMING_METRICS),
        "Algorithm speedup",
        "n",
    ]
    indexed = {
        (
            row["experiment_id"],
            row["dataset"],
            row["family"],
            row["method"],
            row["metric"],
        ): row
        for row in speed_rows
    }
    identities = sorted(
        {
            (
                row["experiment_id"],
                row["dataset"],
                row["family"],
                row["method"],
            )
            for row in speed_rows
        }
    )
    body: list[list[str]] = []
    for experiment_id, dataset, family, method in identities:
        metrics = {
            metric: indexed.get((experiment_id, dataset, family, method, metric))
            for metric in TIMING_METRICS
        }
        algorithm = metrics["algorithm_time_cold_ms"]
        n_value = algorithm["n"] if algorithm is not None else None
        body.append(
            [
                dataset,
                family.capitalize(),
                METHOD_LABELS.get(method, method),
                *(
                    format_interval(
                        metrics[metric]["median"],
                        metrics[metric]["q1"],
                        metrics[metric]["q3"],
                    )
                    if metrics[metric] is not None
                    else "—"
                    for metric in TIMING_METRICS
                ),
                (
                    format_interval(
                        algorithm["speedup_median"],
                        algorithm["speedup_q1"],
                        algorithm["speedup_q3"],
                        digits=3,
                    )
                    + "×"
                    if algorithm is not None
                    and algorithm["speedup_median"] is not None
                    else "—"
                ),
                str(n_value) if n_value is not None else "—",
            ]
        )
    return columns, body


def quality_table_model(
    quality_rows: Sequence[dict[str, Any]]
) -> tuple[list[str], list[list[str]]]:
    columns = (
        "Dataset",
        "Family",
        "Method",
        "Stress kind",
        "Stress mean ± SD",
        "Stress median [Q1, Q3]",
        "Paired stress ratio",
        "seeds",
    )
    body = []
    for row in sorted(quality_rows, key=stable_row_key):
        ratio = format_mean_sd(
            row["stress_ratio_mean"], row["stress_ratio_sample_sd"], digits=4
        )
        body.append(
            [
                str(row["dataset"]),
                str(row["family"]).capitalize(),
                METHOD_LABELS.get(str(row["method"]), str(row["method"])),
                str(row["stress_kind"]),
                format_mean_sd(row["mean"], row["sample_sd"]),
                format_interval(row["median"], row["q1"], row["q3"]),
                ratio,
                str(row["seed_count"]),
            ]
        )
    return list(columns), body


def render_markdown(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def escape(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(escape(column) for column in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines.extend(
        "| " + " | ".join(escape(str(value)) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: str) -> str:
    return "".join(LATEX_ESCAPES.get(character, character) for character in value)


def render_latex(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    alignment = "l" * len(columns)
    line_break = r" \\"
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(latex_escape(column) for column in columns) + line_break,
        r"\hline",
    ]
    lines.extend(
        " & ".join(latex_escape(str(value)) for value in row) + line_break
        for row in rows
    )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def metadata_for(
    experiments: Sequence[LoadedExperiment],
    profile: str,
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "aggregation_rules_version": AGGREGATION_RULES_VERSION,
        "record_schema_version": 1,
        "profile": profile,
        "experiment_ids": sorted(exp.experiment_id for exp in experiments),
        "source_commits": sorted(
            {
                str(row["git_commit"])
                for exp in experiments
                for row in exp.normalized_rows
            }
        ),
        "command": [
            "python3",
            "experiments/aggregate_results.py",
            "--profile",
            profile,
            *[
                value
                for experiment_id in sorted(exp.experiment_id for exp in experiments)
                for value in ("--experiment-dir", experiment_id)
            ],
        ],
        "inputs": [
            {
                "experiment_id": exp.experiment_id,
                "manifest_sha256": exp.checksums.manifest_sha256,
                "environment_sha256": exp.checksums.environment_sha256,
                "results_sha256": exp.checksums.results_sha256,
            }
            for exp in sorted(experiments, key=lambda item: item.experiment_id)
        ],
        "aggregation_sources": {
            relative: file_sha256(REPO_ROOT / relative)
            for relative in (
                "experiments/aggregate_results.py",
                "experiments/aggregation.py",
                "experiments/experiment_plan.py",
            )
        },
        "artifacts": dict(sorted(artifact_hashes.items())),
    }


def generate_artifacts(
    experiments: Sequence[LoadedExperiment],
    profile: str,
    output_dir: Path,
) -> dict[str, Any]:
    if not experiments:
        raise AggregationError("1つ以上のexperiment directoryが必要です")
    report = build_validation_report(experiments, profile)
    if profile != "validation" and report["errors"]:
        raise PublicationGateError(report)
    rows = [row for exp in experiments for row in exp.normalized_rows]
    speed = aggregate_speed(rows)
    quality = aggregate_quality(rows)
    method_stats = aggregate_method_stats(rows)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AggregationError(f"出力先に既存fileがあります: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        write_csv(temporary / "runs.csv", rows, RUN_COLUMNS)
        write_csv(temporary / "speed-summary.csv", speed, SPEED_COLUMNS)
        write_csv(temporary / "quality-summary.csv", quality, QUALITY_COLUMNS)
        write_csv(
            temporary / "method-stats.csv", method_stats, METHOD_STAT_COLUMNS
        )
        speed_columns, speed_body = speed_table_model(speed)
        quality_columns, quality_body = quality_table_model(quality)
        (temporary / "speed-table.md").write_text(
            render_markdown(speed_columns, speed_body), encoding="utf-8"
        )
        (temporary / "speed-table.tex").write_text(
            render_latex(speed_columns, speed_body), encoding="utf-8"
        )
        (temporary / "quality-table.md").write_text(
            render_markdown(quality_columns, quality_body), encoding="utf-8"
        )
        (temporary / "quality-table.tex").write_text(
            render_latex(quality_columns, quality_body), encoding="utf-8"
        )
        write_json(temporary / "validation-report.json", report)
        hashes = {
            path.name: file_sha256(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        write_json(
            temporary / "aggregation-metadata.json",
            metadata_for(experiments, profile, hashes),
        )
        actual = {path.name for path in temporary.iterdir() if path.is_file()}
        expected = set(ARTIFACT_NAMES)
        if actual != expected:
            raise AggregationError(
                f"生成artifact集合が不正です: missing={sorted(expected-actual)}, "
                f"extra={sorted(actual-expected)}"
            )
        if output_dir.exists():
            output_dir.rmdir()
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def dry_run_summary(
    experiments: Sequence[LoadedExperiment], profile: str
) -> dict[str, Any]:
    report = build_validation_report(experiments, profile)
    rows = [row for exp in experiments for row in exp.normalized_rows]
    group_count = len(
        {comparison_group(row) for row in rows if row["status"] == "success"}
    )
    return {
        "profile": profile,
        "planned_runs": report["planned_runs"],
        "current_records": report["current_records"],
        "successful_runs": report["successful_runs"],
        "failed_runs": report["failed_runs"],
        "group_count": group_count,
        "warning_count": len(report["warnings"]),
        "error_count": len(report["errors"]),
        "publication_ready": report["publication_ready"],
        "would_write": list(ARTIFACT_NAMES),
    }
