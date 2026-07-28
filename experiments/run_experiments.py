#!/usr/bin/env python3
"""Manifest-driven runner for Stress SGD experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

try:
    from experiment_plan import (
        METHODS,
        REPO_ROOT,
        PlanError,
        RunDefinition,
        expand_runs,
        load_manifest,
    )
except ModuleNotFoundError:
    from experiments.experiment_plan import (
        METHODS,
        REPO_ROOT,
        PlanError,
        RunDefinition,
        expand_runs,
        load_manifest,
    )

SCHEMA_PATH = REPO_ROOT / "experiments/schema/experiment-record-v1.json"


class RunnerError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "output/experiments"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def verify_inputs_and_binaries(runs: list[RunDefinition]) -> None:
    checked_inputs: set[tuple[Path, str]] = set()
    checked_binaries: set[Path] = set()
    for run in runs:
        key = (run.dataset.path, run.dataset.sha256)
        if key not in checked_inputs:
            if not run.dataset.path.is_file():
                raise RunnerError(f"datasetがありません: {run.dataset.path}")
            actual = file_sha256(run.dataset.path)
            if actual != run.dataset.sha256:
                raise RunnerError(
                    f"dataset checksum不一致: {run.dataset.name}: "
                    f"manifest={run.dataset.sha256}, actual={actual}"
                )
            checked_inputs.add(key)
        if run.binary not in checked_binaries:
            if not run.binary.is_file() or not os.access(run.binary, os.X_OK):
                raise RunnerError(
                    f"release binaryがありません、または実行できません: {run.binary}"
                )
            checked_binaries.add(run.binary)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_for(run: RunDefinition, output_dir: Path, run_id: str | None = None) -> list[str]:
    command = [
        str(run.binary),
        "--run-id",
        run_id or run.run_id,
        "--input",
        str(run.dataset.path),
        "--iterations",
        str(run.iterations),
        "--epsilon",
        format(run.epsilon, ".17g"),
        "--seed",
        str(run.seed),
        "--output-format",
        "json",
        "--output-dir",
        str(output_dir),
        "--run-mode",
        run.run_mode,
    ]
    if run.pivots is not None:
        command.extend(["--pivots", str(run.pivots)])
    return command


def git_environment() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
    }


def prepare_output(
    manifest: dict[str, Any],
    manifest_path: Path,
    output_root: Path,
    resume: bool,
) -> Path:
    experiment_dir = (output_root / manifest["experiment_id"]).resolve()
    snapshot_path = experiment_dir / "manifest.json"
    if experiment_dir.exists():
        if not resume:
            raise RunnerError(
                f"出力先が既に存在します: {experiment_dir}（再開は--resume）"
            )
        if not snapshot_path.is_file():
            raise RunnerError("既存出力にmanifest snapshotがありません")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot != manifest:
            raise RunnerError("再開時のmanifestが保存済みsnapshotと一致しません")
    else:
        experiment_dir.mkdir(parents=True)
        (experiment_dir / "stderr").mkdir()
        (experiment_dir / "artifacts").mkdir()
        shutil.copyfile(manifest_path, snapshot_path)
        (experiment_dir / "environment.json").write_text(
            json.dumps(git_environment(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return experiment_dir


def load_existing_records(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    raw_lines = results_path.read_bytes().splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    for index, raw_line in enumerate(raw_lines):
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if index != len(raw_lines) - 1:
                raise RunnerError(
                    f"results.jsonlの途中に壊れた行があります: line {index + 1}"
                ) from error
            timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
            corrupt_path = results_path.with_name(
                f"{results_path.stem}.corrupt-{timestamp}.jsonl"
            )
            corrupt_path.write_bytes(raw_line)
            rewrite_jsonl(results_path, records)
            print(f"Isolated incomplete final JSONL line: {corrupt_path}", file=sys.stderr)
            break
        if not isinstance(record, dict):
            raise RunnerError(f"results.jsonl line {index + 1} はobjectではありません")
        records.append(record)
    return records


def rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as destination:
        temporary = Path(destination.name)
        for record in records:
            destination.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as destination:
        destination.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        destination.flush()
        os.fsync(destination.fileno())


def run_method(
    run: RunDefinition,
    command: list[str],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["EXPERIMENT_METHOD"] = run.method
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def parse_and_validate_success(
    run: RunDefinition, completed: subprocess.CompletedProcess[str]
) -> dict[str, Any]:
    if completed.returncode != 0:
        raise RunnerError(f"process exit code {completed.returncode}")
    lines = completed.stdout.splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise RunnerError(f"stdoutはJSON 1行である必要があります: lines={len(lines)}")
    try:
        record = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise RunnerError(f"stdoutをJSONとして解析できません: {error}") from error
    if not isinstance(record, dict):
        raise RunnerError("method stdoutのJSON rootはobjectです")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    required = set(schema["required"])
    properties = set(schema["properties"])
    missing = sorted(required - record.keys())
    unknown = sorted(record.keys() - properties)
    if missing:
        raise RunnerError(f"recordの必須fieldがありません: {', '.join(missing)}")
    if unknown:
        raise RunnerError(f"recordに未知のfieldがあります: {', '.join(unknown)}")
    expected = {
        "schema_version": 1,
        "run_id": run.run_id,
        "run_mode": run.run_mode,
        "status": "success",
        "method": run.method,
        "family": METHODS[run.method].family,
        "dataset": run.dataset.name,
        "input_sha256": run.dataset.sha256,
        "seed": run.seed,
        "iterations": run.iterations,
        "pivots": run.pivots,
    }
    for field, value in expected.items():
        if record[field] != value:
            raise RunnerError(
                f"recordの{field}がmanifestと不一致です: "
                f"actual={record[field]!r}, expected={value!r}"
            )
    if not math.isclose(
        float(record["epsilon"]), run.epsilon, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise RunnerError("recordのepsilonがmanifestと不一致です")
    validate_timing(record)
    validate_stress(record)
    validate_null_contract(record, run)
    for field in ("final_positions_path",):
        value = record[field]
        if not isinstance(value, str) or not resolve_record_path(value).is_file():
            raise RunnerError(f"{field}が存在するfileを指していません: {value!r}")
    if run.pivots is not None:
        value = record["vertex_map_path"]
        if not isinstance(value, str) or not resolve_record_path(value).is_file():
            raise RunnerError("Sparse成功recordのvertex_map_pathが存在しません")
    return record


def validate_timing(record: dict[str, Any]) -> None:
    stage_fields = (
        "input_time_ms",
        "common_preprocess_time_ms",
        "method_setup_time_ms",
        "runtime_init_time_ms",
        "upload_time_ms",
        "iteration_time_ms",
        "gpu_device_time_ms",
        "readback_time_ms",
        "postprocess_time_ms",
    )
    for field in stage_fields + (
        "algorithm_time_cold_ms",
        "algorithm_time_warm_ms",
        "cli_total_time_cold_ms",
    ):
        value = record[field]
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise RunnerError(f"{field}はnullまたは有限な非負値です")
    for required in (
        "input_time_ms",
        "common_preprocess_time_ms",
        "iteration_time_ms",
        "postprocess_time_ms",
        "algorithm_time_cold_ms",
        "algorithm_time_warm_ms",
        "cli_total_time_cold_ms",
    ):
        if record[required] is None:
            raise RunnerError(f"成功recordの{required}はnullにできません")
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
    assert_close("algorithm_time_cold_ms", record["algorithm_time_cold_ms"], algorithm)
    assert_close(
        "algorithm_time_warm_ms",
        record["algorithm_time_warm_ms"],
        algorithm - float(record["runtime_init_time_ms"] or 0.0),
    )
    assert_close(
        "cli_total_time_cold_ms",
        record["cli_total_time_cold_ms"],
        float(record["input_time_ms"]) + algorithm,
    )


def assert_close(field: str, actual: float, expected: float) -> None:
    if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-6):
        raise RunnerError(
            f"{field}が内訳と不一致です: actual={actual}, expected={expected}"
        )


def validate_stress(record: dict[str, Any]) -> None:
    kind = record["stress_kind"]
    value = record["stress_value"]
    elapsed = record["stress_eval_time_ms"]
    if kind not in {"exact", "sampled"}:
        raise RunnerError("成功recordのstress_kindはexactまたはsampledです")
    for field, number in (("stress_value", value), ("stress_eval_time_ms", elapsed)):
        if (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(number)
            or number < 0
        ):
            raise RunnerError(f"{field}は有限な非負値です")
    if kind == "exact":
        if record["stress_samples"] is not None or record["stress_seed"] is not None:
            raise RunnerError("exact stressのsamplesとseedはnullです")
    elif not isinstance(record["stress_samples"], int) or record["stress_samples"] <= 0:
        raise RunnerError("sampled stressには正のstress_samplesが必要です")
    elif not isinstance(record["stress_seed"], int) or record["stress_seed"] < 0:
        raise RunnerError("sampled stressには非負stress_seedが必要です")


def validate_null_contract(record: dict[str, Any], run: RunDefinition) -> None:
    if run.pivots is None:
        if record["preprocess_sha256"] is not None:
            raise RunnerError("Full methodのpreprocess_sha256はnullです")
    elif not isinstance(record["preprocess_sha256"], str):
        raise RunnerError("Sparse methodにはpreprocess_sha256が必要です")
    if run.method in {"sgd", "sparse_sgd"}:
        for field in (
            "gpu_name",
            "gpu_backend",
            "runtime_init_time_ms",
            "upload_time_ms",
            "gpu_device_time_ms",
            "readback_time_ms",
        ):
            if record[field] is not None:
                raise RunnerError(f"CPU methodの{field}はnullです")
    if run.method == "sgd" and record["pivots"] is not None:
        raise RunnerError("Full methodのpivotsはnullです")


def validate_paired_preprocessing(
    record: dict[str, Any], existing: Iterable[dict[str, Any]]
) -> None:
    condition_fields = (
        "family",
        "dataset",
        "input_sha256",
        "seed",
        "iterations",
        "epsilon",
        "pivots",
    )
    for previous in existing:
        if previous.get("status") != "success":
            continue
        if any(previous.get(field) != record.get(field) for field in condition_fields):
            continue
        if (
            previous.get("initial_positions_sha256")
            != record.get("initial_positions_sha256")
        ):
            raise RunnerError(
                "同一family・dataset・seed条件でinitial_positions_sha256が一致しません: "
                f"{previous.get('method')} vs {record.get('method')}"
            )
        if record["family"] == "sparse" and (
            previous.get("preprocess_sha256") != record.get("preprocess_sha256")
        ):
            raise RunnerError(
                "同一Sparse条件でpivot・constraint・etaのfingerprintが一致しません: "
                f"{previous.get('method')} vs {record.get('method')}"
            )


def resolve_record_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def failure_record(
    run: RunDefinition,
    stage: str,
    message: str,
    exit_code: int | None,
    stderr_path: Path,
) -> dict[str, Any]:
    environment = git_environment()
    return {
        "schema_version": 1,
        "run_id": run.run_id,
        "run_mode": run.run_mode,
        "status": "failure",
        "method": run.method,
        "family": METHODS[run.method].family,
        "binary": str(run.binary),
        "git_commit": environment["git_commit"],
        "git_dirty": environment["git_dirty"],
        "dataset": run.dataset.name,
        "input_path": str(run.dataset.path),
        "input_sha256": run.dataset.sha256,
        "seed": run.seed,
        "initial_positions_sha256": None,
        "preprocess_sha256": None,
        "nodes": None,
        "edges": None,
        "constraints": None,
        "pivots": run.pivots,
        "iterations": run.iterations,
        "epsilon": run.epsilon,
        "cpu_model": platform.processor() or platform.machine(),
        "gpu_name": None,
        "gpu_backend": None,
        "input_time_ms": None,
        "common_preprocess_time_ms": None,
        "method_setup_time_ms": None,
        "runtime_init_time_ms": None,
        "upload_time_ms": None,
        "iteration_time_ms": None,
        "gpu_device_time_ms": None,
        "readback_time_ms": None,
        "postprocess_time_ms": None,
        "algorithm_time_cold_ms": None,
        "algorithm_time_warm_ms": None,
        "cli_total_time_cold_ms": None,
        "stress_kind": None,
        "stress_value": None,
        "stress_eval_time_ms": None,
        "stress_samples": None,
        "stress_seed": None,
        "attempted_updates": None,
        "completed_updates": None,
        "retry_failures": None,
        "rounds": None,
        "dispatches": None,
        "final_positions_path": None,
        "vertex_map_path": None,
        "error_stage": stage,
        "error_message": message[-4000:],
        "exit_code": exit_code,
        "stderr_log_path": str(stderr_path),
    }


def write_stderr(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def execute_run(
    run: RunDefinition,
    experiment_dir: Path,
    manifest: dict[str, Any],
    results_path: Path,
) -> bool:
    artifact_dir = experiment_dir / "artifacts" / run.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = experiment_dir / "stderr" / f"{run.run_id}.log"

    for warmup in range(manifest["warmups"]):
        with tempfile.TemporaryDirectory(
            prefix=f"{run.run_id}-warmup-", dir=experiment_dir
        ) as warmup_dir:
            warmup_command = command_for(
                run,
                Path(warmup_dir),
                run_id=f"{run.run_id}-warmup-{warmup}",
            )
            try:
                warmed = run_method(run, warmup_command, manifest["timeout_seconds"])
            except subprocess.TimeoutExpired as error:
                message = f"warm-up timeout: {error}"
                write_stderr(stderr_path, error.stderr or "")
                append_record(
                    results_path,
                    failure_record(run, "warmup", message, None, stderr_path),
                )
                return False
            if warmed.returncode != 0:
                write_stderr(stderr_path, warmed.stderr)
                append_record(
                    results_path,
                    failure_record(
                        run,
                        "warmup",
                        warmed.stderr.strip() or f"exit code {warmed.returncode}",
                        warmed.returncode,
                        stderr_path,
                    ),
                )
                return False

    command = command_for(run, artifact_dir)
    try:
        completed = run_method(run, command, manifest["timeout_seconds"])
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr or ""
        write_stderr(stderr_path, stderr)
        append_record(
            results_path,
            failure_record(run, "timeout", str(error), None, stderr_path),
        )
        return False
    write_stderr(stderr_path, completed.stderr)
    try:
        record = parse_and_validate_success(run, completed)
        validate_paired_preprocessing(record, load_existing_records(results_path))
    except RunnerError as error:
        stage = "execute" if completed.returncode != 0 else "validate"
        message = completed.stderr.strip() or str(error)
        append_record(
            results_path,
            failure_record(run, stage, message, completed.returncode, stderr_path),
        )
        return False
    append_record(results_path, record)
    return True


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        manifest_path = arguments.manifest.resolve()
        manifest = load_manifest(manifest_path)
        runs = expand_runs(manifest, manifest_path)
        verify_inputs_and_binaries(runs)
        print(
            f"Experiment {manifest['experiment_id']}: "
            f"{len(runs)} formal runs, warmups={manifest['warmups']}"
        )
        if arguments.dry_run:
            for index, run in enumerate(runs, start=1):
                output_dir = (
                    arguments.output_root.resolve()
                    / manifest["experiment_id"]
                    / "artifacts"
                    / run.run_id
                )
                print(
                    f"[{index}/{len(runs)}] {run.run_id}\n  "
                    + " ".join(command_for(run, output_dir))
                )
            return 0

        experiment_dir = prepare_output(
            manifest, manifest_path, arguments.output_root, arguments.resume
        )
        results_path = experiment_dir / "results.jsonl"
        existing = load_existing_records(results_path)
        successful = {
            record.get("run_id")
            for record in existing
            if record.get("status") == "success"
        }
        failures = 0
        for index, run in enumerate(runs, start=1):
            if run.run_id in successful:
                print(f"[{index}/{len(runs)}] skip success {run.run_id}")
                continue
            print(f"[{index}/{len(runs)}] run {run.run_id}")
            if execute_run(run, experiment_dir, manifest, results_path):
                print(f"[{index}/{len(runs)}] success {run.run_id}")
            else:
                failures += 1
                print(f"[{index}/{len(runs)}] failure {run.run_id}", file=sys.stderr)
                if manifest["fail_fast"]:
                    break
        return 1 if failures else 0
    except (RunnerError, PlanError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
