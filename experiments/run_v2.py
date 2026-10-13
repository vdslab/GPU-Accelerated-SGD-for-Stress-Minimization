#!/usr/bin/env python3
"""Reproducible sequential v2 runs, RR variants, raw evidence, and safe resume."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from experiment_plan import REPO_ROOT, PlanError, RunDefinition, expand_runs, load_manifest
    import run_experiments as common
except ModuleNotFoundError:
    from experiments.experiment_plan import REPO_ROOT, PlanError, RunDefinition, expand_runs, load_manifest
    from experiments import run_experiments as common

DEFAULT_RESULTS_DIR = REPO_ROOT / "experiment_results/v2"
RR_META = {"tile_size": "rr_tile_size", "workgroup_size": "rr_workgroup_size",
           "coordinate_memory": "rr_memory_mode", "outer_shuffle": "rr_outer_shuffle",
           "inner_shuffle": "rr_inner_shuffle"}
RR_DEFAULT = {"tile_size": 1024, "workgroup_size": 1024, "coordinate_memory": "shared",
              "outer_shuffle": True, "inner_shuffle": True}


@dataclasses.dataclass(frozen=True)
class VariantRun:
    run: RunDefinition
    variant_id: str
    variant: dict[str, Any]

    @property
    def run_id(self):
        return self.run.run_id


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def variants_for(manifest: dict, method: str) -> list[dict]:
    variants = manifest.get("variants", {})
    if not isinstance(variants, dict) or any(key not in manifest["methods"] for key in variants):
        raise PlanError("variants must be an object for selected methods")
    configured = variants.get(method, [{"id": "default"}])
    if not isinstance(configured, list) or not configured:
        raise PlanError("each method variants list must be nonempty")
    validated = []
    ids = set()
    for variant in configured:
        if not isinstance(variant, dict) or not isinstance(variant.get("id"), str) or not variant["id"]:
            raise PlanError("each variant needs a nonempty string id")
        if variant["id"] in ids:
            raise PlanError("duplicate variant id")
        ids.add(variant["id"])
        unknown = variant.keys() - {"id", *RR_DEFAULT}
        if unknown or (method != "rr_sgd" and variant.keys() != {"id"}):
            raise PlanError("only rr_sgd accepts RR variant settings")
        parameters = dict(RR_DEFAULT, **{key: value for key, value in variant.items() if key != "id"}) if method == "rr_sgd" else {}
        for field in ("tile_size", "workgroup_size") if parameters else ():
            value = parameters[field]
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1024:
                raise PlanError(f"{field} must be integer 1..1024")
        if parameters and parameters["coordinate_memory"] not in {"shared", "global"}:
            raise PlanError("coordinate_memory must be shared or global")
        for field in ("outer_shuffle", "inner_shuffle") if parameters else ():
            if not isinstance(parameters[field], bool):
                raise PlanError(f"{field} must be boolean")
        validated.append({"id": variant["id"], **parameters})
    return validated


def expand_v2_runs(manifest: dict, path: Path) -> list[VariantRun]:
    result = []
    variants = {method: variants_for(manifest, method) for method in manifest["methods"]}
    for run in expand_runs(manifest, path):
        for setting in variants[run.method]:
            parameters = {key: value for key, value in setting.items() if key != "id"}
            digest = canonical_hash({"base_run_id": run.run_id, "variant_id": setting["id"], "variant": parameters})[:12]
            extended = dataclasses.replace(run, run_id=run.run_id.rsplit("-", 1)[0] + "-" + digest)
            result.append(VariantRun(extended, setting["id"], parameters))
    if len({run.run_id for run in result}) != len(result):
        raise PlanError("v2 run IDs are not unique")
    datasets = [row["name"] for row in manifest["datasets"]]
    methods = manifest["methods"]
    ordering = [(method, variant["id"]) for method in methods for variant in variants[method]]
    def key(item):
        run = item.run
        return (datasets.index(run.dataset.name), run.seed, run.repetition,
                run.iterations, run.epsilon, run.pivots or 0,
                (ordering.index((run.method, item.variant_id)) - run.seed) % len(ordering))
    return sorted(result, key=key)


def command_for(item: VariantRun, output_dir: Path, run_id: str | None = None) -> list[str]:
    command = common.command_for(item.run, output_dir, run_id)
    if item.variant:
        command.extend(["--tile-size", str(item.variant["tile_size"]),
                        "--workgroup-size", str(item.variant["workgroup_size"]),
                        "--coordinate-memory", item.variant["coordinate_memory"]])
        if not item.variant["outer_shuffle"]:
            command.append("--no-outer-shuffle")
        if not item.variant["inner_shuffle"]:
            command.append("--no-inner-shuffle")
    return command


def source_files() -> list[Path]:
    """Only execution source/configuration; generated results and target trees excluded."""
    files = []
    for base in (REPO_ROOT / "implementations", REPO_ROOT / "shared"):
        if not base.exists():
            continue
        for directory, directories, names in os.walk(base):
            directories[:] = [name for name in directories if name not in {"target", ".git", "output", "node_modules"}]
            for name in names:
                path = Path(directory) / name
                if path.suffix in {".rs", ".wgsl", ".metal"} or name in {"Cargo.toml", "Cargo.lock"}:
                    files.append(path)
    for name in ("run_v2.py", "run_experiments.py", "experiment_plan.py", "artifact_paths.py", "record_metadata.py"):
        path = REPO_ROOT / "experiments" / name
        if path.exists():
            files.append(path)
    schema = REPO_ROOT / "experiments/schema/experiment-record-v1.json"
    if schema.exists():
        files.append(schema)
    return sorted(files)


def source_state() -> dict:
    hashes = {str(path.relative_to(REPO_ROOT)): common.file_sha256(path) for path in source_files()}
    return {"sha256": canonical_hash(hashes), "files": hashes}


def atomic_json(path: Path, value: Any):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)


def endpoint(item: VariantRun) -> dict:
    return {"binary_sha256": common.file_sha256(item.run.binary),
            "input_sha256": common.file_sha256(item.run.dataset.path),
            "source_tree_sha256": source_state()["sha256"]}


def freeze_evidence(folder: Path, items: list[VariantRun], resume: bool):
    binary_hashes = {str(binary): common.file_sha256(binary) for binary in sorted({item.run.binary for item in items})}
    sources = source_state()
    evidence_path = folder / "execution-integrity.json"
    evidence = {"binaries": binary_hashes, "source_tree": sources}
    if evidence_path.exists():
        previous = json.loads(evidence_path.read_text())
        if previous != evidence:
            raise common.RunnerError("Resume refused: execution binary or source tree changed")
    elif resume:
        raise common.RunnerError("Resume refused: missing execution-integrity snapshot")
    else:
        atomic_json(evidence_path, evidence)
        source_dir = folder / "source-snapshot"
        for name in sources["files"]:
            target = source_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO_ROOT / name).read_bytes())
        diff = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--binary", "HEAD"],
                              capture_output=True, text=True, check=True)
        (folder / "git-diff.patch").write_text(diff.stdout)
    return evidence


def validate_success(item: VariantRun, completed: subprocess.CompletedProcess) -> dict:
    if len(completed.stdout.splitlines()) != 1:
        raise common.RunnerError("Method stdout must contain exactly one JSON line")
    try:
        raw = json.loads(completed.stdout)
    except (ValueError, TypeError) as error:
        raise common.RunnerError("Method stdout is not valid JSON") from error
    if not isinstance(raw, dict):
        raise common.RunnerError("Method stdout JSON must be an object")
    rr_fields = {key: value for key, value in raw.items() if key.startswith("rr_")}
    if item.run.method != "rr_sgd" and rr_fields:
        raise common.RunnerError("RR metadata only permitted for rr_sgd")
    if item.variant:
        expected = {RR_META[key]: value for key, value in item.variant.items()}
        expected["rr_shared_bytes"] = 16 * item.variant["tile_size"] if item.variant["coordinate_memory"] == "shared" else 0
        if rr_fields != expected:
            raise common.RunnerError(f"RR metadata disagrees with requested variant: {rr_fields!r} vs {expected!r}")
    generic = {key: value for key, value in raw.items() if key not in rr_fields}
    sanitized = subprocess.CompletedProcess(completed.args, completed.returncode, json.dumps(generic) + "\n", completed.stderr)
    common.parse_and_validate_success(item.run, sanitized)
    if item.run.method in {"atomic_sgd", "atomic_sparse_sgd"}:
        constraints = raw["constraints"]
        if not isinstance(constraints, int) or isinstance(constraints, bool) or constraints < 0:
            raise common.RunnerError("Atomic constraints must be a nonnegative integer")
        expected_updates = constraints * item.run.iterations
        for field, expected in (("attempted_updates", expected_updates),
                                ("completed_updates", expected_updates), ("retry_failures", 0)):
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise common.RunnerError(f"Atomic update coverage incomplete: {field}={value!r}, expected={expected}")
    return raw


def enriched(record: dict, item: VariantRun, attempt: int, invocation: str,
             stdout_path: Path, stderr_path: Path, elapsed: float, integrity: dict) -> dict:
    return {**record, "experiment_id": item.run.experiment_id,
            "repetition": item.run.repetition, "variant_id": item.variant_id,
            "variant": item.variant, "attempt": attempt, "invocation_id": invocation,
            "stdout_log_path": str(stdout_path), "stderr_log_path": str(stderr_path),
            "wall_time_ms": elapsed, "integrity": integrity}


def execute(item: VariantRun, folder: Path, manifest: dict, evidence: dict,
            invocation: str, attempt: int, warmup: bool = False, warmup_index: int = 0) -> bool:
    run = item.run
    run_id = f"{run.run_id}-warmup-{invocation}-{warmup_index}" if warmup else run.run_id
    artifacts = folder / ("warmups" if warmup else "artifacts") / run_id / f"attempt-{attempt}"
    artifacts.mkdir(parents=True, exist_ok=False)
    stdout_path, stderr_path = artifacts / "stdout.json", artifacts / "stderr.log"
    command = command_for(item, artifacts, run_id)
    atomic_json(artifacts / "command.json", command)
    before = endpoint(item)
    wanted = {"binary_sha256": evidence["binaries"][str(run.binary)],
              "input_sha256": run.dataset.sha256,
              "source_tree_sha256": evidence["source_tree"]["sha256"]}
    if before != wanted:
        raise common.RunnerError("Input, binary, or execution source changed before run")
    start = time.perf_counter()
    completed = None
    try:
        completed = common.run_method(run, command, manifest["timeout_seconds"])
        common.write_stderr(stdout_path, completed.stdout)
        common.write_stderr(stderr_path, completed.stderr)
        # Warmups must satisfy the same output and configuration contract as formal runs.
        check = dataclasses.replace(item, run=dataclasses.replace(run, run_id=run_id))
        record = validate_success(check, completed)
        if not warmup:
            common.validate_paired_preprocessing(record, common.load_existing_records(folder / "results.jsonl"))
        success = True
    except subprocess.TimeoutExpired as error:
        common.write_stderr(stdout_path, error.stdout or "")
        common.write_stderr(stderr_path, error.stderr or "")
        record = common.failure_record(run, "warmup-timeout" if warmup else "timeout", str(error), None, stderr_path)
        success = False
    except common.RunnerError as error:
        stage = "warmup" if warmup else ("execute" if completed is not None and completed.returncode else "validate")
        record = common.failure_record(run, stage, str(error), completed.returncode if completed else None, stderr_path)
        success = False
    elapsed = (time.perf_counter() - start) * 1000
    after = endpoint(item)
    integrity = {"before": before, "after": after, "unchanged": before == after == wanted}
    if not integrity["unchanged"]:
        record = common.failure_record(run, "integrity", "Input, binary, or execution source changed during run", None, stderr_path)
        success = False
    record = enriched(record, item, attempt, invocation, stdout_path, stderr_path, elapsed, integrity)
    # Hash saved coordinate artifacts independently of input/source integrity.
    if success:
        record["artifacts_sha256"] = {key: common.file_sha256(common.resolve_record_path(record[key]))
                                     for key in ("final_positions_path", "vertex_map_path") if record.get(key)}
    atomic_json(artifacts / "record.json", record)
    common.append_record(folder / ("warmup-results.jsonl" if warmup else "results.jsonl"), record)
    if not integrity["unchanged"]:
        raise common.RunnerError("Execution integrity changed; stopped after preserving evidence")
    return success


def write_checkpoint(folder: Path, items: list[VariantRun], invocation: str):
    latest = {row["run_id"]: row for row in common.load_existing_records(folder / "results.jsonl")}
    success = {key for key, value in latest.items() if value["status"] == "success"}
    atomic_json(folder / "checkpoint.json", {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "invocation_id": invocation,
        "planned": len(items), "success": len(success),
        "failure": sum(row["status"] != "success" for row in latest.values()),
        "pending": [item.run_id for item in items if item.run_id not in success]})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, help="maximum pending formal runs this invocation")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--method", action="append")
    parser.add_argument("--variant", action="append")
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        path = args.manifest.resolve()
        manifest = load_manifest(path)
        items = expand_v2_runs(manifest, path)
        selected = [item for item in items
                    if (not args.dataset or item.run.dataset.name in args.dataset)
                    and (not args.method or item.run.method in args.method)
                    and (not args.variant or item.variant_id in args.variant)
                    and (not args.seed or item.run.seed in args.seed)]
        print(f"{manifest['experiment_id']}: planned={len(items)}, selected={len(selected)}", flush=True)
        if not selected:
            raise common.RunnerError("No runs match requested filters")
        if args.dry_run:
            for item in selected[:args.limit]:
                print(item.run_id, " ".join(command_for(item, Path("/dry-run") / item.run_id)))
            return 0
        common.verify_inputs_and_binaries([item.run for item in items])
        folder = common.prepare_output(manifest, path, args.results_dir.resolve() / "raw", args.resume)
        evidence = freeze_evidence(folder, items, args.resume)
        if not (folder / "run-plan.json").exists():
            atomic_json(folder / "run-plan.json", [{"run_id": item.run_id, "method": item.run.method,
                         "dataset": item.run.dataset.name, "seed": item.run.seed,
                         "iterations": item.run.iterations, "epsilon": item.run.epsilon,
                         "pivots": item.run.pivots, "repetition": item.run.repetition,
                         "variant_id": item.variant_id, "variant": item.variant}
                        for item in items])
        invocation = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        records = common.load_existing_records(folder / "results.jsonl")
        done = {row["run_id"] for row in records if row["status"] == "success"}
        attempts = {key: max(row.get("attempt", 0) for row in group)
                    for key, group in itertools.groupby(sorted(records, key=lambda row: row["run_id"]), lambda row: row["run_id"])}
        # Resume verifies that previously successful artifacts remain available and unchanged.
        for record in records:
            if record["status"] == "success":
                for key, expected in record.get("artifacts_sha256", {}).items():
                    artifact = common.resolve_record_path(record[key])
                    if not artifact.is_file() or common.file_sha256(artifact) != expected:
                        raise common.RunnerError(f"Saved artifact changed: {artifact}")
        warmed = set()
        failures, executed = 0, 0
        invocation_path = folder / "invocations" / (invocation + ".json")
        invocation_path.parent.mkdir(exist_ok=True)
        atomic_json(invocation_path, {"argv": sys.argv if argv is None else argv,
                                     "selected_runs": [item.run_id for item in selected],
                                     "execution_integrity_sha256": canonical_hash(evidence)})
        for item in selected:
            if item.run_id in done:
                continue
            if args.limit is not None and executed >= args.limit:
                break
            key = (item.run.dataset.name, item.run.method, item.variant_id)
            attempt = attempts.get(item.run_id, 0) + 1
            print(f"[{executed + 1}] {item.run.dataset.name} {item.run.method} {item.variant_id} seed={item.run.seed}", flush=True)
            warm_success = True
            if key not in warmed:
                for warmup_index in range(manifest["warmups"]):
                    warm_success = execute(item, folder, manifest, evidence, invocation, attempt, warmup=True, warmup_index=warmup_index)
                    if not warm_success:
                        break
                warmed.add(key)
            if warm_success:
                success = execute(item, folder, manifest, evidence, invocation, attempt)
            else:
                log = folder / "warmups" / f"{item.run_id}-warmup-{invocation}-{warmup_index}" / f"attempt-{attempt}" / "stderr.log"
                record = common.failure_record(item.run, "warmup", "Warmup failed; see warmup-results.jsonl", None, log)
                common.append_record(folder / "results.jsonl", enriched(record, item, attempt, invocation, log.with_name("stdout.json"), log, 0.0, {}))
                success = False
            executed += 1
            failures += not success
            if success:
                done.add(item.run_id)
            write_checkpoint(folder, items, invocation)
            print("success" if success else "FAILURE (saved raw evidence)", flush=True)
            if failures and manifest["fail_fast"]:
                break
        write_checkpoint(folder, items, invocation)
        print(f"Finished invocation: executed={executed}, failed={failures}, successful-total={len(done)}/{len(items)}", flush=True)
        return int(failures > 0)
    except (common.RunnerError, PlanError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
