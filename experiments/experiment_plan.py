"""Pure manifest planning shared by the experiment runner and aggregator."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclasses.dataclass(frozen=True)
class MethodSpec:
    binary: str
    family: str
    sparse: bool


METHODS: dict[str, MethodSpec] = {
    "sgd": MethodSpec(
        "baseline-sgd-non-gpu/target/release/baseline-sgd-non-gpu", "full", False
    ),
    "atomic_sgd": MethodSpec(
        "vram-lock-native/target/release/vram-lock-native", "full", False
    ),
    "rr_sgd": MethodSpec("rr_gpu/target/release/rr-gpu", "full", False),
    "sparse_sgd": MethodSpec(
        "baseline-sparse-sgd-non-gpu/target/release/baseline-sparse-sgd-non-gpu",
        "sparse",
        True,
    ),
    "rr_sparse_sgd": MethodSpec(
        "sparse-sgd-gpu/target/release/sparse-sgd-gpu", "sparse", True
    ),
}


@dataclasses.dataclass(frozen=True)
class Dataset:
    name: str
    path: Path
    sha256: str


@dataclasses.dataclass(frozen=True)
class RunDefinition:
    experiment_id: str
    run_id: str
    run_mode: str
    method: str
    binary: Path
    dataset: Dataset
    seed: int
    iterations: int
    epsilon: float
    pivots: int | None
    repetition: int

    @property
    def family(self) -> str:
        return METHODS[self.method].family

    @property
    def paired_key(self) -> tuple[Any, ...]:
        """Condition key used to pair a method with its family baseline."""
        return (
            self.experiment_id,
            self.dataset.name,
            self.dataset.sha256,
            self.family,
            self.run_mode,
            self.seed,
            self.repetition,
            self.iterations,
            self.epsilon,
            self.pivots,
        )


class PlanError(RuntimeError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"manifestを読めません: {path}: {error}") from error
    if not isinstance(value, dict):
        raise PlanError("manifest rootはJSON objectである必要があります")
    validate_manifest(value)
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "experiment_id",
        "run_mode",
        "methods",
        "datasets",
        "seeds",
        "iterations",
        "epsilon",
        "pivots",
        "repetitions",
        "warmups",
        "timeout_seconds",
        "fail_fast",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise PlanError(f"manifestの必須fieldがありません: {', '.join(missing)}")
    experiment_id = manifest["experiment_id"]
    if not isinstance(experiment_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id
    ):
        raise PlanError("experiment_idには英数字、-、_、.だけを使用してください")
    if manifest["run_mode"] not in {"benchmark", "diagnostic"}:
        raise PlanError("run_modeはbenchmarkまたはdiagnosticです")
    methods = require_nonempty_list(manifest, "methods")
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise PlanError(
            "未実装または未知のmethodです: "
            + ", ".join(str(method) for method in unknown)
        )
    if len(set(methods)) != len(methods):
        raise PlanError("methodsに重複があります")
    datasets = require_nonempty_list(manifest, "datasets")
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise PlanError("datasetsの各要素はobjectです")
        if not all(
            isinstance(dataset.get(field), str) and dataset[field]
            for field in ("name", "path", "sha256")
        ):
            raise PlanError("datasetにはname、path、sha256が必要です")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", dataset["sha256"]):
            raise PlanError(f"dataset {dataset['name']} のsha256が不正です")
    for value in require_nonempty_list(manifest, "seeds"):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PlanError("seedsは0以上のinteger配列です")
    for value in require_nonempty_list(manifest, "iterations"):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PlanError("iterationsは1以上のinteger配列です")
    for value in require_nonempty_list(manifest, "epsilon"):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise PlanError("epsilonは正の有限数の配列です")
    for value in require_nonempty_list(manifest, "pivots"):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PlanError("pivotsは1以上のinteger配列です")
    for field, minimum in (("repetitions", 1), ("warmups", 0), ("timeout_seconds", 1)):
        value = manifest[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise PlanError(f"{field}は{minimum}以上のintegerです")
    if not isinstance(manifest["fail_fast"], bool):
        raise PlanError("fail_fastはbooleanです")
    binaries = manifest.get("binaries", {})
    if not isinstance(binaries, dict) or any(method not in methods for method in binaries):
        raise PlanError("binaries overrideは選択済みmethodのobjectです")


def require_nonempty_list(manifest: dict[str, Any], field: str) -> list[Any]:
    value = manifest.get(field)
    if not isinstance(value, list) or not value:
        raise PlanError(f"{field}は空でない配列です")
    return value


def resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    repo_path = (REPO_ROOT / path).resolve()
    if repo_path.exists():
        return repo_path
    return (manifest_path.parent / path).resolve()


def expand_runs(
    manifest: dict[str, Any], manifest_path: Path
) -> list[RunDefinition]:
    datasets = [
        Dataset(
            name=entry["name"],
            path=resolve_path(entry["path"], manifest_path),
            sha256=entry["sha256"].lower(),
        )
        for entry in manifest["datasets"]
    ]
    binary_overrides = manifest.get("binaries", {})
    runs: list[RunDefinition] = []
    for method in manifest["methods"]:
        method_spec = METHODS[method]
        binary_value = binary_overrides.get(method, method_spec.binary)
        binary = resolve_path(binary_value, manifest_path)
        pivot_values: Iterable[int | None] = (
            manifest["pivots"] if method_spec.sparse else [None]
        )
        for dataset, seed, iterations, epsilon, pivots, repetition in itertools.product(
            datasets,
            manifest["seeds"],
            manifest["iterations"],
            manifest["epsilon"],
            pivot_values,
            range(manifest["repetitions"]),
        ):
            condition = {
                "experiment_id": manifest["experiment_id"],
                "method": method,
                "dataset": dataset.name,
                "input_sha256": dataset.sha256,
                "seed": seed,
                "iterations": iterations,
                "epsilon": float(epsilon),
                "pivots": pivots,
                "repetition": repetition,
                "run_mode": manifest["run_mode"],
            }
            digest = hashlib.sha256(
                json.dumps(
                    condition, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()[:12]
            run_id = (
                f"{manifest['experiment_id']}-{slug(dataset.name)}-"
                f"{method}-{digest}"
            )
            runs.append(
                RunDefinition(
                    experiment_id=manifest["experiment_id"],
                    run_id=run_id,
                    run_mode=manifest["run_mode"],
                    method=method,
                    binary=binary,
                    dataset=dataset,
                    seed=seed,
                    iterations=iterations,
                    epsilon=float(epsilon),
                    pivots=pivots,
                    repetition=repetition,
                )
            )
    if len({run.run_id for run in runs}) != len(runs):
        raise PlanError("run IDが重複しました")
    return runs


def slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return normalized or "dataset"
