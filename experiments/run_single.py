#!/usr/bin/env python3
"""Run one graph and save settings, timings, coordinates, and before/after PNGs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo

try:
    from experiment_plan import METHODS, REPO_ROOT, Dataset, RunDefinition
    from run_experiments import (
        RunnerError, command_for, file_sha256, parse_and_validate_success,
    )
except ModuleNotFoundError:
    from experiments.experiment_plan import METHODS, REPO_ROOT, Dataset, RunDefinition
    from experiments.run_experiments import (
        RunnerError, command_for, file_sha256, parse_and_validate_success,
    )

TOKYO = ZoneInfo("Asia/Tokyo")
RENDERER = "tools/gpu_visualizer/target/release/gpu-visualizer"
TIMINGS = {
    "iteration_time_ms": "Iteration",
    "algorithm_time_cold_ms": "Algorithm (cold)",
    "algorithm_time_warm_ms": "Algorithm (warm)",
    "cli_total_time_cold_ms": "CLI total (cold)",
}


class SingleRunError(RuntimeError):
    pass


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("正の有限数を指定してください")
    return parsed


def seed_value(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 2**64 - 1:
        raise argparse.ArgumentTypeError("seedは0以上のu64整数です")
    return parsed


def image_size(value: str) -> str:
    pieces = value.lower().split("x")
    if len(pieces) == 1:
        pieces *= 2
    try:
        if len(pieces) != 2 or any(not 1 <= int(p) <= 8192 for p in pieces):
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError("画像サイズは1..8192の整数またはWxHです") from error
    return "x".join(str(int(p)) for p in pieces)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="MTX入力")
    parser.add_argument("--method", choices=METHODS, default="rr_sparse_sgd")
    parser.add_argument("--iterations", type=positive_int, default=15)
    parser.add_argument("--epsilon", type=positive_float, default=0.1)
    parser.add_argument("--pivots", type=positive_int, default=200, help="Sparse手法のみ")
    parser.add_argument("--seed", type=seed_value, default=0)
    parser.add_argument("--image-size", type=image_size, default="2048x2048", help="整数またはWxH (default: 2048)")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output")
    return parser.parse_args(argv)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("._-") or "graph"


def create_run_dir(output_root: Path, graph: str, timestamp: str, method: str) -> Path:
    parent = output_root.resolve() / safe_name(graph)
    parent.mkdir(parents=True, exist_ok=True)
    base = f"{timestamp}_{safe_name(method)}"
    index = 1
    while True:
        destination = parent / (base if index == 1 else f"{base}_{index}")
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            index += 1


def save_json(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def log_tail(path: Path) -> str:
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        source.seek(max(0, source.tell() - 8192))
        return source.read().decode("utf-8", errors="replace").strip()


def logged_command(command: list[str], log: Path, *, environment: dict[str, str] | None = None) -> None:
    with log.open("w", encoding="utf-8") as output:
        completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        detail = log_tail(log)
        raise SingleRunError(f"exit code {completed.returncode}: {detail}")


def build(binary: Path, log: Path) -> None:
    crate = binary.parent.parent.parent
    command = ["cargo", "build", "--release", "--locked", "--manifest-path", str(crate / "Cargo.toml"), "--target-dir", str(crate / "target")]
    print(f"Build: {crate.relative_to(REPO_ROOT)}", flush=True)
    logged_command(command, log)
    if not binary.is_file():
        raise SingleRunError(f"build後に実行ファイルがありません: {binary}")


def generated_path(value: str, run_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(run_dir) or not path.is_file():
        raise SingleRunError(f"出力が実行フォルダに存在しません: {path}")
    return path


def layout_paths(record: dict[str, Any], working_dir: Path) -> dict[str, Path]:
    final = generated_path(record["final_positions_path"], working_dir)
    if not final.name.endswith(".final.txt"):
        raise SingleRunError(f"final出力名が不正です: {final.name}")
    initial = generated_path(str(final.with_name(final.name[:-len(".final.txt")] + ".initial.txt")), working_dir)
    return {"before": initial, "after": final}


def write_config(output: TextIO, config: dict[str, Any]) -> None:
    output.write("Experiment settings\n")
    for key, value in config.items():
        output.write(f"  {key}: {value}\n")


def write_speed(output: TextIO, record: dict[str, Any]) -> None:
    output.write("Speed (SGD; build, render, and stress evaluation excluded)\n")
    for field, label in TIMINGS.items():
        value = record.get(field)
        output.write(f"  {label}: {value:.3f} ms\n" if value is not None else f"  {label}: n/a\n")
    if record.get("stress_value") is not None:
        output.write(f"Stress: {record['stress_value']} ({record.get('stress_kind')})\n")


def append_layout(output: TextIO, path: Path, record: dict[str, Any]) -> None:
    output.write("\n# Edges (source target; 0-based vertex IDs)\n")
    section = None
    edges = positions = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if stripped == "# Edges (source target)":
                section = "edges"
                continue
            if stripped == "# Positions (x y)":
                section = "positions"
                output.write("\n# Positions (x y; after SGD; line index = 0-based vertex ID)\n")
                continue
            if not stripped or stripped.startswith("#"):
                continue
            if section == "edges":
                edges += 1
            elif section == "positions":
                positions += 1
            else:
                continue
            output.write(line if line.endswith("\n") else line + "\n")
    if edges != record["edges"] or positions != record["nodes"]:
        raise SingleRunError(f"座標ファイルの件数が不一致です: edges={edges}/{record['edges']}, positions={positions}/{record['nodes']}")


def write_info(path: Path, metadata: dict[str, Any], layout: Path | None = None) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write(f"Status: {metadata['status']}\n\n")
        write_config(output, metadata["config"])
        record = metadata.get("record")
        if record:
            output.write(f"\nGraph: nodes={record['nodes']}, edges={record['edges']}, constraints={record.get('constraints')}\n")
            output.write(f"# Node count: {record['nodes']}\n# Edge count: {record['edges']}\n")
            write_speed(output, record)
        if metadata.get("error"):
            output.write(f"\nError stage: {metadata.get('stage')}\n{metadata['error']}\n")
        output.write("\nFiles\n")
        for label, value in metadata["artifacts"].items():
            if value is not None:
                output.write(f"  {label}: {value}\n")
        if record and layout:
            append_layout(output, layout, record)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1
    started = dt.datetime.now(TOKYO)
    timestamp = started.strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = create_run_dir(args.output_root.expanduser(), input_path.stem, timestamp, args.method)
    prefix = run_dir / timestamp
    result_path = prefix.with_name(f"{timestamp}_result.json")
    info_path = prefix.with_name(f"{timestamp}_info.txt")
    config = {"input": str(input_path), "method": args.method, "iterations": args.iterations, "epsilon": args.epsilon, "pivots": args.pivots if METHODS[args.method].sparse else None, "seed": args.seed, "image_size": args.image_size, "warmups": 0}
    metadata: dict[str, Any] = {"schema_version": 1, "kind": "single_run", "status": "running", "started_at": started.isoformat(), "run_directory": str(run_dir), "config": config, "record": None, "artifacts": {"info": str(info_path), "result": str(result_path)}, "stage": "prepare", "error": None}
    save_json(result_path, metadata)
    write_info(info_path, metadata)
    write_config(sys.stdout, config)
    print(f"Output: {run_dir}", flush=True)
    code = 1
    # Native coordinate files and logs are temporary; only four final artifacts remain.
    with tempfile.TemporaryDirectory(prefix=".work-", dir=run_dir) as temporary:
        working_dir = Path(temporary)
        layouts: dict[str, Path] = {}
        try:
            binary = (REPO_ROOT / METHODS[args.method].binary).resolve()
            renderer = (REPO_ROOT / RENDERER).resolve()
            for label, target in (("method", binary), ("renderer", renderer)):
                metadata["stage"] = f"build_{label}"
                build(target, working_dir / f"build-{label}.log")
            metadata["stage"] = "sgd"
            run = RunDefinition("single", run_dir.name, "benchmark", args.method, binary, Dataset(input_path.stem, input_path, file_sha256(input_path)), args.seed, args.iterations, args.epsilon, config["pivots"], 0)
            environment = os.environ.copy()
            environment["EXPERIMENT_METHOD"] = args.method
            completed = subprocess.run(command_for(run, working_dir), cwd=REPO_ROOT, env=environment, capture_output=True, text=True, check=False)
            if completed.returncode:
                raise SingleRunError(f"SGD exit code {completed.returncode}: {completed.stderr[-8192:].strip()}")
            record = parse_and_validate_success(run, completed)
            metadata["record"] = record
            write_speed(sys.stdout, record)
            metadata["stage"] = "artifacts"
            layouts = layout_paths(record, working_dir)
            record["final_positions_path"] = str(info_path)
            record["vertex_map_path"] = None
            for index, stage in enumerate(("before", "after")):
                metadata["stage"] = f"render_{stage}"
                image = run_dir / f"{timestamp}_{index}.png"
                command = [str(renderer), str(layouts[stage]), str(image), "--size", args.image_size, "--stress", "off"]
                logged_command(command, working_dir / f"render-{stage}.log")
                if not image.is_file() or image.stat().st_size == 0:
                    raise SingleRunError(f"画像が生成されませんでした: {image}")
                metadata["artifacts"][f"{stage}_image"] = str(image)
            metadata.update(status="success", stage="complete")
            code = 0
        except KeyboardInterrupt:
            metadata.update(status="interrupted", error="Interrupted")
            code = 130
        except (OSError, ValueError, KeyError, RunnerError, SingleRunError) as error:
            metadata.update(status="failure", error=str(error))
            print(f"Failed ({metadata['stage']}): {error}", file=sys.stderr)
        finally:
            metadata["finished_at"] = dt.datetime.now(TOKYO).isoformat()
            try:
                write_info(info_path, metadata, layouts.get("after"))
            except (OSError, SingleRunError) as error:
                metadata.update(status="failure", stage="info", error=str(error))
                write_info(info_path, metadata)
                print(f"Failed (info): {error}", file=sys.stderr)
                code = 1
            save_json(result_path, metadata)
    if code == 0:
        print(f"Saved: {run_dir}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
