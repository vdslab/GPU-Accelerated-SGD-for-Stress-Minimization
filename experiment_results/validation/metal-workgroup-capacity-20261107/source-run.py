"""Replay native Metal probes serially, retaining source and input provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
tool = ROOT / "tools/diagnostics/metal-workgroup-capacity"
jobs = [("sweep", ["sweep", "5", "4096"]), ("boundary", ["boundary", "5", "16384"])]
for name, _ in jobs:
    for extension in ["jsonl", "stderr"]:
        if (out / f"{name}.{extension}").exists():
            raise SystemExit(f"Refusing to overwrite {name}.{extension}")
files = [tool / name for name in ["benchmark", "benchmark.swift", "benchmark.metal", "analyze.py", "extract_counters.py"]]
metadata = {
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    "hardware": subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True),
    "commands": [],
}
for name, arguments in jobs:
    command = [str(tool / "benchmark"), str(tool / "benchmark.metal"), *arguments]
    print(f"Running {name}", flush=True)
    with (out / f"{name}.jsonl").open("x") as stdout, (out / f"{name}.stderr").open("x") as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, env=os.environ.copy())
    metadata["commands"].append({"argv": command, "exit_code": result.returncode})
    (out / "replay-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result.check_returncode()
