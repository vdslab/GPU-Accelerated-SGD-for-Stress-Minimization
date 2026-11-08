"""Replay the tile cause comparisons serially into a new output directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
binary = ROOT / "implementations/full/rr_gpu/target/release/examples/tile_cause"
jobs = [
    ("3elt", ["data/v5/common/3elt.mtx", "3", "10"], {"SGD_GPU_TIMESTAMPS": "1"}),
    ("fe_4elt2", ["data/v5/common/fe_4elt2.mtx", "3", "10"], {"SGD_GPU_TIMESTAMPS": "1"}),
    ("micro-controls", ["micro"], {"SGD_GPU_TIMESTAMPS": "1"}),
    ("3elt-no-timestamps", ["data/v5/common/3elt.mtx", "5", "10"],
     {"SGD_GPU_TIMESTAMPS": "0", "TILE_CAUSE_FILTER": "baseline"}),
]
for name, _, _ in jobs:
    for suffix in [".jsonl", ".stderr"]:
        if (output / (name + suffix)).exists():
            raise SystemExit(f"Refusing to overwrite {name + suffix}")
paths = [binary, ROOT / "implementations/full/rr_gpu/examples/tile_cause.rs",
         ROOT / "implementations/full/rr_gpu/src/gpu.rs",
         ROOT / "implementations/full/rr_gpu/src/shader.wgsl",
         ROOT / "shared/gpu-timing/wgpu.rs",
         ROOT / "implementations/full/rr_gpu/Cargo.lock",
         ROOT / "data/v5/common/3elt.mtx", ROOT / "data/v5/common/fe_4elt2.mtx"]
provenance = {
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "jobs": [{"name": name, "command": [str(binary), *argv], "env": env}
             for name, argv, env in jobs],
    "status": {},
}
(output / "replay-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
for name, argv, updates in jobs:
    env = os.environ.copy()
    env.pop("TILE_CAUSE_FILTER", None)
    env.update(updates)
    print(f"Running {name}", flush=True)
    with (output / (name + ".jsonl")).open("x") as stdout, (output / (name + ".stderr")).open("x") as stderr:
        result = subprocess.run([str(binary), *argv], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    provenance["status"][name] = result.returncode
    (output / "replay-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    result.check_returncode()
