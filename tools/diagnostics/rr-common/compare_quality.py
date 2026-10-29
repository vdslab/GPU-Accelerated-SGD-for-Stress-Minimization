#!/usr/bin/env python3
"""Check saved probe runs against unchanged RR output; report quality regressions.

Usage: compare_quality.py AUDIT_DIR [--max-cpu-ratio 1.05 --v3-runs RUNS.csv]
CPU gating is only applied with identical initial/preprocess hashes and sampled
256-source stress; small-graph v3 exact stress must not be mixed with probes.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--v3-runs", type=Path)
    parser.add_argument("--max-cpu-ratio", type=float)
    args = parser.parse_args()
    if args.max_cpu_ratio is not None and args.v3_runs is None:
        parser.error("--max-cpu-ratio requires --v3-runs")
    references = {}
    if args.v3_runs:
        with args.v3_runs.open() as f:
            references = {
                (r["graph"], int(r["seed"])): r
                for r in csv.DictReader(f)
                if r["study"] == "basic" and r["method"] == "sparse_sgd"
                and r["status"] == "success" and r["pivots"] == "200"
            }
    failures = []
    results = []
    for old_path in sorted(args.audit_dir.glob("suite-*.jsonl")):
        old = json.loads(old_path.read_text().splitlines()[0])
        new_path = old_path.with_name(old_path.name.replace("suite-", "optimized-"))
        if not new_path.exists():
            failures.append(f"missing {new_path.name}")
            continue
        new = json.loads(new_path.read_text().splitlines()[0])
        for key in ("preprocess_sha256", "initial_sha256", "final_sha256", "stress"):
            if new[key] != old[key]:
                failures.append(f"{new_path.name}: {key} changed")
        graph = Path(new["graph"]).stem
        result = {"graph": graph, "seed": new["seed"], "exact_match": all(new[k] == old[k] for k in ("preprocess_sha256", "initial_sha256", "final_sha256", "stress"))}
        cpu = references.get((graph, new["seed"]))
        if cpu and cpu["stress_kind"] == "sampled" and cpu["stress_samples"] == "256" and cpu["stress_seed"] == "0":
            if cpu["preprocess_sha256"] != new["preprocess_sha256"] or cpu["initial_positions_sha256"] != new["initial_sha256"]:
                failures.append(f"{new_path.name}: CPU identity mismatch")
            else:
                ratio = new["stress"] / float(cpu["stress_value"])
                result["cpu_stress_ratio"] = ratio
                if args.max_cpu_ratio is not None and ratio > args.max_cpu_ratio:
                    failures.append(f"{new_path.name}: CPU ratio {ratio:.6f} > {args.max_cpu_ratio}")
        results.append(result)
    if not results:
        failures.append("no comparable probe runs")
    matched_cpu = sum("cpu_stress_ratio" in r for r in results)
    if args.max_cpu_ratio is not None and matched_cpu == 0:
        failures.append("no compatible CPU reference for requested quality gate")
    print(json.dumps({"results": results, "cpu_comparisons": matched_cpu, "failures": failures}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
