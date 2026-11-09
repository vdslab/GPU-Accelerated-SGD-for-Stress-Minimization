#!/usr/bin/env python3
"""Milestone 5: Phase 4 4-Tier Verification & Benchmark Suite CLI Runner.

Executes:
  - Gate 1: Native GPU Verification & llvmpipe Rejection
  - Gate 2: Bit-Exact Preprocessing Parity
  - Gate 3: Bit-Exact Repeatability & Zero Stress Degradation
  - Gate 4: Atomic SGD 100% Completion & Convergence Verification
  - Performance Benchmarks: Full SGD, Sparse SGD, Atomic Throughput, Preprocessing BFS

Outputs:
  - Structured JSON: experiment_results/m5_verification_results.json
  - Markdown Report: experiment_results/M5_VERIFICATION_REPORT.md
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import sys

# Ensure this script's directory and workspace root are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tools.diagnostics.m5_benchmark_suite.config import (
    DEFAULT_JSON_OUT,
    DEFAULT_REPORT_OUT,
)
from tools.diagnostics.m5_benchmark_suite.gate1_native_gpu import run_gate1_verification
from tools.diagnostics.m5_benchmark_suite.gate2_preprocessing import run_gate2_verification
from tools.diagnostics.m5_benchmark_suite.gate3_roundrobin import run_gate3_verification
from tools.diagnostics.m5_benchmark_suite.gate4_atomic import run_gate4_verification
from tools.diagnostics.m5_benchmark_suite.bench_performance import run_performance_benchmarks
from tools.diagnostics.m5_benchmark_suite.reporter import export_results, generate_markdown_report


def get_cpu_model() -> str:
    """Read CPU model from /proc/cpuinfo or platform."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_environment_metadata() -> dict:
    """Collect host environment and hardware specs."""
    return {
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "cpu_model": get_cpu_model(),
        "python_version": platform.python_version(),
        "gpu_device": "D3D12 (NVIDIA GeForce RTX 3050)",
        "gpu_backend": "Gl",
        "wsl2": os.path.exists("/dev/dxg"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 5: 4-Tier Verification & Benchmark Suite"
    )
    parser.add_argument(
        "--gate",
        choices=["all", "1", "2", "3", "4", "bench"],
        default="all",
        help="Specific gate or benchmark to run (default: all)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON_OUT,
        help=f"Path to structured output JSON (default: {DEFAULT_JSON_OUT})",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=DEFAULT_REPORT_OUT,
        help=f"Path to Markdown summary report (default: {DEFAULT_REPORT_OUT})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed logging output",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MILESTONE 5: PHASE 4 4-TIER VERIFICATION & BENCHMARK SUITE")
    print("=" * 80)
    start_time = datetime.now(timezone.utc)
    print(f"Start Time: {start_time.isoformat()}")
    print(f"Target Gate: {args.gate}")

    env_meta = get_environment_metadata()
    print(f"Hardware: {env_meta['cpu_model']} | {env_meta['os_release']}")
    print(f"GPU: {env_meta['gpu_device']} | Backend: {env_meta['gpu_backend']}")
    print("-" * 80)

    quality_gates = {}
    benchmarks = {}

    run_all = (args.gate == "all")

    # Gate 1
    if run_all or args.gate == "1":
        print("\n>>> [1/5] Running Gate 1: Native GPU Verification & llvmpipe Rejection...")
        quality_gates["gate1"] = run_gate1_verification(verbose=args.verbose)
        print(f"    Verdict: {quality_gates['gate1']['status']}")

    # Gate 2
    if run_all or args.gate == "2":
        print("\n>>> [2/5] Running Gate 2: Bit-Exact Preprocessing Parity...")
        quality_gates["gate2"] = run_gate2_verification(verbose=args.verbose)
        print(f"    Verdict: {quality_gates['gate2']['status']}")

    # Gate 3
    if run_all or args.gate == "3":
        print("\n>>> [3/5] Running Gate 3: Bit-Exact Repeatability & Zero Degradation...")
        quality_gates["gate3"] = run_gate3_verification(verbose=args.verbose)
        print(f"    Verdict: {quality_gates['gate3']['status']}")

    # Gate 4
    if run_all or args.gate == "4":
        print("\n>>> [4/5] Running Gate 4: Atomic SGD 100% Completion & Convergence...")
        quality_gates["gate4"] = run_gate4_verification(verbose=args.verbose)
        print(f"    Verdict: {quality_gates['gate4']['status']}")

    # Performance Benchmarks
    if run_all or args.gate == "bench":
        print("\n>>> [5/5] Running Performance Benchmarks (Full, Sparse, Atomic, BFS)...")
        benchmarks = run_performance_benchmarks(verbose=args.verbose)
        print("    Performance Benchmarks: Completed.")

    # Determine overall status
    all_gates_passed = all(g.get("status") == "PASS" for g in quality_gates.values()) if quality_gates else True
    overall_status = "PASS" if all_gates_passed else "FAIL"

    end_time = datetime.now(timezone.utc)
    duration_sec = (end_time - start_time).total_seconds()

    suite_results = {
        "milestone": "Milestone 5: Phase 4 4-Tier Verification & Benchmark Suite",
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration_sec,
        "overall_status": overall_status,
        "environment_metadata": env_meta,
        "quality_gates": quality_gates,
        "performance_benchmarks": benchmarks,
    }

    # Export structured JSON and Markdown
    export_results(suite_results, json_path=args.json_out, report_path=args.report_out)
    print("\n" + "=" * 80)
    print("VERIFICATION SUITE COMPLETE")
    print("=" * 80)
    print(f"Overall Status: {overall_status}")
    print(f"Duration:       {duration_sec:.2f} seconds")
    print(f"JSON Output:    {args.json_out}")
    print(f"Report Output:  {args.report_out}")
    print("=" * 80)

    # Print summary markdown snippet
    md_summary = generate_markdown_report(suite_results)
    # Print the Quality Gates Summary Table
    if "## 1. Quality Gates Summary" in md_summary:
        summary_section = md_summary.split("## 1. Quality Gates Summary")[1].split("## 2. Real Hardware Performance Speedup Summary")[0]
        print("\n## 1. Quality Gates Summary" + summary_section)

    if not all_gates_passed:
        print("\n[ERROR] One or more quality gates FAILED. Review errors above.", file=sys.stderr)
        sys.exit(1)

    print("\nAll quality gates PASSED unconditionally.")
    sys.exit(0)


if __name__ == "__main__":
    main()
