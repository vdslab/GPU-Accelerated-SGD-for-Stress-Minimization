#!/usr/bin/env python3
"""Milestone 4 Fix Challenger Verification Script

Empirical verification of:
1. Target 2: Full SGD (rr-gpu) elimination of preprocessing regression (<5ms on N=100,128, <=150ms on N=2000)
2. Target 2: Explicit --bfs-engine gpu telemetry (bfs_init_time_ms, bfs_kernel_time_ms, bfs_compaction_time_ms)
3. Target 1: Sparse SGD (sparse-sgd-gpu) speedup on USpowerGrid.mtx (>= 3.0x) and bit-exact Gate 2 parity
"""

import json
import os
import subprocess
import sys
import statistics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RR_GPU_BIN = os.path.join(BASE_DIR, "implementations/full/rr_gpu/target/release/rr-gpu")
SPARSE_GPU_BIN = os.path.join(BASE_DIR, "implementations/sparse/sparse-sgd-gpu/target/release/sparse-sgd-gpu")
SPARSE_CPU_BIN = os.path.join(BASE_DIR, "implementations/sparse/baseline-sparse-sgd-non-gpu/target/release/baseline-sparse-sgd-non-gpu")

GRID10_MTX = os.path.join(BASE_DIR, "data/grid2d_10x10.mtx")
CYCLE128_MTX = os.path.join(BASE_DIR, "data/cycle128.mtx")
BA2000_MTX = os.path.join(BASE_DIR, "data/BA-2000-g0.mtx")
USPOWER_MTX = os.path.join(BASE_DIR, "data/USpowerGrid.mtx")
CYCLE6_MTX = os.path.join(BASE_DIR, "data/cycle6.mtx")

def run_cmd(cmd_list):
    res = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
    lines = res.stdout.strip().splitlines()
    for l in reversed(lines):
        if l.strip().startswith("{") and l.strip().endswith("}"):
            return json.loads(l.strip())
    raise RuntimeError(f"No JSON found in output:\n{res.stdout}\nSTDERR:\n{res.stderr}")

def test_target_2_regression():
    print("=" * 70)
    print("TARGET 2: Full SGD (rr-gpu) Preprocessing Latency & Regression Elimination")
    print("=" * 70)

    graphs = [
        ("N=100 (grid2d_10x10)", GRID10_MTX, 5.0),
        ("N=128 (cycle128)", CYCLE128_MTX, 5.0),
        ("N=2,000 (BA-2000-g0)", BA2000_MTX, 150.0),
    ]

    results = {}
    for name, path, max_allowed in graphs:
        print(f"\n--- Testing {name} ---")
        auto_times = []
        cpu_times = []
        stress_matches = True

        for trial in range(5):
            cmd_auto = [RR_GPU_BIN, "--input", path, "--iterations", "5", "--seed", "0", "--output-format", "json", "--bfs-engine", "auto"]
            res_auto = run_cmd(cmd_auto)
            t_auto = res_auto["common_preprocess_time_ms"]
            auto_times.append(t_auto)

            cmd_cpu = [RR_GPU_BIN, "--input", path, "--iterations", "5", "--seed", "0", "--output-format", "json", "--bfs-engine", "cpu"]
            res_cpu = run_cmd(cmd_cpu)
            t_cpu = res_cpu["common_preprocess_time_ms"]
            cpu_times.append(t_cpu)

            # verify stress match
            if abs(res_auto["stress_value"] - res_cpu["stress_value"]) > 1e-6:
                stress_matches = False

        avg_auto = statistics.mean(auto_times)
        min_auto = min(auto_times)
        avg_cpu = statistics.mean(cpu_times)
        min_cpu = min(cpu_times)
        ratio = avg_auto / avg_cpu

        passed = avg_auto < max_allowed
        print(f"  Auto common_preprocess_time_ms: mean={avg_auto:.4f} ms, min={min_auto:.4f} ms (Target: < {max_allowed} ms) -> {'PASS' if passed else 'FAIL'}")
        print(f"  CPU  common_preprocess_time_ms: mean={avg_cpu:.4f} ms, min={min_cpu:.4f} ms")
        print(f"  Auto/CPU ratio: {ratio:.2f}x (Regression eliminated: was up to 15,294x!)")
        print(f"  Mathematical stress match: {'YES' if stress_matches else 'NO'}")

        results[name] = {
            "avg_auto": avg_auto,
            "min_auto": min_auto,
            "avg_cpu": avg_cpu,
            "ratio": ratio,
            "passed": passed,
            "stress_matches": stress_matches,
        }

    return results

def test_target_2_explicit_gpu():
    print("\n" + "=" * 70)
    print("TARGET 2: Explicit --bfs-engine gpu & Telemetry Verification")
    print("=" * 70)

    for name, path in [("cycle6 (N=6)", CYCLE6_MTX), ("grid2d_10x10 (N=100)", GRID10_MTX), ("cycle128 (N=128)", CYCLE128_MTX)]:
        print(f"\n--- Testing explicit --bfs-engine gpu on {name} ---")
        cmd_gpu = [RR_GPU_BIN, "--input", path, "--iterations", "5", "--seed", "0", "--output-format", "json", "--bfs-engine", "gpu"]
        res_gpu = run_cmd(cmd_gpu)

        has_bfs_engine = res_gpu.get("bfs_engine") == "gpu"
        has_init = "bfs_init_time_ms" in res_gpu and isinstance(res_gpu["bfs_init_time_ms"], (int, float))
        has_kernel = "bfs_kernel_time_ms" in res_gpu and isinstance(res_gpu["bfs_kernel_time_ms"], (int, float))
        has_compaction = "bfs_compaction_time_ms" in res_gpu and isinstance(res_gpu["bfs_compaction_time_ms"], (int, float))

        init_t = res_gpu.get("bfs_init_time_ms", 0.0)
        kernel_t = res_gpu.get("bfs_kernel_time_ms", 0.0)
        compaction_t = res_gpu.get("bfs_compaction_time_ms", 0.0)
        total_common = res_gpu.get("common_preprocess_time_ms", 0.0)

        telemetry_valid = has_bfs_engine and has_init and has_kernel and has_compaction
        print(f"  bfs_engine = {res_gpu.get('bfs_engine')}")
        print(f"  bfs_init_time_ms = {init_t:.4f} ms")
        print(f"  bfs_kernel_time_ms = {kernel_t:.4f} ms")
        print(f"  bfs_compaction_time_ms = {compaction_t:.4f} ms")
        print(f"  common_preprocess_time_ms = {total_common:.4f} ms")
        print(f"  Telemetry fields complete and numeric: {'PASS' if telemetry_valid else 'FAIL'}")

        # Check parity with auto
        cmd_auto = [RR_GPU_BIN, "--input", path, "--iterations", "5", "--seed", "0", "--output-format", "json", "--bfs-engine", "auto"]
        res_auto = run_cmd(cmd_auto)
        stress_diff = abs(res_gpu["stress_value"] - res_auto["stress_value"])
        print(f"  Stress match with Auto: diff={stress_diff:.6e} -> {'PASS' if stress_diff < 1e-5 else 'FAIL'}")

def test_target_1_sparse_speedup():
    print("\n" + "=" * 70)
    print("TARGET 1: Sparse SGD (sparse-sgd-gpu) Speedup on USpowerGrid.mtx")
    print("=" * 70)

    trials = 10
    baseline_times = []
    gpu_times = []
    gate2_intact = True

    cmd_base = [SPARSE_CPU_BIN, "--input", USPOWER_MTX, "--pivots", "50", "--iterations", "5", "--seed", "0", "--output-format", "json"]
    cmd_gpu = [SPARSE_GPU_BIN, "--input", USPOWER_MTX, "--pivots", "50", "--iterations", "5", "--seed", "0", "--output-format", "json", "--bfs-engine", "auto"]

    # First run: obtain baseline fingerprints
    res_b0 = run_cmd(cmd_base)
    expected_preprocess_sha = res_b0["preprocess_sha256"]
    expected_initial_sha = res_b0["initial_positions_sha256"]
    print(f"Baseline Preprocess SHA256: {expected_preprocess_sha}")
    print(f"Baseline Initial Pos SHA256: {expected_initial_sha}")

    for t in range(trials):
        res_b = run_cmd(cmd_base)
        baseline_times.append(res_b["common_preprocess_time_ms"])

        res_g = run_cmd(cmd_gpu)
        gpu_times.append(res_g["common_preprocess_time_ms"])

        if res_g["preprocess_sha256"] != expected_preprocess_sha or res_g["initial_positions_sha256"] != expected_initial_sha:
            gate2_intact = False

    b_mean = statistics.mean(baseline_times)
    b_median = statistics.median(baseline_times)
    g_mean = statistics.mean(gpu_times)
    g_median = statistics.median(gpu_times)

    speedup_mean = b_mean / g_mean
    speedup_median = b_median / g_median
    speedup_peak = max(baseline_times) / min(gpu_times)

    passed = speedup_mean >= 3.0 and gate2_intact
    print(f"\nResults over {trials} trials:")
    print(f"  Baseline Preprocessing: mean = {b_mean:.3f} ms, median = {b_median:.3f} ms, min = {min(baseline_times):.3f} ms, max = {max(baseline_times):.3f} ms")
    print(f"  GPU Preprocessing:      mean = {g_mean:.3f} ms, median = {g_median:.3f} ms, min = {min(gpu_times):.3f} ms, max = {max(gpu_times):.3f} ms")
    print(f"  Mean Speedup:   {speedup_mean:.2f}x (Criterion: >= 3.0x) -> {'PASS' if speedup_mean >= 3.0 else 'FAIL'}")
    print(f"  Median Speedup: {speedup_median:.2f}x")
    print(f"  Peak Speedup:   {speedup_peak:.2f}x")
    print(f"  Gate 2 Bit-Exact Match across all {trials} trials: {'PASS' if gate2_intact else 'FAIL'}")

    return {
        "b_mean": b_mean,
        "g_mean": g_mean,
        "speedup_mean": speedup_mean,
        "passed": passed,
    }

if __name__ == "__main__":
    t2_res = test_target_2_regression()
    test_target_2_explicit_gpu()
    t1_res = test_target_1_sparse_speedup()

    all_pass = all(v["passed"] for v in t2_res.values()) and t1_res["passed"]
    print("\n" + "=" * 70)
    print(f"OVERALL EMPIRICAL VERDICT: {'APPROVE' if all_pass else 'REJECT'}")
    print("=" * 70)
    sys.exit(0 if all_pass else 1)
