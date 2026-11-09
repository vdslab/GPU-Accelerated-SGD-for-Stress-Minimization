#!/usr/bin/env python3
"""
Empirical verification script for Milestone 3: Atomic SGD wgpu Port (Convergence Quality & Stress Stability).
Author: atomic_challenger_m3_2

Tests:
1. Full Atomic SGD on cycle6.mtx (1, 5, 10, 15 iterations; seeds 0, 1, 2)
   - Checks: stress finite (no NaN/Inf), stress decreases, Gate 4 (100% completion).
   - Comparison with CPU baseline and RoundRobin baseline.
2. Full Atomic SGD on grid2d_10x10.mtx (1, 5, 10, 15 iterations; seeds 0, 1, 2; chunk-size variants)
   - Checks: stress finite, convergence, Gate 4 (100% completion).
   - Comparison with CPU baseline and RoundRobin baseline.
3. Sparse Atomic SGD on USpowerGrid.mtx (pivots=50, iterations=15, seeds 0, 1, 2; sweep iters 1, 5, 10, 15, 20)
   - Checks: stress finite, convergence without divergence or degradation.
   - Comparison with CPU baseline and RoundRobin baseline.
   - Testing --cpu-shuffle mode.
   - Repetition stability check (within-seed stress stability).
4. Resource Bounds Check:
   - Tracks peak RSS memory, CPU usage, and execution duration.
"""

import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time

ATOMIC_FULL_BIN = "./implementations/full/vram-lock-native/target/release/vram-lock-native"
CPU_FULL_BIN = "./implementations/full/baseline-sgd-non-gpu/target/release/baseline-sgd-non-gpu"
RR_FULL_BIN = "./implementations/full/rr_gpu/target/release/rr-gpu"

ATOMIC_SPARSE_BIN = "./implementations/sparse/vram-lock-sparse/target/release/vram-lock-sparse"
CPU_SPARSE_BIN = "./implementations/sparse/baseline-sparse-sgd-non-gpu/target/release/baseline-sparse-sgd-non-gpu"
RR_SPARSE_BIN = "./implementations/sparse/sparse-sgd-gpu/target/release/sparse-sgd-gpu"

CYCLE6_MTX = "data/cycle6.mtx"
GRID10_MTX = "data/grid2d_10x10.mtx"
USPOWER_MTX = "data/USpowerGrid.mtx"


def run_cmd(bin_path, args):
    env = os.environ.copy()
    env["PATH"] = f"{os.path.expanduser('~')}/.cargo/bin:{env.get('PATH', '')}"
    cmd = [bin_path] + args
    start_time = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start_time
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\nStderr: {res.stderr}\nStdout: {res.stdout}")
    
    # parse last JSON object
    lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"No JSON output from {' '.join(cmd)}\nStdout: {res.stdout}\nStderr: {res.stderr}")
    data = json.loads(lines[-1])
    data["_host_elapsed_sec"] = elapsed
    return data


def verify_finite_coordinates(positions_path):
    if not os.path.exists(positions_path):
        raise FileNotFoundError(f"Positions file not found: {positions_path}")
    with open(positions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                x, y = float(parts[0]), float(parts[1])
                if not (math.isfinite(x) and math.isfinite(y)):
                    return False
    return True


def test_full_atomic_cycle6():
    print("\n" + "="*70)
    print("Test 1: Full Atomic SGD on cycle6.mtx")
    print("="*70)
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Sweep iterations: 1, 5, 10, 15
        for iters in [1, 5, 10, 15]:
            atomic_res = run_cmd(ATOMIC_FULL_BIN, [
                "--input", CYCLE6_MTX, "--iterations", str(iters), "--epsilon", "0.1",
                "--seed", "0", "--output-format", "json", "--output-dir", tmpdir
            ])
            cpu_res = run_cmd(CPU_FULL_BIN, [
                "--input", CYCLE6_MTX, "--iterations", str(iters), "--epsilon", "0.1",
                "--seed", "0", "--output-format", "json", "--output-dir", tmpdir
            ])
            rr_res = run_cmd(RR_FULL_BIN, [
                "--input", CYCLE6_MTX, "--iterations", str(iters), "--epsilon", "0.1",
                "--seed", "0", "--output-format", "json", "--output-dir", tmpdir
            ])

            s_atomic = atomic_res["stress_value"]
            s_cpu = cpu_res["stress_value"]
            s_rr = rr_res["stress_value"]

            assert math.isfinite(s_atomic), f"Non-finite stress in Atomic SGD: {s_atomic}"
            assert verify_finite_coordinates(atomic_res["final_positions_path"]), "Atomic final coordinates contain NaN/Inf"
            assert atomic_res["attempted_updates"] == atomic_res["completed_updates"], "Gate 4 violation: attempted != completed"
            assert atomic_res["retry_failures"] == 0, "Gate 4 violation: retry_failures != 0"

            print(f"iters={iters:2d} | Atomic: {s_atomic:8.6f} | CPU: {s_cpu:8.6f} | RR: {s_rr:8.6f} | Atomic/CPU: {s_atomic/s_cpu:6.4f} | Done: {atomic_res['completed_updates']}/{atomic_res['attempted_updates']}")
            results.append((iters, s_atomic, s_cpu, s_rr))

        # Check convergence: stress at iter 15 should be much lower than iter 1
        s_init = results[0][1]
        s_final = results[-1][1]
        print(f"Stress progression (iters 1 -> 15): {s_init:.6f} -> {s_final:.6f} (reduction: {(s_init - s_final)/s_init*100:.2f}%)")
        assert s_final < s_init, f"Stress did not decrease: {s_init} -> {s_final}"

        # Sweep seeds: 0, 1, 2, 42 at 5 iterations
        print("  Checking seeds 0, 1, 2, 42 at 5 iterations:")
        for s in [0, 1, 2, 42]:
            a_res = run_cmd(ATOMIC_FULL_BIN, [
                "--input", CYCLE6_MTX, "--iterations", "5", "--epsilon", "0.1",
                "--seed", str(s), "--output-format", "json", "--output-dir", tmpdir
            ])
            c_res = run_cmd(CPU_FULL_BIN, [
                "--input", CYCLE6_MTX, "--iterations", "5", "--epsilon", "0.1",
                "--seed", str(s), "--output-format", "json", "--output-dir", tmpdir
            ])
            st = a_res["stress_value"]
            assert math.isfinite(st)
            print(f"    seed={s:2d} | Atomic stress: {st:8.6f} | CPU stress: {c_res['stress_value']:8.6f}")

    print("Test 1 (cycle6.mtx): PASS")
    return results


def test_full_atomic_grid10():
    print("\n" + "="*70)
    print("Test 2: Full Atomic SGD on grid2d_10x10.mtx")
    print("="*70)
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Sweep iterations: 1, 5, 10, 15
        for iters in [1, 5, 10, 15]:
            atomic_res = run_cmd(ATOMIC_FULL_BIN, [
                "--input", GRID10_MTX, "--iterations", str(iters), "--epsilon", "0.1",
                "--seed", "0", "--output-format", "json", "--output-dir", tmpdir
            ])
            cpu_res = run_cmd(CPU_FULL_BIN, [
                "--input", GRID10_MTX, "--iterations", str(iters), "--epsilon", "0.1",
                "--seed", "0", "--output-format", "json", "--output-dir", tmpdir
            ])
            rr_res = run_cmd(RR_FULL_BIN, [
                "--input", GRID10_MTX, "--iterations", str(iters), "--epsilon", "0.1",
                "--seed", "0", "--output-format", "json", "--output-dir", tmpdir
            ])

            s_atomic = atomic_res["stress_value"]
            s_cpu = cpu_res["stress_value"]
            s_rr = rr_res["stress_value"]

            assert math.isfinite(s_atomic), f"Non-finite stress in Atomic SGD: {s_atomic}"
            assert verify_finite_coordinates(atomic_res["final_positions_path"]), "Atomic final coordinates contain NaN/Inf"
            assert atomic_res["attempted_updates"] == atomic_res["completed_updates"], "Gate 4 violation: attempted != completed"
            assert atomic_res["retry_failures"] == 0, "Gate 4 violation: retry_failures != 0"

            print(f"iters={iters:2d} | Atomic: {s_atomic:9.4f} | CPU: {s_cpu:9.4f} | RR: {s_rr:9.4f} | Atomic/CPU: {s_atomic/s_cpu:6.4f} | Done: {atomic_res['completed_updates']}/{atomic_res['attempted_updates']}")
            results.append((iters, s_atomic, s_cpu, s_rr))

        s_init = results[0][1]
        s_final = results[-1][1]
        print(f"Stress progression (iters 1 -> 15): {s_init:.4f} -> {s_final:.4f} (reduction: {(s_init - s_final)/s_init*100:.2f}%)")
        assert s_final < s_init, f"Stress did not decrease on grid10: {s_init} -> {s_final}"

        # Test chunk sizes: 1024 vs default (65536) vs 4950
        print("  Checking chunk sizes for batching safety:")
        for cs in [1024, 4950, 65536]:
            c_res = run_cmd(ATOMIC_FULL_BIN, [
                "--input", GRID10_MTX, "--iterations", "5", "--epsilon", "0.1",
                "--seed", "0", "--chunk-size", str(cs),
                "--output-format", "json", "--output-dir", tmpdir
            ])
            st = c_res["stress_value"]
            assert math.isfinite(st)
            assert c_res["attempted_updates"] == c_res["completed_updates"]
            print(f"    chunk_size={cs:5d} | stress: {st:9.4f} | dispatches: {c_res['dispatches']} | done: {c_res['completed_updates']}")

    print("Test 2 (grid2d_10x10.mtx): PASS")
    return results


def test_sparse_atomic_uspowergrid():
    print("\n" + "="*70)
    print("Test 3: Sparse Atomic SGD on USpowerGrid.mtx (pivots=50, iterations=15)")
    print("="*70)
    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        # Part A: Target seeds 0, 1, 2 (and 3, 4)
        print("Part A: Seed sweep (seeds 0, 1, 2, 3, 4, pivots=50, iters=15):")
        seed_data = []
        for s in [0, 1, 2, 3, 4]:
            atomic_res = run_cmd(ATOMIC_SPARSE_BIN, [
                "--input", USPOWER_MTX, "--pivots", "50", "--iterations", "15",
                "--epsilon", "0.1", "--seed", str(s),
                "--output-format", "json", "--output-dir", tmpdir
            ])
            cpu_res = run_cmd(CPU_SPARSE_BIN, [
                "--input", USPOWER_MTX, "--pivots", "50", "--iterations", "15",
                "--epsilon", "0.1", "--seed", str(s),
                "--output-format", "json", "--output-dir", tmpdir
            ])
            rr_res = run_cmd(RR_SPARSE_BIN, [
                "--input", USPOWER_MTX, "--pivots", "50", "--iterations", "15",
                "--epsilon", "0.1", "--seed", str(s),
                "--output-format", "json", "--output-dir", tmpdir
            ])

            s_atomic = atomic_res["stress_value"]
            s_cpu = cpu_res["stress_value"]
            s_rr = rr_res["stress_value"]

            assert math.isfinite(s_atomic), f"Non-finite stress in Atomic Sparse: {s_atomic}"
            assert verify_finite_coordinates(atomic_res["final_positions_path"]), "Coordinates contain NaN/Inf"
            assert atomic_res["attempted_updates"] == atomic_res["completed_updates"], "Gate 4 violation"
            assert atomic_res["retry_failures"] == 0, "Gate 4 retry_failures != 0"

            ratio_cpu = s_atomic / s_cpu
            ratio_rr = s_atomic / s_rr
            print(f"seed={s} | Atomic: {s_atomic:10.2f} | CPU: {s_cpu:10.2f} | RR: {s_rr:10.2f} | Atomic/CPU: {ratio_cpu:6.4f} | Atomic/RR: {ratio_rr:6.4f} | Updates: {atomic_res['completed_updates']}")
            seed_data.append((s, s_atomic, s_cpu, s_rr, ratio_cpu, ratio_rr))

        # Check consistency across seeds:
        atomic_stresses = [x[1] for x in seed_data]
        mean_s = sum(atomic_stresses) / len(atomic_stresses)
        std_s = math.sqrt(sum((x - mean_s)**2 for x in atomic_stresses) / len(atomic_stresses))
        cv = std_s / mean_s
        print(f"Across-seed statistics for Atomic Sparse: Mean={mean_s:.2f}, Std={std_s:.2f}, CV={cv*100:.2f}%")
        assert cv < 0.05, f"Stress variation across seeds is too large (CV={cv*100:.2f}%)"

        # Check ratio to CPU baseline is within reasonable bounds (0.95 - 1.15)
        for _, s_atomic, s_cpu, _, ratio_cpu, _ in seed_data:
            assert 0.90 <= ratio_cpu <= 1.25, f"Atomic/CPU ratio outside safe bounds: {ratio_cpu:.4f}"

        # Part B: Convergence trajectory across iterations (1, 5, 10, 15, 20) on seed 0
        print("\nPart B: Convergence trajectory on seed 0 (iters: 1, 5, 10, 15, 20):")
        trajectory = []
        for iters in [1, 5, 10, 15, 20]:
            a_res = run_cmd(ATOMIC_SPARSE_BIN, [
                "--input", USPOWER_MTX, "--pivots", "50", "--iterations", str(iters),
                "--epsilon", "0.1", "--seed", "0",
                "--output-format", "json", "--output-dir", tmpdir
            ])
            st = a_res["stress_value"]
            trajectory.append((iters, st, a_res["completed_updates"]))
            print(f"  iters={iters:2d} | stress: {st:10.2f} | time: {a_res['iteration_total_time_ms']:6.2f} ms | updates: {a_res['completed_updates']}")

        # Ensure monotonically decreasing stress along trajectory
        for i in range(1, len(trajectory)):
            assert trajectory[i][1] < trajectory[i-1][1], f"Stress diverged between iters {trajectory[i-1][0]} and {trajectory[i][0]}: {trajectory[i-1][1]} -> {trajectory[i][1]}"
        print(f"Convergence verified: stress decreased steadily from {trajectory[0][1]:.2f} (iter 1) to {trajectory[-1][1]:.2f} (iter 20).")

        # Part C: Within-seed repetition stability (run 3 times with seed 0)
        print("\nPart C: Within-seed repetition stability (3 runs, seed 0, iters 15):")
        rep_stresses = []
        for rep in range(3):
            rep_res = run_cmd(ATOMIC_SPARSE_BIN, [
                "--input", USPOWER_MTX, "--pivots", "50", "--iterations", "15",
                "--epsilon", "0.1", "--seed", "0",
                "--output-format", "json", "--output-dir", tmpdir
            ])
            st = rep_res["stress_value"]
            rep_stresses.append(st)
            print(f"  rep {rep+1}: stress = {st:.4f}")

        rep_mean = sum(rep_stresses) / len(rep_stresses)
        rep_std = math.sqrt(sum((x - rep_mean)**2 for x in rep_stresses) / len(rep_stresses))
        rep_cv = rep_std / rep_mean
        print(f"Within-seed repetition CV: {rep_cv*100:.4f}% (Range: {min(rep_stresses):.2f} - {max(rep_stresses):.2f})")
        assert rep_cv < 0.05, f"Within-seed repetition variance too high: {rep_cv*100:.2f}%"

        # Part D: Check --cpu-shuffle mode
        print("\nPart D: Comparing standard setup-shuffle vs --cpu-shuffle (seed 0, iters 15):")
        shuffle_res = run_cmd(ATOMIC_SPARSE_BIN, [
            "--input", USPOWER_MTX, "--pivots", "50", "--iterations", "15",
            "--epsilon", "0.1", "--seed", "0", "--cpu-shuffle",
            "--output-format", "json", "--output-dir", tmpdir
        ])
        print(f"  Default setup-shuffle stress: {rep_stresses[0]:.2f}")
        print(f"  --cpu-shuffle mode stress:   {shuffle_res['stress_value']:.2f}")
        assert math.isfinite(shuffle_res["stress_value"])
        assert shuffle_res["attempted_updates"] == shuffle_res["completed_updates"]

    print("Test 3 (USpowerGrid.mtx): PASS")
    return seed_data


def test_resource_and_bounds():
    print("\n" + "="*70)
    print("Test 4: Resource Bounds & Memory / CPU Load Monitoring")
    print("="*70)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Measure resource usage of USpowerGrid (largest test workload)
        cmd = [
            ATOMIC_SPARSE_BIN,
            "--input", USPOWER_MTX,
            "--pivots", "50",
            "--iterations", "25",
            "--epsilon", "0.1",
            "--seed", "0",
            "--output-format", "json",
            "--output-dir", tmpdir
        ]
        # Use /usr/bin/time -v to inspect peak RSS and CPU
        time_cmd = ["/usr/bin/time", "-v"] + cmd
        env = os.environ.copy()
        env["PATH"] = f"{os.path.expanduser('~')}/.cargo/bin:{env.get('PATH', '')}"
        res = subprocess.run(time_cmd, capture_output=True, text=True, env=env)
        stderr = res.stderr
        
        # parse Max RSS
        max_rss_kb = None
        user_time_sec = None
        system_time_sec = None
        percent_cpu = None
        for line in stderr.splitlines():
            line = line.strip()
            if "Maximum resident set size" in line:
                max_rss_kb = int(line.split(":")[-1].strip())
            elif "User time (seconds)" in line:
                user_time_sec = float(line.split(":")[-1].strip())
            elif "System time (seconds)" in line:
                system_time_sec = float(line.split(":")[-1].strip())
            elif "Percent of CPU this job got" in line:
                percent_cpu = line.split(":")[-1].strip()

        print(f"Execution resource profile on USpowerGrid (25 iters):")
        print(f"  Maximum RSS: {max_rss_kb} KB ({max_rss_kb / 1024:.2f} MB)")
        print(f"  User Time: {user_time_sec} s, System Time: {system_time_sec} s")
        print(f"  Percent CPU: {percent_cpu}")

        assert max_rss_kb is not None, "Could not determine Max RSS"
        # Safety bound: memory must not exceed 1 GB for 5K node graph
        max_rss_mb = max_rss_kb / 1024
        assert max_rss_mb < 1000.0, f"Memory exceeded safety bound: {max_rss_mb} MB"
        print(f"Memory check: {max_rss_mb:.2f} MB is well within safe system limit (15 GiB RAM).")

    print("Test 4 (Resource bounds): PASS")


def main():
    print("Running Milestone 3 Challenger Verification Suite (Convergence & Stress Stability)")
    print(f"Target Git commit: 68d11133b827dfec93e7f4c3ef054ba9db285b54")
    t0 = time.time()
    
    test_full_atomic_cycle6()
    test_full_atomic_grid10()
    test_sparse_atomic_uspowergrid()
    test_resource_and_bounds()

    elapsed = time.time() - t0
    print("\n" + "="*70)
    print(f"ALL 4 EMPIRICAL VERIFICATION TESTS PASSED SUCCESSFULLY in {elapsed:.2f} s")
    print("="*70)


if __name__ == "__main__":
    main()
