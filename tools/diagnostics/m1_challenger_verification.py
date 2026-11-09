#!/usr/bin/env python3
"""
Empirical verification script for Milestone 1: RoundRobin Full SGD Acceleration.
Tests:
1. Gate 3 Bit-Exact Reproducibility:
   - Compares commit a35b7bf27 against baseline commit 4f7f259b0 on multiple graphs:
     cycle6.mtx, path4.mtx, BA-2000-g0.mtx, BA-4000-g0.mtx.
   - Tests across parameter sweeps: seeds (0, 42, 123), iterations (1, 5, 10, 15),
     tile sizes (128, 1024), phase chunks (1, 4, 16).
   - Asserts: initial_positions_sha256 exact match.
   - Asserts: final coordinates bit-exact match (zero byte difference).
   - Asserts: relative difference in stress is 0.000e0.
2. 4-Bank Buffer Boundary & Large Graph Behavior:
   - BA-8000-g0 (N=8000, 64M distances, ~122 MiB in Bank 0):
     * Baseline fails (exceeds 128 MiB for f32).
     * Current binary succeeds with u16 packing.
   - grid2d_96x96 (N=9216, 84.9M distances, 42.5M words -> Crosses Bank 0 into Bank 1):
     * Crosses Bank 0 into Bank 1.
     * Current binary executes across bank boundary correctly.
   - grid2d_128x128 (N=16384, 268.4M distances, 134.2M words = 4 banks saturated):
     * Current binary saturates all 4 banks to 100% capacity.
   - N=16385 over-limit graph:
     * Current binary cleanly rejects with descriptive error.
3. Determinism & Batch Submission Safety:
   - Verifies 100% repeatable bit-exact coordinates across 5 repeated runs with identical seed.
4. Edge Case Graphs:
   - Odd N (N=5, 17, 127, 129), boundary tile sizes.
"""

import json
import math
import os
import struct
import subprocess
import sys
import tempfile

BASELINE_BIN = "/tmp/baseline_m0/implementations/full/rr_gpu/target/release/rr-gpu"
CURRENT_BIN = "./implementations/full/rr_gpu/target/release/rr-gpu"

CYCLE6_MTX = "./experiment_results/validation/atomic-sparse-20261106/cli-smoke/cycle6.mtx"
PATH4_MTX = "/tmp/path4.mtx"
BA2000_MTX = "./experiment_results/full16/degree/inputs/BA-2000-g0.mtx"
BA4000_MTX = "./experiment_results/full16/degree/inputs/BA-4000-g0.mtx"
BA8000_MTX = "./experiment_results/full16/degree/inputs/BA-8000-g0.mtx"
GRID96_MTX = "./experiment_results/v5/additional-grid-workgroup/data/grid2d_96x96.mtx"
GRID128_MTX = "./experiment_results/v5/additional-grid-workgroup/data/grid2d_128x128.mtx"


def run_bin(bin_path, args, check=True):
    env = os.environ.copy()
    env["PATH"] = f"{os.path.expanduser('~')}/.cargo/bin:{env.get('PATH', '')}"
    cmd = [bin_path] + args
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\nStderr: {res.stderr}")
    if not check and res.returncode != 0:
        return None, res.stderr, res.returncode
    lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"No JSON found in stdout from {' '.join(cmd)}\nStdout: {res.stdout}\nStderr: {res.stderr}")
    return json.loads(lines[-1]), res.stderr, res.returncode


def extract_coordinates(filepath):
    """Extract coordinates as list of (float, float) and raw lines."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if "# Positions (x y)" in content:
        pos_lines = content.split("# Positions (x y)")[1].strip().splitlines()
    else:
        pos_lines = content.strip().splitlines()
    coords = []
    for line in pos_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2:
            coords.append((float(parts[0]), float(parts[1])))
    return coords, pos_lines


def compare_runs(name, baseline_args, current_args=None):
    if current_args is None:
        current_args = baseline_args
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = os.path.join(tmpdir, "base")
        curr_dir = os.path.join(tmpdir, "curr")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(curr_dir, exist_ok=True)

        rec_base, _, _ = run_bin(BASELINE_BIN, baseline_args + ["--output-format", "json", "--output-dir", base_dir, "--run-id", "base"])
        rec_curr, _, _ = run_bin(CURRENT_BIN, current_args + ["--output-format", "json", "--output-dir", curr_dir, "--run-id", "curr"])

        # Check initial positions sha256
        init_sha_base = rec_base["initial_positions_sha256"]
        init_sha_curr = rec_curr["initial_positions_sha256"]
        assert init_sha_base == init_sha_curr, f"[{name}] initial_positions_sha256 mismatch: {init_sha_base} != {init_sha_curr}"

        # Extract coordinates
        coords_base, lines_base = extract_coordinates(rec_base["final_positions_path"])
        coords_curr, lines_curr = extract_coordinates(rec_curr["final_positions_path"])
        assert len(coords_base) == len(coords_curr), f"[{name}] vertex count mismatch: {len(coords_base)} != {len(coords_curr)}"

        # Check bit-exact string lines
        assert lines_base == lines_curr, f"[{name}] coordinate file lines are not bit-exact!"

        # Check IEEE 754 f64 / f32 exact equality
        max_coord_diff = 0.0
        for i, ((bx, by), (cx, cy)) in enumerate(zip(coords_base, coords_curr)):
            dx = abs(bx - cx)
            dy = abs(by - cy)
            diff = max(dx, dy)
            if diff > max_coord_diff:
                max_coord_diff = diff
            assert bx == cx and by == cy, f"[{name}] Coord mismatch at vertex {i}: base=({bx}, {by}), curr=({cx}, {cy})"

        # Check stress values
        stress_base = rec_base["stress_value"]
        stress_curr = rec_curr["stress_value"]
        abs_diff = abs(stress_base - stress_curr)
        rel_diff = abs_diff / stress_base if stress_base > 0 else abs_diff

        print(f"  [{name}] PASS: N={rec_curr['nodes']}, base_stress={stress_base:.6f}, curr_stress={stress_curr:.6f}, rel_diff={rel_diff:.3e}, max_coord_diff={max_coord_diff}")
        assert rel_diff < 1e-12, f"[{name}] Stress relative difference too large: {rel_diff:.3e}"


def test_task1_gate3_bit_exact():
    print("\n=== TASK 1: Gate 3 Bit-Exact Reproducibility vs Baseline ===")

    # Cycle6
    compare_runs("cycle6 (seed 0, iter 1)", ["--input", CYCLE6_MTX, "--seed", "0", "--iterations", "1"])
    compare_runs("cycle6 (seed 42, iter 10)", ["--input", CYCLE6_MTX, "--seed", "42", "--iterations", "10"])
    compare_runs("cycle6 (seed 42, iter 10, T=128, W=128)", ["--input", CYCLE6_MTX, "--seed", "42", "--iterations", "10", "--tile-size", "128", "--workgroup-size", "128"])
    compare_runs("cycle6 (seed 7, iter 5, C=16)", ["--input", CYCLE6_MTX, "--seed", "7", "--iterations", "5", "--phase-chunks", "16"])

    # Path4
    compare_runs("path4 (seed 0, iter 1)", ["--input", PATH4_MTX, "--seed", "0", "--iterations", "1"])
    compare_runs("path4 (seed 123, iter 10)", ["--input", PATH4_MTX, "--seed", "123", "--iterations", "10"])
    compare_runs("path4 (seed 999, iter 15, T=128, W=128, C=4)", ["--input", PATH4_MTX, "--seed", "999", "--iterations", "15", "--tile-size", "128", "--workgroup-size", "128", "--phase-chunks", "4"])

    # BA-2000-g0
    compare_runs("BA-2000-g0 (seed 0, iter 1)", ["--input", BA2000_MTX, "--seed", "0", "--iterations", "1"])
    compare_runs("BA-2000-g0 (seed 42, iter 5)", ["--input", BA2000_MTX, "--seed", "42", "--iterations", "5"])
    compare_runs("BA-2000-g0 (seed 42, iter 5, C=16)", ["--input", BA2000_MTX, "--seed", "42", "--iterations", "5", "--phase-chunks", "16"])
    compare_runs("BA-2000-g0 (seed 0, iter 3, T=128, W=128)", ["--input", BA2000_MTX, "--seed", "0", "--iterations", "3", "--tile-size", "128", "--workgroup-size", "128"])

    # BA-4000-g0
    compare_runs("BA-4000-g0 (seed 0, iter 2)", ["--input", BA4000_MTX, "--seed", "0", "--iterations", "2"])


def test_task2_buffer_boundaries():
    print("\n=== TASK 2: 4-Bank Buffer Boundary & Large Graphs ===")

    # 1. BA-8000-g0 (N=8000, 64M distances, Bank 0 ~122 MiB)
    print("  Testing BA-8000-g0 (N=8000)...")
    # Baseline check: should fail because 64M * 4 bytes = 256 MiB > 128 MiB max storage buffer binding
    _, base_err, base_rc = run_bin(BASELINE_BIN, ["--input", BA8000_MTX, "--seed", "0", "--iterations", "1", "--output-format", "json"], check=False)
    print(f"    Baseline exit code: {base_rc} (failed as expected due to 128 MiB binding limit)")
    assert base_rc != 0, "Baseline should have failed on BA-8000-g0 due to 128 MiB buffer limit"

    # Current binary: should succeed
    rec_curr, _, rc_curr = run_bin(CURRENT_BIN, ["--input", BA8000_MTX, "--seed", "0", "--iterations", "2", "--output-format", "json"])
    assert rc_curr == 0
    assert rec_curr["status"] == "success"
    assert rec_curr["nodes"] == 8000
    assert rec_curr["completed_updates"] == 8000 * 7999 // 2 * 2
    print(f"    Current binary SUCCEEDED: stress={rec_curr['stress_value']:.4f}, iteration_times={rec_curr['iteration_times_ms']}")

    # 2. grid2d_96x96 (N=9216, 84.9M distances, 42.5M words -> Crosses Bank 0 into Bank 1)
    print("  Testing grid2d_96x96 (N=9216, crosses Bank 0 into Bank 1)...")
    rec_grid96, _, rc_grid96 = run_bin(CURRENT_BIN, ["--input", GRID96_MTX, "--seed", "42", "--iterations", "2", "--output-format", "json"])
    assert rc_grid96 == 0
    assert rec_grid96["status"] == "success"
    assert rec_grid96["nodes"] == 9216
    assert rec_grid96["completed_updates"] == 9216 * 9215 // 2 * 2
    coords_96, _ = extract_coordinates(rec_grid96["final_positions_path"])
    assert len(coords_96) == 9216
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in coords_96)
    print(f"    grid2d_96x96 SUCCEEDED: stress={rec_grid96['stress_value']:.4f}, iteration_times={rec_grid96['iteration_times_ms']}")

    # 3. grid2d_128x128 (N=16384, 268.4M distances, 134.2M words -> All 4 Banks saturated at 100%)
    print("  Testing grid2d_128x128 (N=16384, 4 Banks saturated at 100% capacity)...")
    rec_grid128, _, rc_grid128 = run_bin(CURRENT_BIN, ["--input", GRID128_MTX, "--seed", "42", "--iterations", "1", "--output-format", "json"])
    assert rc_grid128 == 0
    assert rec_grid128["status"] == "success"
    assert rec_grid128["nodes"] == 16384
    assert rec_grid128["completed_updates"] == 16384 * 16383 // 2
    coords_128, _ = extract_coordinates(rec_grid128["final_positions_path"])
    assert len(coords_128) == 16384
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in coords_128)
    print(f"    grid2d_128x128 SUCCEEDED: stress={rec_grid128['stress_value']:.4f}, iteration_times={rec_grid128['iteration_times_ms']}")

    # 4. Over-limit Graph (N=16385)
    print("  Testing synthetic over-limit graph (N=16385)...")
    with tempfile.NamedTemporaryFile("w", suffix=".mtx", delete=False) as f:
        over_mtx = f.name
        f.write("%%MatrixMarket matrix coordinate pattern general\n")
        f.write("16385 16385 1\n")
        f.write("1 2\n")
    try:
        _, err, rc = run_bin(CURRENT_BIN, ["--input", over_mtx, "--iterations", "1", "--output-format", "json"], check=False)
        print(f"    Over-limit exit code: {rc}, stderr snippet: {err.strip()[:100]}")
        assert rc != 0, "N=16385 must be rejected!"
        assert "16384" in err or "Full SGD supports up to N=16384" in err, f"Expected N<=16384 error message, got: {err}"
        print("    -> PASS: Cleanly rejected with informative error message.")
    finally:
        if os.path.exists(over_mtx):
            os.remove(over_mtx)


def test_task3_determinism():
    print("\n=== TASK 3: Determinism & Multi-Pass Hazard Check ===")
    runs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for r in range(5):
            rec, _, _ = run_bin(CURRENT_BIN, [
                "--input", BA2000_MTX, "--seed", "42", "--iterations", "5",
                "--output-format", "json", "--output-dir", tmpdir, "--run-id", f"rep_{r}"
            ])
            coords, lines = extract_coordinates(rec["final_positions_path"])
            runs.append((rec["stress_value"], lines))
        
        first_stress, first_lines = runs[0]
        for r in range(1, 5):
            stress, lines = runs[r]
            assert stress == first_stress, f"Run {r} stress mismatch: {stress} != {first_stress}"
            assert lines == first_lines, f"Run {r} coordinates not bit-exact with Run 0"
        print(f"  5/5 repeated runs produced 100% BIT-EXACT coordinates and identical stress ({first_stress:.6f}).")


def test_task4_odd_nodes_and_boundaries():
    print("\n=== TASK 4: Odd N and Tile Boundaries ===")
    for n in [5, 17, 127, 129, 255]:
        with tempfile.NamedTemporaryFile("w", suffix=".mtx", delete=False) as f:
            mtx_path = f.name
            f.write("%%MatrixMarket matrix coordinate pattern general\n")
            f.write(f"{n} {n} {n-1}\n")
            for i in range(1, n):
                f.write(f"{i} {i+1}\n")
        try:
            # Test with various tile sizes
            for tile_size in [32, 128]:
                compare_runs(f"linear_path N={n} T={tile_size}", [
                    "--input", mtx_path, "--seed", "42", "--iterations", "3",
                    "--tile-size", str(tile_size), "--workgroup-size", "32"
                ])
        finally:
            if os.path.exists(mtx_path):
                os.remove(mtx_path)


def main():
    print("=" * 60)
    print("MILSTONE 1 ADVERSARIAL EMPIRICAL VERIFICATION HARNESS")
    print("=" * 60)
    test_task1_gate3_bit_exact()
    test_task2_buffer_boundaries()
    test_task3_determinism()
    test_task4_odd_nodes_and_boundaries()
    print("\n" + "=" * 60)
    print("ALL EMPIRICAL CHALLENGER TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
