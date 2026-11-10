#!/usr/bin/env python3
"""
Empirical verification script for Milestone 0: Native GPU Adapter & Warmup.
Tests:
1. JIT Warmup Effectiveness:
   - Measures iteration_times_ms across multiple iterations for both rr_gpu and sparse-sgd-gpu.
   - Verifies iteration_times_ms[0] is free from Mesa NIR->DXIL compilation spike (< 10 ms, sub-ms for small graphs).
   - Verifies gpu_prepare_time_ms absorbed the compilation latency (hundreds to thousands of ms).
2. Bit-exact Initial Coordinates (Gate 3 Preservation):
   - Verifies initial_positions_sha256 matches baseline expectations.
   - Verifies repeatable bit-exact final positions and stress across multiple runs with identical seed.
3. Gate 1 Native GPU Enforcement:
   - Verifies gpu_name is D3D12 (RTX 3050) and gpu_backend is Gl.
   - Verifies llvmpipe/CPU rasterization is strictly rejected.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

RR_GPU_BIN = "./implementations/full/rr_gpu/target/release/rr-gpu"
SPARSE_GPU_BIN = "./implementations/sparse/sparse-sgd-gpu/target/release/sparse-sgd-gpu"
CYCLE6_MTX = "./experiment_results/validation/atomic-sparse-20261106/cli-smoke/cycle6.mtx"
BA2000_MTX = "./experiment_results/full16/degree/inputs/BA-2000-g0.mtx"

def run_cmd(args):
    env = os.environ.copy()
    env["PATH"] = f"{os.path.expanduser('~')}/.cargo/bin:{env.get('PATH', '')}"
    res = subprocess.run(args, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"Command failed: {' '.join(args)}\nStderr: {res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)
    # Parse last line of stdout as json
    lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip().startswith("{")]
    if not lines:
        print(f"No JSON found in stdout: {res.stdout}", file=sys.stderr)
        sys.exit(1)
    return json.loads(lines[-1])

def extract_positions_section(filepath):
    """Extract only the lines after '# Positions (x y)' to compare coordinates directly."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if "# Positions (x y)" in content:
        return content.split("# Positions (x y)")[1].strip()
    return content.strip()

def test_warmup_effectiveness_rr_gpu():
    print("=== Test 1A: JIT Warmup Effectiveness - rr_gpu (cycle6, 10 iterations) ===")
    rec = run_cmd([RR_GPU_BIN, "--input", CYCLE6_MTX, "--iterations", "10", "--seed", "42", "--output-format", "json"])
    
    gpu_name = rec["gpu_name"]
    gpu_backend = rec["gpu_backend"]
    gpu_prep = rec["gpu_prepare_time_ms"]
    iter_times = rec["iteration_times_ms"]
    
    print(f"  GPU Name: {gpu_name}")
    print(f"  GPU Backend: {gpu_backend}")
    print(f"  GPU Prepare Time: {gpu_prep:.2f} ms")
    print(f"  Iteration 0 Time: {iter_times[0]:.4f} ms")
    print(f"  Iterations 1-9: {[round(t, 4) for t in iter_times[1:]]} ms")
    
    assert "D3D12" in gpu_name or "RTX 3050" in gpu_name, f"Gate 1 violation: {gpu_name}"
    assert gpu_backend == "Gl", f"Gate 1 backend violation: {gpu_backend}"
    assert gpu_prep > 1000.0, f"Expected Mesa compile absorbed in gpu_prepare_time_ms (>1000ms), got {gpu_prep}ms"
    assert iter_times[0] < 1.0, f"Expected iteration 0 free from JIT spike (<1ms), got {iter_times[0]}ms"
    print("  -> PASS: Iteration 0 is sub-millisecond, compile spike absorbed into gpu_prepare_time_ms.\n")

def test_warmup_effectiveness_rr_gpu_ba2000():
    print("=== Test 1B: JIT Warmup Effectiveness - rr_gpu (BA-2000-g0, 5 iterations) ===")
    rec = run_cmd([RR_GPU_BIN, "--input", BA2000_MTX, "--iterations", "5", "--seed", "42", "--output-format", "json"])
    
    gpu_prep = rec["gpu_prepare_time_ms"]
    iter_times = rec["iteration_times_ms"]
    
    print(f"  GPU Prepare Time: {gpu_prep:.2f} ms")
    print(f"  Iteration 0 Time: {iter_times[0]:.4f} ms")
    print(f"  Iterations 1-4: {[round(t, 4) for t in iter_times[1:]]} ms")
    
    avg_subsequent = sum(iter_times[1:]) / len(iter_times[1:])
    ratio = iter_times[0] / avg_subsequent
    print(f"  Iteration 0 / Avg Subsequent Ratio: {ratio:.3f}")
    
    assert gpu_prep > 1000.0, f"Expected Mesa compile absorbed (>1000ms), got {gpu_prep}ms"
    assert iter_times[0] < 20.0, f"Expected iteration 0 < 20ms, got {iter_times[0]}ms"
    print("  -> PASS: Iteration 0 has no JIT spike on 2000-node graph.\n")

def test_warmup_effectiveness_sparse_gpu():
    print("=== Test 1C: JIT Warmup Effectiveness - sparse-sgd-gpu (cycle6, 10 iterations) ===")
    rec = run_cmd([SPARSE_GPU_BIN, "--input", CYCLE6_MTX, "--iterations", "10", "--seed", "42", "--output-format", "json"])
    
    gpu_name = rec["gpu_name"]
    gpu_backend = rec["gpu_backend"]
    gpu_prep = rec["gpu_prepare_time_ms"]
    iter_times = rec["iteration_times_ms"]
    
    print(f"  GPU Name: {gpu_name}")
    print(f"  GPU Backend: {gpu_backend}")
    print(f"  GPU Prepare Time: {gpu_prep:.2f} ms")
    print(f"  Iteration 0 Time: {iter_times[0]:.4f} ms")
    print(f"  Iterations 1-9: {[round(t, 4) for t in iter_times[1:]]} ms")
    
    assert "D3D12" in gpu_name or "RTX 3050" in gpu_name, f"Gate 1 violation: {gpu_name}"
    assert gpu_backend == "Gl", f"Gate 1 backend violation: {gpu_backend}"
    assert gpu_prep > 200.0, f"Expected compile absorbed in gpu_prepare_time_ms (>200ms), got {gpu_prep}ms"
    assert iter_times[0] < 1.0, f"Expected iteration 0 free from JIT spike (<1ms), got {iter_times[0]}ms"
    print("  -> PASS: Sparse SGD iteration 0 is sub-millisecond, compile absorbed.\n")

def test_bit_exact_coordinates_rr_gpu():
    print("=== Test 2A: Bit-exact Coordinates Preservation - rr_gpu ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run 1 & Run 2 with identical seed and same run_id
        rec1 = run_cmd([
            RR_GPU_BIN, "--input", CYCLE6_MTX, "--iterations", "1", "--seed", "42",
            "--output-format", "json", "--output-dir", tmpdir, "--run-id", "same_run_id"
        ])
        rec2 = run_cmd([
            RR_GPU_BIN, "--input", CYCLE6_MTX, "--iterations", "1", "--seed", "42",
            "--output-format", "json", "--output-dir", tmpdir, "--run-id", "same_run_id"
        ])
        
        # Verify initial_positions_sha256 matches
        init_sha1 = rec1["initial_positions_sha256"]
        init_sha2 = rec2["initial_positions_sha256"]
        print(f"  Run 1 initial_positions_sha256: {init_sha1}")
        print(f"  Run 2 initial_positions_sha256: {init_sha2}")
        assert init_sha1 == init_sha2, f"Initial positions hash mismatch: {init_sha1} != {init_sha2}"
        
        # Verify final stress values match to floating-point exactness
        stress1 = rec1["stress_value"]
        stress2 = rec2["stress_value"]
        print(f"  Run 1 stress: {stress1}")
        print(f"  Run 2 stress: {stress2}")
        assert stress1 == stress2, f"Stress mismatch: {stress1} != {stress2}"
        
        # Compare output coordinate files
        final_file1 = rec1["final_positions_path"]
        final_file2 = rec2["final_positions_path"]
        pos1 = extract_positions_section(final_file1)
        pos2 = extract_positions_section(final_file2)
        assert pos1 == pos2, "Extracted positions are not bit-exact!"
        print(f"  Final coordinates bit-exact match: {len(pos1)} bytes identical.")
        print("  -> PASS: 100% bit-exact initial and final coordinates in rr_gpu.\n")

def test_bit_exact_baseline_full16():
    print("=== Test 2B: Baseline Initial Coordinates Match - BA-2000-g0 ===")
    rec = run_cmd([
        RR_GPU_BIN, "--input", BA2000_MTX, "--iterations", "1", "--seed", "0",
        "--output-format", "json"
    ])
    actual_init_sha = rec["initial_positions_sha256"]
    expected_init_sha = "e88f214dfdac95427cdbed6f89a1daa505a1fc591b115490c197fcdddc36d976"
    print(f"  Actual initial_positions_sha256:   {actual_init_sha}")
    print(f"  Expected initial_positions_sha256: {expected_init_sha}")
    assert actual_init_sha == expected_init_sha, f"Baseline initial coordinates mismatch: {actual_init_sha} != {expected_init_sha}"
    print("  -> PASS: Exact match with baseline seed 0 initial coordinates.\n")

def test_bit_exact_coordinates_sparse_gpu():
    print("=== Test 2C: Bit-exact Coordinates Preservation - sparse-sgd-gpu ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        rec1 = run_cmd([
            SPARSE_GPU_BIN, "--input", CYCLE6_MTX, "--iterations", "2", "--seed", "7",
            "--output-format", "json", "--output-dir", tmpdir, "--run-id", "same_id"
        ])
        rec2 = run_cmd([
            SPARSE_GPU_BIN, "--input", CYCLE6_MTX, "--iterations", "2", "--seed", "7",
            "--output-format", "json", "--output-dir", tmpdir, "--run-id", "same_id"
        ])
        
        init_sha1 = rec1["initial_positions_sha256"]
        init_sha2 = rec2["initial_positions_sha256"]
        prep_sha1 = rec1["preprocess_sha256"]
        prep_sha2 = rec2["preprocess_sha256"]
        stress1 = rec1["stress_value"]
        stress2 = rec2["stress_value"]
        
        print(f"  Run 1 initial_positions_sha256: {init_sha1}")
        print(f"  Run 2 initial_positions_sha256: {init_sha2}")
        print(f"  Run 1 preprocess_sha256:        {prep_sha1}")
        print(f"  Run 2 preprocess_sha256:        {prep_sha2}")
        print(f"  Run 1 stress: {stress1}")
        print(f"  Run 2 stress: {stress2}")
        
        assert init_sha1 == init_sha2, "Initial coordinates mismatch in sparse-sgd-gpu"
        assert prep_sha1 == prep_sha2, "Preprocess sha mismatch in sparse-sgd-gpu"
        assert stress1 == stress2, f"Stress mismatch: {stress1} != {stress2}"
        
        final_file1 = rec1["final_positions_path"]
        final_file2 = rec2["final_positions_path"]
        
        pos1 = extract_positions_section(final_file1)
        pos2 = extract_positions_section(final_file2)
        assert pos1 == pos2, "Sparse extracted positions are not bit-exact!"
        
        # Also check with same_id that the entire file matches
        with open(final_file1, "rb") as f1, open(final_file2, "rb") as f2:
            data1 = f1.read()
            data2 = f2.read()
        assert data1 == data2, "Sparse full files are not bit-exact when run_id is identical!"
        print(f"  Sparse final coordinates bit-exact match: {len(pos1)} bytes identical.")
        print("  -> PASS: 100% bit-exact initial and final coordinates in sparse-sgd-gpu.\n")

if __name__ == "__main__":
    print("=================================================================")
    print("Milestone 0 Empirical Challenger Verification")
    print("=================================================================\n")
    test_warmup_effectiveness_rr_gpu()
    test_warmup_effectiveness_rr_gpu_ba2000()
    test_warmup_effectiveness_sparse_gpu()
    test_bit_exact_coordinates_rr_gpu()
    test_bit_exact_baseline_full16()
    test_bit_exact_coordinates_sparse_gpu()
    print("=================================================================")
    print("ALL EMPIRICAL CHALLENGER TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")
