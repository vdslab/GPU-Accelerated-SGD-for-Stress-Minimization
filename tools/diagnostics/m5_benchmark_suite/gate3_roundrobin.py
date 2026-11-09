"""Gate 3 Verification: Bit-Exact Repeatability & Zero Stress Degradation."""

from typing import Any, Dict, List
from .config import BINARIES, DATASETS, THRESHOLDS
from .runner import compute_positions_sha256, compute_positions_f64_sha256, run_command_safe


def run_gate3_verification(verbose: bool = False) -> Dict[str, Any]:
    """Execute Gate 3 tests and return structured results and verdict."""
    results: Dict[str, Any] = {
        "gate_id": "Gate 3",
        "name": "Bit-Exact Repeatability & Zero Stress Degradation",
        "status": "PENDING",
        "repeatability_tests": [],
        "convergence_quality_tests": [],
        "verdict": "FAIL",
        "errors": [],
    }

    all_passed = True

    # 1. Repeatability Tests: identical seed must produce 100% bit-exact coordinates and 0.0 stress variance
    repeat_cases = [
        ("rr_gpu", "grid2d_10x10", ["--iterations", "5", "--seed", "123"]),
        ("rr_gpu", "BA-2000-g0", ["--iterations", "5", "--seed", "123"]),
        ("sparse_gpu", "grid2d_10x10", ["--pivots", "100", "--iterations", "5", "--seed", "123"]),
        ("sparse_gpu", "BA-2000-g0", ["--pivots", "50", "--iterations", "5", "--seed", "123"]),
    ]

    for bin_name, dataset_key, extra_args in repeat_cases:
        bin_path = BINARIES[bin_name]
        mtx = DATASETS[dataset_key]["path"]
        tier = DATASETS[dataset_key]["tier"]
        timeout = 60 if tier == "medium" else 30

        cmd_args = ["--input", mtx] + extra_args + ["--output-format", "json"]

        r1 = run_command_safe(bin_path, cmd_args, timeout=timeout)
        r2 = run_command_safe(bin_path, cmd_args, timeout=timeout)

        j1 = r1.get("json") or {}
        j2 = r2.get("json") or {}

        stress1 = j1.get("stress_value")
        stress2 = j2.get("stress_value")
        pos_path1 = j1.get("final_positions_path")
        pos_path2 = j2.get("final_positions_path")

        pos_sha1 = compute_positions_sha256(pos_path1) if pos_path1 else None
        pos_sha2 = compute_positions_sha256(pos_path2) if pos_path2 else None

        f64_sha1 = compute_positions_f64_sha256(pos_path1) if pos_path1 else None
        f64_sha2 = compute_positions_f64_sha256(pos_path2) if pos_path2 else None

        stress_diff = abs(stress1 - stress2) if (stress1 is not None and stress2 is not None) else None
        var_pct = (stress_diff / stress1 * 100.0) if (stress1 and stress_diff is not None) else None

        passed = (
            (pos_sha1 is not None)
            and (pos_sha1 == pos_sha2)
            and (f64_sha1 == f64_sha2)
            and (stress_diff is not None)
            and (stress_diff == 0.0)
        )

        test_record = {
            "binary": bin_name,
            "dataset": dataset_key,
            "args": extra_args,
            "run1_stress": stress1,
            "run2_stress": stress2,
            "stress_diff": stress_diff,
            "variance_pct": var_pct,
            "run1_positions_sha256": pos_sha1,
            "run2_positions_sha256": pos_sha2,
            "run1_positions_f64_sha256": f64_sha1,
            "run2_positions_f64_sha256": f64_sha2,
            "coordinates_bit_exact": (pos_sha1 == pos_sha2),
            "passed": passed,
        }
        results["repeatability_tests"].append(test_record)

        if not passed:
            all_passed = False
            msg = (
                f"Repeatability failure for {bin_name} on {dataset_key}: "
                f"stress1={stress1}, stress2={stress2}, diff={stress_diff}, "
                f"pos_match={pos_sha1 == pos_sha2}"
            )
            results["errors"].append(msg)
            if verbose:
                print(f"  [Gate 3 FAIL] {msg}")
        elif verbose:
            print(
                f"  [Gate 3 PASS] Repeatability {bin_name} on {dataset_key}: "
                f"stress={stress1}, bit-exact coordinates ({pos_sha1[:12]}...), diff=0.000e0"
            )

    # 2. Convergence Quality: compare 15 iterations stress against CPU baseline
    quality_cases = [
        # Full SGD: BA-2000-g0 & grid2d_10x10
        ("full", "grid2d_10x10", BINARIES["baseline_full"], BINARIES["rr_gpu"], ["--iterations", "15", "--seed", "42"], THRESHOLDS["max_stress_ratio_full"]),
        ("full", "BA-2000-g0", BINARIES["baseline_full"], BINARIES["rr_gpu"], ["--iterations", "15", "--seed", "42"], THRESHOLDS["max_stress_ratio_full"]),
        # Sparse SGD: BA-2000-g0 (p=50) & grid2d_10x10 (p=100)
        ("sparse", "grid2d_10x10", BINARIES["baseline_sparse"], BINARIES["sparse_gpu"], ["--pivots", "100", "--iterations", "15", "--seed", "42"], THRESHOLDS["max_stress_ratio_sparse"]),
        ("sparse", "BA-2000-g0", BINARIES["baseline_sparse"], BINARIES["sparse_gpu"], ["--pivots", "50", "--iterations", "15", "--seed", "42"], THRESHOLDS["max_stress_ratio_sparse"]),
    ]

    for kind, dataset_key, cpu_bin, gpu_bin, common_args, max_ratio in quality_cases:
        mtx = DATASETS[dataset_key]["path"]
        tier = DATASETS[dataset_key]["tier"]
        timeout = 60 if tier == "medium" else 30

        cmd_cpu = ["--input", mtx] + common_args + ["--output-format", "json"]
        cmd_gpu = ["--input", mtx] + common_args + ["--output-format", "json"]

        r_cpu = run_command_safe(cpu_bin, cmd_cpu, timeout=timeout)
        r_gpu = run_command_safe(gpu_bin, cmd_gpu, timeout=timeout)

        j_cpu = r_cpu.get("json") or {}
        j_gpu = r_gpu.get("json") or {}

        s_cpu = j_cpu.get("stress_value")
        s_gpu = j_gpu.get("stress_value")

        ratio = (s_gpu / s_cpu) if (s_cpu and s_gpu) else None
        passed = (ratio is not None) and (ratio <= max_ratio) and (s_gpu > 0.0)

        test_record = {
            "kind": kind,
            "dataset": dataset_key,
            "cpu_stress": s_cpu,
            "gpu_stress": s_gpu,
            "stress_ratio": ratio,
            "max_ratio_allowed": max_ratio,
            "passed": passed,
        }
        results["convergence_quality_tests"].append(test_record)

        if not passed:
            all_passed = False
            msg = f"Convergence quality failure on {kind} {dataset_key}: CPU={s_cpu}, GPU={s_gpu}, ratio={ratio} > {max_ratio}"
            results["errors"].append(msg)
            if verbose:
                print(f"  [Gate 3 FAIL] {msg}")
        elif verbose:
            print(
                f"  [Gate 3 PASS] Quality {kind} {dataset_key}: "
                f"CPU={s_cpu:.4f}, GPU={s_gpu:.4f}, ratio={ratio:.4f} <= {max_ratio}"
            )

    results["status"] = "PASS" if all_passed else "FAIL"
    results["verdict"] = results["status"]
    return results
