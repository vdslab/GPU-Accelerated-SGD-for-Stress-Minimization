"""Gate 4 Verification: Atomic SGD 100% Completion & Convergence Verification."""

import math
from typing import Any, Dict, List
from .config import BINARIES, DATASETS, THRESHOLDS
from .runner import run_command_safe, verify_finite_coordinates


def run_gate4_verification(verbose: bool = False) -> Dict[str, Any]:
    """Execute Gate 4 tests and return structured results and verdict."""
    results: Dict[str, Any] = {
        "gate_id": "Gate 4",
        "name": "Atomic SGD 100% Completion & Convergence Verification",
        "status": "PENDING",
        "tests": [],
        "verdict": "FAIL",
        "errors": [],
    }

    all_passed = True

    atomic_test_cases = [
        # Full Atomic
        ("vram_lock_native", "Full", "cycle6", ["--iterations", "15", "--seed", "42"]),
        ("vram_lock_native", "Full", "grid2d_10x10", ["--iterations", "15", "--seed", "42"]),
        # Sparse Atomic
        ("vram_lock_sparse", "Sparse", "cycle6", ["--pivots", "3", "--iterations", "15", "--seed", "42"]),
        ("vram_lock_sparse", "Sparse", "grid2d_10x10", ["--pivots", "100", "--iterations", "15", "--seed", "42"]),
        ("vram_lock_sparse", "Sparse", "USpowerGrid", ["--pivots", "50", "--iterations", "15", "--seed", "42"]),
    ]

    for bin_name, kind, dataset_key, extra_args in atomic_test_cases:
        bin_path = BINARIES[bin_name]
        mtx = DATASETS[dataset_key]["path"]
        tier = DATASETS[dataset_key]["tier"]
        timeout = 60 if tier == "medium" else 30

        # Build args for 1 iteration to measure baseline initial stress
        args_1_core = []
        i = 0
        while i < len(extra_args):
            if extra_args[i] == "--iterations":
                args_1_core.extend(["--iterations", "1"])
                i += 2
            else:
                args_1_core.append(extra_args[i])
                i += 1
        if "--iterations" not in args_1_core:
            args_1_core.extend(["--iterations", "1"])

        args_1 = ["--input", mtx] + args_1_core + ["--output-format", "json"]
        r1 = run_command_safe(bin_path, args_1, timeout=timeout)
        j1 = r1.get("json") or {}
        stress_iter1 = j1.get("stress_value")

        # Run 15 iterations for final convergence test
        args_15 = ["--input", mtx] + extra_args + ["--output-format", "json"]
        r15 = run_command_safe(bin_path, args_15, timeout=timeout)
        j15 = r15.get("json") or {}

        attempted = j15.get("attempted_updates")
        completed = j15.get("completed_updates")
        retry_failures = j15.get("retry_failures")
        stress_iter15 = j15.get("stress_value")
        pos_path = j15.get("final_positions_path")

        # Completion rate
        completion_rate = (completed / attempted) if (attempted and completed is not None) else 0.0
        completion_ok = (attempted is not None) and (attempted > 0) and (completed == attempted)

        # Retry failures
        retry_ok = (retry_failures == THRESHOLDS["max_retry_failures"])

        # Finite coordinates
        finite_coords_ok = False
        coord_count = 0
        if pos_path:
            try:
                finite_coords_ok, coord_count = verify_finite_coordinates(pos_path)
            except Exception as e:
                finite_coords_ok = False

        # Convergence
        finite_stress_ok = (
            (stress_iter15 is not None)
            and math.isfinite(stress_iter15)
            and (stress_iter15 > 0.0)
        )
        convergence_ok = (
            finite_stress_ok
            and (stress_iter1 is not None)
            and (stress_iter15 < stress_iter1)
        )

        passed = (
            completion_ok
            and retry_ok
            and finite_coords_ok
            and convergence_ok
        )

        test_record = {
            "binary": bin_name,
            "kind": kind,
            "dataset": dataset_key,
            "attempted_updates": attempted,
            "completed_updates": completed,
            "completion_rate": completion_rate,
            "retry_failures": retry_failures,
            "stress_iter1": stress_iter1,
            "stress_iter15": stress_iter15,
            "stress_decreased": (stress_iter15 < stress_iter1) if (stress_iter1 and stress_iter15) else False,
            "coordinates_finite": finite_coords_ok,
            "coordinate_count": coord_count,
            "passed": passed,
        }
        results["tests"].append(test_record)

        if not passed:
            all_passed = False
            msg = (
                f"Gate 4 failure for {bin_name} ({kind}) on {dataset_key}: "
                f"completed={completed}/{attempted}, retries={retry_failures}, "
                f"s1={stress_iter1}, s15={stress_iter15}, coords_finite={finite_coords_ok}"
            )
            results["errors"].append(msg)
            if verbose:
                print(f"  [Gate 4 FAIL] {msg}")
        elif verbose:
            print(
                f"  [Gate 4 PASS] {bin_name} ({kind}) on {dataset_key}: "
                f"100% completed ({completed}/{attempted}), retries=0, "
                f"stress: {stress_iter1:.4f} -> {stress_iter15:.4f} (converged)"
            )

    results["status"] = "PASS" if all_passed else "FAIL"
    results["verdict"] = results["status"]
    return results
