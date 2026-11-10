"""Gate 2 Verification: Bit-Exact Preprocessing Parity."""

from typing import Any, Dict, List
from .config import BINARIES, DATASETS
from .runner import run_command_safe


def run_gate2_verification(verbose: bool = False) -> Dict[str, Any]:
    """Execute Gate 2 tests and return structured results and verdict."""
    results: Dict[str, Any] = {
        "gate_id": "Gate 2",
        "name": "Bit-Exact Preprocessing Parity",
        "status": "PENDING",
        "full_sgd_tests": [],
        "sparse_sgd_tests": [],
        "verdict": "FAIL",
        "errors": [],
    }

    all_passed = True
    seed = "42"

    # 1. Full SGD Preprocessing & Initial Coordinates Parity
    # Datasets: cycle6, grid2d_10x10, BA-2000-g0
    full_datasets = ["cycle6", "grid2d_10x10", "BA-2000-g0"]

    for d_name in full_datasets:
        mtx = DATASETS[d_name]["path"]
        tier = DATASETS[d_name]["tier"]
        timeout = 60 if tier == "medium" else 30

        # CPU Baseline
        r_cpu = run_command_safe(
            BINARIES["baseline_full"],
            ["--input", mtx, "--iterations", "1", "--seed", seed, "--output-format", "json"],
            timeout=timeout,
        )
        j_cpu = r_cpu.get("json") or {}
        cpu_init_sha = j_cpu.get("initial_positions_sha256")

        # RR GPU (auto)
        r_rr_auto = run_command_safe(
            BINARIES["rr_gpu"],
            ["--input", mtx, "--iterations", "1", "--seed", seed, "--bfs-engine", "auto", "--output-format", "json"],
            timeout=timeout,
        )
        j_rr_auto = r_rr_auto.get("json") or {}
        rr_auto_init_sha = j_rr_auto.get("initial_positions_sha256")

        # RR GPU (cpu BFS)
        r_rr_cpu = run_command_safe(
            BINARIES["rr_gpu"],
            ["--input", mtx, "--iterations", "1", "--seed", seed, "--bfs-engine", "cpu", "--output-format", "json"],
            timeout=timeout,
        )
        j_rr_cpu = r_rr_cpu.get("json") or {}
        rr_cpu_init_sha = j_rr_cpu.get("initial_positions_sha256")

        # RR GPU (gpu BFS)
        r_rr_gpu = run_command_safe(
            BINARIES["rr_gpu"],
            ["--input", mtx, "--iterations", "1", "--seed", seed, "--bfs-engine", "gpu", "--output-format", "json"],
            timeout=timeout,
        )
        j_rr_gpu = r_rr_gpu.get("json") or {}
        rr_gpu_init_sha = j_rr_gpu.get("initial_positions_sha256")

        # Atomic Full (vram-lock-native)
        r_atomic = run_command_safe(
            BINARIES["vram_lock_native"],
            ["--input", mtx, "--iterations", "1", "--seed", seed, "--output-format", "json"],
            timeout=timeout,
        )
        j_atomic = r_atomic.get("json") or {}
        atomic_init_sha = j_atomic.get("initial_positions_sha256")

        passed = (
            cpu_init_sha is not None
            and (cpu_init_sha == rr_auto_init_sha == rr_cpu_init_sha == rr_gpu_init_sha == atomic_init_sha)
        )

        test_record = {
            "dataset": d_name,
            "path": mtx,
            "cpu_initial_positions_sha256": cpu_init_sha,
            "rr_auto_initial_positions_sha256": rr_auto_init_sha,
            "rr_cpu_initial_positions_sha256": rr_cpu_init_sha,
            "rr_gpu_initial_positions_sha256": rr_gpu_init_sha,
            "atomic_initial_positions_sha256": atomic_init_sha,
            "passed": passed,
        }
        results["full_sgd_tests"].append(test_record)

        if not passed:
            all_passed = False
            msg = (
                f"Full SGD Gate 2 failure on {d_name}: "
                f"CPU={cpu_init_sha}, RR_auto={rr_auto_init_sha}, RR_cpu={rr_cpu_init_sha}, "
                f"RR_gpu={rr_gpu_init_sha}, Atomic={atomic_init_sha}"
            )
            results["errors"].append(msg)
            if verbose:
                print(f"  [Gate 2 FAIL] {msg}")
        elif verbose:
            print(f"  [Gate 2 PASS] Full SGD {d_name}: initial_positions_sha256 100% match ({cpu_init_sha[:12]}...)")

    # 2. Sparse SGD Preprocessing & Initial Coordinates Parity
    # Datasets: cycle6 (p=3), grid2d_10x10 (p=50), USpowerGrid (p=50)
    sparse_configs = [
        ("cycle6", 3),
        ("grid2d_10x10", 50),
        ("USpowerGrid", 50),
    ]

    for d_name, pivots in sparse_configs:
        mtx = DATASETS[d_name]["path"]
        tier = DATASETS[d_name]["tier"]
        timeout = 60 if tier == "medium" else 30

        # CPU Baseline
        r_cpu = run_command_safe(
            BINARIES["baseline_sparse"],
            ["--input", mtx, "--pivots", str(pivots), "--iterations", "1", "--seed", seed, "--output-format", "json"],
            timeout=timeout,
        )
        j_cpu = r_cpu.get("json") or {}
        cpu_pre_sha = j_cpu.get("preprocess_sha256")
        cpu_init_sha = j_cpu.get("initial_positions_sha256")

        # Sparse SGD GPU
        r_gpu = run_command_safe(
            BINARIES["sparse_gpu"],
            ["--input", mtx, "--pivots", str(pivots), "--iterations", "1", "--seed", seed, "--output-format", "json"],
            timeout=timeout,
        )
        j_gpu = r_gpu.get("json") or {}
        gpu_pre_sha = j_gpu.get("preprocess_sha256")
        gpu_init_sha = j_gpu.get("initial_positions_sha256")

        # Atomic Sparse (vram-lock-sparse)
        r_atomic = run_command_safe(
            BINARIES["vram_lock_sparse"],
            ["--input", mtx, "--pivots", str(pivots), "--iterations", "1", "--seed", seed, "--output-format", "json"],
            timeout=timeout,
        )
        j_atomic = r_atomic.get("json") or {}
        atomic_pre_sha = j_atomic.get("preprocess_sha256")
        atomic_init_sha = j_atomic.get("initial_positions_sha256")

        pre_match = (cpu_pre_sha is not None) and (cpu_pre_sha == gpu_pre_sha == atomic_pre_sha)
        init_match = (cpu_init_sha is not None) and (cpu_init_sha == gpu_init_sha == atomic_init_sha)
        passed = pre_match and init_match

        test_record = {
            "dataset": d_name,
            "pivots": pivots,
            "path": mtx,
            "cpu_preprocess_sha256": cpu_pre_sha,
            "gpu_preprocess_sha256": gpu_pre_sha,
            "atomic_preprocess_sha256": atomic_pre_sha,
            "cpu_initial_positions_sha256": cpu_init_sha,
            "gpu_initial_positions_sha256": gpu_init_sha,
            "atomic_initial_positions_sha256": atomic_init_sha,
            "preprocess_match": pre_match,
            "initial_positions_match": init_match,
            "passed": passed,
        }
        results["sparse_sgd_tests"].append(test_record)

        if not passed:
            all_passed = False
            msg = (
                f"Sparse SGD Gate 2 failure on {d_name} (p={pivots}): "
                f"Pre-sha match={pre_match} (CPU={cpu_pre_sha}, GPU={gpu_pre_sha}, Atomic={atomic_pre_sha}), "
                f"Init-sha match={init_match} (CPU={cpu_init_sha}, GPU={gpu_init_sha}, Atomic={atomic_init_sha})"
            )
            results["errors"].append(msg)
            if verbose:
                print(f"  [Gate 2 FAIL] {msg}")
        elif verbose:
            print(
                f"  [Gate 2 PASS] Sparse SGD {d_name} (p={pivots}): "
                f"preprocess_sha256 ({cpu_pre_sha[:12]}...) & initial_positions_sha256 ({cpu_init_sha[:12]}...) 100% match"
            )

    results["status"] = "PASS" if all_passed else "FAIL"
    results["verdict"] = results["status"]
    return results
