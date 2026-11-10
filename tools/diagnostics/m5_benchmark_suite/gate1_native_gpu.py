"""Gate 1 Verification: Native GPU Enforcement & Software Fallback Rejection."""

from typing import Any, Dict, List
from .config import BINARIES, DATASETS, THRESHOLDS
from .runner import run_command_safe


def run_gate1_verification(verbose: bool = False) -> Dict[str, Any]:
    """Execute Gate 1 tests and return structured results and verdict."""
    results: Dict[str, Any] = {
        "gate_id": "Gate 1",
        "name": "Native GPU Enforcement & Software Fallback Rejection",
        "status": "PENDING",
        "positive_tests": [],
        "negative_tests": [],
        "verdict": "FAIL",
        "errors": [],
    }

    test_mtx = DATASETS["cycle6"]["path"]
    gpu_binaries = [
        ("rr_gpu", BINARIES["rr_gpu"], ["--input", test_mtx, "--iterations", "1", "--output-format", "json"]),
        ("sparse_gpu", BINARIES["sparse_gpu"], ["--input", test_mtx, "--pivots", "3", "--iterations", "1", "--output-format", "json"]),
        ("vram_lock_native", BINARIES["vram_lock_native"], ["--input", test_mtx, "--iterations", "1", "--output-format", "json"]),
        ("vram_lock_sparse", BINARIES["vram_lock_sparse"], ["--input", test_mtx, "--pivots", "3", "--iterations", "1", "--output-format", "json"]),
    ]

    all_passed = True

    # 1. Positive Tests: Native GPU execution
    for name, bin_path, args in gpu_binaries:
        res = run_command_safe(bin_path, args, timeout=30)
        json_data = res.get("json") or {}
        rc = res["returncode"]
        
        gpu_name = json_data.get("gpu_name", "")
        gpu_backend = json_data.get("gpu_backend", "")
        status = json_data.get("status", "")

        name_ok = any(sub in gpu_name for sub in THRESHOLDS["expected_gpu_name_substrings"])
        backend_ok = (gpu_backend == THRESHOLDS["expected_gpu_backend"])
        success = (rc == 0) and (status == "success") and name_ok and backend_ok

        test_record = {
            "binary_name": name,
            "binary_path": bin_path,
            "returncode": rc,
            "gpu_name": gpu_name,
            "gpu_backend": gpu_backend,
            "status": status,
            "passed": success,
        }
        results["positive_tests"].append(test_record)

        if not success:
            all_passed = False
            msg = f"Positive test failed for {name}: rc={rc}, gpu_name='{gpu_name}', backend='{gpu_backend}', status='{status}'"
            results["errors"].append(msg)
            if verbose:
                print(f"  [Gate 1 FAIL] {msg}")
        elif verbose:
            print(f"  [Gate 1 PASS] {name}: {gpu_name} ({gpu_backend})")

    # 2. Negative Tests: Software rasterizer rejection
    llvmpipe_env = {
        "GALLIUM_DRIVER": "llvmpipe",
        "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe",
    }
    for name, bin_path, _ in gpu_binaries:
        neg_args = ["--input", test_mtx, "--iterations", "1"]
        res = run_command_safe(bin_path, neg_args, env_overrides=llvmpipe_env, timeout=30)
        rc = res["returncode"]
        stderr = res["stderr"]

        # Expected: non-zero returncode and mention of rejection / llvmpipe
        has_rejection = ("llvmpipe" in stderr) or ("cpu" in stderr.lower()) or ("reject" in stderr.lower())
        passed = (rc != 0) and has_rejection

        test_record = {
            "binary_name": name,
            "binary_path": bin_path,
            "returncode": rc,
            "rejected": (rc != 0),
            "rejection_message_detected": has_rejection,
            "stderr_snippet": stderr.strip().splitlines()[-1] if stderr.strip() else "",
            "passed": passed,
        }
        results["negative_tests"].append(test_record)

        if not passed:
            all_passed = False
            msg = f"Negative test failed for {name}: expected non-zero exit and llvmpipe rejection, got rc={rc}, stderr='{stderr[:100]}'"
            results["errors"].append(msg)
            if verbose:
                print(f"  [Gate 1 FAIL] {msg}")
        elif verbose:
            print(f"  [Gate 1 PASS] {name} strictly rejected llvmpipe (rc={rc})")

    results["status"] = "PASS" if all_passed else "FAIL"
    results["verdict"] = results["status"]
    return results
