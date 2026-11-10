"""Milestone 5 Performance Benchmarks: Full, Sparse, Atomic, and Preprocessing BFS."""

from typing import Any, Dict, List
from .config import BINARIES, DATASETS, THRESHOLDS
from .runner import run_command_safe


def run_performance_benchmarks(verbose: bool = False) -> Dict[str, Any]:
    """Execute all performance benchmarks and return structured results."""
    results: Dict[str, Any] = {
        "full_sgd_benchmarks": {},
        "sparse_sgd_benchmarks": {},
        "atomic_throughput_benchmarks": {},
        "bfs_preprocessing_benchmarks": {},
    }

    # =========================================================================
    # 1. Full SGD Performance Benchmarks
    # =========================================================================
    if verbose:
        print("\n--- 1. Full SGD Performance Benchmarks ---")

    # 1.1 BA-2000-g0 (N=2,000, 5 iters)
    mtx_ba2000 = DATASETS["BA-2000-g0"]["path"]
    r_cpu_ba2000 = run_command_safe(
        BINARIES["baseline_full"],
        ["--input", mtx_ba2000, "--iterations", "5", "--seed", "42", "--output-format", "json"],
        timeout=60,
    )
    j_cpu_ba2000 = r_cpu_ba2000.get("json") or {}
    cpu_iter_ba2000 = j_cpu_ba2000.get("iteration_time_ms", 0.0)

    r_gpu_ba2000 = run_command_safe(
        BINARIES["rr_gpu"],
        [
            "--input", mtx_ba2000,
            "--iterations", "5",
            "--seed", "42",
            "--tile-size", "128",
            "--workgroup-size", "128",
            "--phase-chunks", "1",
            "--output-format", "json",
        ],
        timeout=60,
    )
    j_gpu_ba2000 = r_gpu_ba2000.get("json") or {}
    gpu_wall_iter_ba2000 = j_gpu_ba2000.get("iteration_time_ms", 0.0)
    gpu_dev_total_ba2000 = j_gpu_ba2000.get("iteration_total_time_ms", 0.0)
    gpu_dev_iter_ba2000 = gpu_dev_total_ba2000 / 5.0 if gpu_dev_total_ba2000 else 0.0

    wall_speedup_ba2000 = (cpu_iter_ba2000 / gpu_wall_iter_ba2000) if gpu_wall_iter_ba2000 else 0.0
    dev_speedup_ba2000 = (cpu_iter_ba2000 / gpu_dev_iter_ba2000) if gpu_dev_iter_ba2000 else 0.0

    results["full_sgd_benchmarks"]["BA-2000-g0"] = {
        "nodes": 2000,
        "cpu_iteration_time_ms": cpu_iter_ba2000,
        "gpu_wall_iteration_time_ms": gpu_wall_iter_ba2000,
        "gpu_device_iteration_time_ms": gpu_dev_iter_ba2000,
        "wall_speedup": wall_speedup_ba2000,
        "device_speedup": dev_speedup_ba2000,
        "target_wall_speedup": THRESHOLDS["target_full_sgd_wall_speedup_ba2000"],
        "target_device_speedup": THRESHOLDS["target_full_sgd_device_speedup_ba2000"],
        "passed_target_wall": wall_speedup_ba2000 >= THRESHOLDS["target_full_sgd_wall_speedup_ba2000"],
        "passed_target_device": dev_speedup_ba2000 >= THRESHOLDS["target_full_sgd_device_speedup_ba2000"],
    }
    if verbose:
        print(f"  BA-2000-g0: CPU={cpu_iter_ba2000:.2f}ms, GPU_wall={gpu_wall_iter_ba2000:.2f}ms ({wall_speedup_ba2000:.2f}x), GPU_dev={gpu_dev_iter_ba2000:.2f}ms ({dev_speedup_ba2000:.2f}x)")

    # 1.2 grid2d_64x64 (N=4,096, 5 iters)
    mtx_grid64 = DATASETS["grid2d_64x64"]["path"]
    r_cpu_grid64 = run_command_safe(
        BINARIES["baseline_full"],
        ["--input", mtx_grid64, "--iterations", "5", "--seed", "42", "--output-format", "json"],
        timeout=120,
    )
    j_cpu_grid64 = r_cpu_grid64.get("json") or {}
    cpu_iter_grid64 = j_cpu_grid64.get("iteration_time_ms", 0.0)

    r_gpu_grid64 = run_command_safe(
        BINARIES["rr_gpu"],
        [
            "--input", mtx_grid64,
            "--iterations", "5",
            "--seed", "42",
            "--tile-size", "128",
            "--workgroup-size", "128",
            "--phase-chunks", "1",
            "--output-format", "json",
        ],
        timeout=60,
    )
    j_gpu_grid64 = r_gpu_grid64.get("json") or {}
    gpu_wall_iter_grid64 = j_gpu_grid64.get("iteration_time_ms", 0.0)
    gpu_dev_total_grid64 = j_gpu_grid64.get("iteration_total_time_ms", 0.0)
    gpu_dev_iter_grid64 = gpu_dev_total_grid64 / 5.0 if gpu_dev_total_grid64 else 0.0

    wall_speedup_grid64 = (cpu_iter_grid64 / gpu_wall_iter_grid64) if gpu_wall_iter_grid64 else 0.0
    dev_speedup_grid64 = (cpu_iter_grid64 / gpu_dev_iter_grid64) if gpu_dev_iter_grid64 else 0.0

    results["full_sgd_benchmarks"]["grid2d_64x64"] = {
        "nodes": 4096,
        "cpu_iteration_time_ms": cpu_iter_grid64,
        "gpu_wall_iteration_time_ms": gpu_wall_iter_grid64,
        "gpu_device_iteration_time_ms": gpu_dev_iter_grid64,
        "wall_speedup": wall_speedup_grid64,
        "device_speedup": dev_speedup_grid64,
    }
    if verbose:
        print(f"  grid2d_64x64: CPU={cpu_iter_grid64:.2f}ms, GPU_wall={gpu_wall_iter_grid64:.2f}ms ({wall_speedup_grid64:.2f}x), GPU_dev={gpu_dev_iter_grid64:.2f}ms ({dev_speedup_grid64:.2f}x)")

    # 1.3 Full SGD Scale Limit: BA-8000-g0 (single-bank bypass, N=8,000, 122 MiB in Bank 0)
    mtx_ba8000 = DATASETS["BA-8000-g0"]["path"]
    r_ba8000 = run_command_safe(
        BINARIES["rr_gpu"],
        ["--input", mtx_ba8000, "--iterations", "1", "--output-format", "json"],
        timeout=120,
    )
    j_ba8000 = r_ba8000.get("json") or {}
    ba8000_status = j_ba8000.get("status", "failed")
    ba8000_dev_time = j_ba8000.get("gpu_device_time_ms", 0.0)
    results["full_sgd_benchmarks"]["BA-8000-g0_scale_limit"] = {
        "nodes": 8000,
        "status": ba8000_status,
        "buffer_size_mib": 122.1,
        "banks_used": 1,
        "gpu_device_time_ms": ba8000_dev_time,
        "wall_time_sec": r_ba8000["wall_time_sec"],
        "passed": ba8000_status == "success",
    }
    if verbose:
        print(f"  BA-8000-g0 (Single-Bank Bypass): status={ba8000_status}, dev_time={ba8000_dev_time:.2f}ms, wall={r_ba8000['wall_time_sec']:.2f}s")

    # 1.4 Full SGD Scale Limit: grid2d_128x128 (4-bank saturation, N=16,384, 512 MiB across 4 banks)
    mtx_grid128 = DATASETS["grid2d_128x128"]["path"]
    r_grid128 = run_command_safe(
        BINARIES["rr_gpu"],
        [
            "--input", mtx_grid128,
            "--iterations", "1",
            "--tile-size", "128",
            "--workgroup-size", "128",
            "--phase-chunks", "1",
            "--output-format", "json",
        ],
        timeout=120,
    )
    j_grid128 = r_grid128.get("json") or {}
    grid128_status = j_grid128.get("status", "failed")
    grid128_dev_time = j_grid128.get("gpu_device_time_ms", 0.0)
    results["full_sgd_benchmarks"]["grid2d_128x128_scale_limit"] = {
        "nodes": 16384,
        "status": grid128_status,
        "buffer_size_mib": 512.0,
        "banks_used": 4,
        "gpu_device_time_ms": grid128_dev_time,
        "wall_time_sec": r_grid128["wall_time_sec"],
        "passed": grid128_status == "success",
    }
    if verbose:
        print(f"  grid2d_128x128 (4-Bank Saturation): status={grid128_status}, dev_time={grid128_dev_time:.2f}ms, wall={r_grid128['wall_time_sec']:.2f}s")

    # =========================================================================
    # 2. Sparse SGD Performance Benchmarks
    # =========================================================================
    if verbose:
        print("\n--- 2. Sparse SGD Performance Benchmarks ---")

    # 2.1 USpowerGrid (N=4,941, p=50, 5 iters)
    mtx_uspower = DATASETS["USpowerGrid"]["path"]
    r_cpu_sp = run_command_safe(
        BINARIES["baseline_sparse"],
        ["--input", mtx_uspower, "--pivots", "50", "--iterations", "5", "--seed", "42", "--output-format", "json"],
        timeout=60,
    )
    j_cpu_sp = r_cpu_sp.get("json") or {}
    cpu_iter_sp = j_cpu_sp.get("iteration_time_ms", 0.0)

    r_gpu_sp = run_command_safe(
        BINARIES["sparse_gpu"],
        ["--input", mtx_uspower, "--pivots", "50", "--iterations", "5", "--seed", "42", "--output-format", "json"],
        timeout=60,
    )
    j_gpu_sp = r_gpu_sp.get("json") or {}
    gpu_wall_iter_sp = j_gpu_sp.get("iteration_time_ms", 0.0)
    gpu_dev_total_sp = j_gpu_sp.get("iteration_total_time_ms", 0.0)
    gpu_dev_iter_sp = gpu_dev_total_sp / 5.0 if gpu_dev_total_sp else 0.0

    dev_speedup_sp = (cpu_iter_sp / gpu_dev_iter_sp) if gpu_dev_iter_sp else 0.0
    wall_speedup_sp = (cpu_iter_sp / gpu_wall_iter_sp) if gpu_wall_iter_sp else 0.0

    results["sparse_sgd_benchmarks"]["USpowerGrid"] = {
        "nodes": 4941,
        "pivots": 50,
        "cpu_iteration_time_ms": cpu_iter_sp,
        "gpu_wall_iteration_time_ms": gpu_wall_iter_sp,
        "gpu_device_iteration_time_ms": gpu_dev_iter_sp,
        "device_speedup": dev_speedup_sp,
        "wall_speedup": wall_speedup_sp,
        "target_device_speedup": THRESHOLDS["target_sparse_sgd_device_speedup_uspower"],
        "passed_target_device": dev_speedup_sp >= THRESHOLDS["target_sparse_sgd_device_speedup_uspower"],
    }
    if verbose:
        print(f"  USpowerGrid (p=50): CPU={cpu_iter_sp:.2f}ms, GPU_dev={gpu_dev_iter_sp:.3f}ms ({dev_speedup_sp:.2f}x device speedup)")

    # 2.2 Sparse Scale Limit: luxembourg_osm (N=114,599, p=200, 184 MiB across 2 banks)
    mtx_lux = DATASETS["luxembourg_osm"]["path"]
    r_lux = run_command_safe(
        BINARIES["sparse_gpu"],
        ["--input", mtx_lux, "--pivots", "200", "--iterations", "1", "--output-format", "json"],
        timeout=120,
    )
    j_lux = r_lux.get("json") or {}
    lux_status = j_lux.get("status", "failed")
    lux_dev_time = j_lux.get("gpu_device_time_ms", 0.0)
    results["sparse_sgd_benchmarks"]["luxembourg_osm_scale_limit"] = {
        "nodes": 114599,
        "pivots": 200,
        "status": lux_status,
        "buffer_size_mib": 184.2,
        "banks_used": 2,
        "gpu_device_time_ms": lux_dev_time,
        "wall_time_sec": r_lux["wall_time_sec"],
        "passed": lux_status == "success",
    }
    if verbose:
        print(f"  luxembourg_osm (2 Banks, 184 MiB): status={lux_status}, dev_time={lux_dev_time:.2f}ms, wall={r_lux['wall_time_sec']:.2f}s")

    # 2.3 Sparse Scale Limit: web-Stanford (N=281,903, p=200, 408 MiB across 4 banks)
    mtx_stanford = DATASETS["web-Stanford"]["path"]
    r_stanford = run_command_safe(
        BINARIES["sparse_gpu"],
        ["--input", mtx_stanford, "--pivots", "200", "--iterations", "1", "--output-format", "json"],
        timeout=120,
    )
    j_stanford = r_stanford.get("json") or {}
    stanford_status = j_stanford.get("status", "failed")
    stanford_dev_time = j_stanford.get("gpu_device_time_ms", 0.0)
    results["sparse_sgd_benchmarks"]["web-Stanford_scale_limit"] = {
        "nodes": 281903,
        "pivots": 200,
        "status": stanford_status,
        "buffer_size_mib": 408.4,
        "banks_used": 4,
        "gpu_device_time_ms": stanford_dev_time,
        "wall_time_sec": r_stanford["wall_time_sec"],
        "passed": stanford_status == "success",
    }
    if verbose:
        print(f"  web-Stanford (4 Banks, 408 MiB): status={stanford_status}, dev_time={stanford_dev_time:.2f}ms, wall={r_stanford['wall_time_sec']:.2f}s")

    # =========================================================================
    # 3. Atomic SGD Throughput Benchmarks
    # =========================================================================
    if verbose:
        print("\n--- 3. Atomic SGD Throughput Benchmarks ---")

    mtx_grid10 = DATASETS["grid2d_10x10"]["path"]
    # Full Atomic: 15 iters = 74,250 updates
    r_at_full = run_command_safe(
        BINARIES["vram_lock_native"],
        ["--input", mtx_grid10, "--iterations", "15", "--seed", "42", "--output-format", "json"],
        timeout=30,
    )
    j_at_full = r_at_full.get("json") or {}
    updates_full = j_at_full.get("completed_updates", 0)
    dev_time_full_ms = j_at_full.get("iteration_total_time_ms", 0.0)
    tput_full = (updates_full / (dev_time_full_ms / 1000.0) / 1e6) if dev_time_full_ms else 0.0

    # Sparse Atomic: 15 iters, p=100 = 74,250 updates
    r_at_sp = run_command_safe(
        BINARIES["vram_lock_sparse"],
        ["--input", mtx_grid10, "--pivots", "100", "--iterations", "15", "--seed", "42", "--output-format", "json"],
        timeout=30,
    )
    j_at_sp = r_at_sp.get("json") or {}
    updates_sp = j_at_sp.get("completed_updates", 0)
    dev_time_sp_ms = j_at_sp.get("iteration_total_time_ms", 0.0)
    tput_sp = (updates_sp / (dev_time_sp_ms / 1000.0) / 1e6) if dev_time_sp_ms else 0.0

    results["atomic_throughput_benchmarks"] = {
        "vram_lock_native": {
            "dataset": "grid2d_10x10",
            "iterations": 15,
            "completed_updates": updates_full,
            "gpu_device_time_ms": dev_time_full_ms,
            "throughput_m_updates_per_sec": tput_full,
        },
        "vram_lock_sparse": {
            "dataset": "grid2d_10x10",
            "pivots": 100,
            "iterations": 15,
            "completed_updates": updates_sp,
            "gpu_device_time_ms": dev_time_sp_ms,
            "throughput_m_updates_per_sec": tput_sp,
        },
    }
    if verbose:
        print(f"  Full Atomic (vram-lock-native): {tput_full:.2f} M updates/sec ({updates_full} updates in {dev_time_full_ms:.2f}ms)")
        print(f"  Sparse Atomic (vram-lock-sparse): {tput_sp:.2f} M updates/sec ({updates_sp} updates in {dev_time_sp_ms:.2f}ms)")

    # =========================================================================
    # 4. Preprocessing BFS Speedup & Small Graph Regression Check
    # =========================================================================
    if verbose:
        print("\n--- 4. Preprocessing BFS Benchmarks ---")

    # 4.1 USpowerGrid (p=50) CPU vs GPU BFS speedup
    r_cpu_bfs = run_command_safe(
        BINARIES["baseline_sparse"],
        ["--input", mtx_uspower, "--pivots", "50", "--iterations", "1", "--seed", "42", "--output-format", "json"],
        timeout=60,
    )
    j_cpu_bfs = r_cpu_bfs.get("json") or {}
    cpu_bfs_ms = j_cpu_bfs.get("preprocess_time_ms", 0.0)

    r_gpu_bfs = run_command_safe(
        BINARIES["sparse_gpu"],
        ["--input", mtx_uspower, "--pivots", "50", "--iterations", "1", "--seed", "42", "--output-format", "json"],
        timeout=60,
    )
    j_gpu_bfs = r_gpu_bfs.get("json") or {}
    gpu_bfs_ms = j_gpu_bfs.get("preprocess_time_ms", 0.0)
    bfs_speedup = (cpu_bfs_ms / gpu_bfs_ms) if gpu_bfs_ms else 0.0

    # 4.2 Small Graph Overhead Elimination (< 5.0 ms)
    small_cases = ["cycle128", "grid2d_10x10"]
    small_graph_overheads = {}
    for sc in small_cases:
        m = DATASETS[sc]["path"]
        r_sc = run_command_safe(
            BINARIES["rr_gpu"],
            ["--input", m, "--iterations", "1", "--output-format", "json"],
            timeout=30,
        )
        j_sc = r_sc.get("json") or {}
        prep_ms = j_sc.get("preprocess_time_ms", 0.0)
        small_graph_overheads[sc] = {
            "preprocess_time_ms": prep_ms,
            "passed": prep_ms < THRESHOLDS["max_small_graph_preprocess_time_ms"],
        }
        if verbose:
            print(f"  Small Graph {sc}: preprocess_time = {prep_ms:.3f} ms (< 5.0 ms check)")

    results["bfs_preprocessing_benchmarks"] = {
        "USpowerGrid": {
            "nodes": 4941,
            "pivots": 50,
            "cpu_preprocess_time_ms": cpu_bfs_ms,
            "gpu_preprocess_time_ms": gpu_bfs_ms,
            "bfs_speedup": bfs_speedup,
            "target_bfs_speedup": THRESHOLDS["target_bfs_speedup_uspower"],
            "passed_target_speedup": bfs_speedup >= THRESHOLDS["target_bfs_speedup_uspower"],
        },
        "small_graph_regression_check": small_graph_overheads,
    }
    if verbose:
        print(f"  USpowerGrid BFS: CPU={cpu_bfs_ms:.2f}ms, GPU={gpu_bfs_ms:.2f}ms -> Speedup: {bfs_speedup:.2f}x (Target > 3.0x)")

    return results
