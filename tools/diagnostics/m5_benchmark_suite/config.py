"""Configuration and constants for Milestone 5: 4-Tier Verification & Benchmark Suite."""

import os
from pathlib import Path

# Workspace root
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

# Binaries
BINARIES = {
    "rr_gpu": str(WORKSPACE_ROOT / "implementations/full/rr_gpu/target/release/rr-gpu"),
    "sparse_gpu": str(WORKSPACE_ROOT / "implementations/sparse/sparse-sgd-gpu/target/release/sparse-sgd-gpu"),
    "vram_lock_native": str(WORKSPACE_ROOT / "implementations/full/vram-lock-native/target/release/vram-lock-native"),
    "vram_lock_sparse": str(WORKSPACE_ROOT / "implementations/sparse/vram-lock-sparse/target/release/vram-lock-sparse"),
    "baseline_full": str(WORKSPACE_ROOT / "implementations/full/baseline-sgd-non-gpu/target/release/baseline-sgd-non-gpu"),
    "baseline_sparse": str(WORKSPACE_ROOT / "implementations/sparse/baseline-sparse-sgd-non-gpu/target/release/baseline-sparse-sgd-non-gpu"),
}

# Standard Datasets Registry
DATASETS = {
    # Smoke / Tiny
    "cycle6": {
        "path": str(WORKSPACE_ROOT / "data/cycle6.mtx"),
        "nodes": 6,
        "edges": 6,
        "tier": "tiny",
    },
    # Small
    "grid2d_10x10": {
        "path": str(WORKSPACE_ROOT / "data/grid2d_10x10.mtx"),
        "nodes": 100,
        "edges": 180,
        "tier": "small",
    },
    "cycle128": {
        "path": str(WORKSPACE_ROOT / "data/cycle128.mtx"),
        "nodes": 128,
        "edges": 128,
        "tier": "small",
    },
    # Medium
    "BA-2000-g0": {
        "path": str(WORKSPACE_ROOT / "data/BA-2000-g0.mtx"),
        "nodes": 2000,
        "edges": 9975,
        "tier": "medium",
    },
    "grid2d_64x64": {
        "path": str(WORKSPACE_ROOT / "experiment_results/v5/additional-grid-workgroup/data/grid2d_64x64.mtx"),
        "nodes": 4096,
        "edges": 8064,
        "tier": "medium",
    },
    "USpowerGrid": {
        "path": str(WORKSPACE_ROOT / "data/USpowerGrid.mtx"),
        "nodes": 4941,
        "edges": 6594,
        "tier": "medium",
    },
    # Large / Scale Limit
    "BA-8000-g0": {
        "path": str(WORKSPACE_ROOT / "experiment_results/full16/degree/inputs/BA-8000-g0.mtx"),
        "nodes": 8000,
        "edges": 39975,
        "tier": "stress_full",
    },
    "grid2d_128x128": {
        "path": str(WORKSPACE_ROOT / "experiment_results/v5/additional-grid-workgroup/data/grid2d_128x128.mtx"),
        "nodes": 16384,
        "edges": 32512,
        "tier": "stress_full",
    },
    "luxembourg_osm": {
        "path": "/home/likr/webgpu-bfs/data/luxembourg_osm.mtx",
        "nodes": 114599,
        "edges": 119666,
        "tier": "stress_sparse",
    },
    "web-Stanford": {
        "path": "/home/likr/webgpu-bfs/data/web-Stanford/web-Stanford-undirected.mtx",
        "nodes": 281903,
        "edges": 2312497,
        "tier": "stress_sparse",
    },
}

# Timeouts in seconds (WSL2 freeze protection)
TIMEOUTS = {
    "tiny": 30,
    "small": 30,
    "medium": 60,
    "stress_full": 120,
    "stress_sparse": 120,
    "default": 45,
}

# Quality Gate Thresholds
THRESHOLDS = {
    # Gate 1: Hardware GPU adapter requirements
    "expected_gpu_name_substrings": ["D3D12", "RTX 3050"],
    "expected_gpu_backend": "Gl",
    
    # Gate 3: Maximum allowable stress degradation ratio (S_RR / S_CPU)
    "max_stress_ratio_full": 1.286,
    "max_stress_ratio_sparse": 1.155,
    "max_within_seed_variance_pct": 0.00,
    
    # Gate 4: Atomic SGD
    "min_atomic_completion_rate": 1.0,  # 100%
    "max_retry_failures": 0,
    
    # Performance Targets
    "target_full_sgd_wall_speedup_ba2000": 15.0,
    "target_full_sgd_device_speedup_ba2000": 50.0,
    "target_sparse_sgd_device_speedup_uspower": 10.0,
    "target_bfs_speedup_uspower": 3.0,
    "max_small_graph_preprocess_time_ms": 5.0,
}

# Output file paths
DEFAULT_JSON_OUT = WORKSPACE_ROOT / "experiment_results/m5_verification_results.json"
DEFAULT_REPORT_OUT = WORKSPACE_ROOT / "experiment_results/M5_VERIFICATION_REPORT.md"
