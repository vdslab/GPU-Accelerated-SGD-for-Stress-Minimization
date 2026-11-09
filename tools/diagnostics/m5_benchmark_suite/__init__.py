"""Milestone 5: 4-Tier Verification & Benchmark Suite Package."""

from .config import BINARIES, DATASETS, THRESHOLDS
from .gate1_native_gpu import run_gate1_verification
from .gate2_preprocessing import run_gate2_verification
from .gate3_roundrobin import run_gate3_verification
from .gate4_atomic import run_gate4_verification
from .bench_performance import run_performance_benchmarks
from .reporter import export_results, generate_markdown_report

__all__ = [
    "BINARIES",
    "DATASETS",
    "THRESHOLDS",
    "run_gate1_verification",
    "run_gate2_preprocessing",
    "run_gate3_roundrobin",
    "run_gate4_atomic",
    "run_performance_benchmarks",
    "export_results",
    "generate_markdown_report",
]
