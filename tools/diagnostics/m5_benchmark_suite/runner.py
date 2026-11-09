"""Execution harness and helper utilities for safe subprocess management."""

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import TIMEOUTS, WORKSPACE_ROOT


def run_command_safe(
    binary: str,
    args: List[str],
    env_overrides: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute a binary with safety guards (sequential execution, timeout, clean environment).
    
    Returns a dict containing:
      - 'returncode': int
      - 'stdout': str
      - 'stderr': str
      - 'json': Optional[dict] (parsed last json object)
      - 'wall_time_sec': float
      - 'timed_out': bool
    """
    if cwd is None:
        cwd = WORKSPACE_ROOT
    if timeout is None:
        timeout = TIMEOUTS["default"]

    env = os.environ.copy()
    # Ensure cargo/bin is in PATH
    cargo_bin = str(Path.home() / ".cargo/bin")
    if cargo_bin not in env.get("PATH", ""):
        env["PATH"] = f"{cargo_bin}:{env.get('PATH', '')}"

    if env_overrides:
        env.update(env_overrides)

    cmd = [binary] + [str(a) for a in args]
    t0 = time.perf_counter()
    timed_out = False
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd),
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        stdout = proc.stdout
        stderr = proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - t0
        timed_out = True
        rc = -1
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        stderr += f"\n[TIMED OUT after {timeout} seconds]"

    # Attempt to parse last JSON line
    json_data = None
    if stdout:
        json_lines = [
            line.strip()
            for line in stdout.strip().splitlines()
            if line.strip().startswith("{")
        ]
        if json_lines:
            try:
                json_data = json.loads(json_lines[-1])
            except json.JSONDecodeError:
                json_data = None

    return {
        "command": " ".join(cmd),
        "returncode": rc,
        "stdout": stdout,
        "stderr": stderr,
        "json": json_data,
        "wall_time_sec": elapsed,
        "timed_out": timed_out,
    }


def extract_positions_section(filepath: str) -> str:
    """Extract coordinates section from output txt file."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Positions file not found: {filepath}")
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()
    if "# Positions (x y)" in content:
        return content.split("# Positions (x y)")[1].strip()
    return content.strip()


def compute_positions_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of the extracted coordinate lines."""
    pos_text = extract_positions_section(filepath)
    return hashlib.sha256(pos_text.encode("utf-8")).hexdigest()


def compute_positions_f64_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of little-endian f64 coordinate bytes."""
    pos_text = extract_positions_section(filepath)
    h = hashlib.sha256()
    for line in pos_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            x, y = float(parts[0]), float(parts[1])
            h.update(struct.pack("<dd", x, y))
    return h.hexdigest()


def verify_finite_coordinates(filepath: str) -> Tuple[bool, int]:
    """Verify that all coordinates in the positions file are finite (not NaN/Inf).
    
    Returns (all_finite, count).
    """
    pos_text = extract_positions_section(filepath)
    count = 0
    for line in pos_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            count += 1
            x, y = float(parts[0]), float(parts[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                return False, count
    return True, count
