#!/usr/bin/env python3
"""
Adversarial stress-testing suite for Milestone 3 (Atomic SGD wgpu Port).
Author: atomic_challenger_m3_2

Tests:
1. Star graph (Extreme lock contention: 1 central hub, 200 leaves -> 200 edges, all incident to node 1).
   - Tests lock acquisition under extreme concurrent contention on a single shared endpoint.
   - Tests deadlock / livelock freedom and 100% completion invariant (Gate 4).
2. Complete graph K_40 (1225 edges, 780 pairs).
   - High density all-to-all pair interactions.
3. Path graph (N=500 linear chain).
   - Long diameter, testing coordinate stability and gradient propagation.
4. Input validation and edge-case handling:
   - iterations=0 (rejected cleanly?)
   - epsilon negative or zero (rejected cleanly?)
   - pivots=0 (rejected cleanly?)
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time

ATOMIC_FULL_BIN = "./implementations/full/vram-lock-native/target/release/vram-lock-native"
ATOMIC_SPARSE_BIN = "./implementations/sparse/vram-lock-sparse/target/release/vram-lock-sparse"


def run_test_cmd(bin_path, args):
    env = os.environ.copy()
    env["PATH"] = f"{os.path.expanduser('~')}/.cargo/bin:{env.get('PATH', '')}"
    cmd = [bin_path] + args
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
    return res


def make_star_graph(path, n_leaves=200):
    n = n_leaves + 1
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate pattern general\n")
        f.write(f"% Star graph with {n} nodes\n")
        f.write(f"{n} {n} {n_leaves}\n")
        for leaf in range(2, n + 1):
            f.write(f"1 {leaf}\n")


def make_clique_graph(path, n=40):
    edges = []
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            edges.append((u, v))
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate pattern general\n")
        f.write(f"% Clique graph K_{n}\n")
        f.write(f"{n} {n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")


def make_path_graph(path, n=500):
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate pattern general\n")
        f.write(f"% Path graph P_{n}\n")
        f.write(f"{n} {n} {n - 1}\n")
        for i in range(1, n):
            f.write(f"{i} {i + 1}\n")


def main():
    print("="*70)
    print("Adversarial Stress Testing: Atomic SGD Contention & Robustness")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Challenge 1: Star graph (extreme contention on node 1)
        star_mtx = os.path.join(tmpdir, "star201.mtx")
        make_star_graph(star_mtx, n_leaves=200)
        print("\n[Challenge 1] Testing Star Graph (201 nodes, all edges touch hub 1)...")
        # Full Atomic
        res_full = run_test_cmd(ATOMIC_FULL_BIN, [
            "--input", star_mtx, "--iterations", "10", "--epsilon", "0.1",
            "--seed", "0", "--output-format", "json", "--output-dir", tmpdir
        ])
        assert res_full.returncode == 0, f"Full Atomic failed on Star graph:\n{res_full.stderr}"
        data_full = json.loads([l for l in res_full.stdout.splitlines() if l.strip().startswith("{")][-1])
        assert math.isfinite(data_full["stress_value"]), "Star graph produced NaN/Inf stress in Full"
        assert data_full["attempted_updates"] == data_full["completed_updates"]
        assert data_full["retry_failures"] == 0
        print(f"  Full Atomic: Star graph passed! Stress={data_full['stress_value']:.4f}, updates={data_full['completed_updates']}, retries={data_full['retry_failures']}")

        # Sparse Atomic
        res_sparse = run_test_cmd(ATOMIC_SPARSE_BIN, [
            "--input", star_mtx, "--pivots", "10", "--iterations", "10", "--epsilon", "0.1",
            "--seed", "0", "--output-format", "json", "--output-dir", tmpdir
        ])
        assert res_sparse.returncode == 0, f"Sparse Atomic failed on Star graph:\n{res_sparse.stderr}"
        data_sparse = json.loads([l for l in res_sparse.stdout.splitlines() if l.strip().startswith("{")][-1])
        assert math.isfinite(data_sparse["stress_value"]), "Star graph produced NaN/Inf stress in Sparse"
        assert data_sparse["attempted_updates"] == data_sparse["completed_updates"]
        assert data_sparse["retry_failures"] == 0
        print(f"  Sparse Atomic: Star graph passed! Stress={data_sparse['stress_value']:.4f}, updates={data_sparse['completed_updates']}, retries={data_sparse['retry_failures']}")

        # Challenge 2: Clique graph K_40 (all-to-all pair contention)
        clique_mtx = os.path.join(tmpdir, "clique40.mtx")
        make_clique_graph(clique_mtx, n=40)
        print("\n[Challenge 2] Testing Clique Graph K_40 (780 pairs, all-to-all)...")
        res_clique = run_test_cmd(ATOMIC_FULL_BIN, [
            "--input", clique_mtx, "--iterations", "10", "--epsilon", "0.1",
            "--seed", "0", "--output-format", "json", "--output-dir", tmpdir
        ])
        assert res_clique.returncode == 0
        data_clique = json.loads([l for l in res_clique.stdout.splitlines() if l.strip().startswith("{")][-1])
        assert math.isfinite(data_clique["stress_value"])
        assert data_clique["attempted_updates"] == data_clique["completed_updates"]
        print(f"  Clique K_40 passed! Stress={data_clique['stress_value']:.4f}, updates={data_clique['completed_updates']}")

        # Challenge 3: Path graph (N=500 linear chain)
        path_mtx = os.path.join(tmpdir, "path500.mtx")
        make_path_graph(path_mtx, n=500)
        print("\n[Challenge 3] Testing Path Graph P_500 (500 nodes, diameter 499)...")
        res_path = run_test_cmd(ATOMIC_SPARSE_BIN, [
            "--input", path_mtx, "--pivots", "50", "--iterations", "15", "--epsilon", "0.1",
            "--seed", "0", "--output-format", "json", "--output-dir", tmpdir
        ])
        assert res_path.returncode == 0
        data_path = json.loads([l for l in res_path.stdout.splitlines() if l.strip().startswith("{")][-1])
        assert math.isfinite(data_path["stress_value"])
        assert data_path["attempted_updates"] == data_path["completed_updates"]
        print(f"  Path P_500 passed! Stress={data_path['stress_value']:.4f}, updates={data_path['completed_updates']}")

        # Challenge 4: Boundary & Invalid Arguments
        print("\n[Challenge 4] Testing boundary inputs and error handling...")
        # 0 iterations
        res_zero_iter = run_test_cmd(ATOMIC_FULL_BIN, ["--input", star_mtx, "--iterations", "0"])
        assert res_zero_iter.returncode != 0, "iterations=0 should be rejected"
        print("  iterations=0 correctly rejected.")

        # negative epsilon
        res_neg_eps = run_test_cmd(ATOMIC_FULL_BIN, ["--input", star_mtx, "--epsilon", "-0.1"])
        assert res_neg_eps.returncode != 0, "negative epsilon should be rejected"
        print("  negative epsilon correctly rejected.")

        # 0 pivots in sparse
        res_zero_pivots = run_test_cmd(ATOMIC_SPARSE_BIN, ["--input", star_mtx, "--pivots", "0"])
        assert res_zero_pivots.returncode != 0, "pivots=0 should be rejected"
        print("  pivots=0 correctly rejected.")

        # non-existent input
        res_bad_file = run_test_cmd(ATOMIC_FULL_BIN, ["--input", "non_existent_file.mtx"])
        assert res_bad_file.returncode != 0, "non-existent file should be rejected"
        print("  non-existent file correctly rejected.")

    print("\n" + "="*70)
    print("ALL ADVERSARIAL STRESS TESTS PASSED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
