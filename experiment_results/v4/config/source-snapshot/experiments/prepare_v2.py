#!/usr/bin/env python3
"""Prepare a separately versioned, auditable v2 study using existing LCC inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse.csgraph import connected_components

try:
    from artifact_paths import provenance_path, resolve_artifact_path
except ModuleNotFoundError:
    from experiments.artifact_paths import provenance_path, resolve_artifact_path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT / "experiment_results/v2"
SMALL = ["dwt_419", "qh882", "1138_bus", "dwt_1005", "cage8", "poli",
         "dwt_2680", "EVA", "3elt", "USpowerGrid"]
LARGE = ["fe_4elt2", "bcsstk31", "luxembourg_osm", "web-NotreDame"]
ABLATION = ["dwt_2680", "EVA", "3elt", "USpowerGrid"]
METHODS = ["sgd", "atomic_sgd", "rr_sgd", "sparse_sgd", "atomic_sparse_sgd", "rr_sparse_sgd"]
ROLES = {
    "dwt_419": "small graph / overhead",
    "qh882": "structural matrix graph",
    "1138_bus": "power network",
    "dwt_1005": "structural matrix graph",
    "cage8": "high-connectivity cage graph",
    "poli": "sparse matrix graph",
    "dwt_2680": "elongated structural graph / twist-sensitive layout",
    "EVA": "clustered ownership network / sparse approximation stress test",
    "3elt": "finite-element mesh / paper comparison",
    "USpowerGrid": "power network / paper comparison",
    "G65": "synthetic benchmark graph",
    "bcsstk31": "dense structural matrix graph",
    "web-NotreDame": "large web graph / scalability stress test",
    "fe_4elt2": "finite-element mesh",
    "luxembourg_osm": "large road network / paper comparison",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rr_variant(tile: int, workgroup: int = 256, memory: str = "shared") -> dict:
    return {"id": f"T{tile}-W{workgroup}-{memory}", "tile_size": tile,
            "workgroup_size": workgroup, "coordinate_memory": memory,
            "outer_shuffle": True, "inner_shuffle": True}


def catalog_existing(names: list[str]) -> list[dict]:
    original = json.loads((ROOT / "experiment_results/v1/config/datasets.json").read_text())
    originals = {row["name"]: row for row in original}
    download_catalog = ROOT / "experiments/datasets/data-catalog.json"
    downloaded = json.loads(download_catalog.read_text()) if download_catalog.is_file() else {}
    download_by_hash = {row.get("sha256"): row for row in downloaded.get("datasets", [])
                        if row.get("sha256")}
    rows = []
    for name in names:
        entry = originals[name]
        path = ROOT / "data/v1_lcc" / (name + ".mtx")
        if not path.exists():
            path = resolve_artifact_path(entry["path"], ROOT)
        source = ROOT / "data" / (name + ".mtx")
        if not source.exists():
            source = resolve_artifact_path(entry["source_path"], ROOT)
        mapping = path.with_suffix(".vertex-map.txt")
        if not mapping.exists():
            mapping = resolve_artifact_path(entry["vertex_map_path"], ROOT)
        digest = file_sha256(path)
        source_digest = file_sha256(source)
        if digest != entry["sha256"] or source_digest != entry["source_sha256"]:
            raise ValueError(f"Input provenance checksum mismatch for {name}")
        matrix = mmread(path).tocsr()
        matrix.sum_duplicates()
        if matrix.shape[0] != matrix.shape[1] or matrix.diagonal().any():
            raise ValueError(f"Prepared input must be square with no self loops: {name}")
        adjacency = matrix.maximum(matrix.T)
        components = connected_components(adjacency, directed=False, return_labels=False)
        if components != 1 or adjacency.nnz != 2 * entry["edges"]:
            raise ValueError(f"Prepared input differs from simple LCC catalog: {name}")
        old_map = np.loadtxt(mapping, dtype=np.int64, ndmin=1)
        if old_map.size != entry["nodes"] or len(set(old_map.tolist())) != len(old_map):
            raise ValueError(f"Vertex map has missing or repeated vertices: {name}")
        downloaded_row = download_by_hash.get(source_digest, {})
        row = dict(entry)
        row.update({"path": provenance_path(path, ROOT), "sha256": digest,
                    "source_path": provenance_path(source, ROOT),
                    "source_sha256": source_digest,
                    "vertex_map_path": provenance_path(mapping, ROOT),
                    "vertex_map_sha256": file_sha256(mapping),
                    "prepared_bytes": path.stat().st_size,
                    "source_page_url": downloaded_row.get("page_url"),
                    "source_archive_url": downloaded_row.get("archive_url"),
                    "source_retrieved_on": downloaded.get("retrieved_on"),
                    "preparation": "v1 undirected simple largest connected component, reused unchanged",
                    "role": ROLES.get(name, "fixture"),
                    "full_pairs": entry["nodes"] * (entry["nodes"] - 1) // 2})
        rows.append(row)
    return rows


def manifest(group: str, rows: list[dict], methods: list[str], seeds: int,
             variants: list[dict] | None = None, timeout: int = 180) -> dict:
    result = {"experiment_id": "v2-" + group, "study_group": group,
              "run_mode": "benchmark", "methods": methods,
              "datasets": [{key: row[key] for key in ("name", "path", "sha256")}
                           for row in rows],
              "seeds": list(range(seeds)), "iterations": [15], "epsilon": [0.1],
              "pivots": [200], "repetitions": 1, "warmups": 1,
              "timeout_seconds": timeout, "fail_fast": False}
    if variants is not None:
        result["variants"] = {"rr_sgd": variants}
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--large-seeds", type=int, default=10)
    parser.add_argument("--ablation-seeds", type=int, default=10)
    args = parser.parse_args(argv)
    if min(args.seeds, args.large_seeds, args.ablation_seeds) <= 0:
        parser.error("seed counts must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    destination = args.results_dir.resolve()
    raw = destination / "raw"
    if raw.exists() and any(raw.iterdir()):
        raise SystemExit("Existing raw runs must keep their original config; use another results directory.")
    names = list(dict.fromkeys(SMALL + LARGE + ABLATION))
    catalog = catalog_existing(names)
    by_name = {row["name"]: row for row in catalog}
    config = destination / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "datasets.json").write_text(json.dumps(catalog, indent=2) + "\n")
    recipes = [
        manifest("quality-timing", [by_name[name] for name in SMALL], METHODS, args.seeds,
                 variants=[rr_variant(1024)], timeout=180),
        manifest("large-sparse", [by_name[name] for name in LARGE], METHODS[3:],
                 args.large_seeds, timeout=600),
        manifest("rr-memory-tiling", [by_name[name] for name in ABLATION], ["rr_sgd"],
                 args.ablation_seeds,
                 variants=[rr_variant(tile, memory=memory)
                           for tile in (32, 64, 128, 256, 512, 1024)
                           for memory in ("shared", "global")], timeout=180),
        manifest("rr-workgroup", [by_name[name] for name in ABLATION], ["rr_sgd"],
                 args.ablation_seeds,
                 variants=[rr_variant(1024, workgroup=workgroup)
                           for workgroup in (128, 256, 512, 1024)], timeout=180),
    ]
    for recipe in recipes:
        (config / (recipe["study_group"] + ".json")).write_text(json.dumps(recipe, indent=2) + "\n")
        print(recipe["experiment_id"], len(recipe["datasets"]), "datasets", len(recipe["seeds"]), "seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
