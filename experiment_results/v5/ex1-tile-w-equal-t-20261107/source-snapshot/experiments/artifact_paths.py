"""Resolve preserved historical paths through the repository relocation journal."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_artifact_path(value: str | Path, repository_root: Path | None = None,
                          *, prefer_relocated: bool = False) -> Path:
    """Resolve a path, using recorded moves first for archived experiment reads.

    Raw JSONL and manifests retain their original paths. Only reads use this map;
    the stored evidence is never rewritten. Missing/unmapped paths stay missing.
    """
    root = (repository_root or REPO_ROOT).resolve()
    supplied = Path(value)
    current = supplied if supplied.is_absolute() else root / supplied
    if current.exists() and not prefer_relocated:
        return current
    journal_path = root / "archive/relocations.json"
    if not journal_path.is_file():
        return current
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if journal.get("schema_version") != 1:
        raise ValueError(f"Unsupported relocation journal: {journal_path}")
    original_root = Path(journal["repository_root"])
    if supplied.is_absolute():
        for prefix in (original_root, root):
            try:
                relative = supplied.relative_to(prefix)
                break
            except ValueError:
                pass
        else:
            return current
    else:
        relative = supplied
    rebased = root / relative
    moves = sorted(journal["moves"], key=lambda move: len(Path(move["from"]).parts), reverse=True)
    for move in moves:
        old, new = Path(move["from"]), Path(move["to"])
        if old.is_absolute() or new.is_absolute() or ".." in old.parts + new.parts:
            raise ValueError(f"Relocation paths must stay inside the repository: {move}")
        try:
            suffix = relative.relative_to(old)
        except ValueError:
            continue
        relocated = root / new / suffix
        if relocated.exists():
            return relocated
    if rebased.exists():
        return rebased
    return current


def provenance_path(path: Path, repository_root: Path | None = None) -> str:
    """Keep repository-relative evidence where possible, including external runs."""
    root = (repository_root or REPO_ROOT).resolve()
    path = path.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
