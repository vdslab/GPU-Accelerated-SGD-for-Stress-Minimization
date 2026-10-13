"""Validate optional method-specific metadata without changing legacy records."""
from __future__ import annotations

from typing import Any, Mapping

RR_METADATA_FIELDS = (
    "rr_tile_size", "rr_workgroup_size", "rr_memory_mode", "rr_shared_bytes",
    "rr_outer_shuffle", "rr_inner_shuffle",
)


def validate_rr_metadata(record: Mapping[str, Any]) -> None:
    """Accept legacy records or a complete, consistent Full RR metadata block.

    Unknown fields remain the enclosing schema validator's responsibility.
    A partial block cannot describe the executed schedule unambiguously.
    """
    present = set(RR_METADATA_FIELDS).intersection(record)
    if not present:
        return
    if record.get("method") != "rr_sgd":
        raise ValueError("RR metadata is only permitted for rr_sgd")
    missing = set(RR_METADATA_FIELDS) - present
    if missing:
        raise ValueError("Incomplete RR metadata: missing " + ", ".join(sorted(missing)))
    for field in ("rr_tile_size", "rr_workgroup_size"):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1024:
            raise ValueError(f"{field} must be an integer from 1 to 1024")
    if record["rr_memory_mode"] not in ("shared", "global"):
        raise ValueError("rr_memory_mode must be shared or global")
    for field in ("rr_outer_shuffle", "rr_inner_shuffle"):
        if not isinstance(record[field], bool):
            raise ValueError(f"{field} must be boolean")
    value = record["rr_shared_bytes"]
    expected = 16 * record["rr_tile_size"] if record["rr_memory_mode"] == "shared" else 0
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"rr_shared_bytes must equal {expected} for this tile and memory mode")
