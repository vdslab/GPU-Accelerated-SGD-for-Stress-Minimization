#!/usr/bin/env python3
"""Aggregate experiment JSONL into reproducible paper tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from aggregation import (
        ARTIFACT_NAMES,
        PROFILES,
        AggregationError,
        PublicationGateError,
        dry_run_summary,
        generate_artifacts,
        load_experiment,
    )
except ModuleNotFoundError:
    from experiments.aggregation import (
        ARTIFACT_NAMES,
        PROFILES,
        AggregationError,
        PublicationGateError,
        dry_run_summary,
        generate_artifacts,
        load_experiment,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        action="append",
        required=True,
        type=Path,
        help="runner output directory; repeat to combine experiments",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.dry_run and arguments.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is used")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        experiments = [
            load_experiment(directory)
            for directory in arguments.experiment_dir
        ]
        if len({experiment.experiment_id for experiment in experiments}) != len(
            experiments
        ):
            raise AggregationError("experiment_idが重複しています")
        if arguments.dry_run:
            print(
                json.dumps(
                    dry_run_summary(experiments, arguments.profile),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0
        report = generate_artifacts(
            experiments, arguments.profile, arguments.output_dir
        )
        print(
            f"Generated {len(ARTIFACT_NAMES)} artifacts in "
            f"{arguments.output_dir.resolve()} "
            f"(publication_ready={str(report['publication_ready']).lower()})"
        )
        return 0
    except PublicationGateError as error:
        print(
            json.dumps(error.report, ensure_ascii=False, sort_keys=True, indent=2),
            file=sys.stderr,
        )
        return 2
    except (AggregationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
