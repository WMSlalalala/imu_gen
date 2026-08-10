#!/usr/bin/env python3
"""Assemble the cross-method comparison table from every finished cell grid.

Reads each grid's FAR at the development FRR=5% threshold and prints one table
per modality: rows are action x method, columns are the six detectors.  The
paper's own method is included as a row so the comparison is read off one page.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarise_far5 import DETECTORS, declined_actions, read_grid  # noqa: E402

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
TARGET = 0.6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        action="append",
        required=True,
        metavar="LABEL=CELLS[,DATASET]",
        help=(
            "a cell directory to include, labelled for the table.  Append the "
            "built dataset after a comma so the actions that baseline declined "
            "are dropped rather than reported as its results -- those cells "
            "still hold the source method's own attack."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    grids = {}
    for entry in args.grid:
        label, _, path = entry.partition("=")
        cells_path, _, dataset_path = path.partition(",")
        cells = Path(cells_path)
        if not cells.is_dir():
            print(f"skip {label}: {cells} missing")
            continue
        declined = declined_actions(Path(dataset_path) if dataset_path else None)
        grid = {
            key: value
            for key, value in read_grid(cells).items()
            if key[0] not in declined
        }
        grids[label] = grid
        note = f", declined {sorted(declined)}" if declined else ""
        print(f"loaded {label}: {len(grid)} cells{note}")

    modalities = sorted({key[1] for grid in grids.values() for key in grid})
    report: dict[str, dict] = {}
    for modality in modalities:
        header = f"\n### {modality}\n"
        header += f"{'action':<10s}{'method':<26s}" + "".join(
            f"{name[:12]:>13s}" for name in DETECTORS
        )
        print(header)
        print("-" * (36 + 13 * len(DETECTORS)))
        for action in ACTIONS:
            for label, grid in grids.items():
                keys = [(action, modality, d) for d in DETECTORS]
                if not any(key in grid for key in keys):
                    continue
                row = "".join(
                    f"{grid[key][0]:>13.4f}" if key in grid else f"{'-':>13s}"
                    for key in keys
                )
                print(f"{action:<10s}{label:<26s}{row}")
            print()

        print(f"{'method':<26s}{'n':>5s}{'median':>9s}{'>=0.6':>8s}{'>=0.5':>8s}")
        for label, grid in grids.items():
            values = sorted(v for k, (v, _) in grid.items() if k[1] == modality)
            if not values:
                continue
            report.setdefault(modality, {})[label] = {
                "n": len(values),
                "median": statistics.median(values),
                "at_least_0.6": sum(1 for v in values if v >= TARGET),
                "at_least_0.5": sum(1 for v in values if v >= 0.5),
                "min": values[0],
                "max": values[-1],
            }
            summary = report[modality][label]
            print(
                f"{label:<26s}{summary['n']:>5d}{summary['median']:>9.4f}"
                f"{summary['at_least_0.6']:>8d}{summary['at_least_0.5']:>8d}"
            )

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
