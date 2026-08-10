#!/usr/bin/env python3
"""FAR at the development-selected 5% FRR operating point, for any cell grid.

The cells publish `far` at the development EER threshold, which rejects about
37% of genuine attempts and is not an operating point anyone deploys.
`thresholds.json` also stores `frr5`, the threshold development picked for 5%
genuine rejection; the frozen test scores are cut there instead.

DECLINED ACTIONS ARE NOT RESULTS.  When a baseline cannot address an action it
leaves the carrier alone -- and the carrier's fake signal is the paper's own
attack, not a neutral filler.  Those cells therefore report the paper's numbers
under a baseline's name.  Pass `--dataset` and the summary reads the built
dataset's declared per-action declines, prints those cells as `declined`, and
leaves them out of every aggregate.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
DETECTORS = (
    "hmog_style_svm",
    "hmog_style_rf",
    "paper_svm",
    "paper_xgboost",
    "behaveformer_stdat",
    "authconformer",
)
TARGET = 0.6


def read_grid(cells_dir: Path) -> dict:
    """Read every finished cell.  A cell still running has a directory but no
    scores yet; skipping it keeps a partial grid readable, and the caller's
    denominator check reports how many are missing."""

    cells = {}
    for cell in sorted(cells_dir.iterdir()):
        if not (cell / "thresholds.json").is_file():
            continue
        if not (cell / "test_scores.jsonl").is_file():
            continue
        thresholds = json.loads((cell / "thresholds.json").read_text())
        if thresholds["score_direction"] != "larger_is_more_fake":
            raise SystemExit(f"unexpected score direction in {cell.name}")
        cut = float(thresholds["frr5"])
        fake = accepted = genuine = rejected = 0
        with (cell / "test_scores.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                score = float(row["fake_high_score"])
                if int(row["label"]) == 1:
                    fake += 1
                    accepted += score < cut
                else:
                    genuine += 1
                    rejected += score >= cut
        if not fake or not genuine:
            # A cell whose test split lost one of its classes cannot state a
            # rate.  Reporting 0.0 here would read as "the attack never got in";
            # skipping leaves it visible as a missing cell in the denominator.
            print(
                f"  skipping {cell.name}: {fake} fake / {genuine} genuine rows",
                file=sys.stderr,
            )
            continue
        cells[tuple(cell.name.split("__"))] = (accepted / fake, rejected / genuine)
    return cells


def declined_actions(dataset_dir: Path | None) -> set[str]:
    """Actions whose cells still hold the source method's own fake signal."""

    if dataset_dir is None:
        return set()
    counts_path = dataset_dir / "baseline_counts.json"
    if not counts_path.is_file():
        return set()
    counts = json.loads(counts_path.read_text())
    declined = set()
    for action, tally in counts.get("per_action", {}).items():
        swapped = tally.get("trajectory_swapped", 0) + tally.get("imu_swapped", 0)
        if swapped == 0 and tally.get("fake", 0) > 0:
            declined.add(action)
    return declined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cells", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="the built baseline dataset, so declined actions can be excluded",
    )
    parser.add_argument(
        "--source-grid",
        action="store_true",
        help="this grid is the source method's own, so nothing was declined",
    )
    args = parser.parse_args()

    # A table without the dataset behind it cannot know which cells still hold
    # the source method's own attack, and would report them as the baseline's.
    # Refuse rather than print something that reads correct and is not.
    if args.dataset is None and not args.source_grid:
        raise SystemExit(
            "pass --dataset <built baseline dataset> so declined actions can be "
            "excluded, or --source-grid if this grid is the source method's own"
        )

    cells = read_grid(args.cells)
    declined = declined_actions(args.dataset)
    if declined:
        print(f"declined by this baseline, excluded from every aggregate: "
              f"{sorted(declined)}")
    cells = {key: value for key, value in cells.items() if key[0] not in declined}
    modalities = sorted({key[1] for key in cells})
    header = f"{'action':<10s}{'modality':<24s}" + "".join(
        f"{name[:12]:>13s}" for name in DETECTORS
    )
    print(f"FAR at the development-selected FRR=5% threshold (target >= {TARGET})")
    print(header)
    print("-" * len(header))
    for action in ACTIONS:
        for modality in modalities:
            if action in declined:
                row = "".join(f"{'declined':>13s}" for _ in DETECTORS)
            else:
                row = "".join(
                    f"{cells[(action, modality, d)][0]:>13.4f}"
                    if (action, modality, d) in cells
                    else f"{'-':>13s}"
                    for d in DETECTORS
                )
            print(f"{action:<10s}{modality:<24s}{row}")

    far = sorted(value for value, _ in cells.values())
    frr = sorted(value for _, value in cells.values())
    total = len(far)
    # State the denominator rather than letting it drift: a missing cell would
    # otherwise shrink it silently and every aggregate below would describe a
    # different grid than the reader assumes.
    expected = (len(ACTIONS) - len(declined)) * len(modalities) * len(DETECTORS)
    if total != expected:
        print(
            f"\nWARNING: {total} cells present, {expected} expected for "
            f"{len(ACTIONS) - len(declined)} actions x {len(modalities)} "
            f"modalities x {len(DETECTORS)} detectors -- aggregates below cover "
            f"only what is present"
        )
    print(
        f"\nall {total} cells: min {far[0]:.4f}  median {statistics.median(far):.4f}  "
        f"max {far[-1]:.4f}  mean {sum(far) / total:.4f}"
    )
    print(
        f"FAR >= {TARGET}: {sum(1 for v in far if v >= TARGET)}/{total}"
        f"    FAR >= 0.5: {sum(1 for v in far if v >= 0.5)}/{total}"
    )
    print(f"realised test FRR: median {statistics.median(frr):.4f} (target 0.05)")

    for dimension, index in (("detector", 2), ("action", 0), ("modality", 1)):
        table = collections.defaultdict(list)
        for key, (value, _) in cells.items():
            table[key[index]].append(value)
        print(
            f"\n{dimension:<24s}{'n':>4s}{'min':>9s}{'median':>9s}{'max':>9s}"
            f"{'>=0.6':>7s}"
        )
        for key, values in sorted(
            table.items(), key=lambda item: -statistics.median(item[1])
        ):
            values = sorted(values)
            print(
                f"{key:<24s}{len(values):>4d}{values[0]:>9.3f}"
                f"{statistics.median(values):>9.3f}{values[-1]:>9.3f}"
                f"{sum(1 for v in values if v >= TARGET):>7d}"
            )

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {
                    "__".join(key): {"far_at_frr5": far_value, "test_frr": frr_value}
                    for key, (far_value, frr_value) in sorted(cells.items())
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
