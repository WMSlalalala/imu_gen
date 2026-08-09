#!/usr/bin/env python3
"""The published release is the single source of truth for every comparison.

`/mnt/share/mwang49/data7/direct100k_final/` is the frozen source method.  There
are no version numbers any more: the release picks one build per action and
records the choice in `provenance.json`, and every baseline, ablation and table
in this project reads that choice from here rather than naming a dataset.

WHY THIS MATTERS MORE THAN IT LOOKS
------------------------------------
The release is not one dataset.  It is four bundles, because different actions
were finalised from different builds:

    tap, pinch  -> tap_and_pinch   (was replay_dataset_v3)
    scroll      -> scroll          (was replay_dataset_v15)
    swipe       -> swipe           (was replay_dataset_v8)
    keystroke   -> keystroke       (was replay_dataset_v10)

Each bundle carries all five actions' events, but **only the actions listed
against it are part of the release**.  A baseline built on the scroll bundle
must therefore swap scroll and leave the other four alone -- swapping them would
produce numbers for a carrier the paper does not report, and comparing those to
the release would compare two different things.  `owned_actions` is what keeps
that straight, and every builder here takes it.

A comparison is only fair cell by cell: a baseline's scroll number has to come
from the scroll bundle, against the same genuine events, at the same frozen
window.  Reading the map from the release rather than hardcoding it means the
tables cannot silently drift if the release is rebuilt.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

RELEASE = Path("/mnt/share/mwang49/data7/direct100k_final")
DATASETS = RELEASE / "datasets"
DETECTOR_MODELS = RELEASE / "detector_models"
PROVENANCE = RELEASE / "provenance.json"
BUNDLE_MAP = DATASETS / "ACTION_BUNDLE_MAP.json"

# Where the release's own cells were computed, so the reference row can be read
# without recomputing anything.  These are the working-tree directories named by
# provenance.json; the release itself ships only the fitted models.
WORKING_RESULTS = Path("/mnt/share/mwang49/data7/results/direct100k")

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
MODALITIES = ("trajectory_xytime", "imu_only", "imu_trajectory_xytime")
DETECTORS = (
    "hmog_style_svm",
    "hmog_style_rf",
    "paper_svm",
    "paper_xgboost",
    "behaveformer_stdat",
    "authconformer",
)


@lru_cache(maxsize=1)
def bundle_map() -> dict:
    """{bundle name -> the actions it owns}, straight from the release."""

    data = json.loads(BUNDLE_MAP.read_text())
    return {name: tuple(actions) for name, actions in data["bundles"].items()}


@lru_cache(maxsize=1)
def action_to_bundle() -> dict:
    data = json.loads(BUNDLE_MAP.read_text())
    return {action: entry["dataset_bundle"] for action, entry in data["actions"].items()}


def bundle_dir(name: str) -> Path:
    directory = DATASETS / name
    if not (directory / "shards").is_dir():
        raise SystemExit(f"{name} is not a bundle in {DATASETS}")
    return directory


def bundles() -> list:
    """Every (name, directory, owned actions) a build has to iterate over."""

    return [(name, bundle_dir(name), actions) for name, actions in sorted(bundle_map().items())]


@lru_cache(maxsize=1)
def cell_sources() -> dict:
    """{(action, modality, detector) -> the working directory that computed it}."""

    data = json.loads(PROVENANCE.read_text())
    resolved = {}
    for cell, directory in data["cells"].items():
        action, modality, detector = cell.split("__")
        resolved[(action, modality, detector)] = directory
    return resolved


def reference_scores() -> dict:
    """The release's own FAR at the development-selected FRR=5% threshold.

    Read from the cells `provenance.json` points at, not recomputed: these are
    the numbers the paper reports for the source method, and a baseline table
    that recomputed them could disagree with the paper by a rounding step.

    Returns {(action, modality, detector) -> {"far":, "frr":, "source":}}.
    """

    scores = {}
    for key, directory in sorted(cell_sources().items()):
        action, modality, detector = key
        cell = WORKING_RESULTS / directory / "cells" / f"{action}__{modality}__{detector}"
        thresholds_path = cell / "thresholds.json"
        scores_path = cell / "test_scores.jsonl"
        if not thresholds_path.is_file() or not scores_path.is_file():
            continue
        thresholds = json.loads(thresholds_path.read_text())
        if thresholds["score_direction"] != "larger_is_more_fake":
            raise SystemExit(f"unexpected score direction in {cell}")
        cut = float(thresholds["frr5"])
        fake = accepted = genuine = rejected = 0
        with scores_path.open() as handle:
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
            continue
        scores[key] = {
            "far": accepted / fake,
            "frr": rejected / genuine,
            "source": directory,
            "fake_events": fake,
            "genuine_events": genuine,
        }
    return scores


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    print("bundles and the actions they own:")
    for name, directory, actions in bundles():
        shards = len(list((directory / "shards").glob("*.npz")))
        print(f"  {name:16s} {shards:3d} shards   owns: {', '.join(actions)}")

    scores = reference_scores()
    print(f"\nreference cells recovered: {len(scores)}/90")
    missing = sorted(set(cell_sources()) - set(scores))
    if missing:
        print(f"  missing {len(missing)}, e.g. {missing[:3]}")

    print("\nFAR at FRR=5%, source method (rows = action, cols = detector):")
    for modality in MODALITIES:
        print(f"\n  [{modality}]")
        header = "    " + "action".ljust(11) + "".join(d[:9].ljust(11) for d in DETECTORS)
        print(header)
        for action in ACTIONS:
            cells = [scores.get((action, modality, d)) for d in DETECTORS]
            row = "    " + action.ljust(11)
            for cell in cells:
                row += (f"{cell['far']:.3f}" if cell else "  -  ").ljust(11)
            print(row)

    values = [cell["far"] for cell in scores.values()]
    if values:
        reached = sum(1 for v in values if v >= 0.6)
        print(f"\n  median FAR {sorted(values)[len(values) // 2]:.3f}   "
              f"cells at or above 0.60: {reached}/{len(values)}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {f"{a}__{m}__{d}": v for (a, m, d), v in scores.items()},
                indent=2, sort_keys=True,
            )
        )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
