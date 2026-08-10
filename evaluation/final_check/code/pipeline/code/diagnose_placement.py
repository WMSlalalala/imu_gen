#!/usr/bin/env python3
"""Measure what the endpoint binding actually does to a generated sample bank.

Two numbers decide whether the placement is doing its job or quietly destroying
the generator's output, and both have to be reported next to the FAR:

* the similarity **scale** applied.  Matching a generated chord to the carrier's
  chord can inflate a path that happened to return near its start; a run where
  the upper tail runs to 10x or more is not testing the generator any more.
* how often the placement had to **fall back to translation**, which happens
  when the generated chord is too short relative to the path's own extent for
  its direction to mean anything.

Run this after a bank is sampled and before reading its detector numbers.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sample_bank_baseline import (  # noqa: E402
    MINIMUM_CHORD_RATIO,
    PLACEMENT_DRAWS,
    SCALE_BAND,
    _draw_index,
    screen_dimensions,
)
from hmog_baseline_common import (  # noqa: E402
    fit_to_screen,
    iter_shards,
    load_shard,
    zoh_resample,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=25)
    args = parser.parse_args()

    with open(args.banks, "rb") as handle:
        banks = pickle.load(handle).get("trajectory", {})
    if not banks:
        raise SystemExit("this bank holds no trajectories")
    with open(args.binding, "rb") as handle:
        binding = pickle.load(handle)

    scales: dict[str, list[float]] = defaultdict(list)
    fallbacks: dict[str, int] = defaultdict(int)
    shrinks: dict[str, list[float]] = defaultdict(list)
    seen: dict[str, set] = defaultdict(set)
    for path in list(iter_shards(args.source_dir))[: args.shards]:
        arrays = load_shard(path)
        offsets, labels, actions = arrays["offsets"], arrays["label"], arrays["action"]
        split = str(arrays["split"])
        for index in np.flatnonzero(labels == 1):
            action = str(actions[index])
            bank = banks.get(action)
            if bank is None:
                continue
            start_row, stop_row = int(offsets[index]), int(offsets[index + 1])
            event_id = str(arrays["event_id"][index])
            samples = stop_row - start_row
            orientation, start_px, end_px = binding[event_id]
            width, height = screen_dimensions(orientation)
            dimensions = np.asarray((width, height))
            carrier = (
                np.asarray(
                    arrays["trajectory_flat"][start_row:stop_row, 1:3], np.float64
                )
                * dimensions
            )
            contact = np.flatnonzero(
                arrays["trajectory_flat"][start_row:stop_row, 0] > 0.0
            )
            first, last = (
                (int(contact[0]), int(contact[-1]))
                if len(contact) >= 2
                else (0, samples - 1)
            )
            start = (
                np.asarray(start_px, float)
                if start_px is not None
                else carrier[first]
            )
            end = np.asarray(end_px, float) if end_px is not None else carrier[last]

            # Mirror the builder's draw policy exactly, or the diagnosis would
            # describe a placement the dataset never performed.
            target_probe = float(np.linalg.norm(end - start))
            best_path, best_cost = None, None
            for attempt in range(1 if target_probe < 1e-9 else PLACEMENT_DRAWS):
                probe = _draw_index(
                    event_id, f"trajectory#{attempt}", len(bank), split
                )
                candidate = zoh_resample(bank[probe], samples).astype(np.float64)
                candidate_px = candidate * dimensions
                probe_length = float(
                    np.linalg.norm(candidate_px[last] - candidate_px[first])
                )
                cost = (
                    np.inf
                    if probe_length < 1e-9 or target_probe < 1e-9
                    else abs(np.log(target_probe / probe_length))
                )
                if best_cost is None or cost < best_cost:
                    best_path, best_cost = candidate_px, cost
                if target_probe < 1e-9 or cost <= abs(np.log(SCALE_BAND[1])):
                    break
            path_px = best_path
            anchor = path_px[first]
            source_length = float(np.linalg.norm(path_px[last] - anchor))
            target_length = float(np.linalg.norm(end - start))
            extent = float(np.max(np.linalg.norm(path_px - anchor, axis=1)))
            if target_length < 1e-9 or source_length < max(
                1e-9, MINIMUM_CHORD_RATIO * extent
            ):
                fallbacks[action] += 1
                placed = path_px - anchor + start
            else:
                scale = target_length / source_length
                scales[action].append(scale)
                placed = (path_px - anchor) * scale + start
            owned = arrays["trajectory_flat"][start_row:stop_row, 0] > 0.0
            _, shrink = fit_to_screen(placed, start, dimensions, considered=owned)
            shrinks[action].append(shrink)
            seen[action].add(probe)

    report = {}
    print(f"{'action':10s}{'n':>7s}{'fallback':>10s}{'s p50':>9s}{'s p95':>9s}"
          f"{'s p99':>9s}{'s max':>9s}{'shrunk':>9s}{'distinct':>10s}")
    for action in sorted(shrinks):
        values = np.array(scales[action]) if scales[action] else np.array([1.0])
        shrunk = np.array(shrinks[action])
        total = len(shrunk)
        report[action] = {
            "events": int(total),
            "translation_fallbacks": int(fallbacks[action]),
            "scale_p50": float(np.percentile(values, 50)),
            "scale_p95": float(np.percentile(values, 95)),
            "scale_p99": float(np.percentile(values, 99)),
            "scale_max": float(values.max()),
            "events_shrunk_to_fit": int(np.count_nonzero(shrunk < 1.0)),
            "min_shrink": float(shrunk.min()),
            "distinct_bank_windows_used": len(seen[action]),
        }
        row = report[action]
        print(
            f"{action:10s}{total:>7d}{row['translation_fallbacks']:>10d}"
            f"{row['scale_p50']:>9.2f}{row['scale_p95']:>9.2f}"
            f"{row['scale_p99']:>9.2f}{row['scale_max']:>9.2f}"
            f"{row['events_shrunk_to_fit']:>9d}"
            f"{row['distinct_bank_windows_used']:>10d}"
        )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
