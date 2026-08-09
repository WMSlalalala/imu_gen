#!/usr/bin/env python3
"""Collect one generator's per-action sample files into a single bank pickle.

Two ways to name the inputs, because the generators disagree on layout:

  --samples-dir DIR        one flat directory of samples_<action>_<kind>.npy
  --sample action=PATH     one explicit file per action, repeatable

The second exists because CSDI and ImagenTime write into a subdirectory per
action, and because the same generator can produce more than one arm -- CSDI's
unconditional and five-shot draws sit side by side and must become two separate
banks, not one.  Naming the files explicitly makes which arm went into which
bank a matter of record rather than of directory layout.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, default=None,
                        help="flat directory of samples_<action>_<kind>.npy")
    parser.add_argument("--sample", action="append", default=[],
                        metavar="ACTION=PATH",
                        help="explicit file for one action; repeat per action")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--kind", action="append", choices=("trajectory", "imu"), required=True
    )
    args = parser.parse_args()

    if not args.samples_dir and not args.sample:
        raise SystemExit("give --samples-dir or at least one --sample ACTION=PATH")
    if args.sample and len(args.kind) != 1:
        raise SystemExit("--sample names one file per action, so give exactly one --kind")

    explicit = {}
    for item in args.sample:
        action, _, path = item.partition("=")
        if action not in ACTIONS:
            raise SystemExit(f"unknown action in --sample: {action}")
        if not path:
            raise SystemExit(f"--sample {item} has no path")
        explicit[action] = Path(path)

    banks: dict[str, dict[str, np.ndarray]] = {}
    for kind in args.kind:
        banks[kind] = {}
        for action in ACTIONS:
            if explicit:
                path = explicit.get(action)
                if path is None:
                    print(f"no --sample for {action}, {action}/{kind} will be declined")
                    continue
            else:
                path = args.samples_dir / f"samples_{action}_{kind}.npy"
            if not path.is_file():
                print(f"missing {path}, {action}/{kind} will be declined")
                continue
            values = np.load(path)
            if not np.isfinite(values).all():
                raise SystemExit(f"{path} holds non-finite values")
            if values.ndim != 3:
                raise SystemExit(f"{path} is {values.shape}, expected (N, T, channels)")
            banks[kind][action] = values.astype(np.float32)
            print(f"{action}/{kind}: {values.shape}  <- {path}")
    if not any(banks[kind] for kind in banks):
        raise SystemExit("every action was missing; nothing to write")
    with open(args.out, "wb") as handle:
        pickle.dump(banks, handle)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
