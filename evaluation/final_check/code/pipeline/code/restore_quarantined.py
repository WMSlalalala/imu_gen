#!/usr/bin/env python3
"""Put back samples the resume check quarantined for the wrong reason.

`existing_sample_is_valid` in the upstream cache script hardcodes a five-shot
expectation (`ref_count == 5`, five unique reference indices).  A cache drawn
under a protocol with any other reference count therefore fails its own resume
check: every restart moved perfectly good samples into the quarantine tree and
redrew them.  The write-side invariant is now retargeted along with the resume
one, so this cannot recur -- but the samples already moved out are fine and
worth putting back rather than spending the GPU time again.

Each candidate is re-validated on the way back: it must load, carry the
metadata the cache format requires, name the user, action and split its path
claims, and carry the reference count this protocol actually uses.  Anything
that fails is left where it is.  Nothing is overwritten: if a sample was
already redrawn at the destination, the quarantined copy is dropped, because
the fresh one is the one the manifest describes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

QUARANTINE_SUFFIX = re.compile(r"\.corrupt\.\d+$")


def restore(quarantine: Path, cache: Path, expected_refs: int, apply: bool) -> dict:
    counts = {
        "examined": 0,
        "restored": 0,
        "already_present": 0,
        "wrong_reference_count": 0,
        "unreadable": 0,
        "path_metadata_mismatch": 0,
    }
    for path in sorted(quarantine.rglob("*.corrupt.*")):
        counts["examined"] += 1
        relative = path.relative_to(quarantine)
        original_name = QUARANTINE_SUFFIX.sub("", path.name)
        destination = cache / relative.parent / original_name
        if destination.exists():
            counts["already_present"] += 1
            continue
        try:
            with np.load(str(path), allow_pickle=False) as handle:
                required = {"action", "hz", "window", "active_imu", "mask",
                            "valid_mask", "metadata_json"}
                if not required.issubset(set(handle.files)):
                    counts["unreadable"] += 1
                    continue
                meta = json.loads(str(np.asarray(handle["metadata_json"]).item()))
        except Exception:  # noqa: BLE001 -- a truncated file is simply not restorable
            counts["unreadable"] += 1
            continue

        # The path encodes user/action/split; the metadata has to agree, or this
        # file is not the one that belongs at that destination.
        parts = relative.parts
        if len(parts) < 4 or not parts[0].startswith("user_"):
            counts["path_metadata_mismatch"] += 1
            continue
        user_id, action, split = int(parts[0].split("_")[1]), parts[1], parts[2]
        if (
            int(meta.get("user_id", -1)) != user_id
            or str(meta.get("action", "")) != action
            or str(meta.get("split", "")) != split
        ):
            counts["path_metadata_mismatch"] += 1
            continue
        if int(meta.get("ref_count", -1)) != expected_refs:
            # Genuinely from another protocol -- leave it quarantined.
            counts["wrong_reference_count"] += 1
            continue

        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
        counts["restored"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-refs", type=int, required=True)
    parser.add_argument("--apply", action="store_true",
                        help="without this the run only reports what it would do")
    args = parser.parse_args()

    counts = restore(args.quarantine, args.cache, args.expected_refs, args.apply)
    print(json.dumps(counts, indent=2, sort_keys=True))
    if not args.apply:
        print("dry run -- pass --apply to move the files")


if __name__ == "__main__":
    main()
