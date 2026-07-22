#!/usr/bin/env python3
"""Independent fail-closed audit of a completed 25-pair merge tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.pair_merge import audit_merged_pair_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument(
        "--allow-nonformal",
        action="store_true",
        help="test-only: audit a quick/non-formal fixture instead of enforcing 40/0/500",
    )
    args = parser.parse_args()
    result = audit_merged_pair_tree(
        args.experiment_root,
        require_formal=not args.allow_nonformal,
        write_audit=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
