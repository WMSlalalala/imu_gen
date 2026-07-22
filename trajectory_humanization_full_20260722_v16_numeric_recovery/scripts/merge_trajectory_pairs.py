#!/usr/bin/env python3
"""Strictly merge and audit all 25 formal trajectory PAD pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.pair_merge import merge_and_audit_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    result = merge_and_audit_pairs(args.experiment_root)
    print(json.dumps({
        "status": result["status"],
        "n_pairs": result["n_pairs"],
        "n_operating_rows": result["n_operating_rows"],
        "report": result["outputs"]["report"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
