#!/usr/bin/env python3
"""Independently re-audit a supplementary PAD bundle from current bytes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supplement.bundle import audit_supplement_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--require-variant",
        choices=("fully_user_disjoint", "fully_user_disjoint_reference_excluded"),
    )
    args = parser.parse_args()
    print(json.dumps(
        audit_supplement_bundle(
            args.bundle_dir, require_variant=args.require_variant
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
