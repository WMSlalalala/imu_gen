#!/usr/bin/env python3
"""Build one fully user-disjoint PAD sensitivity bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supplement.bundle import audit_supplement_bundle, build_supplement_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--reference-registry-map", type=Path, required=True)
    parser.add_argument("--exclude-references", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_supplement_bundle(
        args.primary_bundle_dir,
        args.output_dir,
        args.split_json,
        args.reference_registry_map,
        exclude_references=args.exclude_references,
    )
    receipt = audit_supplement_bundle(
        args.output_dir, require_variant=manifest["variant"]
    )
    print(json.dumps({
        "status": receipt["status"],
        "variant": receipt["variant"],
        "bundle_manifest": str(args.output_dir / "bundle_manifest.json"),
        "bundle_audit": str(args.output_dir / "bundle_audit.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
