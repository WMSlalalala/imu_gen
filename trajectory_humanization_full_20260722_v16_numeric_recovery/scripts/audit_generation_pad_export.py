#!/usr/bin/env python3
"""Audit all generated shards at the independent PAD ingress boundary."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generation.pad_export import audit_generated_archive_tree
from generation.protocol import FixedUserSplit


DEFAULT_SPLIT = "/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=Path(DEFAULT_SPLIT))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--require-formal", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d.%s" % (os.getpid(), uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    split = FixedUserSplit.load(str(args.split_json), require_formal=True)
    result = audit_generated_archive_tree(
        args.generation_root, split, require_formal=args.require_formal
    )
    output = args.output_json or args.generation_root / "pad_ingress_audit.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite PAD ingress audit: %s" % output)
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
