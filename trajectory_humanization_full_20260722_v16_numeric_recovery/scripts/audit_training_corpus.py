#!/usr/bin/env python3
"""Run the uncapped, all-event five-action corpus audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.corpus import FORMAL_SPLIT_PATH, atomic_json_dump, audit_corpus_directory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit every event in the formal trajectory corpus.")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=FORMAL_SPLIT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_corpus_directory(
        args.corpus_dir,
        split_path=args.split_json,
        require_pinned_split=True,
        require_all_users=True,
        validate_every_event=True,
    )
    atomic_json_dump(args.output.resolve(), result)
    print(json.dumps(result["totals"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
