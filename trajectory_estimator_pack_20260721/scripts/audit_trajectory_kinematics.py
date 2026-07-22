#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estimator.trajectory_kinematics_audit import audit_trajectory_kinematics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-violation", action="store_true")
    args = parser.parse_args()
    report = audit_trajectory_kinematics(bundle_dir=args.bundle_dir, output_path=args.output)
    print(json.dumps({"passed": report["passed"], "violations": report["violations"], "output_sha256": report["output_sha256"]}, indent=2))
    if args.fail_on_violation and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
