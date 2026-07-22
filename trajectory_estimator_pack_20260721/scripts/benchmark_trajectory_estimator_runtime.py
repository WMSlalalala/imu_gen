#!/usr/bin/env python3
"""Benchmark the formal five-action trajectory estimator release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from estimator.runtime_benchmark import benchmark_trajectory_estimator_latency
from estimator.trajectory_release import ACTIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--iterations-per-action", type=int, default=25)
    parser.add_argument("--warmup-per-action", type=int, default=2)
    args = parser.parse_args()
    report = benchmark_trajectory_estimator_latency(
        manifest_path=args.manifest, bundle_dir=args.bundle_dir,
        output_path=args.output, device=args.device, actions=ACTIONS,
        iterations_per_action=args.iterations_per_action,
        warmup_per_action=args.warmup_per_action, load_deep=True,
    )
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
