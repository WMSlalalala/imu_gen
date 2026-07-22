#!/usr/bin/env python3
"""Benchmark one formal combined IMU+trajectory detector artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from estimator.runtime_benchmark import benchmark_total_detector_latency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    args = parser.parse_args()
    report = benchmark_total_detector_latency(
        artifact_path=args.artifact, dataset_path=args.dataset,
        expected_action=args.action, output_path=args.output,
        iterations=args.iterations, warmup_iterations=args.warmup_iterations,
    )
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
