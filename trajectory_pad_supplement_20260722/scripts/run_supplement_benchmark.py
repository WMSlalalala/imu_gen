#!/usr/bin/env python3
"""Run all 25 PAD models on one already-audited sensitivity bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supplement.bundle import V16_ROOT, audit_supplement_bundle, sha256_file

if str(V16_ROOT) not in sys.path:
    sys.path.insert(0, str(V16_ROOT))

from detectors.benchmark_runner import (  # noqa: E402
    BenchmarkConfig,
    load_benchmark_dataset,
    run_complete_benchmark,
)
from detectors.deep_pad import DeepTrainConfig  # noqa: E402


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--confirm-formal-supplement", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_formal_supplement:
        raise PermissionError("formal supplement requires --confirm-formal-supplement")
    if args.epochs != 40 or args.bootstrap_replicates != 500:
        raise ValueError("formal supplement requires 40 epochs and 500 bootstrap replicates")
    bundle_receipt = audit_supplement_bundle(args.bundle_dir)
    records, features = load_benchmark_dataset(args.bundle_dir)
    config = BenchmarkConfig(
        deep_train=DeepTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=0,
            seed=args.seed,
            bootstrap_replicates=args.bootstrap_replicates,
        ),
        feature_bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    outputs = run_complete_benchmark(
        records,
        features,
        output_dir=args.output_dir,
        config=config,
        device=args.device,
    )
    benchmark_manifest = Path(outputs["manifest"])
    receipt = {
        "schema_version": "trajectory_pad_supplement_benchmark_receipt_v1",
        "status": "complete",
        "variant": bundle_receipt["variant"],
        "bundle_manifest": bundle_receipt["bundle_manifest"],
        "bundle_manifest_sha256": bundle_receipt["bundle_manifest_sha256"],
        "bundle_audit": str(Path(args.bundle_dir) / "bundle_audit.json"),
        "bundle_audit_sha256": sha256_file(Path(args.bundle_dir) / "bundle_audit.json"),
        "benchmark_manifest": str(benchmark_manifest),
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest),
        "device": args.device,
        "seed": args.seed,
        "epochs": args.epochs,
        "bootstrap_replicates": args.bootstrap_replicates,
        "v16_dependencies": {
            "detectors/benchmark_runner.py": sha256_file(V16_ROOT / "detectors/benchmark_runner.py"),
            "detectors/deep_pad.py": sha256_file(V16_ROOT / "detectors/deep_pad.py"),
            "detectors/feature_pad.py": sha256_file(V16_ROOT / "detectors/feature_pad.py"),
        },
    }
    receipt_path = Path(args.output_dir) / "supplement_receipt.json"
    _atomic_json(receipt_path, receipt)
    print(json.dumps({"status": "complete", "receipt": str(receipt_path)}, indent=2))


if __name__ == "__main__":
    main()
