#!/usr/bin/env python3
"""Run or resume exactly one formal action/detector pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.deep_pad import ACTIONS, DEEP_DETECTORS, DeepTrainConfig
from detectors.feature_pad import ALLOWED_DETECTORS
from detectors.pair_runner import run_or_resume_pair, stable_pair_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--fake-user-split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--action", choices=ACTIONS, required=True)
    parser.add_argument("--family", choices=("feature_pad", "deep_pad"), required=True)
    parser.add_argument(
        "--detector", choices=tuple(ALLOWED_DETECTORS) + tuple(DEEP_DETECTORS), required=True
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--real-hash-seed", type=int, default=20260713)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--model-params-json", default="{}")
    parser.add_argument("--batch-probe-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = json.loads(args.model_params_json)
    if not isinstance(params, dict):
        raise ValueError("model-params-json must decode to an object")
    pair_seed = stable_pair_seed(args.seed, args.action, args.family, args.detector)
    deep_config = DeepTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=pair_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    result = run_or_resume_pair(
        dataset_file=args.dataset_dir / (args.action + ".npz"),
        fake_user_split=args.fake_user_split,
        output_root=args.output_root,
        action=args.action,
        family=args.family,
        detector=args.detector,
        deep_config=deep_config,
        feature_bootstrap_replicates=args.bootstrap_replicates,
        seed=pair_seed,
        base_seed=args.seed,
        real_hash_seed=args.real_hash_seed,
        device=args.device,
        feature_model_params=params if args.family == "feature_pad" else None,
        deep_model_params=params if args.family == "deep_pad" else None,
        batch_probe_path=args.batch_probe_json,
    )
    print(json.dumps({
        "status": result["status"],
        "manifest": result["manifest"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
