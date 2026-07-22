#!/usr/bin/env python3
"""Select an OOM-safe formal Deep PAD batch without truncating any event."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.batch_probe import probe_max_safe_batch
from detectors.deep_pad import ACTIONS, DEEP_DETECTORS
from detectors.pair_runner import _load_and_assign_action, stable_pair_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--fake-user-split", type=Path, required=True)
    parser.add_argument("--action", choices=ACTIONS, required=True)
    parser.add_argument("--detector", choices=DEEP_DETECTORS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--requested-batch-size", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=20260713)
    parser.add_argument("--real-hash-seed", type=int, default=20260713)
    parser.add_argument("--model-params-json", default="{}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    params = json.loads(args.model_params_json)
    if not isinstance(params, dict):
        raise ValueError("model-params-json must decode to an object")
    dataset_file = args.dataset_dir / (args.action + ".npz")
    records, _, _ = _load_and_assign_action(
        dataset_file, args.fake_user_split, args.action,
        args.real_hash_seed, True,
    )
    seed = stable_pair_seed(args.base_seed, args.action, "deep_pad", args.detector)
    result = probe_max_safe_batch(
        records, action=args.action, detector=args.detector,
        model_params=params, requested_batch_size=args.requested_batch_size,
        device=args.device, seed=seed, dataset_file=dataset_file,
        fake_user_split=args.fake_user_split, output_path=args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "selected_batch_size": result["selected_batch_size"],
        "longest_observed_train_event_length": result["longest_observed_train_event_length"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
