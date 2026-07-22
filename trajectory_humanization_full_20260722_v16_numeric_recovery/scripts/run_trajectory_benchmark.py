#!/usr/bin/env python3
"""CLI for the strict five-action feature + raw Deep PAD benchmark.

The only shortcut is the explicitly named ``--synthetic-smoke`` gate.  A
formal dataset run always reapplies the two distinct pool rules from the fixed
fake-user split JSON before training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.benchmark_runner import BenchmarkConfig, load_benchmark_dataset, run_complete_benchmark
from detectors.deep_pad import (
    ACTIONS,
    DeepTrainConfig,
    RawTrajectoryRecord,
    assign_strict_protocol_pools,
    load_fake_user_split,
    make_record,
)


def _single_pointer_record(
    action: str, label: int, user: int, pool: str, sample_id: str, rng: np.random.RandomState
) -> RawTrajectoryRecord:
    t = int(rng.randint(7, 13))
    times = np.cumsum(rng.uniform(10.0, 24.0, size=t)).astype(np.float32)
    times -= times[0]
    fake_shift = 35.0 * label
    xy = np.stack(
        (
            np.linspace(100.0, 650.0 + fake_shift, t) + rng.normal(0.0, 3.0, t),
            np.linspace(900.0, 420.0 - fake_shift, t) + rng.normal(0.0, 3.0, t),
        ), axis=-1,
    ).astype(np.float32)
    if action == "tap":
        xy = np.repeat(xy[:1], t, axis=0) + rng.normal(0.0, 0.7 + label, (t, 2))
    values = np.zeros((2, t, 4), dtype=np.float32)
    values[0, :, :2] = xy
    values[0, :, 2] = 0.35 + 0.12 * label + rng.normal(0.0, 0.01, t)
    values[0, :, 3] = 0.20 + 0.06 * label + rng.normal(0.0, 0.01, t)
    contact = np.zeros((2, t), dtype=bool)
    contact[0] = True
    code = np.full((2, t), -1, dtype=np.int16)
    code[0] = 2
    code[0, 0], code[0, -1] = 0, 1
    events = np.full((2, t), -1, dtype=np.int32)
    events[0] = 0
    return make_record(
        action=action, label=label, user_id=user, pool=pool, sample_id=sample_id,
        pointer_continuous=values, global_t_ms=times, contact_mask=contact,
        action_code=code, event_ids=events, event_group_id=sample_id,
    )


def _pinch_record(
    label: int, user: int, pool: str, sample_id: str, rng: np.random.RandomState
) -> RawTrajectoryRecord:
    t = int(rng.randint(10, 15))
    times = np.cumsum(rng.uniform(12.0, 23.0, size=t)).astype(np.float32)
    times -= times[0]
    values = np.zeros((2, t, 4), dtype=np.float32)
    contact = np.zeros((2, t), dtype=bool)
    # Pointer 0 is down before pointer 1 and up after it: a shared global
    # timeline, not two independently normalized sequences.
    contact[0, :t] = True
    contact[1, 2:t - 1] = True
    center = np.stack((np.linspace(420, 500, t), np.linspace(760, 690, t)), axis=-1)
    span = np.linspace(50, 180 + 30 * label, t)
    values[0, :, :2] = center - np.stack((span, np.zeros(t)), axis=-1)
    values[1, 2:t - 1, :2] = (center + np.stack((span, np.zeros(t)), axis=-1))[2:t - 1]
    for pointer in range(2):
        values[pointer, contact[pointer], 2] = 0.4 + 0.08 * label
        values[pointer, contact[pointer], 3] = 0.22 + 0.05 * label
    code = np.full((2, t), -1, dtype=np.int16)
    code[0, :] = 2
    code[0, 0], code[0, -1] = 0, 1
    code[1, 2:t - 1] = 2
    code[1, 2], code[1, t - 2] = 5, 6
    events = np.where(contact, 0, -1).astype(np.int32)
    return make_record(
        action="pinch", label=label, user_id=user, pool=pool, sample_id=sample_id,
        pointer_continuous=values, global_t_ms=times, contact_mask=contact,
        action_code=code, event_ids=events, event_group_id=sample_id,
    )


def _keystroke_record(
    label: int, user: int, pool: str, sample_id: str, rng: np.random.RandomState
) -> RawTrajectoryRecord:
    # Three two-token contacts and two explicit flight tokens.
    t = 8
    contact_indices = ((0, 1), (3, 4), (6, 7))
    gap = np.zeros(t, dtype=bool)
    gap[[2, 5]] = True
    times = np.asarray([0, 35, 95, 100, 138, 215, 220, 258 + 8 * label], dtype=np.float32)
    values = np.zeros((2, t, 4), dtype=np.float32)
    contact = np.zeros((2, t), dtype=bool)
    code = np.full((2, t), -1, dtype=np.int16)
    keycode = np.full((2, t), -1, dtype=np.int32)
    event_ids = np.full((2, t), -1, dtype=np.int32)
    for event, indices in enumerate(contact_indices):
        key = (29, 30, 31)[event]
        for local, index in enumerate(indices):
            contact[0, index] = True
            values[0, index] = np.asarray(
                [220 + 120 * event + label * 8, 1180 + 5 * event, 0.4 + 0.1 * label, 0.24 + 0.04 * label]
            )
            code[0, index] = 0 if local == 0 else 1
            keycode[0, index] = key
            event_ids[0, index] = event
    return make_record(
        action="keystroke", label=label, user_id=user, pool=pool, sample_id=sample_id,
        pointer_continuous=values, global_t_ms=times, contact_mask=contact,
        action_code=code, keycode=keycode, event_ids=event_ids, gap_mask=gap,
        event_group_id=sample_id,
    )


def synthetic_five_action_dataset(seed: int = 13) -> Tuple[List[RawTrajectoryRecord], Dict[str, np.ndarray]]:
    """Small gate dataset; it is never presented as a formal benchmark result."""

    rng = np.random.RandomState(seed)
    records: List[RawTrajectoryRecord] = []
    feature_rows: Dict[str, List[np.ndarray]] = {action: [] for action in ACTIONS}
    for action_index, action in enumerate(ACTIONS):
        for pool_index, pool in enumerate(("train", "val", "test")):
            for label in (0, 1):
                for user_local in range(2):
                    user = pool_index * 20 + label * 5 + user_local
                    for event in range(3):
                        sample_id = "smoke_%s_%s_%d_%d_%d" % (action, pool, label, user, event)
                        if action == "pinch":
                            record = _pinch_record(label, user, pool, sample_id, rng)
                        elif action == "keystroke":
                            record = _keystroke_record(label, user, pool, sample_id, rng)
                        else:
                            record = _single_pointer_record(action, label, user, pool, sample_id, rng)
                        records.append(record)
                        feature_rows[action].append(np.asarray([
                            label * 2.5 + rng.normal(0, 0.15),
                            action_index + rng.normal(0, 0.1),
                            np.mean(record.global_t_ms) / 100.0,
                            np.sum(record.contact_mask),
                            rng.normal(0, 0.3),
                            label + rng.normal(0, 0.15),
                        ], dtype=np.float64))
    return records, {action: np.stack(rows) for action, rows in feature_rows.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--fake-user-split", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        if args.dataset_dir is not None or args.fake_user_split is not None:
            raise ValueError("synthetic smoke cannot be combined with formal dataset arguments")
        records, features = synthetic_five_action_dataset()
        config = BenchmarkConfig(
            feature_model_params={"xgboost": {"n_estimators": 8, "max_depth": 2}},
            deep_model_params={
                "tcn": {"hidden_dim": 12, "n_blocks": 1, "dropout": 0.0},
                "transformer": {"hidden_dim": 12, "n_layers": 1, "n_heads": 2, "feedforward_dim": 24, "dropout": 0.0},
            },
            deep_train=DeepTrainConfig(
                epochs=1, batch_size=24, learning_rate=2e-3, patience=0,
                bootstrap_replicates=4, seed=13,
            ),
            feature_bootstrap_replicates=4,
            seed=13,
        )
        split_audit = {"mode": "synthetic_smoke_only", "formal_result": False}
    else:
        if args.dataset_dir is None or args.fake_user_split is None:
            raise ValueError("formal run requires --dataset-dir and --fake-user-split")
        records, features = load_benchmark_dataset(args.dataset_dir)
        split = load_fake_user_split(args.fake_user_split)
        records, split_audit = assign_strict_protocol_pools(records, split)
        for action in ACTIONS:
            action_rows = [record for record in records if record.action == action]
            coverage = {
                (label, pool): len({record.user_id for record in action_rows if record.label == label and record.pool == pool})
                for label in (0, 1) for pool in ("train", "val", "test")
            }
            if tuple(
                coverage[(0, pool)] for pool in ("train", "val", "test")
            ) != (100, 100, 100):
                raise ValueError(
                    "%s formal real train/val/test event pools must each cover all 100 users"
                    % action
                )
            if tuple(coverage[(1, pool)] for pool in ("train", "val", "test")) != (70, 10, 20):
                raise ValueError("%s formal fake pools do not cover fixed 70/10/20 users" % action)
        config = BenchmarkConfig(
            deep_train=DeepTrainConfig(
                epochs=args.epochs, batch_size=args.batch_size,
                bootstrap_replicates=args.bootstrap_replicates,
            ),
            feature_bootstrap_replicates=args.bootstrap_replicates,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    outputs = run_complete_benchmark(
        records, features, output_dir=args.output_dir, config=config, device=args.device
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
