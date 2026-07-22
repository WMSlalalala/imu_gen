#!/usr/bin/env python3
"""Score one saved trajectory record with the packaged estimator service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PACK_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PROJECT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
)
for path in (PACK_ROOT, TRAJECTORY_PROJECT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from detectors.deep_pad import make_record  # noqa: E402
from estimator.service import TrajectoryEstimatorService  # noqa: E402


def _scalar(data: Any, default: Any = None) -> Any:
    if data is None:
        return default
    arr = np.asarray(data)
    if arr.shape == ():
        value = arr.item()
        return value.decode("utf-8") if isinstance(value, bytes) else value
    return default


def load_record_npz(path: Path):
    with np.load(Path(path), allow_pickle=False) as data:
        required = ("pointer_continuous", "global_t_ms", "contact_mask", "action")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError("record npz missing fields: %s" % missing)
        action = str(_scalar(data["action"]))
        return make_record(
            action=action,
            label=int(_scalar(data["label"], 0)),
            user_id=int(_scalar(data["user_id"], -1)),
            pool=str(_scalar(data["pool"], "test")),
            sample_id=str(_scalar(data["sample_id"], Path(path).stem)),
            pointer_continuous=np.asarray(data["pointer_continuous"], dtype=np.float32),
            global_t_ms=np.asarray(data["global_t_ms"], dtype=np.float32),
            contact_mask=np.asarray(data["contact_mask"], dtype=bool),
            active_mask=np.asarray(data["active_mask"], dtype=bool) if "active_mask" in data else None,
            action_code=np.asarray(data["action_code"], dtype=np.int16) if "action_code" in data else None,
            keycode=np.asarray(data["keycode"], dtype=np.int32) if "keycode" in data else None,
            event_ids=np.asarray(data["event_ids"], dtype=np.int32) if "event_ids" in data else None,
            gap_mask=np.asarray(data["gap_mask"], dtype=bool) if "gap_mask" in data else None,
            event_group_id=str(_scalar(data["event_group_id"], Path(path).stem)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-npz", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--feature-only", action="store_true")
    parser.add_argument("--deep-only", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.feature_only and args.deep_only:
        raise ValueError("--feature-only and --deep-only are mutually exclusive")
    service = TrajectoryEstimatorService.load(
        args.manifest,
        load_feature=not args.deep_only,
        load_deep=not args.feature_only,
        device=args.device,
    )
    record = load_record_npz(args.record_npz)
    result = service.estimate_record(record)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

