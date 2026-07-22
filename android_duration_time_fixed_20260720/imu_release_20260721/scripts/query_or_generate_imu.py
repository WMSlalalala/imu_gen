#!/usr/bin/env python3
"""Query the audited cache or run online five-shot IMU diffusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imu_release import ACTIONS, IMUReleaseService  # noqa: E402


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cache", "online"), default="cache")
    parser.add_argument("--action", choices=ACTIONS, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument("--active-len", type=int)
    parser.add_argument("--start-time-ns", type=int)
    parser.add_argument("--xy-start", nargs=2, type=float)
    parser.add_argument("--xy-end", nargs=2, type=float)
    parser.add_argument("--orientation-id", type=int)
    parser.add_argument("--text")
    parser.add_argument("--n-keys", type=int)
    parser.add_argument("--n-letters", type=int)
    parser.add_argument("--pinch-start-span", type=float)
    parser.add_argument("--pinch-end-span", type=float)
    parser.add_argument("--noise-seed", type=int)
    parser.add_argument("--match-mode", choices=("nearest", "strict"), default="nearest")
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    service = IMUReleaseService(mode=args.mode, device=args.device)
    conditions = {
        name: value for name, value in {
            "user_id": args.user_id, "duration_ms": args.duration_ms,
            "active_len": args.active_len, "start_time_ns": args.start_time_ns,
            "xy_start": args.xy_start, "xy_end": args.xy_end,
            "orientation_id": args.orientation_id, "text": args.text,
            "n_keys": args.n_keys, "n_letters": args.n_letters,
            "pinch_start_span": args.pinch_start_span,
            "pinch_end_span": args.pinch_end_span, "noise_seed": args.noise_seed,
            "match_mode": args.match_mode if args.mode == "cache" else None,
        }.items() if value is not None
    }
    result = service.generate(args.action, **conditions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(args.output), active_imu=result["active_imu"],
        relative_timestamps_ns=result["relative_timestamps_ns"],
        timestamps_ns=(
            np.asarray([], dtype=np.int64) if result.get("timestamps_ns") is None
            else result["timestamps_ns"]
        ),
        metadata_json=np.asarray(json.dumps(_json_safe(result), sort_keys=True)),
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps({"health": service.health(), "result": _json_safe(result)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output": str(args.output), "summary": str(args.summary)}, indent=2))


if __name__ == "__main__":
    main()
