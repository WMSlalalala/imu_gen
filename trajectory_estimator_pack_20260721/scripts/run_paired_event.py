#!/usr/bin/env python3
"""Resolve one shared EventPlan and optionally generate both modalities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


PACK_ROOT = Path(__file__).resolve().parents[1]
IMU_PROJECT = Path(
    "/home/mwang49/real-human/imu_gen/final/android_physical_layer_20260709"
)
for path in (PACK_ROOT, IMU_PROJECT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from android_imu_layer.diffusion_generator import AndroidIMUDiffusionLayer  # noqa: E402
from runtime.paired_layer import PairedGenerationService  # noqa: E402
from runtime.trajectory_layer import TrajectoryDiffusionLayer  # noqa: E402


def _xy(values: Optional[Sequence[float]], action: str, name: str):
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    expected = 4 if action == "pinch" else 2
    if array.size != expected:
        raise ValueError("%s needs %d numbers for action=%s" % (name, expected, action))
    return array.reshape(2, 2).tolist() if action == "pinch" else array.tolist()


def _save_npz(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    np.savez_compressed(str(path), **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-map", required=True)
    parser.add_argument("--reference-registry-map", required=True)
    parser.add_argument("--action", choices=("tap", "scroll", "swipe", "pinch", "keystroke"), required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--duration-ms", type=float, required=True)
    parser.add_argument("--start-time-ns", type=int)
    parser.add_argument("--orientation-id", type=int, required=True)
    parser.add_argument("--start-xy", nargs="+", type=float)
    parser.add_argument("--end-xy", nargs="+", type=float)
    parser.add_argument("--pointer-start-offset-ms", nargs="+", type=float)
    parser.add_argument("--pointer-end-offset-ms", nargs="+", type=float)
    parser.add_argument("--text")
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    layer = TrajectoryDiffusionLayer(
        checkpoint_map=args.checkpoint_map,
        reference_registry_map=args.reference_registry_map,
        device=args.device,
    )
    plan = layer.resolve_plan(
        action=args.action, user_id=args.user_id, sample_index=args.sample_index,
        sample_id=args.sample_id, duration_ms=args.duration_ms,
        start_time_ns=args.start_time_ns, orientation_id=args.orientation_id,
        start_xy=_xy(args.start_xy, args.action, "start_xy"),
        end_xy=_xy(args.end_xy, args.action, "end_xy"),
        pointer_start_offset_ms=args.pointer_start_offset_ms,
        pointer_end_offset_ms=args.pointer_end_offset_ms,
        text=args.text,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "event_plan.json"
    if plan_path.exists():
        raise FileExistsError(str(plan_path))
    plan_path.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.plan_only:
        print(json.dumps({"status": "plan_resolved", "event_plan": str(plan_path)}, indent=2))
        return

    imu_layer = AndroidIMUDiffusionLayer(device=args.device)
    result = PairedGenerationService(layer, imu_layer).generate_plan(plan)
    trajectory = result["trajectory"]
    imu = result["imu"]
    _save_npz(
        output / "trajectory.npz",
        {
            "sample_id": np.asarray(plan.sample_id),
            "event_plan_sha256": np.asarray(plan.plan_sha256),
            "relative_timestamps_ns": trajectory["relative_timestamps_ns"],
            "timestamps_ns": np.asarray([] if trajectory["timestamps_ns"] is None else trajectory["timestamps_ns"], np.int64),
            "x": trajectory["x"], "y": trajectory["y"],
            "pressure": trajectory["pressure"], "size": trajectory["size"],
            "pointer_id": trajectory["pointer_id"], "phase": trajectory["phase"],
            "android_action": trajectory["android_action"],
            "key_index": trajectory["key_index"], "keycode": trajectory["keycode"],
            "frame_index": trajectory["frame_index"],
            "feature_vector": trajectory["feature_vector"],
        },
    )
    _save_npz(
        output / "imu.npz",
        {
            "sample_id": np.asarray(plan.sample_id),
            "event_plan_sha256": np.asarray(plan.plan_sha256),
            "active_imu": np.asarray(imu["active_imu"], np.float32),
            "relative_timestamps_ns": np.asarray(imu["relative_timestamps_ns"], np.int64),
            "timestamps_ns": np.asarray([] if imu.get("timestamps_ns") is None else imu["timestamps_ns"], np.int64),
            "mask": np.asarray(imu["mask"], np.uint8),
            "valid_mask": np.asarray(imu["valid_mask"], np.uint8),
        },
    )
    report = {
        "status": "passed",
        "sample_id": plan.sample_id,
        "event_plan_sha256": plan.plan_sha256,
        "pair_audit": result["pair_audit"],
        "consistency_feature_names": list(result["consistency_feature_names"]),
        "consistency_features": result["consistency_features"].tolist(),
        "paired_generation_wall_ms": result["paired_generation_wall_ms"],
        "files": {
            "event_plan": str(plan_path),
            "trajectory": str(output / "trajectory.npz"),
            "imu": str(output / "imu.npz"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
