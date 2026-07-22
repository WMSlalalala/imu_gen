#!/usr/bin/env python3
"""Run and validate the real one-user HMOG trajectory extraction smoke test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = PROJECT_ROOT / "preprocess" / "extract_hmog_trajectories.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "smoke_one_user"
ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")

FLAT_FIELDS = (
    "flat_system_time_ms",
    "flat_event_time_ms",
    "flat_t_rel_ms",
    "flat_frame_index",
    "flat_pointer_count",
    "flat_pointer_id",
    "flat_action_code",
    "flat_x",
    "flat_y",
    "flat_pressure",
    "flat_size",
    "flat_orientation_id",
    "flat_active_mask",
    "flat_valid_mask",
    "flat_key_index",
    "flat_keycode",
)

KEY_FIELDS = (
    "key_index_in_event",
    "keycode",
    "key_is_letter",
    "key_down_ms",
    "key_up_ms",
    "key_hold_ms",
    "key_flight_from_previous_ms",
    "key_orientation_id",
    "key_raw_gesture_id",
    "key_touch_start_ms",
    "key_touch_end_ms",
    "key_match_start_error_ms",
    "key_match_end_error_ms",
    "key_touch_found",
)


def validate_action(path: Path, expected_action: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        assert str(data["schema_version"]) == "hmog_touch_trajectory_v1"
        assert str(data["action_name"]) == expected_action
        object_fields = [name for name in data.files if data[name].dtype.kind == "O"]
        assert not object_fields, f"object arrays are forbidden: {object_fields}"

        event_offsets = np.asarray(data["event_offsets"], dtype=np.int64)
        event_key_offsets = np.asarray(data["event_key_offsets"], dtype=np.int64)
        key_touch_offsets = np.asarray(data["key_touch_offsets"], dtype=np.int64)
        n_events = len(event_offsets) - 1
        n_rows = len(data["flat_x"])
        n_keys = len(data["keycode"])
        assert n_events > 0, f"{expected_action}: smoke user must have accepted events"
        assert len(event_key_offsets) == n_events + 1
        assert event_offsets[0] == 0 and event_offsets[-1] == n_rows
        assert event_key_offsets[0] == 0 and event_key_offsets[-1] == n_keys
        assert key_touch_offsets[0] == 0
        assert np.all(np.diff(event_offsets) > 0)
        assert np.all(np.diff(event_key_offsets) >= 0)
        assert np.all(np.diff(key_touch_offsets) >= 0)

        for field in FLAT_FIELDS:
            assert len(data[field]) == n_rows, (field, len(data[field]), n_rows)
        for field in KEY_FIELDS:
            assert len(data[field]) == n_keys, (field, len(data[field]), n_keys)
        for field in [
            "event_id",
            "user_id",
            "user_external_id",
            "session_id",
            "action_id",
            "activity_id",
            "orientation_id",
            "label_start_ms",
            "label_end_ms",
            "touch_start_ms",
            "touch_end_ms",
            "n_rows",
            "n_frames",
            "n_pointers",
            "max_pointer_count",
            "active_row_count",
            "n_keys",
            "n_letters",
        ]:
            assert len(data[field]) == n_events, (field, len(data[field]), n_events)

        assert np.all(data["flat_valid_mask"] == 1)
        assert np.all(data["flat_active_mask"] <= data["flat_valid_mask"])
        assert np.array_equal(np.diff(event_offsets), data["n_rows"])
        assert np.all(data["active_row_count"] > 0)

        all_frame_deltas: list[int] = []
        for event_index in range(n_events):
            start, end = event_offsets[event_index : event_index + 2]
            times = np.asarray(data["flat_event_time_ms"][start:end], dtype=np.int64)
            relative = np.asarray(data["flat_t_rel_ms"][start:end], dtype=np.int64)
            frames = np.asarray(data["flat_frame_index"][start:end], dtype=np.int64)
            assert len(times) > 0
            assert times[0] == int(data["touch_start_ms"][event_index])
            assert times[-1] == int(data["touch_end_ms"][event_index])
            assert relative[0] == 0
            assert np.all(np.diff(times) >= 0)
            assert np.all(np.diff(relative) >= 0)
            assert np.all(np.diff(frames) >= 0)
            _, first_indices = np.unique(frames, return_index=True)
            frame_times = times[np.sort(first_indices)]
            all_frame_deltas.extend(np.diff(frame_times).tolist())

        if expected_action in {"tap", "scroll", "swipe"}:
            assert np.all(data["max_pointer_count"] == 1)
            assert np.all(data["n_pointers"] == 1)
            assert np.all(data["flat_pointer_count"] == 1)
            assert np.all(data["flat_pointer_id"] == 0)
            assert n_keys == 0
            assert key_touch_offsets[-1] == 0
        elif expected_action == "pinch":
            assert np.all(data["max_pointer_count"] >= 2)
            assert np.all(data["n_pointers"] >= 2)
            assert n_keys == 0
            assert key_touch_offsets[-1] == 0
            for event_index in range(n_events):
                start, end = event_offsets[event_index : event_index + 2]
                active = data["flat_active_mask"][start:end].astype(bool)
                assert np.max(data["flat_pointer_count"][start:end][active]) >= 2
                assert len(np.unique(data["flat_pointer_id"][start:end][active])) >= 2
        else:
            assert expected_action == "keystroke"
            assert n_keys > 0
            assert key_touch_offsets[-1] == n_rows
            assert np.all(np.diff(key_touch_offsets) > 0)
            assert np.all(data["key_touch_found"] == 1)
            assert np.all(data["max_pointer_count"] == 1)
            assert np.all(data["n_pointers"] == 1)
            assert np.all(data["flat_pointer_count"] == 1)
            for event_index in range(n_events):
                key_start, key_end = event_key_offsets[event_index : event_index + 2]
                assert key_end > key_start
                assert int(data["key_flight_from_previous_ms"][key_start]) == 0
                for global_key_index in range(key_start, key_end):
                    point_start, point_end = key_touch_offsets[
                        global_key_index : global_key_index + 2
                    ]
                    assert point_end > point_start
                    local_key_index = int(data["key_index_in_event"][global_key_index])
                    assert local_key_index == global_key_index - key_start
                    assert np.all(
                        data["flat_key_index"][point_start:point_end] == local_key_index
                    )
                    assert np.all(
                        data["flat_keycode"][point_start:point_end]
                        == data["keycode"][global_key_index]
                    )
                    assert (
                        int(data["flat_event_time_ms"][point_start])
                        == int(data["key_touch_start_ms"][global_key_index])
                    )
                    assert (
                        int(data["flat_event_time_ms"][point_end - 1])
                        == int(data["key_touch_end_ms"][global_key_index])
                    )
                    if global_key_index > key_start:
                        expected_flight = int(data["key_down_ms"][global_key_index]) - int(
                            data["key_up_ms"][global_key_index - 1]
                        )
                        assert (
                            int(data["key_flight_from_previous_ms"][global_key_index])
                            == expected_flight
                        )

        # A 100 Hz synthesized series would have only 10 ms frame deltas.  The
        # raw HMOG touch timeline must retain its variable Android cadence.
        assert all_frame_deltas
        assert any(delta != 10 for delta in all_frame_deltas)
        positive_deltas = np.asarray(
            [value for value in all_frame_deltas if value > 0], dtype=np.int64
        )
        assert len(np.unique(positive_deltas)) > 1

        return {
            "action": expected_action,
            "n_events": int(n_events),
            "n_rows": int(n_rows),
            "n_frames": int(np.asarray(data["n_frames"], dtype=np.int64).sum()),
            "n_keys": int(n_keys),
            "active_rows": int(np.asarray(data["flat_active_mask"]).sum()),
            "frame_delta_median_ms": float(np.median(positive_deltas)),
            "frame_delta_unique_count": int(len(np.unique(positive_deltas))),
            "max_pointer_count_values": [
                int(value) for value in np.unique(data["max_pointer_count"])
            ],
            "allow_pickle_false": True,
            "object_array_count": 0,
        }


def write_report(output_dir: Path, results: list[dict[str, Any]]) -> None:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    payload = {
        "status": "passed",
        "user": manifest["selection"]["processed_users"][0],
        "actions": results,
        "audit_action_status": audit["action_status"],
        "audit_rejection_reasons": audit["rejection_reasons"],
    }
    (output_dir / "smoke_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# One-user trajectory smoke validation",
        "",
        "状态：通过。所有 NPZ 均由 `numpy.load(..., allow_pickle=False)` 打开，",
        "offset 单调且覆盖全部 flat rows；原始不规则触摸时间轴未重采样。",
        "",
        "| action | events | rows | frames | keys | median frame Δt (ms) | pointer counts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['action']} | {item['n_events']} | {item['n_rows']} | "
            f"{item['n_frames']} | {item['n_keys']} | "
            f"{item['frame_delta_median_ms']:.1f} | "
            f"{item['max_pointer_count_values']} |"
        )
    lines.extend(
        [
            "",
            "严格语义检查：",
            "",
            "- tap / scroll / swipe 的每个保留事件都是完整单指 DOWN–UP；",
            "- pinch 的 active phase 内存在两个 pointer；",
            "- keystroke 的每个 key 都有非空 raw touch contact，flight gap 由相邻 UP/DOWN 计算；",
            "- keystroke 的键间没有伪造屏幕轨迹；",
            "- 每类至少存在一个非 10ms 帧间隔，证明没有改写成 100Hz 触摸序列。",
            "",
        ]
    )
    (output_dir / "smoke_validation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate an existing one-user output instead of rerunning extraction.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not args.validate_existing:
        subprocess.run(
            [
                sys.executable,
                str(EXTRACTOR),
                "--output-dir",
                str(output_dir),
                "--max-users",
                "1",
                "--overwrite",
            ],
            check=True,
        )

    results = [
        validate_action(output_dir / f"hmog_trajectory_{action}.npz", action)
        for action in ACTIONS
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["selection"]["processed_users"]) == 1
    for item in results:
        assert manifest["outputs"][item["action"]]["n_events"] == item["n_events"]
        assert manifest["outputs"][item["action"]]["n_flat_rows"] == item["n_rows"]
    write_report(output_dir, results)
    print(
        "[smoke passed] "
        + " ".join(
            f"{item['action']}={item['n_events']}/{item['n_rows']}"
            for item in results
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
