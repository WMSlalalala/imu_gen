#!/usr/bin/env python3
"""Build a fake event end to end, the way the source method builds one.

WHY THIS REPLACES THE CHANNEL-SWAP HARNESS

The first harness kept every fake event's container -- its row count, its clock,
and which rows the device reported a new coordinate on -- and only rewrote the
two coordinate columns.  That was right while those things came from upstream.
They no longer do.  From v11 the source method computes a scroll's duration
itself, from a travel-to-duration curve read off the victim's five recordings,
and the released numbers show what that is worth: the duration it chooses is a
median 1.84x and a 95th-percentile 10.7x the one the carrier plan named.  A
baseline handed that row count is handed the answer to "how long should this
gesture be", which is a component of the method under test, and it is handed it
even when it has no victim data at all.

So a baseline now delivers a whole event.  It decides the duration, the row
count follows at the 100 Hz grid, and every column is produced at that length.

WHAT EACH METHOD OWNS AND WHAT IT INHERITS

* **Duration** is the method's.  A method with the victim's five recordings
  reads it off them; a method without any victim data can only take the carrier
  plan's figure, and that limitation is the baseline's, not the harness's.
* **Update timing** -- which rows repeat the previous coordinate -- is also the
  method's.  Genuine events report a new coordinate at about 55 Hz on scroll and
  51 Hz on swipe against the 100 Hz grid, and the source method inherits exactly
  that because it transports a real recording sample for sample (measured: v12
  fake repeat fraction 0.4487 against 0.4518 genuine on scroll).  A method that
  deforms a real recording inherits it the same way.  A method that draws a
  curve from nothing has no recording to inherit from and must choose an
  injection rate; the choice made here is Android's own dispatch rate rather
  than the victim's observed one, because copying the victim's would hand back
  the very thing the baseline is supposed to lack.
* **The channel a method does not generate** is re-cut from the carrier's own
  padded IMU window, at the new length, with the action kept inside the cut --
  the identical mechanism the source method uses when its own duration law
  lengthens a gesture (``fiveshot_gesture_timing.window_slice_bounds``).  It is
  never time-warped, because warping inertia to fit a duration is exactly the
  artefact the source method avoided.
* **Genuine events are copied through untouched**, so the detector's positive
  class stays bit identical across every dataset built here.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/share/mwang49/data7/code/direct100k")
from security_exp.fiveshot_gesture_timing import (  # noqa: E402
    carrier_window_imu,
    contact_travel_px,
)
from security_exp.replay_dataset_builder import (  # noqa: E402
    _carrier_imu_sources,
    _carrier_imu_window,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmog_baseline_common import sha256_file  # noqa: E402

GRID_PERIOD_MS = 10.0
# UiDevice.swipe spends one MOVE per vsync, so a dispatched stroke updates at
# the display rate.  This is what a baseline with no victim recording has to
# fall back on; it is a property of the dispatch route, not of any victim.
ANDROID_INJECTION_HZ = 60.0
TRAJECTORY_COLUMNS = 9
IMU_CHANNELS = 6


def samples_for_duration(duration_ms: float) -> int:
    """Row count a duration occupies on the detector grid, as the builder does."""

    if not math.isfinite(duration_ms) or duration_ms <= 0.0:
        raise ValueError("a gesture duration must be positive")
    return int(round(float(duration_ms) / GRID_PERIOD_MS)) + 1


def hold_mask_from_rate(samples: int, update_hz: float) -> np.ndarray:
    """Which rows repeat the previous coordinate, for a source at update_hz.

    A source slower than the grid cannot refresh every row.  Row i carries a new
    coordinate when its update index has advanced, which is the same staircase a
    zero-order-hold observer produces from a real device.
    """

    if samples < 1:
        raise ValueError("a window needs at least one row")
    index = np.floor(
        np.arange(samples) * (float(update_hz) * GRID_PERIOD_MS / 1000.0)
    ).astype(np.int64)
    held = np.zeros(samples, dtype=bool)
    if samples > 1:
        held[1:] = index[1:] == index[:-1]
    return held


def apply_hold(path: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Forward-fill a path through the rows the source did not refresh."""

    sources = np.where(~held, np.arange(len(path)), 0)
    np.maximum.accumulate(sources, out=sources)
    return path[sources]


def hold_mask_from_material(material_xy: np.ndarray, samples: int) -> np.ndarray:
    """Carry a real recording's own update timing onto a different row count.

    A method that deforms one of the victim's recordings keeps that recording's
    refresh pattern, stretched to the new length -- the same thing the source
    method gets for free by transporting the recording sample for sample.
    """

    if len(material_xy) < 2:
        return np.zeros(samples, dtype=bool)
    source_held = np.zeros(len(material_xy), dtype=bool)
    source_held[1:] = np.all(np.diff(material_xy, axis=0) == 0.0, axis=1)
    positions = np.clip(
        np.floor(
            np.linspace(0.0, 1.0, samples) * (len(material_xy) - 1) + 0.5
        ).astype(np.int64),
        0,
        len(material_xy) - 1,
    )
    held = source_held[positions]
    if samples:
        held[0] = False
    return held


def recut_carrier_imu(
    trajectory_source: str, archive_index: int, samples: int
) -> tuple[np.ndarray, dict]:
    """Cut `samples` frames of the carrier's own inertial window, never warped."""

    sources = _carrier_imu_sources(str(trajectory_source))
    window, mask = _carrier_imu_window(sources[int(archive_index)])
    capped = min(int(samples), int(len(window)))
    if capped < 2:
        raise ValueError("carrier window cannot supply two frames")
    cut, audit = carrier_window_imu(window=window, mask=mask, samples=capped)
    if len(cut) < samples:
        # The window is shorter than the duration the method asked for.  Holding
        # the last frame would invent stillness the hand never had, so the event
        # is reported as unsupported and the caller decides.
        raise ValueError(
            f"carrier window has {len(window)} frames, {samples} requested"
        )
    return np.asarray(cut, dtype=np.float32), audit


def structural_trajectory(samples: int, duration_ms: float, template_row: np.ndarray) -> np.ndarray:
    """The columns a single-pointer stroke carries besides its coordinates.

    Contact, pressure, pointer count and availability are constants for a
    single-pointer gesture in this dataset and are taken from the carrier's own
    first contact row; elapsed is the grid clock for the chosen duration.
    """

    out = np.zeros((samples, TRAJECTORY_COLUMNS), dtype=np.float32)
    out[:, 0] = template_row[0]
    out[:, 3] = template_row[3]
    out[:, 4] = template_row[4]
    out[:, 8] = template_row[8]
    out[:, 7] = np.linspace(0.0, float(duration_ms) / 1000.0, samples, dtype=np.float32)
    return out


def recompute_deltas(trajectory: np.ndarray) -> None:
    contact = trajectory[:, 0] > 0.0
    deltas = np.zeros((len(trajectory), 2), dtype=np.float32)
    consecutive = contact[1:] & contact[:-1]
    differences = np.diff(trajectory[:, 1:3], axis=0)
    deltas[1:][consecutive] = differences[consecutive]
    trajectory[:, 5:7] = deltas


def rebuild_shard(arrays: dict, event_builder) -> tuple[dict, dict]:
    """Rebuild one shard, letting `event_builder` decide each fake event.

    ``event_builder(event) -> dict | None``.  Returning None leaves the event
    exactly as the source built it, which is how an action a baseline does not
    address is recorded.  A returned dict carries ``trajectory`` (n, 9) and
    ``imu`` (n, 6); n is the method's own choice.
    """

    offsets = arrays["offsets"]
    labels = arrays["label"]
    trajectory_parts: list[np.ndarray] = []
    imu_parts: list[np.ndarray] = []
    counts = {"fake": 0, "genuine": 0, "rebuilt": 0, "declined": 0, "failed": 0}
    per_action: dict[str, dict[str, int]] = {}
    lengths: dict[str, list[int]] = {}

    for index in range(len(labels)):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        trajectory = arrays["trajectory_flat"][start:stop]
        imu = arrays["imu_flat"][start:stop]
        action = str(arrays["action"][index])
        if int(labels[index]) == 0:
            counts["genuine"] += 1
            trajectory_parts.append(trajectory)
            imu_parts.append(imu)
            continue
        counts["fake"] += 1
        tally = per_action.setdefault(
            action, {"fake": 0, "rebuilt": 0, "declined": 0, "failed": 0}
        )
        tally["fake"] += 1
        event = {
            "action": action,
            "user_id": str(arrays["user_id"][index]),
            "event_id": str(arrays["event_id"][index]),
            "split": str(arrays["split"]),
            "samples": stop - start,
            "trajectory": trajectory.copy(),
            "imu": imu.copy(),
        }
        try:
            built = event_builder(event)
        except Exception as error:  # noqa: BLE001
            built = None
            counts.setdefault("failure_reasons", {})
            reasons = counts["failure_reasons"]
            key = type(error).__name__
            reasons[key] = reasons.get(key, 0) + 1
            tally["failed"] += 1
            counts["failed"] += 1
        if built is None:
            counts["declined"] += 1
            tally["declined"] += 1
            trajectory_parts.append(trajectory)
            imu_parts.append(imu)
            continue
        new_trajectory = np.asarray(built["trajectory"], dtype=np.float32)
        new_imu = np.asarray(built["imu"], dtype=np.float32)
        if new_trajectory.shape[1] != TRAJECTORY_COLUMNS or new_imu.shape[1] != IMU_CHANNELS:
            raise ValueError("rebuilt event has the wrong channel count")
        if len(new_trajectory) != len(new_imu) or len(new_trajectory) < 2:
            raise ValueError("rebuilt trajectory and IMU disagree on length")
        if not np.isfinite(new_trajectory).all() or not np.isfinite(new_imu).all():
            raise ValueError("rebuilt event carries non-finite values")
        trajectory_parts.append(new_trajectory)
        imu_parts.append(new_imu)
        counts["rebuilt"] += 1
        tally["rebuilt"] += 1
        lengths.setdefault(action, []).append(len(new_trajectory))

    rebuilt = dict(arrays)
    rebuilt["trajectory_flat"] = np.concatenate(trajectory_parts).astype(np.float32)
    rebuilt["imu_flat"] = np.concatenate(imu_parts).astype(np.float32)
    new_offsets = np.zeros(len(trajectory_parts) + 1, dtype=np.int64)
    np.cumsum([len(part) for part in trajectory_parts], out=new_offsets[1:])
    rebuilt["offsets"] = new_offsets
    counts["per_action"] = per_action
    counts["median_rows"] = {
        action: int(np.median(values)) for action, values in lengths.items()
    }
    return rebuilt, counts


def finalise(
    *,
    source_dir: Path,
    output_dir: Path,
    shard_digests: dict,
    counts: dict,
    method_name: str,
    method_detail: dict,
) -> None:
    """Write the manifest, provenance and release the rebuilt dataset needs."""

    lines = []
    for raw in (source_dir / "event_manifest.jsonl").read_text().splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        for shard in record.get("shards", []):
            name = Path(shard["source"]).name
            shard["source"] = str(output_dir / "shards" / name)
            shard["source_sha256"] = shard_digests[name]
        lines.append(json.dumps(record, sort_keys=True))
    (output_dir / "event_manifest.jsonl").write_text("\n".join(lines) + "\n")
    shutil.copy(source_dir / "provenance.jsonl", output_dir / "provenance.jsonl")

    release = json.loads((source_dir / "release.json").read_text())
    release["event_manifest"] = str(output_dir / "event_manifest.jsonl")
    release["event_manifest_sha256"] = sha256_file(output_dir / "event_manifest.jsonl")
    release["provenance"] = str(output_dir / "provenance.jsonl")
    release["provenance_sha256"] = sha256_file(output_dir / "provenance.jsonl")
    release["baseline_method"] = method_name
    release["baseline_detail"] = method_detail
    release["baseline_source_dataset"] = str(source_dir)
    release["baseline_event_ownership"] = {
        "duration": "chosen by the baseline",
        "row_count": "follows the duration at the 100 Hz grid",
        "update_timing": (
            "inherited from the victim's recording when the baseline deforms "
            "one, otherwise the Android dispatch rate"
        ),
        "unowned_channel": (
            "re-cut from the carrier's padded inertial window at the new "
            "length, never time-warped"
        ),
        "genuine_events": "copied through untouched",
    }
    (output_dir / "release.json").write_text(
        json.dumps(release, indent=2, sort_keys=True)
    )
    (output_dir / "baseline_counts.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True)
    )


__all__ = [
    "ANDROID_INJECTION_HZ",
    "apply_hold",
    "contact_travel_px",
    "finalise",
    "hold_mask_from_material",
    "hold_mask_from_rate",
    "recompute_deltas",
    "recut_carrier_imu",
    "rebuild_shard",
    "samples_for_duration",
    "structural_trajectory",
]
