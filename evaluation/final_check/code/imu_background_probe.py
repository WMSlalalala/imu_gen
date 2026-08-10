#!/usr/bin/env python3
"""Measure the between-gesture IMU background in raw HMOG (Task 2 evidence).

Shows, with numbers, that (1) the raw archive has continuous per-session IMU
that the per-event datasets strip, and (2) a human's hand keeps moving between
gestures -- so a machine that only injects IMU during gestures leaves dead air
no human produces.  See docs/TASK2_IMU_AXIS_CN.md.

    unzip -o /home/mwang49/Human_agent/hmog_dataset.zip public_dataset/100669.zip
    unzip -o public_dataset/100669.zip '100669/100669_session_1/*.csv'
    python imu_background_probe.py 100669/100669_session_1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def probe(session_dir: str, contact_pad_ms: float = 200.0, window_ms: float = 2000.0):
    d = Path(session_dir)
    acc = np.genfromtxt(d / "Accelerometer.csv", delimiter=",", usecols=(0, 3, 4, 5))
    t, xyz = acc[:, 0], acc[:, 1:4]
    mag = np.linalg.norm(xyz, axis=1)
    dur = (t[-1] - t[0]) / 1000.0

    touch = np.unique(np.genfromtxt(d / "TouchEvent.csv", delimiter=",", usecols=(0,)))

    contact = np.zeros(len(t), bool)
    for tt in touch:
        contact |= (t >= tt - contact_pad_ms) & (t <= tt + contact_pad_ms)

    between_windows = []
    for a in np.arange(t[0], t[-1], window_ms):
        m = (t >= a) & (t < a + window_ms) & (~contact)
        if m.sum() > 20:
            between_windows.append(float(np.std(mag[m])))
    bw = np.asarray(between_windows)

    return {
        "samples": int(len(t)),
        "duration_s": round(dur, 1),
        "hz": round(len(t) / dur, 1),
        "touch_events": int(len(touch)),
        "between_gesture_fraction": round(float((~contact).mean()), 3),
        "motion_std_during": round(float(np.std(mag[contact])), 4),
        "motion_std_between": round(float(np.std(mag[~contact])), 4),
        "between_2s_windows": int(len(bw)),
        "between_window_motion_median": round(float(np.median(bw)), 4),
        "between_window_motion_p5": round(float(np.percentile(bw, 5)), 4),
        "near_still_window_fraction": round(float(np.mean(bw < 0.01)), 4),
    }


if __name__ == "__main__":
    import json

    session = sys.argv[1] if len(sys.argv) > 1 else "100669/100669_session_1"
    print(json.dumps(probe(session), indent=2))
