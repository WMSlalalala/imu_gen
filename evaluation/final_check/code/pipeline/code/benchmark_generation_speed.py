#!/usr/bin/env python3
"""Measure how fast each method produces one fake event, at inference time.

Training cost is a one-off an attacker pays in their own lab.  What decides
whether an attack is practical is the marginal cost of the next event, so that
is what this measures, on one machine, with the same clock, warmed up, and with
the GPU synchronised so an asynchronous launch cannot be mistaken for work
finished.

Each method is split into the two stages an attacker actually runs:

* **synthesis** -- producing one window of signal.  For the learned baselines
  this is the model's sampling pass, which for a diffusion model means one
  network evaluation per denoising step and for a GAN means a single forward
  pass.  For pyclick it is the Bezier construction.  For the five-shot method
  it is the analytic transform of one frozen real event.
* **placement** -- putting that window on a specific carrier: the bank draw, the
  endpoint binding, the screen fit, the carrier's dispatch restoration.  Only
  the sample-bank baselines pay this separately; pyclick and the five-shot
  method fold it into synthesis because both are told the endpoints up front.

A learned generator can amortise synthesis over many carriers by sampling a bank
once and reusing it, so both the per-window cost and the per-event cost at the
release's actual reuse factor are reported.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "DiffusionTS"))
sys.path.insert(0, str(BASE / "TTSGAN"))
sys.path.insert(0, str(BASE / "pyclick"))

from hmog_baseline_common import (  # noqa: E402
    ACTION_WINDOW_SAMPLES,
    apply_carrier_dispatch,
    fit_to_screen,
    iter_shards,
    load_shard,
    zoh_resample,
)

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_diffusion_sampling(action: str, kind: str, batch: int, gpu: int) -> dict:
    """One sampling pass of the fitted Diffusion-TS, warmed up and synchronised."""

    import torch
    from run_diffusion_ts import build_model

    torch.cuda.set_device(gpu)
    seq = ACTION_WINDOW_SAMPLES[action]
    features = 2 if kind == "trajectory" else 6
    model = build_model(seq, features, 500).cuda().eval()
    with torch.no_grad():
        model.generate_mts(batch_size=4)  # warm up kernels and allocator
        _sync()
        started = time.perf_counter()
        model.generate_mts(batch_size=batch)
        _sync()
        elapsed = time.perf_counter() - started
    del model
    torch.cuda.empty_cache()
    return {"windows": batch, "seconds": elapsed, "ms_per_window": 1000 * elapsed / batch}


def time_ttsgan_sampling(action: str, kind: str, batch: int, gpu: int) -> dict:
    """One forward pass of the fitted TTS-GAN generator, same protocol."""

    import torch
    from GANModels import Generator

    torch.cuda.set_device(gpu)
    seq = ACTION_WINDOW_SAMPLES[action]
    channels = 2 if kind == "trajectory" else 6
    generator = Generator(
        seq_len=seq, patch_size=2, channels=channels, latent_dim=100
    ).cuda().eval()
    with torch.no_grad():
        generator(torch.randn(4, 100, device="cuda"))
        _sync()
        started = time.perf_counter()
        generator(torch.randn(batch, 100, device="cuda"))
        _sync()
        elapsed = time.perf_counter() - started
    del generator
    torch.cuda.empty_cache()
    return {"windows": batch, "seconds": elapsed, "ms_per_window": 1000 * elapsed / batch}


def time_pyclick(events: list, repeats: int) -> dict:
    """pyclick builds its curve per event; there is no bank to amortise."""

    import random
    import types

    sys.modules.setdefault("pyautogui", types.ModuleType("pyautogui"))
    from pyclick import HumanCurve

    for _ in range(20):  # warm up the interpreter's caches
        HumanCurve((100, 100), (400, 700), targetPoints=40)
    started = time.perf_counter()
    count = 0
    for _ in range(repeats):
        for samples, start, end in events:
            random.seed(count)
            np.random.seed(count)
            HumanCurve(start, end, targetPoints=samples)
            count += 1
    elapsed = time.perf_counter() - started
    return {"events": count, "seconds": elapsed, "ms_per_event": 1000 * elapsed / count}


def time_placement(events: list, bank: np.ndarray, repeats: int) -> dict:
    """Bank draw, endpoint binding, screen fit and dispatch restoration."""

    from build_sample_bank_baseline import _bind_to_endpoints

    dimensions = np.asarray((1080.0, 1920.0))
    started = time.perf_counter()
    count = 0
    for _ in range(repeats):
        for samples, carrier in events:
            window = zoh_resample(bank[count % len(bank)], samples).astype(np.float64)
            first, last = 0, samples - 1
            start = carrier[first, 1:3] * dimensions
            end = carrier[last, 1:3] * dimensions
            placed, _ = _bind_to_endpoints(window * dimensions, start, end, first, last)
            fitted, _ = fit_to_screen(placed, start, dimensions)
            apply_carrier_dispatch(fitted / dimensions, carrier)
            count += 1
    elapsed = time.perf_counter() - started
    return {"events": count, "seconds": elapsed, "ms_per_event": 1000 * elapsed / count}


def time_fiveshot(events: list, repeats: int) -> dict:
    """The paper's own analytic transport of one frozen real event."""

    sys.path.insert(0, "/mnt/share/mwang49/data7/code/direct100k")
    from security_exp.exact_touch_template_generator import (
        _request_direction,
        generate_exact_touch_template,
    )

    started = time.perf_counter()
    count = failures = 0
    for _ in range(repeats):
        for samples, start, end, template in events:
            try:
                generate_exact_touch_template(
                    action="swipe",
                    start_xy_px=start,
                    end_xy_px=end,
                    direction=_request_direction(
                        np.asarray(start, float), np.asarray(end, float)
                    ),
                    duration_ms=samples * 10.0,
                    template_t_ms=template[:, 0],
                    template_x_px=template[:, 1],
                    template_y_px=template[:, 2],
                    template_pressure=template[:, 3],
                    screen_width_px=1080.0,
                    screen_height_px=1920.0,
                )
            except Exception:
                failures += 1
            count += 1
    elapsed = time.perf_counter() - started
    return {
        "events": count,
        "failures": failures,
        "seconds": elapsed,
        "ms_per_event": 1000 * elapsed / count,
    }


def _direction(start, end) -> str:
    import math

    dx, dy = end[0] - start[0], end[1] - start[1]
    if dx == 0 and dy == 0:
        return "stationary"
    angle = math.degrees(math.atan2(-dy, dx)) % 360.0
    names = ("east", "northeast", "north", "northwest", "west", "southwest",
             "south", "southeast")
    return names[int((angle + 22.5) % 360.0 // 45.0)]


def collect_carriers(dataset_dir: Path, action: str, limit: int):
    carriers = []
    for path in iter_shards(dataset_dir):
        arrays = load_shard(path)
        offsets, labels, actions = arrays["offsets"], arrays["label"], arrays["action"]
        for index in np.flatnonzero((labels == 1) & (actions == action)):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            carriers.append(arrays["trajectory_flat"][start:stop].copy())
            if len(carriers) >= limit:
                return carriers
    return carriers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--action", default="swipe")
    parser.add_argument("--kind", default="trajectory",
                        choices=("trajectory", "imu"))
    parser.add_argument("--events", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--reuse-factor", type=float, default=5.0)
    args = parser.parse_args()

    carriers = collect_carriers(args.dataset_dir, args.action, args.events)
    dimensions = np.asarray((1080.0, 1920.0))
    simple = [
        (
            len(c),
            (int(c[0, 1] * dimensions[0]), int(c[0, 2] * dimensions[1])),
            (int(c[-1, 1] * dimensions[0]), int(c[-1, 2] * dimensions[1])),
        )
        for c in carriers
    ]
    report: dict = {
        "action": args.action,
        "kind": args.kind,
        "carriers": len(carriers),
        "median_rows": int(np.median([len(c) for c in carriers])),
        "note": (
            "synthesis is the model/curve pass; placement is binding a window to "
            "one carrier; a bank is reused reuse_factor times in the release"
        ),
        "reuse_factor": args.reuse_factor,
    }

    print(f"action {args.action}: {len(carriers)} carriers, "
          f"median {report['median_rows']} rows")

    if args.kind == "trajectory":
        report["pyclick_synthesis"] = time_pyclick(simple, args.repeats)
        print(f"  pyclick synthesis      "
              f"{report['pyclick_synthesis']['ms_per_event']:8.3f} ms/event")

    templates = [
        np.column_stack(
            (
                np.arange(len(c)) * 10.0,
                c[:, 1] * dimensions[0],
                c[:, 2] * dimensions[1],
                np.clip(c[:, 3], 0.0, 1.0),
            )
        )
        for c in carriers
    ]
    fiveshot_events = [
        (len(c), s, e, t)
        for (n, s, e), c, t in zip(simple, carriers, templates)
        if s != e and len(c) >= 2
    ]
    if args.kind == "trajectory" and args.action in ("scroll", "swipe"):
        report["fiveshot_synthesis"] = time_fiveshot(fiveshot_events, args.repeats)
        print(f"  five-shot synthesis    "
              f"{report['fiveshot_synthesis']['ms_per_event']:8.3f} ms/event"
              f"  ({report['fiveshot_synthesis']['failures']} rejected)")

    bank = np.random.default_rng(0).random(
        (256, ACTION_WINDOW_SAMPLES[args.action], 2 if args.kind == "trajectory" else 6)
    ).astype(np.float32)
    if args.kind == "trajectory":
        report["bank_placement"] = time_placement(
            [(len(c), c) for c in carriers], bank, args.repeats
        )
        print(f"  bank placement         "
              f"{report['bank_placement']['ms_per_event']:8.3f} ms/event")

    for name, fn in (("diffusion_ts", time_diffusion_sampling),
                     ("tts_gan", time_ttsgan_sampling)):
        try:
            measured = fn(args.action, args.kind, args.batch, args.gpu)
            per_event = measured["ms_per_window"] / args.reuse_factor
            measured["ms_per_event_at_reuse"] = per_event
            report[f"{name}_synthesis"] = measured
            print(f"  {name:22s}{measured['ms_per_window']:8.3f} ms/window"
                  f"  -> {per_event:.3f} ms/event at reuse {args.reuse_factor}")
        except Exception as error:  # noqa: BLE001
            report[f"{name}_synthesis"] = {"error": str(error)}
            print(f"  {name}: {error}")

    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
