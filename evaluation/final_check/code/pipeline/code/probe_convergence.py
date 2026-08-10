#!/usr/bin/env python3
"""Find the step count a five-window Diffusion-TS actually needs.

The unrestricted runs use the authors' 12,000 steps over thousands of windows.
Applied to five windows that is roughly 300,000 passes over the same five
sequences, and the honest question is whether anything is still changing after
the first couple of thousand.  Guessing either way is expensive: the per-victim
programme is a thousand models, so every step saved is about fourteen GPU-hours
and every step cut too early is a model that never learned the signal.

One model is trained per probed cell with frequent checkpoints, then each
checkpoint is sampled and scored on statistics that separate the two failure
modes:

* **lag-1 autocorrelation** against the real windows -- an undertrained model
  emits sample-to-sample noise, which is exactly how the first TTS-GAN run
  failed, and no FAR computed from such a model describes the method.
* **per-channel standard deviation ratio** -- catches the opposite failure, a
  model that has collapsed onto the mean of its five windows.
* **nearest-neighbour distance to the five training windows** -- with five
  windows memorisation is expected, not a defect; this records how close the
  model actually gets so the paper can say so rather than speculate.

Convergence is called at the first checkpoint after which none of the three
moves materially.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "DiffusionTS"))

from hmog_baseline_common import ACTION_WINDOW_SAMPLES, collect_genuine_windows  # noqa: E402
from run_diffusion_ts import WindowDataset, _Args, build_model, solver_config  # noqa: E402
from engine.solver import Trainer  # noqa: E402

TRAJECTORY_COLUMNS = (1, 2)


def lag1(windows: np.ndarray) -> float:
    values = []
    for window in windows[:256]:
        for channel in range(window.shape[1]):
            series = window[:, channel]
            if series.std() > 1e-9:
                values.append(np.corrcoef(series[:-1], series[1:])[0, 1])
    return float(np.nanmedian(values)) if values else float("nan")


def nearest_distance(generated: np.ndarray, training: np.ndarray) -> float:
    """Median distance from a sample to its closest training window."""

    scale = training.std() + 1e-9
    distances = []
    for sample in generated[:256]:
        gaps = np.sqrt(((training - sample[None]) ** 2).mean(axis=(1, 2)))
        distances.append(gaps.min() / scale)
    return float(np.median(distances))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--kind", choices=("trajectory", "imu"), required=True)
    parser.add_argument("--train-events", type=int, default=5)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    windows, users = collect_genuine_windows(
        args.dataset_dir, "train", args.action, args.kind
    )
    if args.kind == "trajectory":
        windows = windows[:, :, TRAJECTORY_COLUMNS]
    reference = windows.copy()
    rng = np.random.default_rng(args.seed)
    windows = windows[rng.permutation(len(windows))[: args.train_events]]
    seq_length = ACTION_WINDOW_SAMPLES[args.action]

    dataset = WindowDataset(windows)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    model = build_model(seq_length, windows.shape[-1], 500).cuda()
    config = solver_config(args.steps, str(args.out.parent / f"ckpt_probe_{args.action}_{args.kind}"))
    config["solver"]["save_cycle"] = args.checkpoint_every
    trainer = Trainer(
        config=config,
        args=_Args(f"probe_{args.action}_{args.kind}", str(args.out.parent)),
        model=model,
        dataloader={"dataloader": loader},
        logger=None,
    )
    started = time.perf_counter()
    trainer.train()
    trained = time.perf_counter() - started

    real_lag1 = lag1(reference)
    real_std = reference.std(axis=(0, 1))
    rows = []
    milestones = args.steps // args.checkpoint_every
    for milestone in range(1, milestones + 1):
        trainer.load(milestone)
        raw = trainer.sample(
            num=args.samples, size_every=args.samples,
            shape=[seq_length, windows.shape[-1]]
        )[: args.samples]
        generated = dataset.unnormalize(raw).astype(np.float32)
        rows.append(
            {
                "step": milestone * args.checkpoint_every,
                "lag1": lag1(generated),
                "lag1_real": real_lag1,
                "std_ratio": (generated.std(axis=(0, 1)) / np.maximum(real_std, 1e-9)).tolist(),
                "nearest_training_distance": nearest_distance(generated, windows),
            }
        )
        print(
            f"  step {rows[-1]['step']:6d}  lag1 {rows[-1]['lag1']:.4f}"
            f" (real {real_lag1:.4f})  std ratio med "
            f"{np.median(rows[-1]['std_ratio']):.3f}  nn dist "
            f"{rows[-1]['nearest_training_distance']:.3f}",
            flush=True,
        )

    args.out.write_text(
        json.dumps(
            {
                "action": args.action,
                "kind": args.kind,
                "train_events": int(len(windows)),
                "seq_length": seq_length,
                "total_train_seconds": round(trained, 1),
                "seconds_per_1000_steps": round(1000 * trained / args.steps, 1),
                "real_lag1": real_lag1,
                "checkpoints": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
