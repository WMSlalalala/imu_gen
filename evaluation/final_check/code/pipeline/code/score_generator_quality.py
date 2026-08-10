#!/usr/bin/env python3
"""Score a generator's samples with the metrics its own literature uses.

This exists to separate two very different reasons a baseline can score badly
on the detector grid: the method is genuinely weaker at this task, or it was
fitted badly here.  The discriminative and predictive scores are the standard
pair reported by TimeGAN and by Diffusion-TS, and the code is the authors' own
(``Utils/discriminative_metric.py``, ``Utils/predictive_metric.py``).

Every run also scores a **real-against-real** control: the genuine windows are
split in half and the same metrics are computed between the halves.  That is
the floor the data itself allows.  A generator near the control is fitted as
well as this data permits; one far above it is not, and its detector numbers
should not be reported as the method's ceiling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "DiffusionTS"))

def _authors_metrics():
    """Import the authors' TensorFlow metrics, or say plainly where they live.

    Both are TensorFlow 1.x-style code shipped by Diffusion-TS, and TensorFlow
    is installed in the `Sticker` environment rather than the `cuhkx` one
    everything else here runs in.  Importing at module scope made this script --
    and anything importing it -- die on `No module named 'tensorflow'`, which
    says nothing about the actual fix.
    """

    try:
        from Utils.discriminative_metric import discriminative_score_metrics
        from Utils.predictive_metric import predictive_score_metrics
    except ModuleNotFoundError as error:
        if error.name != "tensorflow":
            raise
        raise SystemExit(
            "the discriminative and predictive metrics are TensorFlow code from "
            "Diffusion-TS, and this interpreter has no tensorflow. Run this "
            "script with /home/mwang49/miniconda3/envs/Sticker/bin/python, "
            "which has tensorflow 2.11."
        ) from error
    return discriminative_score_metrics, predictive_score_metrics


def min_max_scale(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Put both sets on the reference's [0, 1] box, as the metrics assume.

    Deliberately not clipped.  Clipping would fold a generator's out-of-range
    excursions back onto the boundary and hide them from the metric, while the
    detector grid sees those same excursions in full -- the quality score and
    the FAR would then be describing different signals.
    """

    low = reference.min(axis=(0, 1), keepdims=True)
    high = reference.max(axis=(0, 1), keepdims=True)
    span = np.where(high - low > 0, high - low, 1.0)
    return (values - low) / span


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    real = np.load(args.real).astype(np.float64)
    generated = np.load(args.generated).astype(np.float64)
    if real.shape[1:] != generated.shape[1:]:
        raise SystemExit(f"shape mismatch {real.shape} vs {generated.shape}")

    rng = np.random.default_rng(args.seed)
    count = min(args.limit, len(real) // 2, len(generated))
    real_index = rng.permutation(len(real))
    real_a = min_max_scale(real, real[real_index[:count]])
    real_b = min_max_scale(real, real[real_index[count : 2 * count]])
    fake = min_max_scale(real, generated[rng.permutation(len(generated))[:count]])

    discriminative_score_metrics, predictive_score_metrics = _authors_metrics()

    scores = {"n_per_side": int(count)}
    for name, left, right in (
        ("control_real_vs_real", real_a, real_b),
        ("generated_vs_real", real_a, fake),
    ):
        discriminative, predictive = [], []
        for repeat in range(args.repeats):
            np.random.seed(args.seed + repeat)
            # The two metrics differ in what they accept: the discriminative one
            # calls np.asarray on its inputs, the predictive one reads .shape
            # directly and needs an array.
            outcome = discriminative_score_metrics(list(left), list(right))
            discriminative.append(
                float(outcome[0] if isinstance(outcome, tuple) else outcome)
            )
            predictive.append(float(predictive_score_metrics(left, right)))
        scores[name] = {
            "discriminative_mean": float(np.mean(discriminative)),
            "discriminative_std": float(np.std(discriminative)),
            "predictive_mean": float(np.mean(predictive)),
            "predictive_std": float(np.std(predictive)),
            "discriminative_runs": discriminative,
            "predictive_runs": predictive,
        }
        print(f"{name}: {scores[name]['discriminative_mean']:.4f} disc, "
              f"{scores[name]['predictive_mean']:.4f} pred", flush=True)

    args.out.write_text(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
