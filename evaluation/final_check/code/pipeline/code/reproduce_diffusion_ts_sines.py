#!/usr/bin/env python3
"""Reproduce Diffusion-TS's published Sines result with this checkout.

This is the environment check that stands behind every Diffusion-TS number we
report on HMOG.  It runs the authors' own dataset (``Utils/Data_utils.sine_dataset``,
which synthesises the data in code -- no download), their own configuration
(``Config/sines.yaml``), their own model, trainer and sampler, through the same
driver path ``run_diffusion_ts.py`` uses.  The samples are then scored with the
authors' own discriminative and predictive metrics.

The ICLR 2024 paper reports, for 24-length Sines:

    discriminative  0.006 +/- 0.007
    predictive      0.093 +/- 0.000

If this script lands near those numbers, the model, the training loop, the
sampler and the metric plumbing in this checkout all work; a poor HMOG result is
then a fact about HMOG, not about our setup.  Scoring runs in a separate
TensorFlow environment, so this script writes the arrays and prints the command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "DiffusionTS"))

from run_diffusion_ts import _Args, solver_config  # noqa: E402
from engine.solver import Trainer  # noqa: E402
from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS  # noqa: E402
from Utils.Data_utils.sine_dataset import SineDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Config/sines.yaml, verbatim.
    dataset = SineDataset(
        window=24,
        num=10000,
        dim=5,
        save2npy=True,
        neg_one_to_one=True,
        seed=123,
        period="train",
        output_dir=str(args.output_dir),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    model = Diffusion_TS(
        seq_length=24,
        feature_size=5,
        n_layer_enc=1,
        n_layer_dec=2,
        d_model=64,
        timesteps=500,
        sampling_timesteps=500,
        loss_type="l1",
        beta_schedule="cosine",
        n_heads=4,
        mlp_hidden_times=4,
        attn_pd=0.0,
        resid_pd=0.0,
        kernel_size=1,
        padding_size=0,
    ).cuda()
    trainer = Trainer(
        config=solver_config(args.steps, str(args.output_dir / "ckpt_sines")),
        args=_Args("sines", str(args.output_dir)),
        model=model,
        dataloader={"dataloader": loader},
        logger=None,
    )
    trainer.train()
    raw = trainer.sample(num=len(dataset), size_every=2001, shape=[24, 5])
    # The authors compare in the [0, 1] space their dataset saves as ground truth.
    samples = np.clip((raw + 1.0) / 2.0, 0.0, 1.0)[: len(dataset)]
    np.save(args.output_dir / "sines_generated.npy", samples.astype(np.float32))
    truth = np.load(args.output_dir / "samples" / "sine_ground_truth_24_train.npy")
    np.save(args.output_dir / "sines_real.npy", truth.astype(np.float32))
    print(f"generated {samples.shape}, truth {truth.shape}")
    print("score with:")
    print(
        f"  TF_USE_LEGACY_KERAS=1 <tsmetric-python> {BASE}/score_generator_quality.py "
        f"--real {args.output_dir}/sines_real.npy "
        f"--generated {args.output_dir}/sines_generated.npy "
        f"--out {args.output_dir}/sines_quality.json --limit 2000 --repeats 5"
    )


if __name__ == "__main__":
    main()
