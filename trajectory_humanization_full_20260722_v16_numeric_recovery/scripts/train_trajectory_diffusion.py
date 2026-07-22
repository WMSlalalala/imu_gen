#!/usr/bin/env python3
"""Formal action-specific trajectory diffusion training entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_determinism import CUBLAS_WORKSPACE_CONFIG  # noqa: F401,E402
from trajectory.data import KEYCODE_VOCAB_SIZE
from training.corpus import ACTIONS, FORMAL_SPLIT_PATH
from training.engine import TrainingConfig, train_action


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Train one strict five-reference action-specific trajectory diffusion model."
    )
    value.add_argument("--action", required=True, choices=ACTIONS)
    value.add_argument("--corpus-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--split-json", type=Path, default=FORMAL_SPLIT_PATH)
    value.add_argument("--resume", type=Path)
    value.add_argument("--epochs", type=int, default=100)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--learning-rate", type=float, default=2e-4)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--grad-clip-norm", type=float, default=1.0)
    value.add_argument("--ema-decay", type=float, default=0.999)
    value.add_argument("--diffusion-steps", type=int, default=1000)
    value.add_argument("--base-channels", type=int, default=96)
    value.add_argument("--cond-dim", type=int, default=192)
    value.add_argument("--time-dim", type=int, default=96)
    value.add_argument("--n-blocks", type=int, default=8)
    value.add_argument("--dropout", type=float, default=0.05)
    value.add_argument("--keycode-vocab", type=int, default=KEYCODE_VOCAB_SIZE)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--num-workers", type=int, default=0)
    value.add_argument("--checkpoint-every-steps", type=int, default=1000)
    value.add_argument("--reference-cache-size", type=int, default=2048)
    value.add_argument("--amp-overflow-max-retries", type=int, default=4)
    value.add_argument("--device", default="cuda")
    value.add_argument("--no-amp", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    corpus_npz = args.corpus_dir.resolve() / ("hmog_trajectory_%s.npz" % args.action)
    config = TrainingConfig(
        action=args.action,
        corpus_npz=str(corpus_npz),
        split_json=str(args.split_json.resolve()),
        output_dir=str(args.output_dir.resolve()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        ema_decay=args.ema_decay,
        diffusion_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        cond_dim=args.cond_dim,
        time_dim=args.time_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        keycode_vocab=args.keycode_vocab,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        checkpoint_every_steps=args.checkpoint_every_steps,
        reference_cache_size=args.reference_cache_size,
        amp_overflow_max_retries=args.amp_overflow_max_retries,
        device=args.device,
    )
    result = train_action(config, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
