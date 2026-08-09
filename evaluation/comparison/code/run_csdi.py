#!/usr/bin/env python3
"""CSDI (Tashiro et al., NeurIPS 2021) as an IMU baseline on HMOG.

WHAT IS THE AUTHORS' AND WHAT IS MINE
-------------------------------------
The model is theirs, unmodified: `CSDI_Physio` from `CSDI/main_model.py`, built
from their own `config/base.yaml` hyper-parameters, trained by their own loss.
Nothing in `CSDI/` is edited -- the one import that fails out of the box
(`linear_attention_transformer`, pulled in at module scope by `diff_models.py`
even though `is_linear: False` means it is never constructed) was fixed by
installing the package, not by touching their file.

Three things are mine, and each is a decision the paper has to state:

1.  **A Dataset.**  CSDI ships loaders for three medical/air-quality corpora and
    none for inertial windows.  Mine returns exactly the four keys their
    `process_data` reads, in the (T, K) layout it expects.

2.  **Per-channel z-scoring.**  There is no normalisation layer anywhere in the
    model, and all three of their datasets standardise before the DataLoader.
    Ours must too: the accelerometer and gyroscope differ by roughly 7x in
    scale, and CSDI adds one shared noise level across all channels, so raw
    input would let the accelerometer dominate the loss and leave the gyroscope
    under-trained.  Statistics come from the training stack only and are
    inverted after sampling.

3.  **An unconditional draw.**  CSDI is an imputer and has no generation entry
    point.  Passing an all-zero conditioning mask turns their own `impute` into
    a plain DDPM over the whole window -- their code path, driven differently.
    The window handed in is then a pure shape carrier; its contents never reach
    the result.

TWO TRAPS THIS CODE STEPS AROUND, BOTH OF WHICH SILENTLY PRODUCE NOTHING
------------------------------------------------------------------------
*   `target_strategy` must stay `random`.  With a fully-observed mask -- which
    is what regularly-sampled IMU always gives -- their historical-masking
    branch returns `cond_mask == observed_mask`, so the target mask is
    identically zero and the loss is exactly 0.0.  The run would train to
    completion and learn nothing.  This code asserts the setting rather than
    trusting it.

*   Their `train()` writes one file, after the final epoch, and the
    `best_valid_loss` it computes is printed and never used to gate a save.  A
    crash at epoch 199 of 200 loses the run.  This driver keeps their optimiser,
    schedule and loss exactly as written but checkpoints every epoch and keeps
    the best by validation loss, for the same reason TTS-GAN needed it here.

FIVE-SHOT MODE
--------------
`--mode conditional` builds the layout their forecasting task already uses: a
sequence of length 6*T whose first five blocks are the victim's five real
windows with the conditioning mask on, and whose last block is the target with
the mask off.  That is cross-window conditioning expressed inside the single
(K, L) grid CSDI attends over, and it needs no model change.  Note that their
`impute` returns model-invented values at conditioned positions too -- it never
re-pastes the context -- so this driver slices out the target block explicitly
rather than trusting the returned grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "CSDI"))

IMU_CHANNELS = 6
# The detector's frozen window per action.  One model per action, because the
# sequence length is baked into the positional encoding and the attention grid.
ACTION_SAMPLES = {
    "tap": 16,
    "pinch": 100,
    "swipe": 176,
    "scroll": 208,
    "keystroke": 512,
}


class WindowDataset(torch.utils.data.Dataset):
    """The four keys `CSDI_Physio.process_data` reads, and nothing else.

    Their PM2.5 loader also emits `cut_length` and `hist_mask`, and their
    forecasting loader emits `feature_id`; both belong to other subclasses and
    would be silently ignored here.  Keeping to the four keys makes it obvious
    which subclass this data is for.
    """

    def __init__(self, windows: np.ndarray, conditioning: np.ndarray | None = None):
        self.windows = np.asarray(windows, dtype=np.float32)
        # `gt_mask` is only consulted when the model is in validation or
        # sampling mode; during training the conditioning mask is drawn
        # internally at random.  Passing ones here would make the validation
        # loss identically zero, so unconditioned rows default to zeros.
        if conditioning is None:
            conditioning = np.zeros_like(self.windows)
        self.conditioning = np.asarray(conditioning, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        window = self.windows[index]
        return {
            "observed_data": window,
            "observed_mask": np.ones_like(window),
            "gt_mask": self.conditioning[index],
            # Their positional encoding takes the raw integer sample index, not
            # seconds and not a normalised position (`dataset_physio.py:136`).
            "timepoints": np.arange(len(window), dtype=np.float32),
        }


def load_config(sample_steps: int | None) -> dict:
    """The authors' own base.yaml, with only the two edits this task forces."""

    config = yaml.safe_load((BASE / "CSDI" / "config" / "base.yaml").read_text())
    # Quadratic attention: their linear variant hardcodes max_seq_len=256, which
    # is shorter than the keystroke window (512) and than every conditional
    # layout here.  base.yaml already says False; this asserts it stayed that way.
    assert not config["diffusion"]["is_linear"], "is_linear must stay False at T>256"
    assert config["model"]["target_strategy"] == "random", (
        "target_strategy must stay 'random': with a fully-observed mask the "
        "historical-masking branch makes the training loss identically zero"
    )
    if sample_steps is not None:
        config["diffusion"]["num_steps"] = int(sample_steps)
    return config


def z_score(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel statistics over every cell of the training stack."""

    mean = stack.mean(axis=(0, 1))
    std = stack.std(axis=(0, 1))
    return mean, np.where(std > 1e-8, std, 1.0)


def train(model, loader, valid_loader, config, folder: Path, args) -> dict:
    """The authors' optimiser, schedule and loss -- with a checkpoint per epoch.

    Everything numerical here is copied from `CSDI/utils.py::train`: Adam at
    `config['lr']` with weight decay 1e-6, MultiStepLR at 75% and 90% of the
    epoch count with gamma 0.1, no clipping, no AMP, no EMA.  The only
    difference is that this loop saves, and keeps the best rather than the last.
    """

    epochs = int(args.epochs if args.epochs is not None else config["epochs"])
    optimiser = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimiser, milestones=[int(0.75 * epochs), int(0.9 * epochs)], gamma=0.1
    )

    checkpoint_path = folder / "checkpoint.pt"
    first_epoch, best = 0, float("inf")
    history: list = []
    if checkpoint_path.is_file() and not args.restart:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        first_epoch = int(state["epoch"]) + 1
        best = float(state["best_valid_loss"])
        history = list(state.get("history", []))
        print(f"resumed at epoch {first_epoch} (best valid {best:.5f})", flush=True)

    started = time.time()
    for epoch in range(first_epoch, epochs):
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            optimiser.zero_grad()
            loss = model(batch)
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            batches += 1
        scheduler.step()
        train_loss = total / max(batches, 1)

        # Their validation costs ~50x a training step because calc_loss_valid
        # loops every diffusion step, so it runs on an interval rather than
        # every epoch -- their default is 20.
        valid_loss = None
        if valid_loader is not None and (epoch + 1) % args.valid_every == 0:
            model.eval()
            total, batches = 0.0, 0
            with torch.no_grad():
                for batch in valid_loader:
                    total += float(model(batch, is_train=0).item())
                    batches += 1
            valid_loss = total / max(batches, 1)
            if valid_loss < best:
                best = valid_loss
                torch.save(model.state_dict(), folder / "best.pth")

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        )
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_valid_loss": best,
                "history": history,
            },
            folder / "checkpoint.pt.partial",
        )
        (folder / "checkpoint.pt.partial").replace(checkpoint_path)
        if epoch % 10 == 0 or epoch == epochs - 1:
            message = f"epoch {epoch}/{epochs} train {train_loss:.5f}"
            if valid_loss is not None:
                message += f" valid {valid_loss:.5f}"
            print(message, flush=True)

    # Prefer the best validation checkpoint; fall back to the terminal weights
    # when validation never ran, which is what their own code always hands back.
    if (folder / "best.pth").is_file():
        model.load_state_dict(torch.load(folder / "best.pth", map_location="cpu",
                                         weights_only=False))
        selected = "best_by_validation"
    else:
        selected = "final_epoch"
    return {
        "epochs_run": epochs - first_epoch,
        "train_seconds": round(time.time() - started, 1),
        "selected_checkpoint": selected,
        "best_valid_loss": None if best == float("inf") else best,
        "history": history,
    }


@torch.no_grad()
def draw_unconditional(model, samples: int, length: int, batch: int, device) -> np.ndarray:
    """Turn their imputer into a DDPM by conditioning on nothing.

    `impute` initialises `current_sample` from `randn_like(observed_data)` and,
    with `cond_mask` all zeros, never reads the observed values again -- so the
    tensor passed in is a shape carrier and its contents are irrelevant.  It is
    passed as zeros to make that explicit.
    """

    model.eval()
    drawn = []
    while sum(len(chunk) for chunk in drawn) < samples:
        count = min(batch, samples - sum(len(chunk) for chunk in drawn))
        carrier = torch.zeros(count, IMU_CHANNELS, length, device=device)
        conditioning = torch.zeros_like(carrier)
        timepoints = (
            torch.arange(length, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(count, 1)
        )
        side_info = model.get_side_info(timepoints, conditioning)
        # (count, 1, K, L) -- one trajectory per carrier window.
        imputed = model.impute(carrier, conditioning, side_info, 1)
        drawn.append(imputed[:, 0].permute(0, 2, 1).cpu().numpy())
    return np.concatenate(drawn)[:samples]


@torch.no_grad()
def draw_conditional(
    model, contexts: np.ndarray, length: int, batch: int, device
) -> np.ndarray:
    """Five victim windows as context, in the layout their forecasting task uses.

    Each sequence is 6*T long: five reference blocks with the conditioning mask
    on, then one target block with it off.  Only the target block is returned --
    `impute` writes model output across the whole grid, including the
    conditioned cells, so returning it unsliced would hand back a mangled copy
    of the references alongside the sample.

    `contexts` is (N, 5, T, 6): one row per window to be drawn, already carrying
    the five references of the victim that row belongs to.  Batching matters:
    each draw is a full reverse diffusion over 6*T steps, and running them one
    at a time would make the five-shot arm cost six times the unconditional one
    for no reason -- the grids are independent, so they batch exactly.
    """

    model.eval()
    count, reference_count = contexts.shape[0], contexts.shape[1]
    total_length = (reference_count + 1) * length
    drawn = []
    for start in range(0, count, batch):
        block = contexts[start : start + batch]
        size = len(block)
        carrier = torch.zeros(size, IMU_CHANNELS, total_length, device=device)
        conditioning = torch.zeros_like(carrier)
        # (size, 5, T, 6) -> (size, 6, 5*T), which is the model's channel-first grid.
        context = torch.tensor(
            block.reshape(size, reference_count * length, IMU_CHANNELS),
            dtype=torch.float32, device=device,
        ).permute(0, 2, 1)
        carrier[:, :, : reference_count * length] = context
        conditioning[:, :, : reference_count * length] = 1.0
        timepoints = (
            torch.arange(total_length, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(size, 1)
        )
        side_info = model.get_side_info(timepoints, conditioning)
        imputed = model.impute(carrier, conditioning, side_info, 1)
        target = imputed[:, 0, :, reference_count * length :]
        drawn.append(target.permute(0, 2, 1).cpu().numpy())
    return np.concatenate(drawn)[:count]


def victim_contexts(
    shots_cache: Path, action: str, length: int, samples: int, mean, std
) -> tuple[np.ndarray, list]:
    """Resample each victim's five real events onto the frozen detector window.

    The cache stores the five shots at their natural lengths (a tap can be 8 or
    13 rows), because that is what was actually recorded.  CSDI attends over a
    fixed grid, so they have to be put on the action's window first -- the same
    resampling the rest of this harness applies, and the same information a
    five-shot attacker has.

    The draw is spread evenly over victims rather than pooled, so this arm is
    per-victim five-shot in the same sense the source method is, and the
    returned owner list lets the dataset builder route each window back.
    """

    import pickle

    from hmog_baseline_common import linear_resample

    with shots_cache.open("rb") as handle:
        cache = pickle.load(handle)
    victims = sorted({user for user, act in cache if act == action})
    if not victims:
        raise SystemExit(f"the shots cache holds no {action} events")

    references = {}
    for victim in victims:
        shots = cache[(victim, action)][:5]
        stack = np.stack([linear_resample(np.asarray(s, np.float32), length) for s in shots])
        references[victim] = (stack - mean) / std

    contexts, owners = [], []
    for index in range(samples):
        victim = victims[index % len(victims)]
        contexts.append(references[victim])
        owners.append(victim)
    return np.stack(contexts), owners


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, required=True,
                        help="directory of real_train_<action>_imu.npy window stacks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action", required=True, choices=sorted(ACTION_SAMPLES))
    parser.add_argument("--mode", choices=("unconditional", "conditional"),
                        default="unconditional")
    parser.add_argument("--shots-cache", type=Path, default=None,
                        help="five-shot references, required by --mode conditional")
    parser.add_argument("--epochs", type=int, default=None,
                        help="default is the authors' 200 from base.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-batch", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--valid-every", type=int, default=20)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from main_model import CSDI_Physio

    windows = np.load(args.real_dir / f"real_train_{args.action}_imu.npy")
    windows = np.asarray(windows, dtype=np.float32)
    length = windows.shape[1]
    if length != ACTION_SAMPLES[args.action]:
        raise SystemExit(
            f"{args.action} windows are {length} samples, the frozen detector "
            f"window is {ACTION_SAMPLES[args.action]}"
        )
    if windows.shape[2] != IMU_CHANNELS:
        raise SystemExit(f"expected {IMU_CHANNELS} channels, got {windows.shape[2]}")

    mean, std = z_score(windows)
    standardised = (windows - mean) / std

    order = np.random.permutation(len(standardised))
    cut = max(1, int(len(order) * args.valid_fraction))
    valid_windows = standardised[order[:cut]]
    train_windows = standardised[order[cut:]]

    config = load_config(args.sample_steps)
    model = CSDI_Physio(config, device, target_dim=IMU_CHANNELS).to(device)
    parameters = sum(p.numel() for p in model.parameters())

    loader = torch.utils.data.DataLoader(
        WindowDataset(train_windows), batch_size=args.batch_size, shuffle=True,
        num_workers=2, drop_last=True,
    )
    # Validation holds out whole cells at random so the loss it reports is
    # computed on something: with gt_mask all ones the target mask is empty and
    # the number would be a constant zero.
    rng = np.random.default_rng(args.seed)
    valid_conditioning = (rng.random(valid_windows.shape) > 0.5).astype(np.float32)
    valid_loader = torch.utils.data.DataLoader(
        WindowDataset(valid_windows, valid_conditioning),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    summary = train(model, loader, valid_loader, config["train"], args.output_dir, args)

    started = time.time()
    owners = None
    if args.mode == "unconditional":
        raw = draw_unconditional(model, args.samples, length, args.sample_batch, device)
    else:
        if args.shots_cache is None:
            raise SystemExit("--mode conditional needs --shots-cache")
        contexts, owners = victim_contexts(
            args.shots_cache, args.action, length, args.samples, mean, std
        )
        # The conditional grid is six times longer, so the batch shrinks to keep
        # peak memory where the unconditional arm put it.
        raw = draw_conditional(
            model, contexts, length, max(1, args.sample_batch // 6), device
        )
    sampled = time.time() - started

    generated = (raw * std + mean).astype(np.float32)
    tag = f"{args.action}_{args.mode}"
    np.save(args.output_dir / f"samples_{tag}.npy", generated)
    if owners is not None:
        # Which victim's five shots produced each window, so the dataset builder
        # can give a victim only the windows conditioned on that victim.
        np.save(args.output_dir / f"owners_{tag}.npy", np.asarray(owners))
    (args.output_dir / f"summary_{tag}.json").write_text(
        json.dumps(
            {
                "method": "csdi",
                "citation": "Tashiro et al., CSDI, NeurIPS 2021",
                "action": args.action,
                "mode": args.mode,
                "seq_length": int(length),
                "channels": IMU_CHANNELS,
                "parameters": int(parameters),
                "train_windows": int(len(train_windows)),
                "valid_windows": int(len(valid_windows)),
                "samples": int(len(generated)),
                "sample_seconds": round(sampled, 2),
                "sample_ms_per_event": round(1000.0 * sampled / max(len(generated), 1), 4),
                "diffusion_steps": int(config["diffusion"]["num_steps"]),
                "authors_config": "CSDI/config/base.yaml, unmodified except num_steps",
                "generator_modified": False,
                "normalisation": "per-channel z-score over the training stack",
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"saved {generated.shape} to samples_{tag}.npy", flush=True)


if __name__ == "__main__":
    main()
