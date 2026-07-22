"""Batched 50-step DDIM generation with per-sample reproducible noise."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from runtime_determinism import runtime_determinism_matches
from trajectory.constraints import ConstrainedTrajectory, constrain_and_decode
from trajectory.data import TrajectoryBatch
from trajectory.model import TrajectoryDiffusion


def checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_model_checkpoint(
    path: str, action: str, device: torch.device, require_best_ema: bool = True,
    expected_registry_sha256: Optional[str] = None,
    expected_split_sha256: Optional[str] = None,
    allow_e2e_smoke_checkpoint: bool = False,
) -> Tuple[TrajectoryDiffusion, str]:
    """Load a trained checkpoint; formal code never falls back to random weights."""
    checkpoint_path = Path(path).resolve()
    manifest = None
    best_entry = None
    manifest_protocol = None
    checkpoint_digest = None
    if require_best_ema:
        manifest_path = checkpoint_path.parent / "best_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("formal sampling requires best_manifest validation-selected checkpoint")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping) or not isinstance(manifest.get("best"), Mapping):
            raise ValueError("best_manifest is not a valid mapping")
        best_entry = dict(manifest["best"])
        manifest_protocol = manifest.get("protocol_version")
        allowed_manifest_protocols = {"trajectory_diffusion_strict_five_ref_v2"}
        if allow_e2e_smoke_checkpoint:
            allowed_manifest_protocols.add("trajectory_finalized_v2_e2e_smoke_v2")
        expected_metric = {
            "trajectory_diffusion_strict_five_ref_v2":
            "full_val_masked_epsilon_mse_ema",
            "trajectory_finalized_v2_e2e_smoke_v2":
            "fixed_smoke_val_masked_epsilon_mse_ema",
        }.get(manifest_protocol)
        recorded = Path(str(best_entry.get("path", "")))
        if not recorded.is_absolute():
            recorded = (checkpoint_path.parent / recorded).resolve()
        else:
            recorded = recorded.resolve()
        manifest_ok = (
            manifest_protocol in allowed_manifest_protocols
            and recorded == checkpoint_path
            and manifest.get("selection_split") == "val"
            and manifest.get("selection_metric") == expected_metric
            and manifest.get("lower_is_better") is True
            and manifest.get("test_used_for_selection") is False
            and manifest.get("checkpoint_role") == "validation_selected_best"
            and manifest.get("inference_weights") == "ema.shadow"
            and best_entry.get("checkpoint_role") == "validation_selected_best"
            and best_entry.get("inference_weights") == "ema.shadow"
            and isinstance(best_entry.get("checkpoint_sha256"), str)
            and len(best_entry["checkpoint_sha256"]) == 64
            and isinstance(manifest.get("history"), list)
            and bool(manifest["history"])
            and manifest["history"][-1] == best_entry
        )
        if not manifest_ok:
            raise ValueError("formal sampling requires a fully bound validation-selected best manifest")

    # Verify the exact bytes before unpickling them.  Loading from the same
    # in-memory byte string also closes the manifest-check/Torch-load race.
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    if require_best_ema and best_entry.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("best_manifest checkpoint SHA-256 does not match checkpoint bytes")
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    checkpoint_action = checkpoint.get(
        "action",
        checkpoint.get("model_action", checkpoint.get("config", {}).get("action", checkpoint.get("model_config", {}).get("action"))),
    )
    if checkpoint_action != action:
        raise ValueError("checkpoint action mismatch: %r != %r" % (checkpoint_action, action))
    config = dict(checkpoint.get("model_config", {}))
    config.pop("action", None)
    allowed = {
        "diffusion_steps", "beta_start", "beta_end", "base_channels", "cond_dim",
        "time_dim", "n_blocks", "dropout", "keycode_vocab",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError("unknown model config fields: %r" % sorted(unknown))
    if require_best_ema and int(config.get("diffusion_steps", -1)) != 1000:
        raise ValueError("formal best checkpoint must record the audited 1000-step training schedule")
    checkpoint_protocol = checkpoint.get("protocol_version")
    allowed_protocols = {"trajectory_diffusion_strict_five_ref_v2"}
    if allow_e2e_smoke_checkpoint:
        allowed_protocols.add("trajectory_finalized_v2_e2e_smoke_v2")
    if require_best_ema and (
        checkpoint_protocol not in allowed_protocols
        or checkpoint_protocol != manifest_protocol
        or not runtime_determinism_matches(checkpoint.get("runtime_determinism"))
    ):
        raise ValueError("formal best checkpoint lacks the strict runtime determinism contract")
    model = TrajectoryDiffusion(action=action, **config)
    if require_best_ema:
        source = checkpoint.get("source", {})
        if (
            not isinstance(source, Mapping)
            or manifest.get("source") != source
            or manifest.get("diffusion_schedule") != checkpoint.get("diffusion_schedule")
            or best_entry.get("source_sha256") != source.get("corpus_sha256")
            or best_entry.get("split_sha256") != source.get("split_sha256")
            or best_entry.get("reference_registry_sha256")
            != source.get("reference_registry_sha256")
        ):
            raise ValueError("best_manifest source/schedule binding mismatch")
        if checkpoint_protocol == "trajectory_diffusion_strict_five_ref_v2":
            progress = checkpoint.get("progress", {})
            last_validation = progress.get("last_validation", {}) if isinstance(progress, Mapping) else {}
            checkpoint_role_ok = (
                checkpoint.get("checkpoint_role")
                == "training_state_with_raw_model_and_ema"
                and checkpoint.get("inference_weights_for_validation_selected_best")
                == "ema.shadow"
            )
            progress_ok = (
                isinstance(progress, Mapping)
                and isinstance(last_validation, Mapping)
                and int(best_entry.get("completed_epoch", -1))
                == int(progress.get("epoch_index", -2))
                == int(last_validation.get("completed_epoch", -3))
                and int(best_entry.get("global_step", -1))
                == int(progress.get("global_step", -2))
                and float(best_entry.get("val_loss", float("nan")))
                == float(progress.get("best_val_loss", float("inf")))
                == float(last_validation.get("val_loss", float("inf")))
            )
        else:
            selection = checkpoint.get("selection", {})
            checkpoint_role_ok = (
                checkpoint.get("checkpoint_role") == "smoke_validation_best_ema"
                and checkpoint.get("inference_weights") == "ema.shadow"
            )
            progress_ok = (
                isinstance(selection, Mapping)
                and selection.get("split") == "val"
                and selection.get("metric") == manifest.get("selection_metric")
                and selection.get("test_used") is False
                and int(best_entry.get("step", -1))
                == int(selection.get("best_step", -2))
                and float(best_entry.get("val_loss", float("nan")))
                == float(selection.get("best_value", float("inf")))
            )
        if not checkpoint_role_ok or not progress_ok:
            raise ValueError("best checkpoint role/progress binding mismatch")
        if expected_registry_sha256 is not None and source.get("reference_registry_sha256") != expected_registry_sha256:
            raise ValueError("checkpoint/reference registry hash mismatch")
        if expected_split_sha256 is not None and source.get("split_sha256") != expected_split_sha256:
            raise ValueError("checkpoint/fixed split hash mismatch")
        schedule_audit = checkpoint.get("diffusion_schedule", {})
        if not schedule_audit.get("terminal_gaussian_gate_passed", False) or float(
            schedule_audit.get("alpha_bar_final", 1.0)
        ) > 1.0e-3:
            raise ValueError("checkpoint failed the terminal-Gaussian schedule gate")
        state_keys = ("ema_model_state_dict", "ema_state_dict", "training_ema_shadow")
    else:
        state_keys = ("ema_model_state_dict", "ema_state_dict", "model_state_dict", "state_dict")
    state = None
    for key in state_keys:
        if key in checkpoint:
            state = checkpoint[key]
            break
    if state is None and require_best_ema and isinstance(checkpoint.get("ema"), Mapping):
        state = checkpoint["ema"].get("shadow")
    if state is None:
        raise ValueError("checkpoint has no required EMA shadow state dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError("checkpoint state mismatch: missing=%r unexpected=%r" % (missing, unexpected))
    if model.diffusion_steps < 50:
        raise ValueError("checkpoint training schedule is shorter than formal 50-step DDIM")
    model.to(device)
    model.eval()
    return model, checkpoint_digest


@torch.no_grad()
def sample_ddim_seeded_batch(
    model: TrajectoryDiffusion,
    batch: TrajectoryBatch,
    noise_seeds: Sequence[int],
    inference_steps: int = 50,
) -> ConstrainedTrajectory:
    """Exactly N denoiser calls, fresh per-sample Gaussian noise, eta=0.

    Per-sample generators make the initial noise exactly independent of batch
    grouping, sharding and resume boundaries.  The masked denoiser excludes
    padding from hidden normalization/convolution; valid outputs are therefore
    numerically invariant to a longer co-batched request (strict regression
    tolerance), but floating batched kernels are not claimed bitwise identical
    across different batch shapes.  Formal generation fixes batch_size=32 and
    deterministic 200-sample unit boundaries, so resume preserves the original
    batch grouping exactly.
    """
    batch.validate(require_references=True)
    if batch.action != model.action or len(noise_seeds) != batch.features.shape[0]:
        raise ValueError("model/batch/seed mismatch")
    if inference_steps != 50:
        raise ValueError("formal generation fixes exactly 50 DDIM steps")
    schedule = model.ddim_timesteps(inference_steps).to(batch.features.device)
    mask = batch.feature_mask.to(batch.features.dtype)
    samples = []
    for seed in noise_seeds:
        generator = torch.Generator(device=batch.features.device)
        generator.manual_seed(int(seed))
        samples.append(torch.randn(
            batch.features.shape[1:], dtype=batch.features.dtype,
            device=batch.features.device, generator=generator,
        ))
    x = torch.stack(samples, dim=0) * mask
    encoded_condition = model.denoiser.encode_condition(batch)
    for schedule_index in reversed(range(schedule.numel())):
        step = int(schedule[schedule_index].item())
        timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
        predicted_noise = model.denoiser(x, timesteps, batch, encoded_metadata=encoded_condition)
        alpha_bar_t = model.alpha_bar[step].to(dtype=x.dtype, device=x.device)
        predicted_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        alpha_bar_previous = x.new_tensor(1.0) if schedule_index == 0 else model.alpha_bar[
            int(schedule[schedule_index - 1].item())
        ].to(dtype=x.dtype, device=x.device)
        direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_previous, min=0.0)) * predicted_noise
        x = (torch.sqrt(alpha_bar_previous) * predicted_x0 + direction) * mask
    return constrain_and_decode(x, batch)
