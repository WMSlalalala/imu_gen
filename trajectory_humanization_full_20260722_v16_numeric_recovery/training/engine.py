"""Action-specific diffusion trainer with reproducible, atomic state."""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from runtime_determinism import (
    CUBLAS_WORKSPACE_CONFIG,
    require_strict_runtime_determinism,
    runtime_determinism_audit,
    runtime_determinism_matches,
    seed_everything,
)

import numpy as np
import torch

from trajectory.data import ACTIONS, KEYCODE_VOCAB_SIZE, TrajectoryBatch
from trajectory.model import TrajectoryDiffusion

from .corpus import NumericTrajectoryCorpus, SplitDefinition, atomic_json_dump
from .fewshot_dataset import ReferenceRegistry, StrictFiveReferenceDataset, make_epoch_loader


TRAINING_PROTOCOL_VERSION = "trajectory_diffusion_strict_five_ref_v2"
TRAINING_PROGRESS_SCHEMA = "trajectory_training_progress_v1"
LAST_STATE_SCHEMA = "trajectory_last_state_v1"
PROGRESS_HEARTBEAT_INTERVAL_SECONDS = 30.0
NUMERIC_RECOVERY_SCHEMA = "trajectory_amp_same_batch_retry_v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_loss_result(
    result: Mapping[str, Any], phase: str,
) -> Tuple[torch.Tensor, float]:
    """Validate the two scalars that make a masked loss auditable.

    A finite loss alone is insufficient: a NaN/zero feature count would make
    the epoch weighting invalid while still allowing optimizer/checkpoint
    progress.  Validate both before any optimizer step or aggregation.
    """

    loss = result.get("loss")
    count = result.get("valid_feature_count")
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise FloatingPointError("%s loss is not a scalar tensor" % phase)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite %s loss" % phase)
    if not isinstance(count, torch.Tensor) or count.numel() != 1:
        raise FloatingPointError("%s feature count is not a scalar tensor" % phase)
    count_value = float(count.detach().item())
    if not math.isfinite(count_value) or count_value <= 0.0:
        raise FloatingPointError(
            "non-finite or non-positive %s feature count" % phase
        )
    return loss, count_value


class TrainingProgressWriter:
    """Publish worker-owned progress without requiring checkpoint deserialization.

    Writes happen only at successful workflow boundaries (optimizer/EMA steps,
    validation batches, and checkpoint transactions).  Therefore a fresh file
    means useful worker progress, not merely that a PID or watchdog thread is
    alive.
    """

    def __init__(
        self,
        path: Path,
        config: "TrainingConfig",
        source: Mapping[str, Any],
        interval_seconds: float = PROGRESS_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self.path = Path(path)
        self.config_sha256 = _canonical_sha256(asdict(config))
        self.source = dict(source)
        self.interval_seconds = float(interval_seconds)
        self.run_instance_id = uuid.uuid4().hex
        self.started_unix_time = time.time()
        self.last_written_monotonic = float("-inf")
        self.sequence = 0
        self.last_successful_progress_unix_time: Optional[float] = None
        self.last_successful_step = 0
        self.last_loss: Optional[float] = None
        self.grad_norm: Optional[float] = None
        self.amp_overflow_retries_total = 0
        self.action = str(config.action)
        self.device = str(config.device)

    def publish(
        self,
        *,
        phase: str,
        epoch_index: int,
        next_batch_in_epoch: int,
        global_step: int,
        examples_seen_in_epoch: int,
        successful_progress: bool = False,
        last_loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        validation_batch_index: Optional[int] = None,
        validation_batches_total: Optional[int] = None,
        force: bool = False,
    ) -> bool:
        if phase not in {
            "init", "train", "validation", "checkpoint_commit", "complete"
        }:
            raise ValueError("invalid training progress phase: %s" % phase)
        now = time.time()
        monotonic_now = time.monotonic()
        if successful_progress:
            self.last_successful_progress_unix_time = now
            self.last_successful_step = int(global_step)
            if last_loss is not None:
                value = float(last_loss)
                if not math.isfinite(value) or value <= 0.0:
                    raise FloatingPointError(
                        "training progress loss must be finite and positive"
                    )
                self.last_loss = value
            if grad_norm is not None:
                value = float(grad_norm)
                if not math.isfinite(value) or value < 0.0:
                    raise FloatingPointError(
                        "training progress gradient norm must be finite and non-negative"
                    )
                self.grad_norm = value
        if not force and monotonic_now - self.last_written_monotonic < self.interval_seconds:
            return False
        self.sequence += 1
        payload = {
            "schema_version": TRAINING_PROGRESS_SCHEMA,
            "protocol_version": TRAINING_PROTOCOL_VERSION,
            "run_instance_id": self.run_instance_id,
            "action": self.action,
            "pid": int(os.getpid()),
            "source": self.source,
            "config_sha256": self.config_sha256,
            "phase": phase,
            "epoch_index": int(epoch_index),
            "next_batch_in_epoch": int(next_batch_in_epoch),
            "global_step": int(global_step),
            "examples_seen_in_epoch": int(examples_seen_in_epoch),
            "last_successful_step": int(self.last_successful_step),
            "last_successful_progress_unix_time": self.last_successful_progress_unix_time,
            "last_loss": self.last_loss,
            "grad_norm": self.grad_norm,
            "amp_overflow_retries_total": int(self.amp_overflow_retries_total),
            "device": self.device,
            "validation_batch_index": (
                None if validation_batch_index is None else int(validation_batch_index)
            ),
            "validation_batches_total": (
                None if validation_batches_total is None else int(validation_batches_total)
            ),
            "heartbeat_sequence": int(self.sequence),
            "started_unix_time": float(self.started_unix_time),
            "updated_unix_time": float(now),
        }
        atomic_json_dump(self.path, payload)
        self.last_written_monotonic = monotonic_now
        return True


def _write_last_state(
    path: Path,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    config: "TrainingConfig",
    source: Mapping[str, Any],
    run_instance_id: str,
) -> None:
    progress = checkpoint.get("progress")
    if not isinstance(progress, Mapping):
        raise ValueError("last checkpoint lacks progress metadata")
    stat = checkpoint_path.stat()
    atomic_json_dump(path, {
        "schema_version": LAST_STATE_SCHEMA,
        "protocol_version": TRAINING_PROTOCOL_VERSION,
        "run_instance_id": str(run_instance_id),
        "action": str(config.action),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "checkpoint_size_bytes": int(stat.st_size),
        "progress": {
            "epoch_index": int(progress["epoch_index"]),
            "next_batch_in_epoch": int(progress["next_batch_in_epoch"]),
            "examples_seen_in_epoch": int(progress["examples_seen_in_epoch"]),
            "global_step": int(progress["global_step"]),
        },
        "source": dict(source),
        "config_sha256": _canonical_sha256(asdict(config)),
        "updated_unix_time": time.time(),
    })


@dataclass(frozen=True)
class TrainingConfig:
    action: str
    corpus_npz: str
    split_json: str
    output_dir: str
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    diffusion_steps: int = 1000
    base_channels: int = 96
    cond_dim: int = 192
    time_dim: int = 96
    n_blocks: int = 8
    dropout: float = 0.05
    keycode_vocab: int = KEYCODE_VOCAB_SIZE
    seed: int = 42
    num_workers: int = 0
    amp: bool = True
    checkpoint_every_steps: int = 1000
    reference_cache_size: int = 2048
    device: str = "cuda"
    amp_overflow_max_retries: int = 4
    allow_non_gaussian_terminal_for_test: bool = False

    def validate(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError("unsupported action: %s" % self.action)
        positive_ints = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "diffusion_steps": self.diffusion_steps,
            "base_channels": self.base_channels,
            "cond_dim": self.cond_dim,
            "time_dim": self.time_dim,
            "n_blocks": self.n_blocks,
            "keycode_vocab": self.keycode_vocab,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "amp_overflow_max_retries": self.amp_overflow_max_retries,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError("%s must be positive" % name)
        if self.diffusion_steps < 2 or self.keycode_vocab < KEYCODE_VOCAB_SIZE:
            raise ValueError(
                "diffusion_steps>=2 and keycode_vocab>=%d are required"
                % KEYCODE_VOCAB_SIZE
            )
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("invalid optimizer configuration")
        if not 0.0 < self.ema_decay < 1.0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid EMA/dropout configuration")
        if self.num_workers < 0 or self.reference_cache_size < 0:
            raise ValueError("num_workers/cache_size cannot be negative")
        if self.alpha_bar_final > 1.0e-3 and not self.allow_non_gaussian_terminal_for_test:
            raise ValueError(
                "diffusion terminal is not close to N(0,I): alpha_bar_final=%.8g > 1e-3; "
                "formal training requires the 1000-step schedule"
                % self.alpha_bar_final
            )

    @property
    def alpha_bar_final(self) -> float:
        betas = np.linspace(1.0e-4, 2.0e-2, int(self.diffusion_steps), dtype=np.float64)
        return float(np.prod(1.0 - betas))

    @property
    def diffusion_schedule_audit(self) -> Dict[str, Any]:
        return {
            "schedule": "linear_beta",
            "beta_start": 1.0e-4,
            "beta_end": 2.0e-2,
            "diffusion_steps": int(self.diffusion_steps),
            "alpha_bar_final": self.alpha_bar_final,
            "formal_max_alpha_bar_final": 1.0e-3,
            "terminal_gaussian_gate_passed": self.alpha_bar_final <= 1.0e-3,
            "test_only_override": bool(self.allow_non_gaussian_terminal_for_test),
        }

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "diffusion_steps": int(self.diffusion_steps),
            "base_channels": int(self.base_channels),
            "cond_dim": int(self.cond_dim),
            "time_dim": int(self.time_dim),
            "n_blocks": int(self.n_blocks),
            "dropout": float(self.dropout),
            "keycode_vocab": int(self.keycode_vocab),
        }

    @property
    def numeric_recovery_policy(self) -> Dict[str, Any]:
        return {
            "schema_version": NUMERIC_RECOVERY_SCHEMA,
            "amp_enabled": bool(self.amp),
            "max_overflow_retries_per_batch": int(self.amp_overflow_max_retries),
            "retry_same_batch": True,
            "restore_pre_attempt_rng": True,
            "count_examples_only_after_finite_optimizer_step": True,
            "update_ema_only_after_finite_optimizer_step": True,
            "scale_backoff_factor_source": "torch_grad_scaler_state",
        }


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise ValueError("EMA/model state keys changed")
        for name, value in current.items():
            if torch.is_floating_point(value):
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value.detach())

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any], device: torch.device) -> None:
        if abs(float(state["decay"]) - self.decay) > 1e-12:
            raise ValueError("EMA decay differs from checkpoint")
        shadow = state["shadow"]
        if shadow.keys() != self.shadow.keys():
            raise ValueError("EMA checkpoint keys differ")
        self.shadow = {name: value.to(device) for name, value in shadow.items()}

    def copy_to(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        original = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return original

    @staticmethod
    def restore(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
        model.load_state_dict(state, strict=True)


def validation_epochs(total_epochs: int) -> Tuple[int, ...]:
    """Five milestones at 20%,40%,60%,80%,100% of completed epochs."""
    if int(total_epochs) <= 0:
        raise ValueError("total_epochs must be positive")
    return tuple(sorted(set(int(math.ceil(total_epochs * fraction / 5.0)) for fraction in range(1, 6))))


def make_seeded_generator(device: torch.device, seed: int) -> torch.Generator:
    """Bind RNG to the exact device, including a non-default CUDA index."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([
            value.detach().cpu() for value in state["torch_cuda"]
        ])


def _backoff_amp_scaler_without_optimizer_step(
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, float]:
    """Back off an AMP scale while guaranteeing no optimizer update occurs."""

    state = scaler.state_dict()
    before = float(state["scale"])
    factor = float(state["backoff_factor"])
    after = before * factor
    if not (
        math.isfinite(before) and before > 0.0
        and math.isfinite(factor) and 0.0 < factor < 1.0
        and math.isfinite(after) and 0.0 < after < before
    ):
        raise FloatingPointError("invalid AMP scale backoff state")
    # update(new_scale=...) clears GradScaler's per-optimizer UNSCALED state,
    # but never calls optimizer.step.  Reset the growth tracker exactly as a
    # normal found-inf backoff would before retrying the same batch.
    scaler.update(new_scale=after)
    backed_off = scaler.state_dict()
    backed_off["_growth_tracker"] = 0
    scaler.load_state_dict(backed_off)
    return before, after


def atomic_torch_save(path: Path, payload: Mapping[str, Any], overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError("versioned checkpoint already exists: %s" % path)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(dict(payload), temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError("versioned checkpoint appeared concurrently: %s" % path)
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fault_inject(phase: str) -> None:
    """Test-only crash hook; formal runs leave the environment unset."""
    if os.environ.get("TRAJECTORY_TRAINING_FAULT_AFTER") == phase:
        raise RuntimeError("injected training commit fault after %s" % phase)


def _append_jsonl_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one epoch/milestone record exactly once across crash recovery."""
    identity = (str(payload.get("type")), int(payload.get("completed_epoch", -1)))
    matches = []
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if (str(row.get("type")), int(row.get("completed_epoch", -1))) == identity:
                    matches.append((line_number, row))
    if len(matches) > 1:
        raise ValueError("duplicate committed metric identity %r in %s" % (identity, path))
    if matches:
        if matches[0][1] != dict(payload):
            raise ValueError("metric identity %r exists with different payload" % (identity,))
        return
    _append_jsonl(path, payload)


def _checkpoint_progress_epoch(path: Path) -> int:
    checkpoint = torch.load(str(path), map_location="cpu")
    return int(checkpoint.get("progress", {}).get("epoch_index", -1))


def _same_file_sha256(first: Path, second: Path) -> bool:
    import hashlib

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                value.update(block)
        return value.hexdigest()

    return first.stat().st_size == second.stat().st_size and digest(first) == digest(second)


def _checkpoint_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _without_best_file_binding(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(value)
    for key in ("checkpoint_sha256", "checkpoint_role", "inference_weights"):
        result.pop(key, None)
    return result


def _bind_journal_best_to_checkpoint(
    journal: Mapping[str, Any], checkpoint: Path,
) -> Dict[str, Any]:
    """Bind best metadata to immutable bytes without self-hashing the .pt."""

    result = dict(journal)
    best_entry = result.get("best_entry")
    if best_entry is None:
        return result
    manifest = result.get("best_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("best transaction is missing its manifest")
    if _without_best_file_binding(manifest.get("best", {})) != _without_best_file_binding(best_entry):
        raise ValueError("best manifest does not match its transaction entry")
    digest = _checkpoint_sha256(checkpoint)
    existing = best_entry.get("checkpoint_sha256")
    if existing is not None and existing != digest:
        raise ValueError("best checkpoint bytes contradict the epoch journal SHA-256")
    enriched = _without_best_file_binding(best_entry)
    enriched.update({
        "checkpoint_sha256": digest,
        "checkpoint_role": "validation_selected_best",
        "inference_weights": "ema.shadow",
    })
    history = [dict(row) for row in manifest.get("history", [])]
    if not history or _without_best_file_binding(history[-1]) != _without_best_file_binding(enriched):
        raise ValueError("best history tail does not match its transaction entry")
    history[-1] = dict(enriched)
    enriched_manifest = dict(manifest)
    enriched_manifest.update({
        "best": dict(enriched),
        "history": history,
        "checkpoint_role": "validation_selected_best",
        "inference_weights": "ema.shadow",
    })
    result["best_entry"] = dict(enriched)
    result["best_manifest"] = enriched_manifest
    return result


def _promote_immutable_checkpoint(source: Path, destination: Path) -> None:
    if destination.exists():
        if not _same_file_sha256(source, destination):
            raise FileExistsError("immutable best checkpoint collision: %s" % destination)
        return
    try:
        os.link(str(source), str(destination))
    except FileExistsError:
        if not _same_file_sha256(source, destination):
            raise
    _fsync_directory(destination.parent)


def _reconcile_epoch_commit(
    output_dir: Path,
    config: TrainingConfig,
    corpus: NumericTrajectoryCorpus,
    registry: ReferenceRegistry,
) -> bool:
    """Finish an interrupted epoch transaction before loading ``last.pt``.

    The journal is published only after a complete next-epoch checkpoint has
    been staged.  Therefore recovery never has to reconstruct model/optimizer
    state from JSON and never repeats an already committed metric row.
    """
    journal_path = output_dir / "epoch_commit.json"
    if not journal_path.is_file():
        pending = sorted(output_dir.glob(".epoch_*_next.pt.pending"))
        if not pending:
            return False
        if len(pending) != 1:
            raise ValueError("multiple orphan epoch commit checkpoints found")
        orphan = torch.load(str(pending[0]), map_location="cpu")
        _check_resume_compatibility(orphan, config, corpus, registry)
        embedded = orphan.get("epoch_commit")
        if not isinstance(embedded, dict):
            raise ValueError("orphan staged checkpoint lacks embedded commit metadata")
        journal_from_checkpoint = dict(embedded)
        if (
            journal_from_checkpoint.get("schema_version")
            != "trajectory_epoch_commit_v1"
            or journal_from_checkpoint.get("protocol_version")
            != TRAINING_PROTOCOL_VERSION
            or not runtime_determinism_matches(
                journal_from_checkpoint.get("runtime_determinism")
            )
            or journal_from_checkpoint.get("runtime_determinism")
            != orphan.get("runtime_determinism")
            or journal_from_checkpoint.get("model_config") != config.model_config
            or journal_from_checkpoint.get("source")
            != _source_identity(corpus, registry)
        ):
            raise ValueError("orphan embedded epoch journal/config/runtime mismatch")
        journal_from_checkpoint["staged_checkpoint"] = str(pending[0])
        atomic_json_dump(journal_path, journal_from_checkpoint)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if (
        journal.get("schema_version") != "trajectory_epoch_commit_v1"
        or journal.get("protocol_version") != TRAINING_PROTOCOL_VERSION
        or not runtime_determinism_matches(journal.get("runtime_determinism"))
        or journal.get("model_config") != config.model_config
        or journal.get("source") != _source_identity(corpus, registry)
    ):
        raise ValueError("epoch commit journal/config/source mismatch")
    completed_epoch = int(journal["completed_epoch"])
    staged = Path(journal["staged_checkpoint"])
    last_path = output_dir / "last.pt"
    source_checkpoint = staged if staged.is_file() else last_path
    if not source_checkpoint.is_file():
        raise FileNotFoundError("epoch journal has neither staged nor last checkpoint")
    checkpoint = torch.load(str(source_checkpoint), map_location="cpu")
    _check_resume_compatibility(checkpoint, config, corpus, registry)
    if checkpoint.get("runtime_determinism") != journal.get("runtime_determinism"):
        raise ValueError("epoch journal/checkpoint runtime determinism mismatch")
    observed_epoch = _checkpoint_progress_epoch(source_checkpoint)
    if observed_epoch != completed_epoch:
        raise ValueError(
            "epoch journal checkpoint progress mismatch: %d != %d"
            % (observed_epoch, completed_epoch)
        )
    journal = _bind_journal_best_to_checkpoint(journal, source_checkpoint)
    atomic_json_dump(journal_path, journal)
    best_entry = journal.get("best_entry")
    best_manifest = journal.get("best_manifest")
    if best_entry is not None:
        best_path = Path(str(best_entry["path"]))
        _promote_immutable_checkpoint(source_checkpoint, best_path)
        if not isinstance(best_manifest, dict) or best_manifest.get("best") != best_entry:
            raise ValueError("epoch journal best manifest is incomplete")
        atomic_json_dump(output_dir / "best_manifest.json", best_manifest)
        if _checkpoint_sha256(best_path) != best_entry["checkpoint_sha256"]:
            raise ValueError("promoted best checkpoint SHA-256 mismatch")
    if staged.is_file():
        os.replace(str(staged), str(last_path))
        _fsync_directory(output_dir)
    _append_jsonl_once(output_dir / "metrics.jsonl", journal["train_record"])
    if journal.get("validation_record") is not None:
        _append_jsonl_once(output_dir / "metrics.jsonl", journal["validation_record"])
    journal_path.unlink()
    _fsync_directory(output_dir)
    return True


def _commit_epoch_transaction(
    output_dir: Path,
    config: TrainingConfig,
    corpus: NumericTrajectoryCorpus,
    registry: ReferenceRegistry,
    next_checkpoint: Mapping[str, Any],
    train_record: Mapping[str, Any],
    validation_record: Optional[Mapping[str, Any]],
    best_entry: Optional[Mapping[str, Any]],
    desired_best_manifest: Optional[Mapping[str, Any]],
) -> None:
    """Atomically recoverable best/manifest/last/metrics epoch transaction."""
    completed_epoch = int(next_checkpoint["progress"]["epoch_index"])
    staged = output_dir / (".epoch_%04d_next.pt.pending" % completed_epoch)
    journal_path = output_dir / "epoch_commit.json"
    journal = {
        "schema_version": "trajectory_epoch_commit_v1",
        "protocol_version": TRAINING_PROTOCOL_VERSION,
        "runtime_determinism": require_strict_runtime_determinism(),
        "numeric_recovery_policy": config.numeric_recovery_policy,
        "model_config": config.model_config,
        "source": _source_identity(corpus, registry),
        "completed_epoch": completed_epoch,
        "staged_checkpoint": str(staged),
        "train_record": dict(train_record),
        "validation_record": None if validation_record is None else dict(validation_record),
        "best_entry": None if best_entry is None else dict(best_entry),
        "best_manifest": None if desired_best_manifest is None else dict(desired_best_manifest),
    }
    checkpoint_with_commit = dict(next_checkpoint)
    # This closes the only gap between the large checkpoint publication and
    # the small JSON journal publication.  If power is lost in that window,
    # startup reconstructs the exact journal from the staged checkpoint.
    checkpoint_with_commit["epoch_commit"] = dict(journal)
    atomic_torch_save(staged, checkpoint_with_commit, overwrite=True)
    _fault_inject("staged_checkpoint")
    journal = _bind_journal_best_to_checkpoint(journal, staged)
    atomic_json_dump(journal_path, journal)
    _fault_inject("journal")
    if best_entry is not None:
        best_entry = journal["best_entry"]
        desired_best_manifest = journal["best_manifest"]
        best_path = Path(str(best_entry["path"]))
        _promote_immutable_checkpoint(staged, best_path)
        if _checkpoint_sha256(best_path) != best_entry["checkpoint_sha256"]:
            raise ValueError("promoted best checkpoint SHA-256 mismatch")
        _fault_inject("best_checkpoint")
        atomic_json_dump(output_dir / "best_manifest.json", desired_best_manifest)
        _fault_inject("best_manifest")
    os.replace(str(staged), str(output_dir / "last.pt"))
    _fsync_directory(output_dir)
    _fault_inject("last_checkpoint")
    _append_jsonl_once(output_dir / "metrics.jsonl", train_record)
    _fault_inject("train_metric")
    if validation_record is not None:
        _append_jsonl_once(output_dir / "metrics.jsonl", validation_record)
        _fault_inject("validation_metric")
    journal_path.unlink()
    _fsync_directory(output_dir)


def _source_identity(
    corpus: NumericTrajectoryCorpus,
    registry: Optional[ReferenceRegistry] = None,
) -> Dict[str, Any]:
    result = {
        "corpus_npz": str(corpus.path),
        "corpus_sha256": corpus.sha256,
        "schema_version": corpus.schema_version,
        "action": corpus.action,
        "split_json": str(corpus.splits.path),
        "split_sha256": corpus.splits.sha256,
        "split_seed": corpus.splits.seed,
        "extraction_manifest": str(corpus.extraction_manifest_path) if corpus.extraction_manifest_path.is_file() else None,
        "extraction_manifest_sha256": corpus.extraction_manifest_sha256,
    }
    if registry is not None:
        result["reference_registry_sha256"] = registry.sha256
        result["reference_registry_protocol"] = registry.PROTOCOL
    return result


def _make_checkpoint(
    config: TrainingConfig,
    corpus: NumericTrajectoryCorpus,
    registry: ReferenceRegistry,
    model: TrajectoryDiffusion,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch_index: int,
    next_batch_in_epoch: int,
    examples_seen_in_epoch: int,
    global_step: int,
    best_val_loss: float,
    last_validation: Optional[Mapping[str, Any]],
    epoch_weighted_loss: float,
    epoch_feature_count: float,
    epoch_new_batches: int,
    amp_overflow_retries_total: int,
    epoch_amp_overflow_events: Sequence[Mapping[str, Any]],
    last_step_loss: Optional[float],
    last_grad_norm: Optional[float],
) -> Dict[str, Any]:
    return {
        "protocol_version": TRAINING_PROTOCOL_VERSION,
        "runtime_determinism": require_strict_runtime_determinism(),
        "checkpoint_role": "training_state_with_raw_model_and_ema",
        "inference_weights_for_validation_selected_best": "ema.shadow",
        "created_unix_time": time.time(),
        "config": asdict(config),
        "model_config": config.model_config,
        "diffusion_schedule": config.diffusion_schedule_audit,
        "numeric_recovery_policy": config.numeric_recovery_policy,
        "source": _source_identity(corpus, registry),
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp_scaler": scaler.state_dict(),
        "progress": {
            "epoch_index": int(epoch_index),
            "next_batch_in_epoch": int(next_batch_in_epoch),
            "examples_seen_in_epoch": int(examples_seen_in_epoch),
            "global_step": int(global_step),
            "best_val_loss": float(best_val_loss),
            "last_validation": None if last_validation is None else dict(last_validation),
            "epoch_weighted_loss": float(epoch_weighted_loss),
            "epoch_feature_count": float(epoch_feature_count),
            "epoch_new_batches": int(epoch_new_batches),
            "amp_overflow_retries_total": int(amp_overflow_retries_total),
            "epoch_amp_overflow_events": [
                dict(value) for value in epoch_amp_overflow_events
            ],
            "last_step_loss": (
                None if last_step_loss is None else float(last_step_loss)
            ),
            "last_grad_norm": (
                None if last_grad_norm is None else float(last_grad_norm)
            ),
        },
        "rng_state": _rng_state(),
    }


def _check_resume_compatibility(
    checkpoint: Mapping[str, Any],
    config: TrainingConfig,
    corpus: NumericTrajectoryCorpus,
    registry: ReferenceRegistry,
) -> None:
    if checkpoint.get("protocol_version") != TRAINING_PROTOCOL_VERSION:
        raise ValueError("checkpoint protocol version mismatch")
    if not runtime_determinism_matches(checkpoint.get("runtime_determinism")):
        raise ValueError("checkpoint runtime determinism contract mismatch")
    if checkpoint.get("model_config") != config.model_config:
        raise ValueError("model configuration differs from checkpoint")
    if checkpoint.get("source") != _source_identity(corpus, registry):
        raise ValueError("source/split identity differs from checkpoint")
    old = checkpoint["config"]
    immutable = (
        "action", "batch_size", "learning_rate", "weight_decay", "grad_clip_norm",
        "ema_decay", "seed", "amp", "amp_overflow_max_retries",
    )
    for name in immutable:
        if old.get(name) != asdict(config).get(name):
            raise ValueError("resume changed immutable training field: %s" % name)
    if checkpoint.get("numeric_recovery_policy") != config.numeric_recovery_policy:
        raise ValueError("resume numeric recovery policy mismatch")
    if int(config.epochs) < int(checkpoint["progress"]["epoch_index"]):
        raise ValueError("new epochs ends before checkpoint progress")


@torch.no_grad()
def evaluate_full_validation(
    model: TrajectoryDiffusion,
    ema: ExponentialMovingAverage,
    dataset: StrictFiveReferenceDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_enabled: bool,
    seed: int,
    completed_epoch: int,
    total_epochs: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    dataset.set_epoch(0)
    loader = make_epoch_loader(
        dataset, batch_size=batch_size, epoch=0, num_workers=num_workers,
        pin_memory=device.type == "cuda", shuffle=False,
    )
    original = ema.copy_to(model)
    model.eval()
    weighted_loss = 0.0
    valid_features = 0.0
    examples = 0
    batch_count = 0
    generator = make_seeded_generator(device, int(seed) + 7000001)
    try:
        total_batches = int(math.ceil(len(dataset) / float(batch_size)))
        for batch in loader:
            batch = batch.to(device)
            b = int(batch.features.shape[0])
            timesteps = torch.randint(
                0, model.diffusion_steps, (b,), device=device, dtype=torch.long,
                generator=generator,
            )
            noise = torch.randn(
                batch.features.shape, dtype=batch.features.dtype, device=device,
                generator=generator,
            )
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                result = model.training_loss(batch, timesteps=timesteps, noise=noise)
            loss, count = _validated_loss_result(result, "validation")
            weighted_loss += float(loss.item()) * count
            valid_features += count
            examples += b
            batch_count += 1
            if progress_callback is not None:
                progress_callback(batch_count, total_batches)
    finally:
        ExponentialMovingAverage.restore(model, original)
        model.train()
    if examples != len(dataset) or valid_features <= 0:
        raise AssertionError("validation did not consume the full split")
    val_loss = weighted_loss / valid_features
    if not math.isfinite(val_loss) or val_loss <= 0.0:
        raise FloatingPointError("non-finite or non-positive aggregate validation loss")
    return {
        "completed_epoch": int(completed_epoch),
        "fraction": float(completed_epoch) / float(total_epochs),
        "val_loss": val_loss,
        "n_examples": examples,
        "n_batches": batch_count,
        "valid_feature_count": valid_features,
        "ema_weights": True,
        "full_validation_split": True,
    }


def train_action(config: TrainingConfig, resume: Optional[Path] = None) -> Dict[str, Any]:
    """Train one action on the complete train split; never uses test targets."""
    config.validate()
    seed_everything(config.seed)
    runtime_determinism = require_strict_runtime_determinism()
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protected_outputs = [
        output_dir / "last.pt",
        output_dir / "last_state.json",
        output_dir / "training_progress.json",
        output_dir / "best_manifest.json",
        output_dir / "metrics.jsonl",
        output_dir / "run_manifest.json",
        output_dir / "epoch_commit.json",
    ]
    if resume is None and (
        any(path.exists() for path in protected_outputs)
        or any(output_dir.glob(".epoch_*_next.pt.pending"))
    ):
        raise FileExistsError(
            "training output already exists; pass --resume explicitly instead of overwriting"
        )
    split = SplitDefinition.load(Path(config.split_json), require_pinned_hash=True)
    corpus = NumericTrajectoryCorpus(
        Path(config.corpus_npz), split, expected_action=config.action,
        verify_sha256=True,
    )
    # Full schema/event/reference audit occurs before optimizer construction.
    source_audit = corpus.audit(require_all_users=True, validate_every_event=True)
    source_audit["training_schedule_gate"] = config.diffusion_schedule_audit
    atomic_json_dump(output_dir / "source_audit.json", source_audit)
    registry = ReferenceRegistry.build(corpus, seed=config.seed)
    registry_path = output_dir / "reference_registry.json"
    registry.save(registry_path)
    source_identity = _source_identity(corpus, registry)
    train_dataset = StrictFiveReferenceDataset(
        corpus, "train", registry, seed=config.seed, cache_size=config.reference_cache_size
    )
    val_dataset = StrictFiveReferenceDataset(
        corpus, "val", registry, seed=config.seed, cache_size=config.reference_cache_size
    )
    # Test construction proves reference availability, but test samples are not
    # iterated, scored or used to choose a checkpoint.
    test_dataset = StrictFiveReferenceDataset(
        corpus, "test", registry, seed=config.seed, cache_size=0
    )
    dataset_audit = {
        "train": train_dataset.reference_audit(),
        "val": val_dataset.reference_audit(),
        "test": test_dataset.reference_audit(),
        "test_used_for_training_or_selection": False,
        "no_sample_cap": True,
    }
    atomic_json_dump(output_dir / "reference_audit.json", dataset_audit)

    # Complete any durable epoch transaction before the requested resume path
    # is opened.  In particular, a crash after journal publication but before
    # ``last.pt`` promotion is recoverable when the caller names the eventual
    # ``last.pt`` path even though it did not exist at process start.
    _reconcile_epoch_commit(output_dir, config, corpus, registry)

    requested_device = torch.device(config.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = requested_device
    model = TrajectoryDiffusion(**config.model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema = ExponentialMovingAverage(model, decay=config.ema_decay)

    start_epoch = 0
    resume_batch = 0
    resume_examples = 0
    global_step = 0
    best_val_loss = float("inf")
    last_validation: Optional[Mapping[str, Any]] = None
    resume_epoch_weighted_loss = 0.0
    resume_epoch_features = 0.0
    resume_epoch_batches = 0
    amp_overflow_retries_total = 0
    resume_epoch_amp_overflow_events = []
    last_step_loss: Optional[float] = None
    last_grad_norm: Optional[float] = None
    best_history = []
    best_manifest_path = output_dir / "best_manifest.json"
    if best_manifest_path.is_file():
        best_manifest = json.loads(best_manifest_path.read_text(encoding="utf-8"))
        best_history = list(best_manifest.get("history", []))

    if resume is not None:
        # Load on CPU so torch_cpu / torch_cuda RNG ByteTensors retain the
        # device type required by torch.set_rng_state(_all).  Model,
        # optimizer, EMA and scaler loaders move their own state afterward.
        checkpoint = torch.load(str(Path(resume).resolve()), map_location="cpu")
        _check_resume_compatibility(checkpoint, config, corpus, registry)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["amp_scaler"])
        ema.load_state_dict(checkpoint["ema"], device)
        progress = checkpoint["progress"]
        start_epoch = int(progress["epoch_index"])
        resume_batch = int(progress["next_batch_in_epoch"])
        resume_examples = int(progress["examples_seen_in_epoch"])
        global_step = int(progress["global_step"])
        best_val_loss = float(progress["best_val_loss"])
        last_validation = progress.get("last_validation")
        resume_epoch_weighted_loss = float(progress.get("epoch_weighted_loss", 0.0))
        resume_epoch_features = float(progress.get("epoch_feature_count", 0.0))
        resume_epoch_batches = int(progress.get("epoch_new_batches", 0))
        amp_overflow_retries_total = int(
            progress.get("amp_overflow_retries_total", 0)
        )
        resume_epoch_amp_overflow_events = [
            dict(value) for value in progress.get("epoch_amp_overflow_events", [])
        ]
        if (
            amp_overflow_retries_total < 0
            or len(resume_epoch_amp_overflow_events) > amp_overflow_retries_total
        ):
            raise ValueError("invalid AMP overflow retry counters in checkpoint")
        if progress.get("last_step_loss") is not None:
            last_step_loss = float(progress["last_step_loss"])
        if progress.get("last_grad_norm") is not None:
            last_grad_norm = float(progress["last_grad_norm"])
        if not (
            last_step_loss is not None and math.isfinite(last_step_loss)
            and last_step_loss > 0.0
            and last_grad_norm is not None and math.isfinite(last_grad_norm)
            and last_grad_norm >= 0.0
        ):
            raise ValueError("resume checkpoint lacks finite last-step loss/gradient progress")
        _restore_rng_state(checkpoint["rng_state"])

    milestones = set(validation_epochs(config.epochs))
    metrics_path = output_dir / "metrics.jsonl"
    last_path = output_dir / "last.pt"
    last_state_path = output_dir / "last_state.json"
    run_manifest = {
        "protocol_version": TRAINING_PROTOCOL_VERSION,
        "action": config.action,
        "config": asdict(config),
        "source": source_identity,
        "diffusion_schedule": config.diffusion_schedule_audit,
        "numeric_recovery_policy": config.numeric_recovery_policy,
        "reference_registry": str(registry_path),
        "counts": {
            "train": len(train_dataset), "val": len(val_dataset), "test_reserved": len(test_dataset)
        },
        "validation_completed_epochs": sorted(milestones),
        "validation_fractions": [0.2, 0.4, 0.6, 0.8, 1.0],
        "full_corpus_no_sample_cap": True,
        "drop_last": False,
        "truncation": False,
        "amp_effective": amp_enabled,
        "runtime_determinism": runtime_determinism,
        "started_unix_time": time.time(),
        "resume_from": None if resume is None else str(Path(resume).resolve()),
        "resume_checkpoint_sha256": (
            None if resume is None else _checkpoint_sha256(Path(resume).resolve())
        ),
        "status": "running",
    }
    atomic_json_dump(output_dir / "run_manifest.json", run_manifest)

    progress_writer = TrainingProgressWriter(
        output_dir / "training_progress.json", config, source_identity,
    )
    progress_writer.publish(
        phase="init",
        epoch_index=start_epoch,
        next_batch_in_epoch=resume_batch,
        global_step=global_step,
        examples_seen_in_epoch=resume_examples,
        force=True,
    )
    # A resume can legitimately need no further optimizer step (for example,
    # power loss after final last.pt but before run_manifest=complete).  Seed
    # the new run-instance writer from checkpoint-bound scalars *after* its
    # init receipt so the eventual complete receipt remains fully auditable.
    if resume is not None:
        progress_writer.last_loss = last_step_loss
        progress_writer.grad_norm = last_grad_norm
        progress_writer.amp_overflow_retries_total = amp_overflow_retries_total
    if resume is not None and last_path.is_file():
        _write_last_state(
            last_state_path, last_path, checkpoint, config, source_identity,
            progress_writer.run_instance_id,
        )

    model.train()
    for epoch_index in range(start_epoch, config.epochs):
        loader = make_epoch_loader(
            train_dataset,
            batch_size=config.batch_size,
            epoch=epoch_index,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            shuffle=True,
        )
        skip_batches = resume_batch if epoch_index == start_epoch else 0
        expected_seen_before_resume = resume_examples if epoch_index == start_epoch else 0
        skipped_examples = 0
        examples_seen = expected_seen_before_resume
        epoch_weighted_loss = resume_epoch_weighted_loss if epoch_index == start_epoch else 0.0
        epoch_features = resume_epoch_features if epoch_index == start_epoch else 0.0
        epoch_batches = resume_epoch_batches if epoch_index == start_epoch else 0
        epoch_amp_overflow_events = (
            list(resume_epoch_amp_overflow_events)
            if epoch_index == start_epoch else []
        )
        epoch_ids = set()
        for batch_index, batch in enumerate(loader):
            if batch_index < skip_batches:
                skipped_examples += int(batch.features.shape[0])
                continue
            if batch_index == skip_batches and skipped_examples != expected_seen_before_resume:
                raise ValueError(
                    "resume examples mismatch: deterministic loader gives %d, checkpoint says %d"
                    % (skipped_examples, expected_seen_before_resume)
                )
            for sample_id in batch.target_sample_ids:
                if sample_id in epoch_ids:
                    raise AssertionError("duplicate target within epoch remainder: %s" % sample_id)
                epoch_ids.add(sample_id)
            batch = batch.to(device)
            pre_attempt_rng_state = _rng_state()
            for retry_index in range(config.amp_overflow_max_retries + 1):
                if retry_index > 0:
                    _restore_rng_state(pre_attempt_rng_state)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    result = model.training_loss(batch)
                    loss, count = _validated_loss_result(result, "training")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip_norm
                )
                if torch.isfinite(torch.as_tensor(grad_norm)):
                    break
                if not amp_enabled:
                    raise FloatingPointError("non-finite gradient norm without AMP")
                if retry_index >= config.amp_overflow_max_retries:
                    raise FloatingPointError(
                        "non-finite gradient norm after %d same-batch AMP retries"
                        % config.amp_overflow_max_retries
                    )
                scale_before, scale_after = _backoff_amp_scaler_without_optimizer_step(
                    scaler
                )
                amp_overflow_retries_total += 1
                progress_writer.amp_overflow_retries_total = (
                    amp_overflow_retries_total
                )
                epoch_amp_overflow_events.append({
                    "schema_version": NUMERIC_RECOVERY_SCHEMA,
                    "epoch_index": int(epoch_index),
                    "batch_index": int(batch_index),
                    "next_global_step": int(global_step + 1),
                    "retry_index": int(retry_index + 1),
                    "scale_before": float(scale_before),
                    "scale_after": float(scale_after),
                    "same_batch": True,
                    "pre_attempt_rng_restored": True,
                })
            else:  # pragma: no cover - the bounded loop always breaks or raises
                raise AssertionError("AMP retry loop exhausted without a decision")
            last_step_loss = float(loss.item())
            last_grad_norm = float(torch.as_tensor(grad_norm).item())
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            global_step += 1
            batch_examples = int(batch.features.shape[0])
            examples_seen += batch_examples
            epoch_weighted_loss += float(loss.item()) * count
            epoch_features += count
            epoch_batches += 1
            progress_writer.publish(
                phase="train",
                epoch_index=epoch_index,
                next_batch_in_epoch=batch_index + 1,
                global_step=global_step,
                examples_seen_in_epoch=examples_seen,
                successful_progress=True,
                last_loss=last_step_loss,
                grad_norm=last_grad_norm,
            )

            if global_step % config.checkpoint_every_steps == 0:
                progress_writer.publish(
                    phase="checkpoint_commit",
                    epoch_index=epoch_index,
                    next_batch_in_epoch=batch_index + 1,
                    global_step=global_step,
                    examples_seen_in_epoch=examples_seen,
                    force=True,
                )
                checkpoint = _make_checkpoint(
                    config, corpus, registry, model, ema, optimizer, scaler,
                    epoch_index=epoch_index,
                    next_batch_in_epoch=batch_index + 1,
                    examples_seen_in_epoch=examples_seen,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    last_validation=last_validation,
                    epoch_weighted_loss=epoch_weighted_loss,
                    epoch_feature_count=epoch_features,
                    epoch_new_batches=epoch_batches,
                    amp_overflow_retries_total=amp_overflow_retries_total,
                    epoch_amp_overflow_events=epoch_amp_overflow_events,
                    last_step_loss=last_step_loss,
                    last_grad_norm=last_grad_norm,
                )
                atomic_torch_save(last_path, checkpoint, overwrite=True)
                _write_last_state(
                    last_state_path, last_path, checkpoint, config, source_identity,
                    progress_writer.run_instance_id,
                )
                progress_writer.publish(
                    phase="train",
                    epoch_index=epoch_index,
                    next_batch_in_epoch=batch_index + 1,
                    global_step=global_step,
                    examples_seen_in_epoch=examples_seen,
                    force=True,
                )

        if examples_seen != len(train_dataset):
            raise AssertionError(
                "epoch did not consume full train split: %d != %d"
                % (examples_seen, len(train_dataset))
            )
        completed_epoch = epoch_index + 1
        aggregate_train_loss = epoch_weighted_loss / max(epoch_features, 1.0)
        if not math.isfinite(aggregate_train_loss) or aggregate_train_loss <= 0.0:
            raise FloatingPointError(
                "non-finite or non-positive aggregate training loss"
            )
        train_record = {
            "type": "train_epoch",
            "completed_epoch": completed_epoch,
            "global_step": global_step,
            "loss": aggregate_train_loss,
            "batches_total_in_epoch": epoch_batches,
            "examples_total_in_epoch": examples_seen,
            "valid_feature_count_total": epoch_features,
            "full_train_split_consumed": True,
            "amp_overflow_retries": int(len(epoch_amp_overflow_events)),
            "amp_overflow_events": [
                dict(value) for value in epoch_amp_overflow_events
            ],
            "unix_time": time.time(),
        }
        validation_record = None
        new_best_entry = None
        desired_best_manifest = None
        if completed_epoch in milestones:
            progress_writer.publish(
                phase="validation",
                epoch_index=epoch_index,
                next_batch_in_epoch=epoch_batches,
                global_step=global_step,
                examples_seen_in_epoch=examples_seen,
                force=True,
            )

            def validation_progress(batch_number: int, total_batches: int) -> None:
                progress_writer.publish(
                    phase="validation",
                    epoch_index=epoch_index,
                    next_batch_in_epoch=epoch_batches,
                    global_step=global_step,
                    examples_seen_in_epoch=examples_seen,
                    successful_progress=True,
                    validation_batch_index=batch_number,
                    validation_batches_total=total_batches,
                )

            validation = evaluate_full_validation(
                model, ema, val_dataset, config.batch_size, config.num_workers,
                device, amp_enabled, config.seed, completed_epoch, config.epochs,
                progress_callback=validation_progress,
            )
            validation["fraction"] = float(completed_epoch) / float(config.epochs)
            validation_record = dict(validation)
            validation_record.update({"type": "validation", "global_step": global_step, "unix_time": time.time()})
            last_validation = validation
            metric = float(validation["val_loss"])
            if metric < best_val_loss:
                best_val_loss = metric
                filename = "best_epoch_%04d_step_%09d_valloss_%.8f.pt" % (
                    completed_epoch, global_step, metric
                )
                best_path = output_dir / filename
                new_best_entry = {
                    "path": str(best_path),
                    "filename": filename,
                    "completed_epoch": completed_epoch,
                    "global_step": global_step,
                    "val_loss": metric,
                    "source_sha256": corpus.sha256,
                    "split_sha256": corpus.splits.sha256,
                    "reference_registry_sha256": registry.sha256,
                }
                best_history.append(new_best_entry)
                desired_best_manifest = {
                    "protocol_version": TRAINING_PROTOCOL_VERSION,
                    "selection_split": "val",
                    "selection_metric": "full_val_masked_epsilon_mse_ema",
                    "lower_is_better": True,
                    "best": best_history[-1],
                    "history": best_history,
                    "test_used_for_selection": False,
                    "source": source_identity,
                    "diffusion_schedule": config.diffusion_schedule_audit,
                    "numeric_recovery_policy": config.numeric_recovery_policy,
                }

        # Atomic last always represents the start of the next epoch.
        checkpoint = _make_checkpoint(
            config, corpus, registry, model, ema, optimizer, scaler,
            epoch_index=completed_epoch,
            next_batch_in_epoch=0,
            examples_seen_in_epoch=0,
            global_step=global_step,
            best_val_loss=best_val_loss,
            last_validation=last_validation,
            epoch_weighted_loss=0.0,
            epoch_feature_count=0.0,
            epoch_new_batches=0,
            amp_overflow_retries_total=amp_overflow_retries_total,
            epoch_amp_overflow_events=[],
            last_step_loss=last_step_loss,
            last_grad_norm=last_grad_norm,
        )
        progress_writer.publish(
            phase="checkpoint_commit",
            epoch_index=completed_epoch,
            next_batch_in_epoch=0,
            global_step=global_step,
            examples_seen_in_epoch=0,
            force=True,
        )
        _commit_epoch_transaction(
            output_dir, config, corpus, registry, checkpoint, train_record,
            validation_record, new_best_entry, desired_best_manifest,
        )
        _write_last_state(
            last_state_path, last_path, checkpoint, config, source_identity,
            progress_writer.run_instance_id,
        )
        progress_writer.publish(
            phase="train" if completed_epoch < config.epochs else "complete",
            epoch_index=completed_epoch,
            next_batch_in_epoch=0,
            global_step=global_step,
            examples_seen_in_epoch=0,
            successful_progress=True,
            force=True,
        )
        if new_best_entry is not None:
            published_best = json.loads(best_manifest_path.read_text(encoding="utf-8"))
            best_history = list(published_best["history"])
        resume_batch = 0
        resume_examples = 0
        resume_epoch_weighted_loss = 0.0
        resume_epoch_features = 0.0
        resume_epoch_batches = 0
        resume_epoch_amp_overflow_events = []

    if not best_history:
        raise AssertionError("no validation milestone produced a best checkpoint")
    run_manifest.update(
        {
            "status": "complete",
            "completed_unix_time": time.time(),
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_checkpoint": best_history[-1]["path"],
            "last_checkpoint": str(last_path),
            "amp_overflow_retries_total": int(amp_overflow_retries_total),
        }
    )
    atomic_json_dump(output_dir / "run_manifest.json", run_manifest)
    progress_writer.publish(
        phase="complete",
        epoch_index=config.epochs,
        next_batch_in_epoch=0,
        global_step=global_step,
        examples_seen_in_epoch=0,
        successful_progress=True,
        force=True,
    )
    return run_manifest
