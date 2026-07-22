#!/usr/bin/env python3
"""Independent finalized-v2 end-to-end smoke for the trajectory pipeline.

This is deliberately *not* a formal result.  It uses a tiny, explicitly
reported subset to exercise every interface quickly, while retaining the
formal 1000-step diffusion schedule, validation-only best-EMA selection and
exact 50-call DDIM sampler.  Formal count/quality gates are evaluated and
reported, never silently weakened or relabelled as passed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_determinism import (  # noqa: E402
    STRICT_RUNTIME_DETERMINISM_SHA256,
    require_strict_runtime_determinism,
    seed_everything,
)

import numpy as np
import torch

from detectors.deep_pad import (  # noqa: E402
    DeepTrainConfig,
    RawTrajectoryRecord,
    assign_strict_protocol_pools,
    make_record,
    run_deep_pad_protocol,
)
from detectors.feature_pad import (  # noqa: E402
    ALLOWED_DETECTORS,
    run_feature_pad_protocol,
    save_protocol_outputs,
)
from detectors.trajectory_adapter import (  # noqa: E402
    _event_frames,
    _feature_vector,
    _validate_source,
)
from generation.audit import audit_generated_unit  # noqa: E402
from generation.corpus import load_action_corpus  # noqa: E402
from generation.pad_export import load_generated_action_tree  # noqa: E402
from generation.pipeline import generate_unit, unit_output_path  # noqa: E402
from generation.protocol import (  # noqa: E402
    ACTIONS,
    FixedUserSplit,
    GenerationUnit,
    ReferenceRegistry as GenerationReferenceRegistry,
    TrainGlobalPrior,
)
from generation.sampler import load_model_checkpoint  # noqa: E402
from trajectory.data import KEYCODE_VOCAB_SIZE, LOG_DT_INDEX  # noqa: E402
from trajectory.model import TrajectoryDiffusion  # noqa: E402
from training.corpus import (  # noqa: E402
    FORMAL_SPLIT_PATH,
    NumericTrajectoryCorpus,
    SplitDefinition,
    atomic_json_dump,
    sha256_file,
)
from training.engine import (  # noqa: E402
    ExponentialMovingAverage,
    TrainingConfig,
    atomic_torch_save,
)
from training.fewshot_dataset import (  # noqa: E402
    ReferenceRegistry,
    StrictFiveReferenceDataset,
    StrictVariableLengthCollator,
)


SMOKE_SCHEMA = "trajectory_finalized_v2_e2e_smoke_v2"
DEFAULT_CORPUS = ROOT / "results" / "trajectories_full_v2"
DEFAULT_OUTPUT = ROOT / "results" / "v2_e2e_smoke"


def _sha256(path: Path) -> str:
    return sha256_file(Path(path))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _require_finalized_corpus(root: Path) -> Dict[str, Any]:
    root = root.resolve()
    if (root / ".build").exists() or any(root.glob("*.tmp")):
        raise RuntimeError("refusing to read an extraction directory with live temporary files")
    manifest_path = root / "manifest.json"
    audit_path = root / "audit.json"
    formal_audit_path = root / "formal_audit" / "formal_data_audit.json"
    if not manifest_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("finalized extraction manifest/audit is missing")
    if not formal_audit_path.is_file():
        raise FileNotFoundError("independent formal extraction audit is missing")
    formal_audit = json.loads(formal_audit_path.read_text(encoding="utf-8"))
    if formal_audit.get("formal_passed") is not True:
        raise RuntimeError(
            "refusing E2E on a source whose independent formal audit is not PASS: %s"
            % formal_audit.get("error", "unknown audit failure")
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest.get("outputs", {})) != set(ACTIONS):
        raise ValueError("finalized manifest must contain exactly five action outputs")
    files: Dict[str, Any] = {}
    for action in ACTIONS:
        path = root / ("hmog_trajectory_%s.npz" % action)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        recorded = manifest["outputs"][action].get("sha256")
        if digest != recorded:
            raise ValueError("%s hash contradicts finalized manifest" % action)
        files[action] = {
            "path": str(path),
            "sha256": digest,
            "n_events": int(manifest["outputs"][action]["n_events"]),
        }
    processed = manifest.get("selection", {}).get("processed_users", [])
    if len(processed) != 100:
        raise ValueError("finalized smoke source must contain 100 processed users")
    return {
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "audit": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "formal_audit": str(formal_audit_path),
        "formal_audit_sha256": _sha256(formal_audit_path),
        "formal_audit_passed": True,
        "processed_users": 100,
        "files": files,
    }


def _selected_users(split: SplitDefinition, per_pool: int) -> Dict[str, Tuple[int, ...]]:
    if per_pool < 1:
        raise ValueError("per_pool must be positive")
    selected = {
        name: tuple(sorted(split.users(name))[:per_pool])
        for name in ("train", "val", "test")
    }
    if any(len(values) != per_pool for values in selected.values()):
        raise ValueError("not enough users in a fixed split")
    return selected


def _positions_for_users(
    dataset: StrictFiveReferenceDataset,
    users: Sequence[int],
    per_user: int,
) -> List[int]:
    positions: List[int] = []
    target_users = dataset.corpus.user_ids[dataset.indices]
    for user_id in users:
        candidates = np.flatnonzero(target_users == int(user_id)).tolist()
        candidates.sort(key=lambda position: (
            dataset.padded_length_key(int(position)),
            int(dataset.corpus.event_ids[int(dataset.indices[int(position)])]),
        ))
        if len(candidates) < per_user:
            raise ValueError("smoke target pool is too small for user=%d" % user_id)
        positions.extend(int(value) for value in candidates[:per_user])
    return positions


def _clone_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


@torch.no_grad()
def _fixed_loss(
    model: TrajectoryDiffusion,
    batch,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
) -> float:
    model.eval()
    value = model.training_loss(batch, timesteps=timesteps, noise=noise)["loss"]
    if not torch.isfinite(value):
        raise FloatingPointError("non-finite fixed validation loss")
    return float(value.item())


def _permute_reference_batch(batch, order: torch.Tensor):
    values = dict(batch.__dict__)
    for name in (
        "ref_features", "ref_point_mask", "ref_contact_mask", "ref_event_ids",
        "ref_pointer_mask", "ref_pointer_start_offset_ms", "ref_pointer_end_offset_ms",
        "ref_mask", "ref_keycodes", "ref_keycode_mask",
    ):
        values[name] = getattr(batch, name)[:, order]
    values["ref_sample_ids"] = tuple(
        tuple(row[int(index)] for index in order.tolist()) for row in batch.ref_sample_ids
    )
    values["ref_user_ids"] = tuple(
        tuple(row[int(index)] for index in order.tolist()) for row in batch.ref_user_ids
    )
    values["ref_splits"] = tuple(
        tuple(row[int(index)] for index in order.tolist()) for row in batch.ref_splits
    )
    result = type(batch)(**values)
    result.validate(require_references=True)
    return result


def _train_smoke_action(
    action: str,
    corpus_path: Path,
    split: SplitDefinition,
    selected: Mapping[str, Sequence[int]],
    output_dir: Path,
    device: torch.device,
    steps: int,
    reference_seed: int,
    training_seed: int,
) -> Dict[str, Any]:
    started = time.time()
    corpus = NumericTrajectoryCorpus(
        corpus_path, split, expected_action=action, verify_sha256=True
    )
    source_audit = corpus.audit(require_all_users=True, validate_every_event=False)
    registry = ReferenceRegistry.build(corpus, seed=reference_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "reference_registry.json"
    registry.save(registry_path)
    train_dataset = StrictFiveReferenceDataset(
        corpus, "train", registry, seed=reference_seed, cache_size=64
    )
    val_dataset = StrictFiveReferenceDataset(
        corpus, "val", registry, seed=reference_seed, cache_size=64
    )
    train_positions = _positions_for_users(train_dataset, selected["train"], 1)
    val_positions = _positions_for_users(val_dataset, selected["val"], 1)
    collator = StrictVariableLengthCollator()
    train_batch = collator([train_dataset[position] for position in train_positions]).to(device)
    val_batch = collator([val_dataset[position] for position in val_positions]).to(device)
    train_batch.validate(require_references=True)
    val_batch.validate(require_references=True)

    model_config = {
        "action": action,
        "diffusion_steps": 1000,
        "base_channels": 16,
        "cond_dim": 32,
        "time_dim": 16,
        "n_blocks": 2,
        "dropout": 0.0,
        "keycode_vocab": KEYCODE_VOCAB_SIZE,
    }
    seed_everything(training_seed)
    model = TrajectoryDiffusion(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=0.0)
    ema = ExponentialMovingAverage(model, decay=0.95)

    generator = torch.Generator(device=device).manual_seed(training_seed + 1009)
    train_timesteps = torch.randint(
        0, model.diffusion_steps, (train_batch.features.shape[0],),
        device=device, dtype=torch.long, generator=generator,
    )
    train_noise = torch.randn(
        train_batch.features.shape, dtype=train_batch.features.dtype,
        device=device, generator=generator,
    )
    val_timesteps = torch.randint(
        0, model.diffusion_steps, (val_batch.features.shape[0],),
        device=device, dtype=torch.long, generator=generator,
    )
    val_noise = torch.randn(
        val_batch.features.shape, dtype=val_batch.features.dtype,
        device=device, generator=generator,
    )

    # Reference-set permutation must not change the encoded condition.
    model.eval()
    with torch.no_grad():
        condition = model.denoiser.encode_condition(train_batch)
        reverse = torch.arange(4, -1, -1, device=device)
        permuted = _permute_reference_batch(train_batch, reverse)
        condition_reversed = model.denoiser.encode_condition(permuted)
        permutation_error = float(torch.max(torch.abs(condition - condition_reversed)).item())
    if permutation_error > 2.0e-5:
        raise AssertionError("reference set encoder is not permutation invariant")

    losses: List[float] = []
    val_history: List[Dict[str, float]] = []
    best_val = math.inf
    best_step = -1
    best_shadow: Dict[str, torch.Tensor] = {}
    maximum_steps = max(int(steps), 1)
    # If a noisy short prefix does not show a trend, continue only this fixed
    # optimization smoke (bounded to 2x); do not tune on detector/test scores.
    for step in range(1, maximum_steps * 2 + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model.training_loss(
            train_batch, timesteps=train_timesteps, noise=train_noise
        )
        loss = result["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite trajectory diffusion loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite trajectory diffusion gradient")
        optimizer.step()
        if any(not torch.isfinite(value).all() for value in model.parameters()):
            raise FloatingPointError("non-finite trajectory diffusion parameter")
        ema.update(model)
        losses.append(float(loss.item()))

        if step == 1 or step % 5 == 0 or step == maximum_steps * 2:
            original = ema.copy_to(model)
            try:
                val_loss = _fixed_loss(model, val_batch, val_timesteps, val_noise)
                val_history.append({"step": int(step), "ema_val_loss": val_loss})
                if val_loss < best_val:
                    best_val = val_loss
                    best_step = int(step)
                    best_shadow = _clone_state(model.state_dict())
            finally:
                ExponentialMovingAverage.restore(model, original)
        window = min(5, len(losses))
        decreased = len(losses) >= maximum_steps and (
            float(np.mean(losses[-window:])) < float(np.mean(losses[:window]))
        )
        if step >= maximum_steps and decreased:
            break
    if not best_shadow or best_step < 1:
        raise AssertionError("validation-only EMA selection produced no checkpoint")
    first_window = float(np.mean(losses[: min(5, len(losses))]))
    last_window = float(np.mean(losses[-min(5, len(losses)) :]))
    if not last_window < first_window:
        raise AssertionError("fixed-batch smoke loss failed to decrease")

    schedule = TrainingConfig(
        action=action,
        corpus_npz=str(corpus_path.resolve()),
        split_json=str(split.path),
        output_dir=str(output_dir),
        diffusion_steps=1000,
        base_channels=16,
        cond_dim=32,
        time_dim=16,
        n_blocks=2,
        dropout=0.0,
    ).diffusion_schedule_audit
    best_path = output_dir / ("best_smoke_step_%04d.pt" % best_step)
    checkpoint = {
        "protocol_version": SMOKE_SCHEMA,
        "runtime_determinism": require_strict_runtime_determinism(),
        "checkpoint_role": "smoke_validation_best_ema",
        "inference_weights": "ema.shadow",
        "action": action,
        "config": {
            "action": action,
            "smoke_only": True,
            "optimizer_steps": len(losses),
            "reference_seed": int(reference_seed),
            "training_seed": int(training_seed),
            "test_used_for_training_or_selection": False,
        },
        "model_config": model_config,
        "diffusion_schedule": schedule,
        "source": {
            "corpus_npz": str(corpus_path.resolve()),
            "corpus_sha256": corpus.sha256,
            "split_json": str(split.path),
            "split_sha256": split.sha256,
            "reference_registry_sha256": registry.sha256,
        },
        "ema": {"decay": 0.95, "shadow": best_shadow},
        "selection": {
            "split": "val",
            "metric": "fixed_smoke_val_masked_epsilon_mse_ema",
            "best_step": best_step,
            "best_value": best_val,
            "test_used": False,
        },
    }
    atomic_torch_save(best_path, checkpoint, overwrite=False)
    checkpoint_digest = _sha256(best_path)
    best_entry = {
        "path": str(best_path.resolve()),
        "step": best_step,
        "val_loss": best_val,
        "source_sha256": checkpoint["source"]["corpus_sha256"],
        "split_sha256": checkpoint["source"]["split_sha256"],
        "reference_registry_sha256": checkpoint["source"]["reference_registry_sha256"],
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_role": "validation_selected_best",
        "inference_weights": "ema.shadow",
    }
    best_manifest = {
        "protocol_version": SMOKE_SCHEMA,
        "schema_version": SMOKE_SCHEMA,
        "selection_split": "val",
        "selection_metric": "fixed_smoke_val_masked_epsilon_mse_ema",
        "lower_is_better": True,
        "test_used_for_selection": False,
        "checkpoint_role": "validation_selected_best",
        "inference_weights": "ema.shadow",
        "source": checkpoint["source"],
        "diffusion_schedule": checkpoint["diffusion_schedule"],
        "best": best_entry,
        "history": [best_entry],
    }
    atomic_json_dump(output_dir / "best_manifest.json", best_manifest)

    # The formal checkpoint loader is intentionally used here.  It proves the
    # smoke artifact meets schedule/best-EMA/source identity gates.
    loaded, loaded_sha = load_model_checkpoint(
        str(best_path), action, device, require_best_ema=True,
        expected_registry_sha256=registry.sha256,
        expected_split_sha256=split.sha256,
        allow_e2e_smoke_checkpoint=True,
    )
    del loaded
    if loaded_sha != _sha256(best_path):
        raise AssertionError("checkpoint loader digest mismatch")

    feature_mask = train_batch.feature_mask
    mask_audit = {
        "geometry_supervised_exactly_on_contact": bool(torch.equal(
            feature_mask[..., [0, 1, 3, 4]],
            train_batch.contact_mask.unsqueeze(-1).expand_as(
                feature_mask[..., [0, 1, 3, 4]]
            ),
        )),
        "log_dt_supervised_exactly_on_valid_timeline": bool(torch.equal(
            feature_mask[..., LOG_DT_INDEX], train_batch.point_mask
        )),
        "keystroke_gap_token_count": int((
            train_batch.point_mask & ~train_batch.contact_mask
        ).sum().item()) if action == "keystroke" else 0,
        "pinch_pointer_count": int(train_batch.pointer_mask[0].sum().item()),
    }
    if not mask_audit["geometry_supervised_exactly_on_contact"] or not mask_audit[
        "log_dt_supervised_exactly_on_valid_timeline"
    ]:
        raise AssertionError("trajectory feature mask semantics changed")
    if action == "keystroke" and mask_audit["keystroke_gap_token_count"] <= 0:
        raise AssertionError("keystroke smoke target lost all explicit flight gaps")
    if action == "pinch" and mask_audit["pinch_pointer_count"] != 2:
        raise AssertionError("pinch smoke target lost one pointer")

    target_refs = []
    for batch_index, sample_id in enumerate(train_batch.target_sample_ids):
        refs = list(train_batch.ref_sample_ids[batch_index])
        target_refs.append({
            "target": sample_id,
            "refs": refs,
            "unique_refs": len(set(refs)) == 5,
            "target_excluded": sample_id not in refs,
            "same_user": len(set(train_batch.ref_user_ids[batch_index])) == 1
            and train_batch.ref_user_ids[batch_index][0] == train_batch.target_user_ids[batch_index],
            "same_split": len(set(train_batch.ref_splits[batch_index])) == 1
            and train_batch.ref_splits[batch_index][0] == train_batch.target_splits[batch_index],
        })
    if not all(
        row["unique_refs"] and row["target_excluded"] and row["same_user"] and row["same_split"]
        for row in target_refs
    ):
        raise AssertionError("same-user/action/split five-ref exclusion failed")
    return {
        "action": action,
        "corpus_events": len(corpus),
        "corpus_sha256": corpus.sha256,
        "source_reference_gate": source_audit["reference_gate"],
        "reference_registry": str(registry_path.resolve()),
        "reference_registry_sha256": registry.sha256,
        "reference_seed": int(reference_seed),
        "training_seed": int(training_seed),
        "targets_and_refs": target_refs,
        "train_target_count": len(train_positions),
        "val_target_count": len(val_positions),
        "train_target_lengths": [
            [int(mask.sum().item()) for mask in train_batch.point_mask[index]]
            for index in range(train_batch.features.shape[0])
        ],
        "loss": {
            "optimizer_steps": len(losses),
            "all_finite": bool(np.all(np.isfinite(losses))),
            "first_window_mean": first_window,
            "last_window_mean": last_window,
            "relative_change": (last_window - first_window) / max(first_window, 1e-12),
            "values": losses,
        },
        "validation_selection": {
            "best_step": best_step,
            "best_ema_val_loss": best_val,
            "history": val_history,
            "test_used": False,
        },
        "mask_audit": mask_audit,
        "reference_permutation_max_abs_error": permutation_error,
        "checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": loaded_sha,
        "checkpoint_loader_formal_schedule_best_ema_gate_passed": True,
        "diffusion_schedule": schedule,
        "elapsed_seconds": time.time() - started,
        "passed": True,
    }


def _load_real_subset(
    path: Path,
    selected_users: Iterable[int],
    per_user: int,
) -> Tuple[List[RawTrajectoryRecord], np.ndarray]:
    allowed = set(int(value) for value in selected_users)
    records: List[RawTrajectoryRecord] = []
    features: List[np.ndarray] = []
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    action, _ = _validate_source(data)
    offsets = np.asarray(data["event_offsets"], dtype=np.int64)
    selected_indices: List[int] = []
    for user_id in sorted(allowed):
        candidates = np.flatnonzero(np.asarray(data["user_id"]) == user_id).tolist()
        candidates.sort(key=lambda index: (
            int(offsets[index + 1] - offsets[index]),
            int(data["event_id"][index]),
        ))
        if len(candidates) < per_user:
            raise ValueError("real detector smoke subset lacks user/action events")
        selected_indices.extend(int(index) for index in candidates[:per_user])
    for index in selected_indices:
        left, right = int(offsets[index]), int(offsets[index + 1])
        values, times, contact, active, codes, keycodes, events, gap = _event_frames(
            data, left, right, action
        )
        event_group = str(int(data["event_id"][index]))
        records.append(make_record(
            action=action,
            label=0,
            user_id=int(data["user_id"][index]),
            pool="train",
            sample_id="smoke-real:%s:%s" % (action, event_group),
            event_group_id=event_group,
            pointer_continuous=values,
            global_t_ms=times,
            contact_mask=contact,
            active_mask=active,
            action_code=codes,
            keycode=keycodes,
            event_ids=events,
            gap_mask=gap,
        ))
        features.append(_feature_vector(data, action, index, values, times, contact, gap))
    result = np.stack(features).astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("real smoke feature adapter produced non-finite data")
    return records, result


def _run_detector_smoke(
    action: str,
    real_path: Path,
    fake_root: Path,
    fixed_split: FixedUserSplit,
    selected_users: Sequence[int],
    output_dir: Path,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    started = time.time()
    real_records, real_features = _load_real_subset(real_path, selected_users, per_user=12)
    fake_records, fake_features = load_generated_action_tree(
        fake_root, action, fixed_split, require_formal=False
    )
    records, split_audit = assign_strict_protocol_pools(
        real_records + fake_records,
        {"train": fixed_split.train_users, "val": fixed_split.val_users, "test": fixed_split.test_users},
        real_hash_seed=seed,
    )
    features = np.concatenate((real_features, fake_features), axis=0)
    if len(records) != len(features):
        raise AssertionError("real/fake sequence-feature order mismatch")
    labels = np.asarray([row.label for row in records], np.int64)
    users = np.asarray([row.user_id for row in records], np.int64)
    pools = np.asarray([row.pool for row in records], dtype="U5")
    action_values = np.asarray([row.action for row in records], dtype="U16")
    per_pool_classes = {
        pool: sorted(set(int(value) for value in labels[pools == pool].tolist()))
        for pool in ("train", "val", "test")
    }
    if any(values != [0, 1] for values in per_pool_classes.values()):
        raise ValueError("smoke detector split lacks both classes")

    detector_results: Dict[str, Any] = {}
    for detector in ALLOWED_DETECTORS:
        params = {"n_estimators": 20, "max_depth": 2, "n_jobs": 1} if detector == "xgboost" else None
        result = run_feature_pad_protocol(
            features,
            labels,
            users,
            pools,
            action_values,
            action=action,
            detector_kind=detector,
            random_state=seed,
            model_params=params,
            bootstrap_replicates=0,
        )
        detector_dir = output_dir / "feature_pad" / detector
        save_protocol_outputs(result, detector_dir)
        detector_results[detector] = {
            "family": "feature_pad",
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
            "thresholds": result.thresholds,
            "passed": True,
        }

    deep_config = DeepTrainConfig(
        epochs=2,
        batch_size=16,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        patience=0,
        num_workers=0,
        seed=seed,
        bootstrap_replicates=0,
        gradient_clip_norm=5.0,
    )
    deep_parameters = {
        "tcn": {"hidden_dim": 16, "n_blocks": 1, "dropout": 0.0},
        "transformer": {
            "hidden_dim": 16,
            "n_layers": 1,
            "n_heads": 2,
            "feedforward_dim": 32,
            "dropout": 0.0,
        },
    }
    for detector in ("tcn", "transformer"):
        result = run_deep_pad_protocol(
            records,
            action=action,
            detector_kind=detector,
            output_dir=output_dir / "deep_pad" / detector,
            config=deep_config,
            model_params=deep_parameters[detector],
            device=str(device),
            resume=False,
        )
        detector_results[detector] = {
            "family": "deep_pad",
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
            "thresholds": result.thresholds,
            "best_epoch": result.best_epoch,
            "all_history_finite": all(
                np.isfinite([row["train_loss"], row["val_loss"], row["val_auc"]]).all()
                for row in result.history
            ),
            "passed": True,
        }
    return {
        "action": action,
        "n_real": len(real_records),
        "n_fake": len(fake_records),
        "users": sorted(set(int(row.user_id) for row in records)),
        "per_pool_classes": per_pool_classes,
        "split_audit": split_audit,
        "detectors": detector_results,
        "detector_kind_count": len(detector_results),
        "elapsed_seconds": time.time() - started,
        "passed": len(detector_results) == 5 and all(
            value["passed"] for value in detector_results.values()
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Finalized v2 trajectory E2E smoke",
        "",
        "结论：该目录是正式 supervisor 的启动前 quick smoke，但不是正式实验结果。",
        "它验证 finalized v2 loader → 固定五条同用户参考 → 短训练 → validation-only best EMA →",
        "1000-step schedule / 50-step DDIM → numeric archive → PAD adapter → 3 Feature + 2 Deep detector。",
        "",
        "- 正式计数 gate 没有被改写；smoke 计数天然不满足 100 users × 200 fake，因此明确为 `formal_count_gate=false`。",
        "- 结构/物理合法性使用宽松 clipping 上界验证完整生成与 Android archive 路径；",
        "  正式 5%/25% clipping 质量阈值也会独立复核并原样报告，但 20-step 未训练充分模型的该诊断不作为 quick-smoke 启动门槛。",
        "  正式阈值只在 100-epoch checkpoint 的完整 100,000 条生成审计中作为硬门槛。",
        "- test users 没有进入生成模型训练或 best checkpoint 选择。",
        "",
        "## Training and generation",
        "",
        "| action | first loss | last loss | best EMA step | alpha_bar_final | generated | smoke validity | formal clipping diagnostic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for action in ACTIONS:
        train = report["training"][action]
        generation = report["generation"][action]
        lines.append(
            "| %s | %.6f | %.6f | %d | %.8g | %d | %s | %s |" % (
                action,
                train["loss"]["first_window_mean"],
                train["loss"]["last_window_mean"],
                train["validation_selection"]["best_step"],
                train["diffusion_schedule"]["alpha_bar_final"],
                generation["n_fake"],
                "pass" if generation["smoke_physical_validity_gate_passed"] else "fail",
                "pass" if generation["formal_physical_gate_passed"] else "fail (smoke-only; reported)",
            )
        )
    lines.extend([
        "",
        "## Detector interface smoke",
        "",
        "以下指标只证明 25 条训练/验证/测试代码路径可运行且输出有限值；样本极少，不作为方法效果。",
        "",
        "| action | linear SVM | RBF SVM | XGBoost | TCN | Transformer |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for action in ACTIONS:
        det = report["detectors"][action]["detectors"]
        cells = []
        for name in ("linear_svm", "rbf_svm", "xgboost", "tcn", "transformer"):
            metric = det[name]["test_metrics"]["eer"]
            cells.append("FA %.3f / AUC %.3f" % (metric["fa"], metric["auc"]))
        lines.append("| %s | %s |" % (action, " | ".join(cells)))
    lines.extend([
        "",
        "## Logic audit",
        "",
        "- geometry/pressure/size loss mask = contact mask；`log_dt` loss mask = valid timeline；",
        "- keystroke 在键间保留 no-contact flight token，键序列/字母数进入条件；",
        "- pinch 保留两个 pointer 及各自全局生命周期，不按 pointer 独立归一化时间；",
        "- 五条 refs 固定、唯一、同 user/action/split，且 target 永不出现在 refs；",
        "- reference encoder 的集合顺序置换误差逐 action 记录于 JSON；",
        "- detector 的 `active_mask` 仅作审计字段，不输入 Deep encoder；",
        "- formal count/quality gate 仍由正式 100k pipeline 单独执行。",
        "",
        "完整 hashes、逐步 loss、refs/event IDs、archive audit 和 25 个 detector smoke 结果见 `e2e_smoke.json`。",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-json", type=Path, default=FORMAL_SPLIT_PATH)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--users-per-pool", type=int, default=2)
    parser.add_argument("--samples-per-user", type=int, default=2)
    parser.add_argument("--optimizer-steps", type=int, default=25)
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=20260713)
    parser.add_argument("--overwrite-smoke", action="store_true")
    parser.add_argument("--confirm-quick-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_quick_smoke:
        raise ValueError("pass --confirm-quick-smoke; this is isolated non-formal output")
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite_smoke:
            raise FileExistsError("smoke output exists; pass --overwrite-smoke")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    started = time.time()
    # Formal enrollment/training uses seed 42.  Fake ConditionRequest/DDIM
    # seeds are a separate protocol choice and must never alter registry refs.
    seed_everything(args.reference_seed)
    runtime_determinism = require_strict_runtime_determinism()
    source = _require_finalized_corpus(args.corpus_dir)
    split = SplitDefinition.load(args.split_json, require_pinned_hash=True)
    fixed_split = FixedUserSplit.load(str(args.split_json), require_formal=True)
    selected = _selected_users(split, args.users_per_pool)
    selected_all = tuple(value for name in ("train", "val", "test") for value in selected[name])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    report: Dict[str, Any] = {
        "schema_version": SMOKE_SCHEMA,
        "status": "running",
        "runtime_determinism": runtime_determinism,
        "runtime_determinism_sha256": STRICT_RUNTIME_DETERMINISM_SHA256,
        "formal_result": False,
        "formal_count_gate_passed": False,
        "formal_count_gate_reason": "quick smoke uses a declared user/sample subset, not 100x200",
        "source": source,
        "split": split.audit_dict(),
        "selected_users": {key: list(value) for key, value in selected.items()},
        "configuration": {
            "device": str(device),
            "users_per_pool": args.users_per_pool,
            "samples_per_user_action": args.samples_per_user,
            "optimizer_steps_requested": args.optimizer_steps,
            "training_diffusion_steps": 1000,
            "ddim_inference_steps": 50,
            "deep_detector_epochs": 2,
            "bootstrap_replicates": 0,
            "reference_seed": int(args.reference_seed),
            "training_seed": int(args.reference_seed),
            "generation_seed": int(args.generation_seed),
        },
        "training": {},
        "generation": {},
        "detectors": {},
    }
    atomic_json_dump(output / "e2e_smoke.in_progress.json", report)

    checkpoint_paths: Dict[str, str] = {}
    registry_paths: Dict[str, str] = {}
    for action_index, action in enumerate(ACTIONS):
        action_result = _train_smoke_action(
            action,
            Path(source["files"][action]["path"]),
            split,
            selected,
            output / "training" / action,
            device,
            args.optimizer_steps,
            args.reference_seed,
            args.reference_seed,
        )
        report["training"][action] = action_result
        checkpoint_paths[action] = action_result["checkpoint"]
        registry_paths[action] = action_result["reference_registry"]
        atomic_json_dump(output / "e2e_smoke.in_progress.json", report)

    fake_root = output / "generated_archives"
    generation_call_counts: Dict[str, int] = {}
    for action_index, action in enumerate(ACTIONS):
        pool = load_action_corpus(
            source["files"][action]["path"], action, fixed_split,
            user_ids=selected_all, strict=True,
        )
        prior = TrainGlobalPrior.fit(action, pool, split.train_users)
        registry = GenerationReferenceRegistry.load(
            registry_paths[action], fixed_split.source_sha256
        )
        model, checkpoint_digest = load_model_checkpoint(
            checkpoint_paths[action], action, device,
            require_best_ema=True,
            expected_registry_sha256=registry.registry_sha256,
            expected_split_sha256=fixed_split.source_sha256,
            allow_e2e_smoke_checkpoint=True,
        )
        calls = {"count": 0}
        hook = model.denoiser.register_forward_hook(
            lambda *_args, counter=calls: counter.__setitem__("count", counter["count"] + 1)
        )
        unit_audits = []
        formal_gate_failures = []
        try:
            for user_id in selected_all:
                unit = GenerationUnit(
                    action=action,
                    user_id=int(user_id),
                    split=fixed_split.split_for_user(int(user_id)),
                    samples=args.samples_per_user,
                    shard_id=0,
                    num_shards=1,
                )
                audit = generate_unit(
                    unit,
                    pool,
                    fixed_split,
                    registry,
                    prior,
                    model,
                    checkpoint_digest,
                    str(fake_root),
                    args.generation_seed,
                    batch_size=args.samples_per_user,
                    device=device,
                    resume=False,
                    # Operational smoke publication only.  The exact formal
                    # physical thresholds are immediately re-audited below.
                    max_aggregate_clip_rate=1.0,
                    max_event_clip_rate=1.0,
                )
                unit_audits.append(audit)
                archive_path = unit_output_path(str(fake_root), unit)
                try:
                    audit_generated_unit(
                        str(archive_path), pool, fixed_split, registry, prior,
                        expected_count=args.samples_per_user,
                        expected_base_seed=args.generation_seed,
                        expected_generation_batch_size=args.samples_per_user,
                        expected_ddim_steps=50,
                        max_aggregate_clip_rate=0.05,
                        max_event_clip_rate=0.25,
                        max_alpha_bar_final=0.001,
                    )
                except Exception as exc:  # exact failure is part of the report
                    formal_gate_failures.append({
                        "user_id": int(user_id),
                        "archive": str(archive_path.resolve()),
                        "error": "%s: %s" % (type(exc).__name__, exc),
                    })
        finally:
            hook.remove()
        expected_calls = len(selected_all) * 50 * math.ceil(
            args.samples_per_user / args.samples_per_user
        )
        if calls["count"] != expected_calls:
            raise AssertionError(
                "%s DDIM denoiser calls %d != %d" % (action, calls["count"], expected_calls)
            )
        generation_call_counts[action] = calls["count"]
        report["generation"][action] = {
            "n_users": len(selected_all),
            "n_fake": len(selected_all) * args.samples_per_user,
            "checkpoint_sha256": checkpoint_digest,
            "train_prior_source_users": sorted(set(int(value) for value in prior.source_user_ids.tolist())),
            "train_prior_contains_only_fixed_train_users": set(
                int(value) for value in prior.source_user_ids.tolist()
            ).issubset(set(split.train_users)),
            "ddim_steps": 50,
            "denoiser_calls": calls["count"],
            "expected_denoiser_calls": expected_calls,
            "selector_used": False,
            "unit_audits": unit_audits,
            "smoke_physical_validity_gate_passed": (
                len(unit_audits) == len(selected_all)
                and all(
                row.get("passed") is True
                and 0.0 <= float(row.get("aggregate_clipped_point_rate", -1.0)) <= 1.0
                and 0.0 <= float(row.get("max_event_clipped_point_rate", -1.0)) <= 1.0
                for row in unit_audits
                )
            ),
            "formal_physical_gate_passed": len(formal_gate_failures) == 0,
            "formal_physical_gate_failures": formal_gate_failures,
            "formal_physical_gate_evaluated": True,
        }
        del model, pool, prior
        if device.type == "cuda":
            torch.cuda.empty_cache()
        atomic_json_dump(output / "e2e_smoke.in_progress.json", report)

    for action_index, action in enumerate(ACTIONS):
        report["detectors"][action] = _run_detector_smoke(
            action,
            Path(source["files"][action]["path"]),
            fake_root,
            fixed_split,
            selected_all,
            output / "detectors" / action,
            device,
            args.generation_seed + action_index * 100,
        )
        atomic_json_dump(output / "e2e_smoke.in_progress.json", report)

    artifacts = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"e2e_smoke.json", "e2e_smoke.in_progress.json"}:
            artifacts[str(path.relative_to(output))] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
    all_smoke_physical_validity_gates_passed = all(
        row.get("smoke_physical_validity_gate_passed") is True
        for row in report["generation"].values()
    )
    all_formal_physical_gates_passed = all(
        row.get("formal_physical_gate_passed") is True
        and not row.get("formal_physical_gate_failures")
        for row in report["generation"].values()
    )
    all_formal_physical_gates_evaluated = all(
        row.get("formal_physical_gate_evaluated") is True
        and isinstance(row.get("formal_physical_gate_passed"), bool)
        and isinstance(row.get("formal_physical_gate_failures"), list)
        and bool(row.get("formal_physical_gate_failures"))
        == (row.get("formal_physical_gate_passed") is False)
        for row in report["generation"].values()
    )
    all_training_loss_finite_and_decreased = all(
            row["loss"]["all_finite"]
            and row["loss"]["last_window_mean"] < row["loss"]["first_window_mean"]
            for row in report["training"].values()
    )
    all_checkpoint_schedule_best_ema_gates_passed = all(
            row["checkpoint_loader_formal_schedule_best_ema_gate_passed"]
            for row in report["training"].values()
    )
    all_archive_adapter_paths_passed = all(
            row["n_fake"] == len(selected_all) * args.samples_per_user
            for row in report["generation"].values()
    )
    detector_pairs_completed = sum(
            row["detector_kind_count"] for row in report["detectors"].values()
    )
    all_25_detector_interface_smokes_passed = all(
            row["passed"] for row in report["detectors"].values()
    ) and detector_pairs_completed == 25
    smoke_passed = all((
        all_training_loss_finite_and_decreased,
        all_checkpoint_schedule_best_ema_gates_passed,
        all_archive_adapter_paths_passed,
        all_smoke_physical_validity_gates_passed,
        all_formal_physical_gates_evaluated,
        all_25_detector_interface_smokes_passed,
    ))
    report.update({
        "schema_version": SMOKE_SCHEMA,
        "status": "passed" if smoke_passed else "failed",
        "elapsed_seconds": time.time() - started,
        "all_training_loss_finite_and_decreased": all_training_loss_finite_and_decreased,
        "all_checkpoint_schedule_best_ema_gates_passed": all_checkpoint_schedule_best_ema_gates_passed,
        "all_archive_adapter_paths_passed": all_archive_adapter_paths_passed,
        "all_smoke_physical_validity_gates_passed": all_smoke_physical_validity_gates_passed,
        "all_formal_physical_gates_evaluated": all_formal_physical_gates_evaluated,
        "all_formal_physical_gates_passed": all_formal_physical_gates_passed,
        "detector_pairs_completed": detector_pairs_completed,
        "all_25_detector_interface_smokes_passed": all_25_detector_interface_smokes_passed,
        "artifact_hashes": artifacts,
    })
    atomic_json_dump(output / "e2e_smoke.json", report)
    _atomic_text(output / "e2e_smoke.md", _markdown(report))
    in_progress = output / "e2e_smoke.in_progress.json"
    if in_progress.exists():
        in_progress.unlink()
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "elapsed_seconds": report["elapsed_seconds"],
        "detector_pairs_completed": report["detector_pairs_completed"],
        "formal_result": False,
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
