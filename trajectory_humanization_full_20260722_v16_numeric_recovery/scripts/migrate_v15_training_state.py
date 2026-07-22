#!/usr/bin/env python3
"""Migrate verified v15 training checkpoints into the v16 numeric policy.

Only metadata and paths are rewritten.  Model, EMA, optimizer, scaler and RNG
state are required to have identical recursive content digests before and
after migration.  Both actions are staged in one training directory and
atomically renamed into place; partial or pre-existing targets fail closed.
A receipt stored inside that directory makes the public receipt recoverable
if the process stops immediately after the atomic rename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.corpus import atomic_json_dump  # noqa: E402
from training.engine import (  # noqa: E402
    TrainingConfig,
    _canonical_sha256,
    atomic_torch_save,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recursive_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(b"\0")
            # Raw uint8 storage supports every torch dtype, including
            # bfloat16, without requiring a matching NumPy dtype.
            digest.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            )
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(array.dtype.str.encode())
            digest.update(b"\0")
            digest.update(json.dumps(list(array.shape)).encode())
            digest.update(b"\0")
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda value: str(value)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            digest.update(b"scalar\0")
            digest.update(repr(item).encode("utf-8"))
            digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def expected_training_config(
    config: Mapping[str, Any], action: str, batch_size: int, output: Path,
) -> TrainingConfig:
    training = config["training"]
    corpus = Path(config["corpus_dir"]) / ("hmog_trajectory_%s.npz" % action)
    return TrainingConfig(
        action=action,
        corpus_npz=str(corpus.resolve()),
        split_json=str(Path(config["split_json"]).resolve()),
        output_dir=str(output.resolve()),
        epochs=int(training["epochs"]),
        batch_size=int(batch_size),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        grad_clip_norm=float(training["grad_clip_norm"]),
        ema_decay=float(training["ema_decay"]),
        diffusion_steps=int(training["diffusion_steps"]),
        base_channels=int(training["base_channels"]),
        cond_dim=int(training["cond_dim"]),
        time_dim=int(training["time_dim"]),
        n_blocks=int(training["n_blocks"]),
        dropout=float(training["dropout"]),
        keycode_vocab=int(training["keycode_vocab"]),
        seed=int(training["seed"]),
        num_workers=int(training["num_workers"]),
        amp=True,
        checkpoint_every_steps=int(training["checkpoint_every_steps"]),
        reference_cache_size=int(training["reference_cache_size"]),
        device=str(config["action_device"][action]),
        amp_overflow_max_retries=int(training["amp_overflow_max_retries"]),
    )


def migrate_checkpoint(
    source: Path, target: Path, expected: TrainingConfig,
    lineage_base: Mapping[str, Any], published_target: Path | None = None,
) -> Dict[str, Any]:
    source_sha = sha256_file(source)
    checkpoint = torch.load(str(source), map_location="cpu")
    source_config = checkpoint.get("config", {})
    expected_config = asdict(expected)
    stable_config_names = tuple(
        key for key in expected_config
        if key not in ("output_dir", "amp_overflow_max_retries")
    )
    if any(source_config.get(key) != expected_config[key] for key in stable_config_names):
        raise ValueError(
            "source checkpoint training config is incompatible with v16: %s" % source
        )
    protected_names = ("model", "ema", "optimizer", "amp_scaler", "rng_state")
    protected_before = {
        name: recursive_digest(checkpoint[name]) for name in protected_names
    }
    migrated = dict(checkpoint)
    migrated["config"] = asdict(expected)
    migrated["numeric_recovery_policy"] = expected.numeric_recovery_policy
    progress = dict(migrated["progress"])
    progress["amp_overflow_retries_total"] = 0
    progress["epoch_amp_overflow_events"] = []
    migrated["progress"] = progress
    migrated["migration_lineage"] = {
        **dict(lineage_base),
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": source_sha,
        "protected_content_sha256": protected_before,
    }
    protected_after = {
        name: recursive_digest(migrated[name]) for name in protected_names
    }
    if protected_after != protected_before:
        raise RuntimeError("protected checkpoint state changed during migration")
    atomic_torch_save(target, migrated, overwrite=False)
    return {
        "source_path": str(source.resolve()),
        "source_sha256": source_sha,
        "target_path": str(
            (target if published_target is None else published_target).resolve()
        ),
        "target_sha256": sha256_file(target),
        "protected_content_sha256": protected_before,
        "progress": {
            key: progress[key]
            for key in (
                "epoch_index", "next_batch_in_epoch", "examples_seen_in_epoch",
                "global_step", "amp_overflow_retries_total",
            )
        },
    }


def migrate_action(
    action: str, source_root: Path, staged_training: Path,
    final_training: Path,
    config: Mapping[str, Any], selection: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> Dict[str, Any]:
    source = source_root / "training" / action
    target = final_training / action
    staged = staged_training / action
    staged.mkdir(parents=True, exist_ok=False)
    expected = expected_training_config(
        config, action, int(selection["selected_batch_size_by_action"][action]), target,
    )
    expected.validate()
    source_last = source / "last.pt"
    expected_last_sha = bootstrap["actions"][action]["last_checkpoint_sha256"]
    if sha256_file(source_last) != expected_last_sha:
        raise ValueError("source last checkpoint SHA mismatch for %s" % action)
    lineage_base = {
        "schema_version": "trajectory_v15_to_v16_numeric_recovery_lineage_v1",
        "source_run_root": str(source_root.resolve()),
        "source_tree_sha256": bootstrap["source_tree_sha256"],
        "reason": "P-TRJ-29 AMP overflow and P-TRJ-30 CUDA RNG resume fix",
        "target_numeric_recovery_policy": expected.numeric_recovery_policy,
    }
    copied = []
    for name in ("source_audit.json", "reference_audit.json", "reference_registry.json"):
        shutil.copy2(source / name, staged / name)
        copied.append({"name": name, "sha256": sha256_file(staged / name)})

    checkpoint_receipts: Dict[str, Any] = {}
    source_best_manifest = source / "best_manifest.json"
    if source_best_manifest.is_file():
        manifest = json.loads(source_best_manifest.read_text(encoding="utf-8"))
        migrated_history = []
        for row in manifest["history"]:
            source_checkpoint = Path(str(row["path"]))
            target_checkpoint = staged / str(row["filename"])
            receipt = migrate_checkpoint(
                source_checkpoint, target_checkpoint, expected, lineage_base,
                published_target=target / str(row["filename"]),
            )
            checkpoint_receipts[str(row["filename"])] = receipt
            migrated_row = dict(row)
            migrated_row["path"] = str((target / str(row["filename"])).resolve())
            migrated_row["checkpoint_sha256"] = receipt["target_sha256"]
            migrated_row["migration_source_sha256"] = receipt["source_sha256"]
            migrated_history.append(migrated_row)
        if not migrated_history:
            raise ValueError("source best manifest has empty history: %s" % action)
        manifest["history"] = migrated_history
        manifest["best"] = migrated_history[-1]
        manifest["numeric_recovery_policy"] = expected.numeric_recovery_policy
        manifest["migration_lineage"] = dict(lineage_base)
        atomic_json_dump(staged / "best_manifest.json", manifest)

    last_receipt = migrate_checkpoint(
        source_last, staged / "last.pt", expected, lineage_base,
        published_target=target / "last.pt",
    )
    checkpoint_receipts["last.pt"] = last_receipt
    last_checkpoint = torch.load(str(staged / "last.pt"), map_location="cpu")
    progress = last_checkpoint["progress"]
    atomic_json_dump(staged / "last_state.json", {
        "schema_version": "trajectory_last_state_v1",
        "protocol_version": last_checkpoint["protocol_version"],
        "run_instance_id": "bootstrap_" + uuid.uuid4().hex,
        "action": action,
        "checkpoint_path": str((target / "last.pt").resolve()),
        "checkpoint_sha256": last_receipt["target_sha256"],
        "checkpoint_size_bytes": int((staged / "last.pt").stat().st_size),
        "progress": {
            key: int(progress[key]) for key in (
                "epoch_index", "next_batch_in_epoch", "examples_seen_in_epoch",
                "global_step",
            )
        },
        "source": last_checkpoint["source"],
        "config_sha256": _canonical_sha256(asdict(expected)),
        "updated_unix_time": time.time(),
        "migration_lineage": dict(lineage_base),
    })

    metrics = []
    for line in (source / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "train_epoch":
            row["amp_overflow_retries"] = 0
            row["amp_overflow_events"] = []
        metrics.append(row)
    with (staged / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in metrics:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    run = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    run.update({
        "config": asdict(expected),
        "numeric_recovery_policy": expected.numeric_recovery_policy,
        "resume_from": str((target / "last.pt").resolve()),
        "resume_checkpoint_sha256": last_receipt["target_sha256"],
        "status": "migrated_awaiting_resume",
        "migration_lineage": dict(lineage_base),
    })
    atomic_json_dump(staged / "run_manifest.json", run)
    return {
        "action": action,
        "source_last_checkpoint_sha256": expected_last_sha,
        "target_last_checkpoint_sha256": last_receipt["target_sha256"],
        "expected_config_sha256": _canonical_sha256(asdict(expected)),
        "checkpoint_receipts": checkpoint_receipts,
        "copied_files": copied,
        "metrics_rows": len(metrics),
    }


def main() -> int:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--selection", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    args = value.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    selection = json.loads(args.selection.resolve().read_text(encoding="utf-8"))
    bootstrap = config["training_bootstrap"]
    source_root = Path(bootstrap["source_run_root"]).resolve()
    target_root = Path(config["run_root"]).resolve()
    source_state = json.loads(
        (source_root / "supervisor_status.json").read_text(encoding="utf-8")
    )
    if (
        source_state.get("status") != "failed"
        or source_state.get("source_tree_sha256") != bootstrap["source_tree_sha256"]
    ):
        raise ValueError("source failed-run identity mismatch")
    actions = list(bootstrap["actions"])
    if actions != ["tap", "scroll"]:
        raise ValueError("bootstrap actions must be exactly tap,scroll")
    final_training = target_root / "training"
    internal_receipt = final_training / ".v15_to_v16_bootstrap_receipt.json"
    if final_training.exists():
        if not internal_receipt.is_file():
            raise FileExistsError(
                "bootstrap training target exists without an internal receipt: %s"
                % final_training
            )
        result = json.loads(internal_receipt.read_text(encoding="utf-8"))
        if (
            result.get("schema_version")
            != "trajectory_v15_to_v16_training_bootstrap_v1"
            or result.get("passed") is not True
            or result.get("source_tree_sha256") != bootstrap["source_tree_sha256"]
            or result.get("target_run_root") != str(target_root)
            or result.get("selection_sha256") != sha256_file(args.selection.resolve())
        ):
            raise ValueError("existing internal bootstrap receipt is not reusable")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(args.output.resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    staged_training = target_root / (".training.bootstrap." + uuid.uuid4().hex)
    staged_training.mkdir(parents=True, exist_ok=False)
    try:
        receipts = [
            migrate_action(
                action, source_root, staged_training, final_training,
                config, selection, bootstrap,
            )
            for action in actions
        ]
        result = {
            "schema_version": "trajectory_v15_to_v16_training_bootstrap_v1",
            "passed": True,
            "source_run_root": str(source_root),
            "source_tree_sha256": bootstrap["source_tree_sha256"],
            "target_run_root": str(target_root),
            "config_sha256": _canonical_sha256({
                key: item for key, item in config.items()
                if key != "formal_launch_authorized"
            }),
            "selection_path": str(args.selection.resolve()),
            "selection_sha256": sha256_file(args.selection.resolve()),
            "actions": receipts,
            "completed_unix_time": time.time(),
        }
        atomic_json_dump(
            staged_training / ".v15_to_v16_bootstrap_receipt.json", result,
        )
        os.replace(str(staged_training), str(final_training))
    finally:
        if staged_training.exists():
            shutil.rmtree(staged_training)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
