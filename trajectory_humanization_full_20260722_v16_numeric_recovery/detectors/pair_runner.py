"""Power-loss-safe execution and audit for one action/PAD-detector pair.

Formal orchestration runs the 25 independent pairs in parallel and merges only
complete, independently audited pair manifests.  No pair reads another pair's
scores and no threshold is selected from test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from detectors.benchmark_runner import _plot_curve, _row_from_metrics
from detectors.deep_pad import (
    ACTIONS,
    DEEP_DETECTORS,
    DeepTrainConfig,
    RawSequenceNormalizer,
    RawTrajectoryRecord,
    assign_strict_protocol_pools,
    load_fake_user_split,
    load_raw_sequence_bundle,
    run_deep_pad_protocol,
)
from detectors.feature_pad import (
    ALLOWED_DETECTORS,
    fa_frr_curve,
    operating_metrics,
    run_feature_pad_protocol,
    save_protocol_outputs,
    select_validation_thresholds,
    user_level_bootstrap,
)


PAIR_SCHEMA = "trajectory_pad_pair_v2"
FAMILIES = ("feature_pad", "deep_pad")
FORMAL_MIN_EPOCHS = 40
FORMAL_MIN_BOOTSTRAP = 500


def stable_pair_seed(base_seed: int, action: str, family: str, detector: str) -> int:
    """Schedule-order-independent RNG seed unique to one formal pair."""

    _validate_pair_identity(action, family, detector)
    payload = "%d|%s|%s|%s" % (int(base_seed), action, family, detector)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _archive_unbound_feature_result(
    result_dir: Path, pair_root: Path, *, dataset_sha256: str, config_sha256: str
) -> Optional[Path]:
    """Move scores lacking a valid pair commit out of the active result path.

    Feature models are deliberately not deserialized from an untrusted pickle,
    so a summary/score tree alone cannot prove which feature matrix produced
    its scores.  Only ``pair_manifest.json`` is the source/config commit.  If
    that commit is absent, archive the whole tree and deterministically retrain
    instead of relabelling stale scores with current hashes.
    """

    root = Path(result_dir)
    if not root.exists() or not any(root.iterdir()):
        return None
    archive_root = Path(pair_root) / "orphaned_unbound_feature_results"
    archive_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    stem = "dataset_%s_config_%s_artifacts_%s" % (
        dataset_sha256[:12], config_sha256[:12], digest.hexdigest()[:12],
    )
    destination = archive_root / stem
    suffix = 1
    while destination.exists():
        destination = archive_root / (stem + "_%02d" % suffix)
        suffix += 1
    os.replace(str(root), str(destination))
    return destination


def _archive_unbound_deep_outputs(result_dir: Path, pair_root: Path) -> Optional[Path]:
    """Archive uncommitted derived Deep outputs, retaining source-bound states.

    When the pair commit is absent, saved scores cannot be trusted even though
    best/last checkpoints are independently source-bound.  Preserve only the
    checkpoint transaction, archive every derived top-level artifact, then let
    ``run_deep_pad_protocol(..., resume=True)`` reload the selected best and
    deterministically regenerate scores/curves/bootstrap without another
    training epoch.
    """

    root = Path(result_dir)
    if not root.is_dir():
        return None
    candidates = sorted(
        (path for path in root.iterdir() if path.name != "checkpoints"),
        key=lambda path: path.name,
    )
    if not candidates:
        return None
    digest = hashlib.sha256()
    for item in candidates:
        if item.is_file():
            digest.update(item.name.encode("utf-8"))
            digest.update(sha256_file(item).encode("ascii"))
        else:
            digest.update((item.name + "/").encode("utf-8"))
    archive_root = Path(pair_root) / "orphaned_unbound_deep_outputs"
    archive_root.mkdir(parents=True, exist_ok=True)
    stem = "derived_%s" % digest.hexdigest()[:16]
    destination = archive_root / stem
    suffix = 1
    while destination.exists():
        destination = archive_root / (stem + "_%02d" % suffix)
        suffix += 1
    destination.mkdir()
    for item in candidates:
        os.replace(str(item), str(destination / item.name))
    return destination


def _preflight_deep_checkpoint_identity(
    result_dir: Path,
    *,
    action: str,
    detector: str,
    train_config: DeepTrainConfig,
    model_params: Optional[Mapping[str, Any]],
    input_identity: Mapping[str, Any],
) -> None:
    checkpoint_dir = Path(result_dir) / "checkpoints"
    if not checkpoint_dir.is_dir():
        return
    last = checkpoint_dir / "last.pt"
    immutable = sorted(checkpoint_dir.glob("best_epoch_*.pt"))
    candidates = [last] if last.is_file() else immutable
    if not candidates:
        return
    expected = {
        "schema_version": "trajectory_deep_pad_run_identity_v2",
        "action": action,
        "detector_kind": detector,
        "model_params": dict(model_params or {}),
        "train_config": asdict(train_config),
        "selection_pool": "validation_only",
        "input_identity": dict(input_identity),
    }
    expected_sha = _config_digest(expected)
    for path in candidates:
        checkpoint = torch.load(str(path), map_location="cpu")
        if (
            checkpoint.get("schema_version") != "trajectory_deep_pad_v2"
            or checkpoint.get("run_identity") != expected
            or checkpoint.get("run_identity_sha256") != expected_sha
        ):
            raise ValueError(
                "Deep checkpoint source/config identity mismatch before recovery"
            )


def _config_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_deep_pair_input_identity(
    *,
    dataset_file: Path,
    dataset_sha256: str,
    fake_user_split: Path,
    fake_user_split_sha256: str,
    real_hash_seed: int,
    action: str,
    detector: str,
    pair_config: Mapping[str, Any],
    pair_config_sha256: str,
) -> Dict[str, Any]:
    """Canonical source/config identity embedded in every Deep checkpoint."""

    return {
        "schema_version": "trajectory_deep_pad_pair_input_v1",
        "dataset_file": str(Path(dataset_file).resolve()),
        "dataset_sha256": str(dataset_sha256),
        "fake_user_split": str(Path(fake_user_split).resolve()),
        "fake_user_split_sha256": str(fake_user_split_sha256),
        "real_hash_seed": int(real_hash_seed),
        "action": str(action),
        "family": "deep_pad",
        "detector": str(detector),
        "pair_config": dict(pair_config),
        "pair_config_sha256": str(pair_config_sha256),
    }


def _require_close(actual: float, expected: float, message: str) -> None:
    if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(actual, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError("%s: %r != %r" % (message, actual, expected))


def _require_finite_nested(value: Any, context: str) -> None:
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise ValueError("%s contains a non-finite tensor" % context)
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError("%s contains a non-finite array" % context)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_nested(item, "%s.%s" % (context, key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_nested(item, "%s[%d]" % (context, index))
        return
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("%s contains a non-finite scalar" % context)


def _assert_normalizer_state_equal(
    observed: Mapping[str, Any], expected: Mapping[str, Any], context: str
) -> None:
    if set(observed) != set(expected):
        raise ValueError("%s normalizer fields mismatch" % context)
    for name in ("pointer_mean", "pointer_scale"):
        left = np.asarray(observed[name], dtype=np.float64)
        right = np.asarray(expected[name], dtype=np.float64)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError("%s normalizer %s mismatch" % (context, name))
    for name in ("log_dt_mean", "log_dt_scale"):
        if float(observed[name]) != float(expected[name]):
            raise ValueError("%s normalizer %s mismatch" % (context, name))
    if list(observed.get("fit_sample_ids", [])) != list(
        expected.get("fit_sample_ids", [])
    ):
        raise ValueError("%s normalizer train sample IDs mismatch" % context)


def _validate_pair_identity(action: str, family: str, detector: str) -> None:
    if action not in ACTIONS:
        raise ValueError("unknown action: %s" % action)
    if family not in FAMILIES:
        raise ValueError("family must be feature_pad or deep_pad")
    allowed = ALLOWED_DETECTORS if family == "feature_pad" else DEEP_DETECTORS
    if detector not in allowed:
        raise ValueError("detector %s is invalid for %s" % (detector, family))


def _load_and_assign_action(
    dataset_file: Path,
    fake_user_split: Path,
    action: str,
    real_hash_seed: int,
    require_formal: bool,
) -> Tuple[Sequence[RawTrajectoryRecord], np.ndarray, Dict[str, Any]]:
    records, features = load_raw_sequence_bundle(dataset_file)
    if not records or any(record.action != action for record in records):
        raise ValueError("dataset file does not contain only %s records" % action)
    fixed_split = load_fake_user_split(fake_user_split)
    assigned, split_audit = assign_strict_protocol_pools(
        records, fixed_split, real_hash_seed=real_hash_seed
    )
    if len(features) != len(assigned):
        raise ValueError("feature rows and raw records disagree")
    coverage = {
        "%d:%s" % (label, pool): len({
            row.user_id for row in assigned if row.label == label and row.pool == pool
        })
        for label in (0, 1) for pool in ("train", "val", "test")
    }
    if require_formal and tuple(
        coverage["0:%s" % pool] for pool in ("train", "val", "test")
    ) != (100, 100, 100):
        raise ValueError(
            "formal real event pools must each include all 100 users"
        )
    if require_formal and tuple(coverage["1:%s" % pool] for pool in ("train", "val", "test")) != (70, 10, 20):
        raise ValueError("formal fake pools must use fixed disjoint 70/10/20 users")
    fake_counts = {
        pool: sum(row.label == 1 and row.pool == pool for row in assigned)
        for pool in ("train", "val", "test")
    }
    if require_formal and fake_counts != {"train": 14000, "val": 2000, "test": 4000}:
        raise ValueError(
            "formal fake action must contain 200/user: expected 14k/2k/4k, got %r"
            % fake_counts
        )
    fake_user_values = sorted({row.user_id for row in assigned if row.label == 1})
    per_fake_user = {
        user: sum(row.label == 1 and row.user_id == user for row in assigned)
        for user in fake_user_values
    }
    if require_formal and (set(fake_user_values) != set(range(100)) or set(per_fake_user.values()) != {200}):
        raise ValueError("formal fake action must contain exactly 200 samples per user")
    return assigned, np.asarray(features, dtype=np.float64), {
        "coverage_users": coverage,
        "fake_sample_counts": fake_counts,
        "fake_samples_per_user_min_max": [
            min(per_fake_user.values()), max(per_fake_user.values())
        ],
        "split_audit": split_audit,
    }


def _audit_score_rows_against_dataset(
    scores: Mapping[str, np.ndarray],
    *,
    records: Sequence[RawTrajectoryRecord],
    action: str,
    family: str,
) -> Dict[str, Any]:
    """Reconnect every persisted validation/test score row to its source row."""

    all_identities = []
    counts: Dict[str, int] = {}
    for pool in ("val", "test"):
        expected_indices = np.asarray(
            [index for index, row in enumerate(records) if row.pool == pool],
            dtype=np.int64,
        )
        expected_rows = [records[int(index)] for index in expected_indices]
        expected_count = len(expected_rows)
        if expected_count == 0:
            raise ValueError("dataset relink found no %s rows" % pool)

        required = ("score", "label", "user_id", "pool", "action")
        for field in required:
            key = "%s_%s" % (pool, field)
            if key not in scores:
                raise ValueError("score dump missing dataset-link field %s" % key)
            value = np.asarray(scores[key])
            if value.ndim != 1 or len(value) != expected_count:
                raise ValueError(
                    "%s score row count mismatch: expected %d, observed %d"
                    % (pool, expected_count, len(value) if value.ndim == 1 else -1)
                )

        observed_label = np.asarray(scores[pool + "_label"], dtype=np.int64)
        observed_user = np.asarray(scores[pool + "_user_id"], dtype=np.int64)
        observed_pool = np.asarray(scores[pool + "_pool"]).astype(str)
        observed_action = np.asarray(scores[pool + "_action"]).astype(str)
        expected_label = np.asarray([row.label for row in expected_rows], dtype=np.int64)
        expected_user = np.asarray([row.user_id for row in expected_rows], dtype=np.int64)
        expected_pool = np.full(expected_count, pool, dtype="U5")
        expected_action = np.full(expected_count, action, dtype="U16")
        for field, observed, expected in (
            ("label", observed_label, expected_label),
            ("user_id", observed_user, expected_user),
            ("pool", observed_pool, expected_pool),
            ("action", observed_action, expected_action),
        ):
            if not np.array_equal(observed, expected):
                raise ValueError("%s score %s rows do not relink to dataset" % (pool, field))

        if family == "feature_pad":
            key = pool + "_row_index"
            if key not in scores:
                raise ValueError("score dump missing %s" % key)
            raw_identity = np.asarray(scores[key])
            if raw_identity.dtype.kind not in "iu" or raw_identity.ndim != 1:
                raise ValueError("%s row_index must be a one-dimensional integer array" % pool)
            identity = raw_identity.astype(np.int64, copy=False)
            if not np.array_equal(identity, expected_indices):
                raise ValueError("%s row_index does not exactly relink dataset order" % pool)
            identities = [("row_index", int(value)) for value in identity]
        else:
            key = pool + "_sample_id"
            if key not in scores:
                raise ValueError("score dump missing %s" % key)
            raw_identity = np.asarray(scores[key])
            if raw_identity.ndim != 1 or raw_identity.dtype.kind not in "US":
                raise ValueError("%s sample_id must be a one-dimensional string array" % pool)
            identity = raw_identity.astype(str)
            expected_identity = np.asarray(
                [row.sample_id for row in expected_rows], dtype=str
            )
            if not np.array_equal(identity, expected_identity):
                raise ValueError("%s sample_id does not exactly relink dataset order" % pool)
            identities = [
                ("label+sample_id", int(label), str(value))
                for label, value in zip(observed_label, identity)
            ]

        if len(set(identities)) != expected_count:
            raise ValueError("%s score identities are not unique" % pool)
        all_identities.extend(identities)
        counts[pool] = int(expected_count)

    if len(set(all_identities)) != len(all_identities):
        raise ValueError("validation/test score identities overlap")
    return {
        "protocol": "every_score_row_exactly_relinked_to_assigned_dataset_v1",
        "identity_field": (
            "row_index" if family == "feature_pad" else "label+sample_id"
        ),
        "expected_and_observed_rows": counts,
        "unique_identity_count": int(len(all_identities)),
    }


def audit_protocol_result(
    result_dir: Path,
    *,
    action: str,
    family: str,
    detector: str,
    expected_bootstrap_replicates: int,
    dataset_file: Optional[Path] = None,
    fake_user_split: Optional[Path] = None,
    real_hash_seed: Optional[int] = None,
    expected_dataset_sha256: Optional[str] = None,
    expected_fake_user_split_sha256: Optional[str] = None,
    expected_bootstrap_seed: Optional[int] = None,
    expected_deep_run_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Recompute metrics and, when supplied, relink every score to the dataset."""

    _validate_pair_identity(action, family, detector)
    root = Path(result_dir)
    required = (
        "summary.json", "score_dump.npz", "curves.npz",
        "bootstrap_summary.json", "bootstrap_replicates.npz",
    )
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError("incomplete pair result: %s" % (root / name))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("action") != action or summary.get("detector_kind") != detector:
        raise ValueError("pair summary identity mismatch")
    if summary.get("score_direction") != "fake_high" or summary.get("acceptance_rule") != "score < threshold":
        raise ValueError("pair score semantics mismatch")
    if summary.get("threshold_selection_pool") != "validation_only":
        raise ValueError("pair threshold was not selected on validation only")
    deep_paths: Dict[str, Path] = {}
    deep_checkpoints: Dict[str, Mapping[str, Any]] = {}
    deep_training_audit: Dict[str, Any] = {}
    if family == "deep_pad":
        if summary.get("schema_version") != "trajectory_deep_pad_result_v2":
            raise ValueError("deep result schema mismatch")
        if summary.get("checkpoint_selection_pool") != "validation_only":
            raise ValueError("deep checkpoint was not selected on validation only")
        if summary.get("uses_critic") is not False or summary.get("uses_selector") is not False:
            raise ValueError("Deep PAD must be independent of generator critic/selector")
        for checkpoint in ("best", "last"):
            path = Path(summary["checkpoint_paths"][checkpoint])
            if not path.is_file():
                raise FileNotFoundError("deep checkpoint is missing: %s" % path)
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError("deep checkpoint escapes its pair result directory") from exc
            deep_paths[checkpoint] = path
        if not (root / "history.csv").is_file():
            raise FileNotFoundError("deep history.csv is missing")
        with (root / "history.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != (
                "epoch", "train_loss", "val_loss", "val_auc"
            ):
                raise ValueError("deep history.csv schema mismatch")
            history_rows = list(reader)
        if not history_rows:
            raise ValueError("deep history.csv is empty")
        history_epochs = [int(row["epoch"]) for row in history_rows]
        if history_epochs != list(range(1, len(history_rows) + 1)):
            raise ValueError("deep history epochs are not contiguous from one")
        for row in history_rows:
            numeric = np.asarray(
                [row["train_loss"], row["val_loss"], row["val_auc"]],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(numeric)):
                raise ValueError("deep history contains non-finite values")
        canonical_history = [
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "val_auc": float(row["val_auc"]),
            }
            for row in history_rows
        ]
        last_epoch = int(summary.get("last_epoch", -1))
        best_epoch = int(summary.get("best_epoch", -1))
        if last_epoch != len(history_rows):
            raise ValueError("deep summary last_epoch/history length mismatch")
        if not (1 <= best_epoch <= last_epoch):
            raise ValueError("deep summary best_epoch is outside completed history")
        train_config = summary.get("train_config")
        if not isinstance(train_config, dict):
            raise ValueError("deep summary train_config is missing")
        if int(train_config.get("epochs", -1)) < last_epoch:
            raise ValueError("deep completed epochs exceed configured epoch budget")
        selected_best = canonical_history[0]
        for row in canonical_history[1:]:
            if (
                row["val_auc"] > selected_best["val_auc"] + 1.0e-12
                or (
                    abs(row["val_auc"] - selected_best["val_auc"]) <= 1.0e-12
                    and row["val_loss"] < selected_best["val_loss"] - 1.0e-12
                )
            ):
                selected_best = row
        if int(selected_best["epoch"]) != best_epoch:
            raise ValueError("deep summary best_epoch violates validation selection rule")
        deep_training_audit = {
            "history_epoch_count": len(history_rows),
            "history_first_epoch": history_epochs[0],
            "history_last_epoch": history_epochs[-1],
            "summary_last_epoch": last_epoch,
            "summary_best_epoch": best_epoch,
        }
        identity = summary.get("run_identity")
        identity_sha = summary.get("run_identity_sha256")
        if not isinstance(identity, dict) or _config_digest(identity) != identity_sha:
            raise ValueError("deep summary run identity/digest mismatch")
        if expected_deep_run_identity is not None:
            expected_wrapper = {
                "schema_version": "trajectory_deep_pad_run_identity_v2",
                "action": action,
                "detector_kind": detector,
                "model_params": dict(summary.get("model_params", {})),
                "train_config": dict(summary.get("train_config", {})),
                "selection_pool": "validation_only",
                "input_identity": dict(expected_deep_run_identity),
            }
            if identity != expected_wrapper or identity_sha != _config_digest(expected_wrapper):
                raise ValueError("deep summary does not match expected pair run identity")
        for checkpoint_name, checkpoint_path in deep_paths.items():
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            if (
                checkpoint.get("schema_version") != "trajectory_deep_pad_v2"
                or checkpoint.get("action") != action
                or checkpoint.get("detector_kind") != detector
                or checkpoint.get("run_identity") != identity
                or checkpoint.get("run_identity_sha256") != identity_sha
                or _config_digest(checkpoint.get("run_identity", {})) != identity_sha
            ):
                raise ValueError(
                    "deep %s checkpoint run identity mismatch" % checkpoint_name
                )
            if (
                checkpoint.get("model_params") != summary.get("model_params", {})
                or checkpoint.get("train_config") != train_config
                or checkpoint.get("selection_pool") != "validation_only"
                or checkpoint.get("checkpoint_commit_policy")
                != "immutable_best_then_last_with_exact_epoch_replay_v3_source_bound"
            ):
                raise ValueError("deep %s checkpoint config/policy mismatch" % checkpoint_name)
            if not isinstance(checkpoint.get("model_state"), dict) or not checkpoint["model_state"]:
                raise ValueError("deep %s checkpoint model_state is empty" % checkpoint_name)
            optimizer_state = checkpoint.get("optimizer_state")
            if (
                not isinstance(optimizer_state, dict)
                or not isinstance(optimizer_state.get("state"), dict)
                or not optimizer_state["state"]
            ):
                raise ValueError("deep %s checkpoint optimizer state is empty" % checkpoint_name)
            _require_finite_nested(checkpoint["model_state"], "deep %s model" % checkpoint_name)
            _require_finite_nested(optimizer_state, "deep %s optimizer" % checkpoint_name)
            # Also schema/finite-check the serialized normalizer before any
            # dataset-dependent equality test below.
            RawSequenceNormalizer.from_state_dict(checkpoint.get("normalizer", {}))
            deep_checkpoints[checkpoint_name] = checkpoint

        best_checkpoint = deep_checkpoints["best"]
        last_checkpoint = deep_checkpoints["last"]
        if (
            int(best_checkpoint.get("epoch", -1)) != best_epoch
            or int(best_checkpoint.get("best_epoch", -1)) != best_epoch
            or int(last_checkpoint.get("epoch", -1)) != last_epoch
            or int(last_checkpoint.get("best_epoch", -1)) != best_epoch
            or Path(str(best_checkpoint.get("best_path", ""))).resolve()
            != deep_paths["best"].resolve()
            or Path(str(last_checkpoint.get("best_path", ""))).resolve()
            != deep_paths["best"].resolve()
        ):
            raise ValueError("deep best/last checkpoint epoch/path semantics mismatch")
        if (
            best_checkpoint.get("history") != canonical_history[:best_epoch]
            or last_checkpoint.get("history") != canonical_history
        ):
            raise ValueError("deep checkpoint history does not equal history.csv")
        for checkpoint_name, checkpoint in deep_checkpoints.items():
            if (
                int(checkpoint.get("best_epoch", -1)) != best_epoch
                or float(checkpoint.get("best_auc", float("nan")))
                != float(selected_best["val_auc"])
                or float(checkpoint.get("best_loss", float("nan")))
                != float(selected_best["val_loss"])
            ):
                raise ValueError(
                    "deep %s checkpoint best-selection metadata mismatch"
                    % checkpoint_name
                )
        _assert_normalizer_state_equal(
            best_checkpoint["normalizer"], summary.get("normalizer", {}),
            "deep best/summary",
        )

    with np.load(root / "score_dump.npz", allow_pickle=False) as archive:
        scores = {name: archive[name] for name in archive.files}
    identity_field = "row_index" if family == "feature_pad" else "sample_id"
    expected_score_keys = {
        "%s_%s" % (pool, field)
        for pool in ("val", "test")
        for field in (
            "score", "label", "user_id", "pool", "action", identity_field,
        )
    }
    if set(scores) != expected_score_keys:
        raise ValueError("score dump array schema mismatch")
    for pool in ("val", "test"):
        for field in ("score", "label", "user_id", identity_field):
            key = "%s_%s" % (pool, field)
            if key not in scores:
                raise ValueError("score dump missing %s" % key)
        score = np.asarray(scores[pool + "_score"], dtype=np.float64)
        raw_label = np.asarray(scores[pool + "_label"])
        raw_user = np.asarray(scores[pool + "_user_id"])
        raw_pool = np.asarray(scores[pool + "_pool"])
        raw_action = np.asarray(scores[pool + "_action"])
        if raw_label.dtype.kind not in "iu" or raw_user.dtype.kind not in "iu":
            raise ValueError("%s score label/user_id must use integer dtype" % pool)
        if raw_pool.dtype.kind not in "US" or raw_action.dtype.kind not in "US":
            raise ValueError("%s score pool/action must use string dtype" % pool)
        label = raw_label.astype(np.int64, copy=False)
        if score.ndim != 1 or label.shape != score.shape or not np.all(np.isfinite(score)):
            raise ValueError("%s score dump is invalid/non-finite" % pool)
        if set(np.unique(label).tolist()) != {0, 1}:
            raise ValueError("%s score dump lacks both classes" % pool)

    dataset_arguments = (dataset_file, fake_user_split, real_hash_seed)
    if any(value is not None for value in dataset_arguments) and not all(
        value is not None for value in dataset_arguments
    ):
        raise ValueError(
            "dataset relink requires dataset_file, fake_user_split, and real_hash_seed"
        )
    dataset_relink_audit: Dict[str, Any] = {}
    if dataset_file is not None and fake_user_split is not None and real_hash_seed is not None:
        actual_dataset_sha = sha256_file(dataset_file)
        actual_split_sha = sha256_file(fake_user_split)
        if (
            expected_dataset_sha256 is not None
            and actual_dataset_sha != expected_dataset_sha256
        ):
            raise ValueError("dataset bytes changed before score-row relink")
        if (
            expected_fake_user_split_sha256 is not None
            and actual_split_sha != expected_fake_user_split_sha256
        ):
            raise ValueError("fake-user split bytes changed before score-row relink")
        assigned_records, assigned_features, _ = _load_and_assign_action(
            Path(dataset_file), Path(fake_user_split), action,
            int(real_hash_seed), require_formal=False,
        )
        dataset_relink_audit = _audit_score_rows_against_dataset(
            scores, records=assigned_records, action=action, family=family,
        )
        if family == "feature_pad":
            train_indices = np.asarray(
                [
                    index for index, row in enumerate(assigned_records)
                    if row.pool == "train"
                ],
                dtype=np.int64,
            )
            expected_scaler = StandardScaler().fit(assigned_features[train_indices])
            observed_mean = np.asarray(summary.get("scaler_mean", []), dtype=np.float64)
            observed_scale = np.asarray(summary.get("scaler_scale", []), dtype=np.float64)
            if (
                int(summary.get("train_row_count", -1)) != len(train_indices)
                or not np.array_equal(observed_mean, expected_scaler.mean_)
                or not np.array_equal(observed_scale, expected_scaler.scale_)
            ):
                raise ValueError(
                    "feature scaler/train-row audit does not match current train features"
                )
            dataset_relink_audit.update({
                "feature_train_row_count": int(len(train_indices)),
                "feature_scaler_recomputed_from_current_train_rows": True,
            })
        else:
            expected_normalizer = RawSequenceNormalizer().fit(
                [row for row in assigned_records if row.pool == "train"]
            ).state_dict()
            _assert_normalizer_state_equal(
                summary.get("normalizer", {}), expected_normalizer,
                "deep summary/dataset",
            )
            for checkpoint_name, checkpoint in deep_checkpoints.items():
                _assert_normalizer_state_equal(
                    checkpoint.get("normalizer", {}), expected_normalizer,
                    "deep %s/dataset" % checkpoint_name,
                )
        dataset_relink_audit.update({
            "dataset_sha256": actual_dataset_sha,
            "fake_user_split_sha256": actual_split_sha,
            "real_hash_seed": int(real_hash_seed),
        })
    recomputed_thresholds = select_validation_thresholds(
        scores["val_label"], scores["val_score"], target_frr=0.05
    )
    for point in ("eer", "val_frr_le_5pct"):
        _require_close(
            float(summary["thresholds"][point]), float(recomputed_thresholds[point]),
            point + " threshold",
        )
        for pool, summary_key in (("val", "validation_metrics"), ("test", "test_metrics")):
            metric = operating_metrics(
                scores[pool + "_label"], scores[pool + "_score"],
                float(summary["thresholds"][point]),
            )
            for name in ("fa", "frr", "auc"):
                _require_close(
                    float(summary[summary_key][point][name]), float(metric[name]),
                    "%s/%s/%s" % (pool, point, name),
                )

    with np.load(root / "curves.npz", allow_pickle=False) as archive:
        curve_arrays = {name: archive[name] for name in archive.files}
    curves = {
        pool: fa_frr_curve(scores[pool + "_label"], scores[pool + "_score"])
        for pool in ("val", "test")
    }
    expected_curve_keys = {
        "%s_%s" % (pool, name)
        for pool in ("val", "test") for name in ("threshold", "fa", "frr")
    }
    if set(curve_arrays) != expected_curve_keys:
        raise ValueError("saved curve array schema mismatch")
    for pool in ("val", "test"):
        for name in ("threshold", "fa", "frr"):
            key = pool + "_" + name
            if not np.array_equal(curve_arrays[key], curves[pool][name]):
                raise ValueError("saved %s curve does not match score dump: %s" % (pool, name))

    bootstrap = json.loads((root / "bootstrap_summary.json").read_text(encoding="utf-8"))
    if int(bootstrap.get("n_replicates", -1)) != int(expected_bootstrap_replicates):
        raise ValueError("bootstrap replicate count mismatch")
    if bootstrap.get("protocol") != "separate_real_fake_user_resampling_all_windows_fixed_val_thresholds":
        raise ValueError("bootstrap protocol mismatch")
    with np.load(root / "bootstrap_replicates.npz", allow_pickle=False) as archive:
        replicate_arrays = {name: archive[name] for name in archive.files}
    replicate_names = (
        "auc", "eer_fa", "eer_frr",
        "val_frr_le_5pct_fa", "val_frr_le_5pct_frr",
    )
    if set(replicate_arrays) != set(replicate_names):
        raise ValueError("bootstrap replicate array schema mismatch")
    for name in replicate_names:
        values = np.asarray(replicate_arrays.get(name), dtype=np.float64)
        if values.shape != (expected_bootstrap_replicates,) or not np.all(np.isfinite(values)):
            raise ValueError("bootstrap replicate array invalid: %s" % name)
    if expected_bootstrap_seed is None:
        expected_bootstrap_seed = (
            int(summary["random_state"]) + 31
            if family == "feature_pad"
            else int(summary["train_config"]["seed"]) + 17
        )
    if int(bootstrap.get("seed", -1)) != int(expected_bootstrap_seed):
        raise ValueError("bootstrap seed does not match the fixed pair seed policy")
    operating_thresholds = {
        point: float(summary["thresholds"][point])
        for point in ("eer", "val_frr_le_5pct")
    }
    recomputed_bootstrap = user_level_bootstrap(
        scores["test_label"], scores["test_score"], scores["test_user_id"],
        operating_thresholds,
        n_replicates=int(expected_bootstrap_replicates),
        seed=int(expected_bootstrap_seed),
    )
    expected_bootstrap_summary = {
        key: value for key, value in recomputed_bootstrap.items()
        if key != "replicates"
    }
    if _config_digest(bootstrap) != _config_digest(expected_bootstrap_summary):
        raise ValueError("bootstrap summary is not the exact fixed-seed recomputation")
    for name in replicate_names:
        if not np.array_equal(
            np.asarray(replicate_arrays[name]),
            np.asarray(recomputed_bootstrap["replicates"][name]),
        ):
            raise ValueError(
                "bootstrap replicate values do not equal fixed-seed recomputation: %s"
                % name
            )

    rows = [
        _row_from_metrics(
            action=action, family=family, detector=detector,
            operating_point=point, threshold=float(summary["thresholds"][point]),
            validation=summary["validation_metrics"][point],
            test=summary["test_metrics"][point],
            best_epoch=summary.get("best_epoch") if family == "deep_pad" else None,
            bootstrap=bootstrap,
        )
        for point in ("eer", "val_frr_le_5pct")
    ]
    artifact_hashes = {name: sha256_file(root / name) for name in required}
    if family == "deep_pad":
        artifact_hashes.update({
            "history.csv": sha256_file(root / "history.csv"),
            "checkpoint_best:" + deep_paths["best"].name: sha256_file(deep_paths["best"]),
            "checkpoint_last:" + deep_paths["last"].name: sha256_file(deep_paths["last"]),
        })
    return {
        "summary": summary,
        "bootstrap": bootstrap,
        "rows": rows,
        "test_curve": curves["test"],
        "artifact_hashes": artifact_hashes,
        "deep_training_audit": deep_training_audit,
        "dataset_relink_audit": dataset_relink_audit,
    }


def _require_formal_deep_completion(audited: Mapping[str, Any]) -> None:
    """Require evidence that a formal Deep result really executed 40 epochs."""

    summary = audited.get("summary", {})
    training = audited.get("deep_training_audit", {})
    if (
        int(summary.get("last_epoch", -1)) != FORMAL_MIN_EPOCHS
        or int(training.get("history_epoch_count", -1)) != FORMAL_MIN_EPOCHS
        or int(training.get("history_last_epoch", -1)) != FORMAL_MIN_EPOCHS
    ):
        raise ValueError("formal Deep pair must contain exactly 40 completed history epochs")


def run_or_resume_pair(
    *,
    dataset_file: Path,
    fake_user_split: Path,
    output_root: Path,
    action: str,
    family: str,
    detector: str,
    deep_config: DeepTrainConfig,
    feature_bootstrap_replicates: int,
    seed: int,
    real_hash_seed: int,
    device: Optional[str] = None,
    feature_model_params: Optional[Mapping[str, Any]] = None,
    deep_model_params: Optional[Mapping[str, Any]] = None,
    require_formal: bool = True,
    base_seed: Optional[int] = None,
    batch_probe_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one pair, resume Deep PAD from last.pt, and atomically commit it."""

    _validate_pair_identity(action, family, detector)
    if feature_bootstrap_replicates <= 0 or deep_config.bootstrap_replicates <= 0:
        raise ValueError("formal pair runs require positive user bootstrap replicates")
    resolved_base_seed = int(seed if base_seed is None else base_seed)
    if require_formal:
        expected_seed = stable_pair_seed(resolved_base_seed, action, family, detector)
        if int(seed) != expected_seed or int(deep_config.seed) != expected_seed:
            raise ValueError("formal pair seed must use stable_pair_seed(base, action, family, detector)")
        if deep_config.epochs != FORMAL_MIN_EPOCHS:
            raise ValueError("formal pair fixes configured epochs at exactly 40")
        if deep_config.patience != 0:
            raise ValueError("formal pair disables early stopping: patience must equal 0")
        if (
            feature_bootstrap_replicates != FORMAL_MIN_BOOTSTRAP
            or deep_config.bootstrap_replicates != FORMAL_MIN_BOOTSTRAP
        ):
            raise ValueError("formal pair fixes user-bootstrap replicates at exactly 500")
        if family == "deep_pad" and batch_probe_path is None:
            raise ValueError("formal Deep PAD pair requires an audited longest-event batch probe")
    pair_root = Path(output_root) / "pairs" / action / family / detector
    result_dir = pair_root / "result"
    manifest_path = pair_root / "pair_manifest.json"
    expected_replicates = (
        feature_bootstrap_replicates if family == "feature_pad"
        else deep_config.bootstrap_replicates
    )
    dataset_sha = sha256_file(dataset_file)
    split_sha = sha256_file(fake_user_split)
    batch_probe = None
    if batch_probe_path is not None:
        batch_probe = json.loads(Path(batch_probe_path).read_text(encoding="utf-8"))
        if (
            batch_probe.get("schema_version") != "trajectory_deep_batch_probe_v1"
            or batch_probe.get("status") != "passed"
            or batch_probe.get("action") != action
            or batch_probe.get("detector") != detector
            or batch_probe.get("dataset_sha256") != dataset_sha
            or batch_probe.get("fake_user_split_sha256") != split_sha
            or int(batch_probe.get("seed", -1)) != int(seed)
            or batch_probe.get("model_params", {}) != dict(deep_model_params or {})
            or int(batch_probe.get("selected_batch_size", -1)) != int(deep_config.batch_size)
            or batch_probe.get("truncation") is not False
            or batch_probe.get("resampling") is not False
        ):
            raise ValueError("batch probe identity/source/model/batch/no-truncation audit mismatch")
        if device is None or str(batch_probe.get("device")) != str(torch.device(device)):
            raise ValueError("batch probe device does not match Deep pair device")
    elif family == "feature_pad" and require_formal:
        # Feature PAD has no sequence-memory probe and must not inherit an
        # unrelated Deep artifact.
        batch_probe = None
    config_payload = {
        "action": action,
        "family": family,
        "detector": detector,
        "seed": int(seed),
        "base_seed": resolved_base_seed,
        "seed_policy": "sha256(base_seed|action|family|detector)_uint32",
        "formal_protocol": bool(require_formal),
        "real_hash_seed": int(real_hash_seed),
        "feature_bootstrap_replicates": int(feature_bootstrap_replicates),
        "deep_train": asdict(deep_config),
        "feature_model_params": dict(feature_model_params or {}),
        "deep_model_params": dict(deep_model_params or {}),
        "batch_probe": None if batch_probe_path is None else {
            "path": str(Path(batch_probe_path).resolve()),
            "sha256": sha256_file(batch_probe_path),
            "selected_batch_size": int(batch_probe["selected_batch_size"]),
            "longest_observed_train_event_length": int(batch_probe["longest_observed_train_event_length"]),
            "truncation": False,
            "resampling": False,
        },
    }
    config_sha = _config_digest(config_payload)
    deep_run_identity = (
        _build_deep_pair_input_identity(
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            fake_user_split=fake_user_split,
            fake_user_split_sha256=split_sha,
            real_hash_seed=real_hash_seed,
            action=action,
            detector=detector,
            pair_config=config_payload,
            pair_config_sha256=config_sha,
        )
        if family == "deep_pad" else None
    )
    expected_bootstrap_seed = int(seed) + (31 if family == "feature_pad" else 17)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audited = audit_protocol_result(
            result_dir, action=action, family=family, detector=detector,
            expected_bootstrap_replicates=expected_replicates,
            dataset_file=dataset_file, fake_user_split=fake_user_split,
            real_hash_seed=real_hash_seed,
            expected_dataset_sha256=dataset_sha,
            expected_fake_user_split_sha256=split_sha,
            expected_bootstrap_seed=expected_bootstrap_seed,
            expected_deep_run_identity=deep_run_identity,
        )
        if require_formal and family == "deep_pad":
            _require_formal_deep_completion(audited)
        if (
            manifest.get("schema_version") != PAIR_SCHEMA
            or manifest.get("status") != "complete"
            or (manifest.get("action"), manifest.get("family"), manifest.get("detector"))
            != (action, family, detector)
            or manifest.get("config") != config_payload
            or manifest.get("config_sha256") != config_sha
            or str(Path(manifest.get("dataset_file", "")).resolve())
            != str(Path(dataset_file).resolve())
            or manifest.get("dataset_sha256") != dataset_sha
            or str(Path(manifest.get("fake_user_split", "")).resolve())
            != str(Path(fake_user_split).resolve())
            or manifest.get("fake_user_split_sha256") != split_sha
            or str(Path(manifest.get("result_dir", "")).resolve())
            != str(result_dir.resolve())
            or manifest.get("artifact_hashes") != audited["artifact_hashes"]
            or manifest.get("dataset_relink_audit")
            != audited["dataset_relink_audit"]
            or manifest.get("operating_rows") != audited["rows"]
        ):
            raise ValueError("completed pair manifest/config/source/artifact audit mismatch")
        plot_path = pair_root / "test_fa_frr.png"
        if (
            str(Path(manifest.get("plot", "")).resolve()) != str(plot_path.resolve())
            or not plot_path.is_file()
            or manifest.get("plot_sha256") != sha256_file(plot_path)
        ):
            raise ValueError("completed pair plot path/hash audit mismatch")
        return {"status": "already_complete", "manifest": str(manifest_path), **audited}

    records, features, split_audit = _load_and_assign_action(
        dataset_file, fake_user_split, action, real_hash_seed, require_formal
    )
    if batch_probe is not None:
        observed_max = max(len(record.global_t_ms) for record in records if record.pool == "train")
        if int(batch_probe["longest_observed_train_event_length"]) != int(observed_max):
            raise ValueError("batch probe no longer matches the longest untruncated train event")
    archived_unbound_feature_result = None
    archived_unbound_deep_outputs = None
    if family == "feature_pad":
        archived_unbound_feature_result = _archive_unbound_feature_result(
            result_dir, pair_root,
            dataset_sha256=dataset_sha, config_sha256=config_sha,
        )
    else:
        _preflight_deep_checkpoint_identity(
            result_dir, action=action, detector=detector,
            train_config=deep_config, model_params=deep_model_params,
            input_identity=deep_run_identity,
        )
        archived_unbound_deep_outputs = _archive_unbound_deep_outputs(
            result_dir, pair_root,
        )
    summary_exists = (result_dir / "summary.json").exists()
    if not summary_exists:
        if family == "feature_pad":
            if result_dir.exists() and any(result_dir.iterdir()):
                raise RuntimeError(
                    "incomplete feature pair is not resumable; archive it explicitly before rerun: %s"
                    % result_dir
                )
            labels = np.asarray([row.label for row in records], dtype=np.int64)
            users = np.asarray([row.user_id for row in records], dtype=np.int64)
            pools = np.asarray([row.pool for row in records], dtype="U5")
            actions = np.full(len(records), action, dtype="U16")
            result = run_feature_pad_protocol(
                features, labels, users, pools, actions,
                action=action, detector_kind=detector,
                random_state=seed, model_params=feature_model_params,
                bootstrap_replicates=feature_bootstrap_replicates,
                bootstrap_seed=seed + 31,
            )
            save_protocol_outputs(result, result_dir)
        else:
            last_exists = (result_dir / "checkpoints" / "last.pt").exists()
            immutable_best_exists = any(
                (result_dir / "checkpoints").glob("best_epoch_*.pt")
            )
            if (
                result_dir.exists()
                and any(result_dir.iterdir())
                and not last_exists
                and not immutable_best_exists
            ):
                raise RuntimeError("deep partial result has no resumable last.pt: %s" % result_dir)
            run_deep_pad_protocol(
                records, action=action, detector_kind=detector,
                output_dir=result_dir, config=deep_config,
                model_params=deep_model_params, device=device,
                # A power loss may occur after the immutable best for epoch k
                # was atomically committed but before last.pt.  The Deep
                # runner recognizes only that exact transaction residue,
                # deterministically replays epoch k and compares every value.
                resume=last_exists or immutable_best_exists,
                run_identity=deep_run_identity,
            )

    audited = audit_protocol_result(
        result_dir, action=action, family=family, detector=detector,
        expected_bootstrap_replicates=expected_replicates,
        dataset_file=dataset_file, fake_user_split=fake_user_split,
        real_hash_seed=real_hash_seed,
        expected_dataset_sha256=dataset_sha,
        expected_fake_user_split_sha256=split_sha,
        expected_bootstrap_seed=expected_bootstrap_seed,
        expected_deep_run_identity=deep_run_identity,
    )
    if family == "deep_pad":
        if audited["summary"].get("train_config") != asdict(deep_config):
            raise ValueError("completed deep result train_config does not match requested pair config")
        if audited["summary"].get("model_params", {}) != dict(deep_model_params or {}):
            raise ValueError("completed deep result model_params do not match requested pair config")
        if require_formal:
            _require_formal_deep_completion(audited)
    else:
        if audited["summary"].get("model_params", {}) != dict(feature_model_params or {}):
            raise ValueError("completed feature result model_params do not match requested pair config")
        if int(audited["summary"].get("random_state", seed)) != int(seed):
            raise ValueError("completed feature result random_state does not match requested pair config")
    plot_path = pair_root / "test_fa_frr.png"
    _plot_curve(
        audited["test_curve"], audited["summary"]["test_metrics"],
        action, detector, plot_path,
    )
    manifest = {
        "schema_version": PAIR_SCHEMA,
        "status": "complete",
        "action": action,
        "family": family,
        "detector": detector,
        "dataset_file": str(Path(dataset_file).resolve()),
        "dataset_sha256": dataset_sha,
        "fake_user_split": str(Path(fake_user_split).resolve()),
        "fake_user_split_sha256": split_sha,
        "config": config_payload,
        "config_sha256": config_sha,
        "split_audit": split_audit,
        "result_dir": str(result_dir.resolve()),
        "plot": str(plot_path.resolve()),
        "plot_sha256": sha256_file(plot_path),
        "artifact_hashes": audited["artifact_hashes"],
        "dataset_relink_audit": audited["dataset_relink_audit"],
        "archived_unbound_feature_result": (
            None if archived_unbound_feature_result is None
            else str(archived_unbound_feature_result.resolve())
        ),
        "archived_unbound_deep_outputs": (
            None if archived_unbound_deep_outputs is None
            else str(archived_unbound_deep_outputs.resolve())
        ),
        "operating_rows": audited["rows"],
    }
    _atomic_json(manifest_path, manifest)
    return {"status": "completed", "manifest": str(manifest_path), **audited}


__all__ = [
    "PAIR_SCHEMA", "FAMILIES", "audit_protocol_result", "run_or_resume_pair",
    "sha256_file", "stable_pair_seed",
]
