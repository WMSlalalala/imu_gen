"""Relink five formal trajectory PAD score dumps to exact paired events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .fake_imu_pairs import FAKE_IMU_PAIR_SCHEMA, sha256_file
from .paired_dataset_builder import COMPONENT_TABLE_SCHEMA, DetectorComponentTable
from .real_pair_index import REAL_PAIR_INDEX_SCHEMA


DETECTORS: Tuple[Tuple[str, str], ...] = (
    ("feature_pad", "linear_svm"),
    ("feature_pad", "rbf_svm"),
    ("feature_pad", "xgboost"),
    ("deep_pad", "tcn"),
    ("deep_pad", "transformer"),
)


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError("%s must be a scalar string" % name)
    return str(array.item())


def _load_bundle(path: Path, action: str) -> Dict[str, np.ndarray]:
    required = {
        "schema_version", "sequence_offsets", "flat_global_t_ms", "label", "user_id",
        "pool", "action", "sample_id",
    }
    with np.load(str(path), allow_pickle=False) as source:
        missing = required - set(source.files)
        if missing:
            raise ValueError("trajectory bundle lacks fields: %s" % sorted(missing))
        if _scalar_text(source["schema_version"], "schema_version") != "trajectory_pad_bundle_v2":
            raise ValueError("trajectory bundle schema mismatch")
        arrays = {name: np.asarray(source[name]) for name in required if name != "schema_version"}
    n = len(arrays["sample_id"])
    offsets = np.asarray(arrays["sequence_offsets"], dtype=np.int64)
    if offsets.shape != (n + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 2):
        raise ValueError("trajectory bundle offsets are invalid")
    if offsets[-1] != len(arrays["flat_global_t_ms"]):
        raise ValueError("trajectory bundle flat timeline disagrees with offsets")
    for name in ("label", "user_id", "pool", "action"):
        if np.asarray(arrays[name]).shape != (n,):
            raise ValueError("trajectory bundle %s length mismatch" % name)
    sample_ids = np.asarray(arrays["sample_id"]).astype(str)
    if np.any(sample_ids == "") or len(np.unique(sample_ids)) != n:
        raise ValueError("trajectory bundle sample ids must be unique and non-empty")
    actions = np.asarray(arrays["action"]).astype(str)
    if set(actions.tolist()) != {action}:
        raise ValueError("trajectory bundle action mismatch")
    pools = np.asarray(arrays["pool"]).astype(str)
    if set(pools.tolist()) != {"train", "val", "test"}:
        raise ValueError("trajectory bundle must cover train/val/test")
    labels = np.asarray(arrays["label"], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("trajectory bundle must contain real=0/fake=1")
    timeline = np.asarray(arrays["flat_global_t_ms"], dtype=np.float64)
    duration = timeline[offsets[1:] - 1] - timeline[offsets[:-1]]
    if not np.all(np.isfinite(duration)) or np.any(duration <= 0):
        raise ValueError("trajectory bundle contains invalid event durations")
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "user_ids": np.asarray(arrays["user_id"], dtype=np.int64),
        "pools": pools,
        "actions": actions,
        "duration_ms": duration,
    }


def _load_real_pairs(path: Path, action: str) -> Dict[int, Dict[str, Any]]:
    required = {
        "schema_version", "action", "sample_ids", "pair_identity_sha256", "labels",
        "user_ids", "pools", "duration_ms", "trajectory_event_ids",
    }
    with np.load(str(path), allow_pickle=False) as source:
        missing = required - set(source.files)
        if missing:
            raise ValueError("real pair index lacks fields: %s" % sorted(missing))
        if _scalar_text(source["schema_version"], "schema_version") != REAL_PAIR_INDEX_SCHEMA:
            raise ValueError("real pair index schema mismatch")
        if _scalar_text(source["action"], "action") != action:
            raise ValueError("real pair index action mismatch")
        arrays = {name: np.asarray(source[name]) for name in required - {"schema_version", "action"}}
    n = len(arrays["sample_ids"])
    if any(np.asarray(value).shape != (n,) for value in arrays.values()):
        raise ValueError("real pair index vector length mismatch")
    event_ids = np.asarray(arrays["trajectory_event_ids"], dtype=np.int64)
    if len(np.unique(event_ids)) != n or np.any(np.asarray(arrays["labels"]) != 0):
        raise ValueError("real pair trajectory ids must be unique and label=0")
    return {
        int(event_id): {
            "sample_id": str(arrays["sample_ids"][i]),
            "identity": str(arrays["pair_identity_sha256"][i]),
            "label": 0,
            "user": int(arrays["user_ids"][i]),
            "pool": str(arrays["pools"][i]),
            "duration": float(arrays["duration_ms"][i]),
        }
        for i, event_id in enumerate(event_ids.tolist())
    }


def _load_fake_pairs(root: Path, action: str, require_formal: bool) -> Tuple[Dict[str, Dict[str, Any]], Sequence[Path]]:
    paths = sorted((Path(root) / action).glob("user_*.npz"))
    if not paths or (require_formal and len(paths) != 100):
        raise ValueError("paired fake IMU unit count mismatch for %s" % action)
    result: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with np.load(str(path), allow_pickle=False) as source:
            required = {
                "schema_version", "action", "sample_ids", "event_plan_sha256", "user_ids",
                "pools", "duration_ms",
            }
            missing = required - set(source.files)
            if missing:
                raise ValueError("paired fake IMU unit lacks fields: %s" % sorted(missing))
            if _scalar_text(source["schema_version"], "schema_version") != FAKE_IMU_PAIR_SCHEMA:
                raise ValueError("paired fake IMU schema mismatch")
            if _scalar_text(source["action"], "action") != action:
                raise ValueError("paired fake IMU action mismatch")
            ids = np.asarray(source["sample_ids"]).astype(str)
            identities = np.asarray(source["event_plan_sha256"]).astype(str)
            users = np.asarray(source["user_ids"], dtype=np.int64)
            pools = np.asarray(source["pools"]).astype(str)
            durations = np.asarray(source["duration_ms"], dtype=np.float64)
        n = len(ids)
        if any(value.shape != (n,) for value in (identities, users, pools, durations)):
            raise ValueError("paired fake IMU metadata length mismatch")
        for i, sample_id in enumerate(ids.tolist()):
            if sample_id in result:
                raise ValueError("duplicate paired fake sample id")
            result[sample_id] = {
                "sample_id": sample_id, "identity": str(identities[i]), "label": 1,
                "user": int(users[i]), "pool": str(pools[i]), "duration": float(durations[i]),
            }
    if require_formal and len(result) != 20_000:
        raise ValueError("formal paired fake IMU action must contain 20,000 events")
    return result, paths


def _detector_scores(
    *, detector_root: Path, bundle: Mapping[str, np.ndarray], action: str,
) -> Tuple[np.ndarray, Tuple[str, ...], Sequence[Path]]:
    n = len(bundle["sample_ids"])
    id_to_row = {value: i for i, value in enumerate(bundle["sample_ids"].tolist())}
    common_rows = None
    columns = []
    names = []
    paths = []
    for family, detector in DETECTORS:
        path = Path(detector_root) / action / family / detector / "result" / "score_dump.npz"
        if not path.is_file():
            raise ValueError("trajectory detector score dump is missing: %s" % path)
        paths.append(path)
        observed_rows = []
        observed_scores = []
        with np.load(str(path), allow_pickle=False) as source:
            for pool in ("val", "test"):
                required = {
                    "%s_score" % pool, "%s_label" % pool, "%s_user_id" % pool,
                    "%s_pool" % pool, "%s_action" % pool,
                }
                identity_name = "%s_row_index" % pool if family == "feature_pad" else "%s_sample_id" % pool
                required.add(identity_name)
                missing = required - set(source.files)
                if missing:
                    raise ValueError("score dump lacks fields: %s" % sorted(missing))
                scores = np.asarray(source["%s_score" % pool], dtype=np.float64)
                if family == "feature_pad":
                    rows = np.asarray(source[identity_name], dtype=np.int64)
                else:
                    ids = np.asarray(source[identity_name]).astype(str)
                    try:
                        rows = np.asarray([id_to_row[value] for value in ids.tolist()], dtype=np.int64)
                    except KeyError as exc:
                        raise ValueError("deep score sample id is absent from trajectory bundle") from exc
                if rows.shape != scores.shape or rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= n):
                    raise ValueError("score dump row identities are invalid")
                if len(np.unique(rows)) != len(rows) or not np.all(np.isfinite(scores)):
                    raise ValueError("score dump contains duplicate rows or non-finite scores")
                exact = (
                    np.array_equal(np.asarray(source["%s_label" % pool], dtype=np.int64), bundle["labels"][rows]),
                    np.array_equal(np.asarray(source["%s_user_id" % pool], dtype=np.int64), bundle["user_ids"][rows]),
                    np.array_equal(np.asarray(source["%s_pool" % pool]).astype(str), bundle["pools"][rows]),
                    np.array_equal(np.asarray(source["%s_action" % pool]).astype(str), bundle["actions"][rows]),
                    bool(np.all(bundle["pools"][rows] == pool)),
                )
                if not all(exact):
                    raise ValueError("score dump metadata does not relink exactly to bundle")
                observed_rows.extend(rows.tolist())
                observed_scores.extend(scores.tolist())
        rows = np.asarray(observed_rows, dtype=np.int64)
        scores = np.asarray(observed_scores, dtype=np.float64)
        order = np.argsort(rows, kind="stable")
        rows, scores = rows[order], scores[order]
        if len(np.unique(rows)) != len(rows):
            raise ValueError("detector validation/test score rows overlap")
        expected = np.flatnonzero(np.isin(bundle["pools"], ["val", "test"]))
        if not np.array_equal(rows, expected):
            raise ValueError("detector scores do not cover the exact bundle validation/test rows")
        if common_rows is None:
            common_rows = rows
        elif not np.array_equal(rows, common_rows):
            raise ValueError("trajectory detectors cover different events")
        columns.append(scores)
        names.append("trajectory__%s_%s_score" % (family, detector))
    return np.stack(columns, axis=1), tuple(names), paths


def build_trajectory_score_component(
    *, action: str, bundle_path: Path, detector_root: Path, real_pair_index_path: Path,
    fake_imu_root: Path, output_path: Path, manifest_path: Path, require_formal: bool = True,
) -> Dict[str, Any]:
    """Build a val/test trajectory component with exact cross-modal identities."""

    bundle = _load_bundle(bundle_path, action)
    real = _load_real_pairs(real_pair_index_path, action)
    fake, fake_paths = _load_fake_pairs(fake_imu_root, action, require_formal)
    scores, feature_names, score_paths = _detector_scores(
        detector_root=detector_root, bundle=bundle, action=action
    )
    bundle_rows = np.flatnonzero(np.isin(bundle["pools"], ["val", "test"]))
    metadata = []
    kept_scores = []
    seen_real_events = set()
    seen_fake = set()
    for score_index, row in enumerate(bundle_rows.tolist()):
        label = int(bundle["labels"][row])
        bundle_id = str(bundle["sample_ids"][row])
        if label == 0:
            prefix = "real:%s:" % action
            if not bundle_id.startswith(prefix):
                raise ValueError("real trajectory bundle sample id has an unexpected format")
            try:
                event_id = int(bundle_id[len(prefix):])
            except ValueError as exc:
                raise ValueError("real trajectory bundle event id is not numeric") from exc
            if event_id not in real:
                continue  # A trajectory without an exact IMU match is intentionally excluded.
            item = real[event_id]
            seen_real_events.add(event_id)
        else:
            if bundle_id not in fake:
                raise ValueError("fake trajectory score has no exact paired IMU EventPlan")
            item = fake[bundle_id]
            seen_fake.add(bundle_id)
        if (
            item["label"] != label or item["user"] != int(bundle["user_ids"][row])
            or item["pool"] != str(bundle["pools"][row])
            or not np.isclose(item["duration"], bundle["duration_ms"][row], rtol=0.0, atol=1.0e-5)
        ):
            raise ValueError("paired identity metadata disagrees with trajectory bundle")
        metadata.append(item)
        kept_scores.append(scores[score_index])
    expected_real = {event_id for event_id, item in real.items() if item["pool"] in {"val", "test"}}
    expected_fake = {sample_id for sample_id, item in fake.items() if item["pool"] in {"val", "test"}}
    if seen_real_events != expected_real or seen_fake != expected_fake:
        raise ValueError("trajectory scores do not cover every paired validation/test event")
    features = np.asarray(kept_scores, dtype=np.float64)
    sample_ids = np.asarray([item["sample_id"] for item in metadata])
    order = np.argsort(sample_ids, kind="stable")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        str(temporary), schema_version=np.asarray(COMPONENT_TABLE_SCHEMA),
        component=np.asarray("trajectory"), features=features[order],
        feature_names=np.asarray(feature_names), sample_ids=sample_ids[order],
        labels=np.asarray([item["label"] for item in metadata], dtype=np.int64)[order],
        user_ids=np.asarray([item["user"] for item in metadata], dtype=np.int64)[order],
        pools=np.asarray([item["pool"] for item in metadata])[order],
        actions=np.full(len(metadata), action)[order],
        duration_ms=np.asarray([item["duration"] for item in metadata], dtype=np.float64)[order],
        pair_identity_sha256=np.asarray([item["identity"] for item in metadata])[order],
    )
    temporary.replace(output)
    table = DetectorComponentTable.load(output, "trajectory")
    report: Dict[str, Any] = {
        "schema_version": "paired_trajectory_score_component_manifest_v1",
        "status": "complete", "action": action, "rows": int(len(table.labels)),
        "features": int(table.features.shape[1]), "feature_names": list(table.feature_names),
        "pool_counts": {pool: int(np.sum(table.pools == pool)) for pool in ("val", "test")},
        "label_counts": {"real": int(np.sum(table.labels == 0)), "fake": int(np.sum(table.labels == 1))},
        "base_train_scores_used": False,
        "join_policy": "feature_row_index_or_deep_sample_id_then_exact_pair_identity",
        "unpaired_real_trajectory_events_excluded": int(
            np.sum((bundle["labels"][bundle_rows] == 0)) - len(expected_real)
        ),
        "bundle": str(Path(bundle_path).resolve()), "bundle_sha256": sha256_file(bundle_path),
        "real_pair_index": str(Path(real_pair_index_path).resolve()),
        "real_pair_index_sha256": sha256_file(real_pair_index_path),
        "fake_imu_units": len(fake_paths),
        "fake_imu_unit_sha256": {str(path.resolve()): sha256_file(path) for path in fake_paths},
        "score_dumps": {str(path.resolve()): sha256_file(path) for path in score_paths},
        "output": str(output.resolve()), "output_sha256": sha256_file(output),
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = target.with_name(target.name + ".tmp")
    temporary_manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(target)
    return report


__all__ = ["DETECTORS", "build_trajectory_score_component"]
