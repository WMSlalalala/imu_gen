"""Exact paired-event IMU feature cache and train-only level-1 scorers."""

from __future__ import annotations

import importlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .fake_imu_pairs import FAKE_IMU_PAIR_SCHEMA, sha256_file
from .paired_dataset_builder import COMPONENT_TABLE_SCHEMA, DetectorComponentTable
from .real_pair_index import REAL_PAIR_INDEX_SCHEMA


IMU_FEATURE_TABLE_SCHEMA = "paired_imu_feature_table_v1"
FORMAL_IMU_SCORERS = (
    "hmog_style_svm", "hmog_style_rf", "hmog_style_xgboost",
    "paper_svm", "paper_xgboost",
)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.%d.%s" % (os.getpid(), uuid.uuid4().hex))
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError("%s must be a scalar string" % name)
    return str(array.item())


def _feature_functions(pad_detectors_root: Path):
    root = Path(pad_detectors_root).resolve()
    required = ("hmog_features.py", "riskcog_features.py", "dsn17_features.py")
    if any(not (root / name).is_file() for name in required):
        raise ValueError("pad detector feature source is incomplete")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    hmog = importlib.import_module("hmog_features").hmog_features
    risk = importlib.import_module("riskcog_features").riskcog_features
    dsn = importlib.import_module("dsn17_features").dsn17_features
    return hmog, risk, dsn, {name: sha256_file(root / name) for name in required}


def _extract_feature_batches(
    sequences: Iterable[np.ndarray], *, count: int, pad_detectors_root: Path,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    if batch_size < 1:
        raise ValueError("feature batch size must be positive")
    hmog_fn, risk_fn, dsn_fn, source_hashes = _feature_functions(pad_detectors_root)
    hmog_parts: List[np.ndarray] = []
    paper_parts: List[np.ndarray] = []
    batch: List[np.ndarray] = []

    def consume(values: Sequence[np.ndarray]) -> None:
        lengths = np.asarray([len(value) for value in values], dtype=np.int64)
        if np.any(lengths < 1):
            raise ValueError("paired IMU event has no active frames")
        width = int(np.max(lengths))
        windows = np.zeros((len(values), width, 6), dtype=np.float32)
        mask = np.zeros((len(values), width), dtype=np.float32)
        for index, value in enumerate(values):
            array = np.asarray(value, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != 6 or not np.all(np.isfinite(array)):
                raise ValueError("paired IMU waveform must be finite [T,6]")
            windows[index, :len(array)] = array
            mask[index, :len(array)] = 1.0
        hmog = np.asarray(hmog_fn(windows, lengths, mask, hz=100.0), dtype=np.float32)
        risk = np.asarray(risk_fn(windows, lengths, mask, hz=100.0), dtype=np.float32)
        dsn = np.asarray(dsn_fn(windows, lengths, mask, hz=100.0), dtype=np.float32)
        paper = np.concatenate((hmog, risk, dsn), axis=1)
        if not np.all(np.isfinite(hmog)) or not np.all(np.isfinite(paper)):
            raise ValueError("paired IMU handcrafted features are non-finite")
        hmog_parts.append(hmog)
        paper_parts.append(paper)

    observed = 0
    for sequence in sequences:
        batch.append(sequence)
        observed += 1
        if len(batch) == batch_size:
            consume(batch)
            batch = []
    if batch:
        consume(batch)
    if observed != count or not hmog_parts:
        raise ValueError("paired IMU sequence count mismatch")
    return np.concatenate(hmog_parts), np.concatenate(paper_parts), source_hashes


def _real_source(pair_index_path: Path, imu_source_path: Path, action: str):
    with np.load(str(pair_index_path), allow_pickle=False) as pair:
        required = {
            "schema_version", "action", "sample_ids", "pair_identity_sha256", "labels",
            "user_ids", "pools", "duration_ms", "imu_row_offsets", "imu_rows",
        }
        missing = required - set(pair.files)
        if missing:
            raise ValueError("real pair index lacks fields: %s" % sorted(missing))
        if _scalar_text(pair["schema_version"], "schema_version") != REAL_PAIR_INDEX_SCHEMA:
            raise ValueError("real pair index schema mismatch")
        if _scalar_text(pair["action"], "action") != action:
            raise ValueError("real pair index action mismatch")
        arrays = {name: np.asarray(pair[name]) for name in required - {"schema_version", "action"}}
    n = len(arrays["sample_ids"])
    for name in ("pair_identity_sha256", "labels", "user_ids", "pools", "duration_ms"):
        if np.asarray(arrays[name]).shape != (n,):
            raise ValueError("real pair index %s length mismatch" % name)
    offsets = np.asarray(arrays["imu_row_offsets"], dtype=np.int64)
    rows = np.asarray(arrays["imu_rows"], dtype=np.int64)
    if offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(rows) or np.any(np.diff(offsets) <= 0):
        raise ValueError("real pair IMU offsets are invalid")
    if np.any(np.asarray(arrays["labels"], dtype=np.int64) != 0):
        raise ValueError("real pair index contains non-real labels")
    with np.load(str(imu_source_path), allow_pickle=False) as imu:
        required_imu = {"windows", "mask", "active_len", "hz"}
        missing = required_imu - set(imu.files)
        if missing:
            raise ValueError("real IMU source lacks fields: %s" % sorted(missing))
        windows = np.asarray(imu["windows"], dtype=np.float32)
        masks = np.asarray(imu["mask"], dtype=bool)
        active_len = np.asarray(imu["active_len"], dtype=np.int64)
        hz = float(np.asarray(imu["hz"]).item())
    if windows.ndim != 3 or windows.shape[2] != 6 or masks.shape != windows.shape[:2]:
        raise ValueError("real IMU source waveform/mask shapes are invalid")
    if active_len.shape != (len(windows),) or np.any(np.sum(masks, axis=1) != active_len):
        raise ValueError("real IMU source mask disagrees with active_len")
    if not np.isclose(hz, 100.0, rtol=0.0, atol=1.0e-6):
        raise ValueError("paired IMU scorer requires 100 Hz")
    if len(rows) and (np.min(rows) < 0 or np.max(rows) >= len(windows)):
        raise ValueError("real pair index references an out-of-range IMU row")

    def sequences():
        for index in range(n):
            selected = rows[offsets[index]:offsets[index + 1]]
            yield np.concatenate([windows[row][masks[row]] for row in selected], axis=0)

    metadata = {
        "sample_ids": np.asarray(arrays["sample_ids"]).astype(str),
        "pair_identity_sha256": np.asarray(arrays["pair_identity_sha256"]).astype(str),
        "labels": np.zeros(n, dtype=np.int64),
        "user_ids": np.asarray(arrays["user_ids"], dtype=np.int64),
        "pools": np.asarray(arrays["pools"]).astype(str),
        "duration_ms": np.asarray(arrays["duration_ms"], dtype=np.float64),
    }
    return metadata, sequences(), n


def _fake_source(fake_imu_root: Path, action: str, require_formal: bool):
    paths = sorted((Path(fake_imu_root) / action).glob("user_*.npz"))
    if not paths or (require_formal and len(paths) != 100):
        raise ValueError("paired fake IMU unit count mismatch")
    metadata = {name: [] for name in (
        "sample_ids", "pair_identity_sha256", "labels", "user_ids", "pools", "duration_ms"
    )}
    sequence_list = []
    for path in paths:
        with np.load(str(path), allow_pickle=False) as source:
            required = {
                "schema_version", "action", "sample_ids", "event_plan_sha256", "user_ids",
                "pools", "duration_ms", "imu_offsets", "flat_active_imu",
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
            offsets = np.asarray(source["imu_offsets"], dtype=np.int64)
            flat = np.asarray(source["flat_active_imu"], dtype=np.float32)
        n = len(ids)
        if any(value.shape != (n,) for value in (identities, users, pools, durations)):
            raise ValueError("paired fake IMU metadata length mismatch")
        if offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(flat) or np.any(np.diff(offsets) <= 0):
            raise ValueError("paired fake IMU offsets are invalid")
        sequence_list.extend([flat[offsets[i]:offsets[i + 1]] for i in range(n)])
        metadata["sample_ids"].extend(ids.tolist())
        metadata["pair_identity_sha256"].extend(identities.tolist())
        metadata["labels"].extend([1] * n)
        metadata["user_ids"].extend(users.tolist())
        metadata["pools"].extend(pools.tolist())
        metadata["duration_ms"].extend(durations.tolist())
    count = len(sequence_list)
    if require_formal and count != 20_000:
        raise ValueError("formal paired fake IMU action must contain 20,000 events")
    result = {
        "sample_ids": np.asarray(metadata["sample_ids"]),
        "pair_identity_sha256": np.asarray(metadata["pair_identity_sha256"]),
        "labels": np.asarray(metadata["labels"], dtype=np.int64),
        "user_ids": np.asarray(metadata["user_ids"], dtype=np.int64),
        "pools": np.asarray(metadata["pools"]),
        "duration_ms": np.asarray(metadata["duration_ms"], dtype=np.float64),
    }
    return result, iter(sequence_list), count, paths


def build_paired_imu_feature_table(
    *, action: str, pair_index_path: Path, real_imu_source_path: Path,
    fake_imu_root: Path, pad_detectors_root: Path, output_path: Path,
    manifest_path: Path, batch_size: int = 512, require_formal: bool = True,
) -> Dict[str, Any]:
    real_meta, real_sequences, n_real = _real_source(pair_index_path, real_imu_source_path, action)
    fake_meta, fake_sequences, n_fake, fake_paths = _fake_source(fake_imu_root, action, require_formal)
    real_hmog, real_paper, source_hashes = _extract_feature_batches(
        real_sequences, count=n_real, pad_detectors_root=pad_detectors_root, batch_size=batch_size
    )
    fake_hmog, fake_paper, observed_hashes = _extract_feature_batches(
        fake_sequences, count=n_fake, pad_detectors_root=pad_detectors_root, batch_size=batch_size
    )
    if source_hashes != observed_hashes:
        raise RuntimeError("IMU feature source changed during extraction")
    metadata = {name: np.concatenate((real_meta[name], fake_meta[name])) for name in real_meta}
    if len(np.unique(metadata["sample_ids"])) != n_real + n_fake:
        raise ValueError("paired IMU feature table sample ids are not globally unique")
    if set(metadata["pools"].tolist()) != {"train", "val", "test"}:
        raise ValueError("paired IMU feature table lacks train/val/test")
    if require_formal:
        fake_counts = {
            pool: int(np.sum((metadata["labels"] == 1) & (metadata["pools"] == pool)))
            for pool in ("train", "val", "test")
        }
        if fake_counts != {"train": 14_000, "val": 2_000, "test": 4_000}:
            raise ValueError("formal paired fake IMU split is not 14k/2k/4k")
    output = Path(output_path)
    _atomic_npz(output, {
        "schema_version": np.asarray(IMU_FEATURE_TABLE_SCHEMA), "action": np.asarray(action),
        "hmog_features": np.concatenate((real_hmog, fake_hmog)),
        "paper_features": np.concatenate((real_paper, fake_paper)),
        "actions": np.full(n_real + n_fake, action), **metadata,
    })
    # Strictly re-open before committing the manifest.
    table = load_paired_imu_feature_table(output, expected_action=action)
    report = {
        "schema_version": "paired_imu_feature_manifest_v1", "status": "complete",
        "action": action, "rows": int(len(table["labels"])), "real_rows": n_real,
        "fake_rows": n_fake, "hmog_features": int(table["hmog_features"].shape[1]),
        "paper_features": int(table["paper_features"].shape[1]),
        "feature_source_sha256": source_hashes,
        "pair_index": str(Path(pair_index_path).resolve()), "pair_index_sha256": sha256_file(pair_index_path),
        "real_imu_source": str(Path(real_imu_source_path).resolve()),
        "real_imu_source_sha256": sha256_file(real_imu_source_path),
        "fake_imu_units": {str(path.resolve()): sha256_file(path) for path in fake_paths},
        "output": str(output.resolve()), "output_sha256": sha256_file(output),
    }
    _write_json(manifest_path, report)
    return report


def load_paired_imu_feature_table(path: Path, *, expected_action: str) -> Dict[str, np.ndarray]:
    required = {
        "schema_version", "action", "hmog_features", "paper_features", "sample_ids",
        "pair_identity_sha256", "labels", "user_ids", "pools", "duration_ms", "actions",
    }
    with np.load(str(path), allow_pickle=False) as source:
        missing = required - set(source.files)
        if missing:
            raise ValueError("paired IMU feature table lacks fields: %s" % sorted(missing))
        if _scalar_text(source["schema_version"], "schema_version") != IMU_FEATURE_TABLE_SCHEMA:
            raise ValueError("paired IMU feature table schema mismatch")
        if _scalar_text(source["action"], "action") != expected_action:
            raise ValueError("paired IMU feature table action mismatch")
        arrays = {name: np.asarray(source[name]) for name in required - {"schema_version", "action"}}
    n = len(arrays["labels"])
    for name in ("sample_ids", "pair_identity_sha256", "user_ids", "pools", "duration_ms", "actions"):
        if arrays[name].shape != (n,):
            raise ValueError("paired IMU feature metadata length mismatch: %s" % name)
    for name in ("hmog_features", "paper_features"):
        if arrays[name].ndim != 2 or len(arrays[name]) != n or not np.all(np.isfinite(arrays[name])):
            raise ValueError("paired IMU %s are invalid" % name)
    arrays["sample_ids"] = arrays["sample_ids"].astype(str)
    arrays["pair_identity_sha256"] = arrays["pair_identity_sha256"].astype(str)
    arrays["pools"] = arrays["pools"].astype(str)
    arrays["actions"] = arrays["actions"].astype(str)
    arrays["labels"] = arrays["labels"].astype(np.int64)
    arrays["user_ids"] = arrays["user_ids"].astype(np.int64)
    arrays["duration_ms"] = arrays["duration_ms"].astype(np.float64)
    if len(np.unique(arrays["sample_ids"])) != n or set(arrays["labels"].tolist()) != {0, 1}:
        raise ValueError("paired IMU feature identities/labels are invalid")
    if set(arrays["actions"].tolist()) != {expected_action}:
        raise ValueError("paired IMU feature row action mismatch")
    return arrays


def _make_classifier(name: str, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    if name == "hmog_style_rf":
        return RandomForestClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    if name in {"hmog_style_svm", "paper_svm"}:
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=5000, random_state=seed),
        )
    if name in {"hmog_style_xgboost", "paper_xgboost"}:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss",
            random_state=seed, n_jobs=8, tree_method="hist",
        )
    raise ValueError("unsupported paired IMU scorer: %s" % name)


def _score(classifier: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        values = classifier.predict_proba(features)[:, 1]
    else:
        values = classifier.decision_function(features)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(features),) or not np.all(np.isfinite(values)):
        raise ValueError("paired IMU scorer returned invalid values")
    return values


def train_paired_imu_scorers(
    *, feature_table_path: Path, action: str, output_component_path: Path,
    artifact_path: Path, manifest_path: Path, seed: int = 42,
    scorers: Sequence[str] = FORMAL_IMU_SCORERS, require_formal: bool = True,
) -> Dict[str, Any]:
    names = tuple(str(value) for value in scorers)
    if not names or len(set(names)) != len(names):
        raise ValueError("paired IMU scorer list must be non-empty and unique")
    if require_formal and names != FORMAL_IMU_SCORERS:
        raise ValueError("formal paired IMU scorer set is fixed")
    table = load_paired_imu_feature_table(feature_table_path, expected_action=action)
    train = table["pools"] == "train"
    evaluate = np.isin(table["pools"], ["val", "test"])
    if set(table["labels"][train].tolist()) != {0, 1}:
        raise ValueError("paired IMU base train lacks real/fake rows")
    models = {}
    score_columns = []
    feature_names = []
    for name in names:
        key = "paper_features" if name.startswith("paper_") else "hmog_features"
        model = _make_classifier(name, seed)
        model.fit(table[key][train], table["labels"][train])
        score_columns.append(_score(model, table[key][evaluate]))
        feature_names.append("imu__%s_score" % name)
        models[name] = {"model": model, "feature_key": key}
    import joblib
    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = artifact.with_name(artifact.name + ".tmp")
    joblib.dump({
        "schema_version": "paired_imu_scorer_artifact_v1", "action": action,
        "seed": int(seed), "scorers": names, "models": models,
        "feature_table_sha256": sha256_file(feature_table_path),
        "fit_pool": "base_train_only", "scored_pools": ("base_val", "base_test"),
    }, temporary_artifact)
    temporary_artifact.replace(artifact)
    selected = np.flatnonzero(evaluate)
    component = Path(output_component_path)
    _atomic_npz(component, {
        "schema_version": np.asarray(COMPONENT_TABLE_SCHEMA), "component": np.asarray("imu"),
        "features": np.stack(score_columns, axis=1), "feature_names": np.asarray(feature_names),
        "sample_ids": table["sample_ids"][selected], "labels": table["labels"][selected],
        "user_ids": table["user_ids"][selected], "pools": table["pools"][selected],
        "actions": table["actions"][selected], "duration_ms": table["duration_ms"][selected],
        "pair_identity_sha256": table["pair_identity_sha256"][selected],
    })
    strict = DetectorComponentTable.load(component, "imu")
    report = {
        "schema_version": "paired_imu_scorer_training_manifest_v1", "status": "complete",
        "action": action, "scorers": list(names), "seed": int(seed),
        "training_rows": int(np.sum(train)), "scored_rows": int(np.sum(evaluate)),
        "normalization_and_model_fit_pool": "base_train_only",
        "output_pools": ["base_validation", "base_test"],
        "base_train_scores_exported": False,
        "feature_table": str(Path(feature_table_path).resolve()),
        "feature_table_sha256": sha256_file(feature_table_path),
        "artifact": str(artifact.resolve()), "artifact_sha256": sha256_file(artifact),
        "component": str(component.resolve()), "component_sha256": sha256_file(component),
        "component_rows": int(len(strict.labels)), "component_features": int(strict.features.shape[1]),
    }
    _write_json(manifest_path, report)
    return report


__all__ = [
    "IMU_FEATURE_TABLE_SCHEMA", "FORMAL_IMU_SCORERS", "build_paired_imu_feature_table",
    "load_paired_imu_feature_table", "train_paired_imu_scorers",
]
