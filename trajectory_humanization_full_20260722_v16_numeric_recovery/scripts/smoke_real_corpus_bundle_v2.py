#!/usr/bin/env python3
"""Non-formal five-action real-corpus adapter/bundle/Feature+Deep smoke.

Each invocation loads one authoritative HMOG v2 action archive, selects six
complete observed events, creates label-mirrored rows only to exercise the
binary detector API, persists/loads bundle v2, runs one Feature PAD protocol,
and forwards both raw Deep PAD architectures.  The mirrored rows are explicitly
not generator output and no metric from this script is a scientific result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.deep_pad import (
    ACTIONS,
    RawSequenceNormalizer,
    RawTCNPAD,
    RawTransformerPAD,
    collate_raw_sequences,
    load_raw_sequence_bundle,
    save_raw_sequence_bundle,
)
from detectors.feature_pad import run_feature_pad_protocol
from detectors.trajectory_adapter import load_extracted_trajectory_npz
from trajectory.features import TRAJECTORY_FEATURE_SCHEMA_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=ACTIONS, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_npz = args.output_dir / (args.action + ".npz")
    output_json = args.output_dir / (args.action + ".json")
    if output_npz.exists() or output_json.exists():
        raise FileExistsError("refusing to overwrite real-corpus smoke output")
    source = args.corpus_dir / ("hmog_trajectory_%s.npz" % args.action)
    records, features = load_extracted_trajectory_npz(
        source, label=0, default_pool="train", sample_prefix="real_smoke_source:"
    )
    if len(records) < 6 or len(features) != len(records):
        raise ValueError("real action corpus needs at least six aligned events")
    source_event_count = len(records)
    selected_records = records[:6]
    selected_features = features[:6].copy()
    del records, features
    gc.collect()

    smoke_records = []
    smoke_features = []
    pools = ("train", "train", "val", "val", "test", "test")
    for base, feature, pool in zip(selected_records, selected_features, pools):
        for label in (0, 1):
            smoke_records.append(replace(
                base,
                label=label,
                pool=pool,
                sample_id="%s:mirror_label%d" % (base.sample_id, label),
            ))
            smoke_features.append(feature.copy())
    smoke_feature_array = np.stack(smoke_features).astype(np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_raw_sequence_bundle(output_npz, smoke_records, smoke_feature_array)
    with np.load(output_npz, allow_pickle=False) as bundle:
        bundle_schema = str(bundle["schema_version"].item())
        feature_schema = str(bundle["feature_schema_version"].item())
    loaded, loaded_features = load_raw_sequence_bundle(output_npz)
    np.testing.assert_array_equal(loaded_features, smoke_feature_array)

    labels = np.asarray([row.label for row in loaded], np.int64)
    users = np.asarray([row.user_id for row in loaded], np.int64)
    pool_values = np.asarray([row.pool for row in loaded], dtype="U5")
    action_values = np.asarray([row.action for row in loaded], dtype="U16")
    feature_result = run_feature_pad_protocol(
        loaded_features, labels, users, pool_values, action_values,
        action=args.action, detector_kind="linear_svm", random_state=20260713,
        bootstrap_replicates=0,
    )

    train_rows = [row for row in loaded if row.pool == "train"]
    normalizer = RawSequenceNormalizer().fit(train_rows)
    batch = collate_raw_sequences(loaded, normalizer)
    deep = {}
    models = {
        "tcn": RawTCNPAD(hidden_dim=12, n_blocks=1, dropout=0.0),
        "transformer": RawTransformerPAD(
            hidden_dim=12, n_layers=1, n_heads=2,
            feedforward_dim=24, dropout=0.0,
        ),
    }
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            score = model(batch).detach().cpu().numpy()
        if score.shape != (len(loaded),) or not np.all(np.isfinite(score)):
            raise RuntimeError("%s Deep smoke produced invalid score" % name)
        deep[name] = {
            "n_scores": int(len(score)),
            "all_finite": True,
            "min": float(np.min(score)),
            "max": float(np.max(score)),
        }

    if bundle_schema != "trajectory_pad_bundle_v2":
        raise AssertionError("real-corpus smoke did not persist bundle v2")
    if feature_schema != TRAJECTORY_FEATURE_SCHEMA_VERSION:
        raise AssertionError("real-corpus smoke feature schema drift")
    report = {
        "schema_version": "real_corpus_bundle_v2_smoke_v1",
        "status": "passed",
        "formal_result": False,
        "metric_interpretation": (
            "pipeline-only; fake labels are exact mirrors of six real events and are not neural generation"
        ),
        "action": args.action,
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "source_event_count": int(source_event_count),
        "selected_real_event_group_ids": [row.event_group_id for row in selected_records],
        "bundle": str(output_npz.resolve()),
        "bundle_sha256": _sha256(output_npz),
        "bundle_schema": bundle_schema,
        "feature_schema_version": feature_schema,
        "feature_shape": list(loaded_features.shape),
        "feature_all_finite": bool(np.all(np.isfinite(loaded_features))),
        "feature_linear_svm": {
            "checkpoint_selection_pool": "validation_only",
            "thresholds": feature_result.thresholds,
            "validation_metrics": feature_result.validation_metrics,
            "test_metrics": feature_result.test_metrics,
        },
        "deep_forward": deep,
        "raw_timeline_lengths": [int(len(row.global_t_ms)) for row in loaded],
    }
    _atomic_json(output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
