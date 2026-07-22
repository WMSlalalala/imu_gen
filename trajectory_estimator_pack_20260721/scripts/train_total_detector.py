#!/usr/bin/env python3
"""Train a formal total IMU+trajectory detector from one numeric paired table."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from estimator.duration_metrics import duration_stratified_metrics, fit_duration_bins  # noqa: E402
from estimator.paired_dataset import PairedDetectorTable  # noqa: E402
from estimator.total_detector import (  # noqa: E402
    ALLOWED_MODELS,
    TotalDetectorArtifact,
    write_training_outputs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-kind", choices=ALLOWED_MODELS, default="logistic")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--duration-bins", type=int, default=4)
    args = parser.parse_args()

    table = PairedDetectorTable.load(args.dataset)
    selected = table.actions == str(args.action)
    if not np.any(selected):
        raise ValueError("requested action has no rows: %s" % args.action)
    if np.any(table.actions[selected] != str(args.action)):
        raise AssertionError("action selection failed")
    artifact = TotalDetectorArtifact.train(
        features=table.features[selected], labels=table.labels[selected],
        user_ids=table.user_ids[selected], pools=table.pools[selected],
        sample_ids=table.sample_ids[selected], action=str(args.action),
        feature_names=table.feature_names, model_kind=args.model_kind,
        seed=args.seed, bootstrap_replicates=args.bootstrap_replicates,
        require_consistency=True,
    )
    paths = write_training_outputs(args.output_dir, artifact)
    bin_spec = fit_duration_bins(
        table.duration_ms[selected & (table.pools == "train")], args.duration_bins
    )
    cache = artifact._training_cache
    duration_report = {
        "schema_version": "total_detector_duration_report_v1",
        "action": str(args.action),
        "bin_spec": bin_spec,
        "pools": {},
    }
    for pool in ("val", "test"):
        pool_mask = selected & (table.pools == pool)
        dump = cache["score_dumps"][pool]
        duration_report["pools"][pool] = duration_stratified_metrics(
            labels=dump["label"], scores=dump["score"],
            duration_ms=table.duration_ms[pool_mask],
            thresholds={
                "eer": artifact.thresholds["eer"],
                "val_frr_le_5pct": artifact.thresholds["val_frr_le_5pct"],
            },
            bin_spec=bin_spec, pool=pool,
        )
    duration_path = Path(args.output_dir) / "duration_stratified_metrics.json"
    _atomic_json(duration_path, duration_report)
    paths["duration_stratified_metrics"] = str(duration_path)
    manifest = {
        "schema_version": "total_detector_training_manifest_v1",
        "status": "complete",
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "action": str(args.action),
        "model_kind": args.model_kind,
        "seed": int(args.seed),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "duration_bins": int(args.duration_bins),
        "rows": int(selected.sum()),
        "feature_names": list(table.feature_names),
        "normalization_fit_pool": "train_only",
        "threshold_selection_pool": "validation_only",
        "test_role": "fixed_threshold_reporting_only",
        "paths": paths,
    }
    manifest_path = Path(args.output_dir) / "training_manifest.json"
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "complete", "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
