#!/usr/bin/env python3
"""Fail-closed, resumable orchestration for the formal five-action run.

This file only coordinates the audited CLIs.  It does not implement model,
generation, export, detector or metric logic.  A stage is published complete
only after its own durable output gate passes.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_determinism import (
    EXPECTED_RUNTIME_DETERMINISM,
    STRICT_RUNTIME_DETERMINISM_SHA256,
    runtime_determinism_matches,
)

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
OPERATIONAL_CONFIG_KEYS = frozenset(("formal_launch_authorized",))
FORMAL_SOURCE_ROOTS = (
    "orchestration", "preprocess", "trajectory", "training",
    "generation", "detectors", "scripts", "tests",
)
FORMAL_SOURCE_FILES = ("runtime_determinism.py",)


def numeric_recovery_policy(config: Mapping[str, Any]) -> Dict[str, Any]:
    training = config["training"]
    return {
        "schema_version": "trajectory_amp_same_batch_retry_v1",
        "amp_enabled": True,
        "max_overflow_retries_per_batch": int(
            training["amp_overflow_max_retries"]
        ),
        "retry_same_batch": True,
        "restore_pre_attempt_rng": True,
        "count_examples_only_after_finite_optimizer_step": True,
        "update_ema_only_after_finite_optimizer_step": True,
        "scale_backoff_factor_source": "torch_grad_scaler_state",
    }


def throughput_projection_matches(
    value: Mapping[str, Any], measured_steps: int, warmup_steps: int
) -> bool:
    """Recompute the self-contained full-epoch projection contract."""
    try:
        dataset_count = int(value["dataset_target_count"])
        batch_size = int(value["batch_size"])
        if dataset_count <= 0 or batch_size <= 0:
            return False
        expected_batches = int(math.ceil(dataset_count / float(batch_size)))
        if value.get("profile_epoch_batch_counts") != [expected_batches] * 5:
            return False
        if int(value.get("profile_epoch_count", -1)) != 5:
            return False
        if value.get("profile_each_epoch_covers_dataset_once") is not True:
            return False
        if int(value.get("profile_target_occurrences", -1)) != 5 * dataset_count:
            return False
        if int(value.get("measured_optimizer_steps", -1)) != int(measured_steps):
            return False
        if int(value.get("projection_measurement_count", -1)) != int(measured_steps) + 1:
            return False
        if int(value.get("optimizer_state_initialization_steps", -1)) != int(warmup_steps):
            return False
        if value.get("optimizer_state_initialization_excluded_from_projection") is not True:
            return False
        shape_warmups = int(value.get("shape_specific_warmup_optimizer_steps", -1))
        if not (1 <= shape_warmups <= measured_steps + 1):
            return False
        if value.get("shape_specific_warmup_excluded_from_projection") is not True:
            return False
        if int(value.get("total_unmeasured_optimizer_steps", -1)) != warmup_steps + shape_warmups:
            return False
        if value.get("projection_has_extrapolation") is not False:
            return False
        if any(int(value.get(name, 0)) <= 0 for name in (
            "worst_case_padded_t", "worst_case_reference_padded_t",
            "worst_case_keycode_padded_k", "worst_case_reference_keycode_padded_k",
            "worst_case_padded_work",
        )):
            return False
        measurements = value.get("projection_measurements")
        if not isinstance(measurements, list) or len(measurements) != measured_steps + 1:
            return False
        worst = [row for row in measurements if row.get("label") == "artificial_global_worst_case"]
        if len(worst) != 1:
            return False
        if not math.isclose(
            float(worst[0]["elapsed_seconds"]),
            float(value["worst_case_elapsed_seconds"]),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            return False
        for row in measurements:
            if (
                float(row.get("elapsed_seconds", 0.0)) <= 0.0
                or int(row.get("padded_work", 0)) <= 0
                or int(row.get("batch_size", 0)) <= 0
                or int(row.get("target_padded_t", 0)) <= 0
                or int(row.get("reference_padded_t", 0)) <= 0
                or int(row.get("target_keycode_padded_k", 0)) <= 0
                or int(row.get("reference_keycode_padded_k", 0)) <= 0
            ):
                return False
        projection = value.get("epoch_projection")
        if not isinstance(projection, dict):
            return False
        if (
            projection.get("method")
            != "monotone_piecewise_linear_exact_t_tr_k_kr_padding_v2"
            or projection.get("projection_has_extrapolation") is not False
            or projection.get("profile_epochs") != [0, 1, 2, 3, 4]
        ):
            return False
        fit_x = [float(item) for item in projection.get("fit_padded_work", [])]
        fit_y = [float(item) for item in projection.get("fit_elapsed_seconds", [])]
        if (
            not fit_x or len(fit_x) != len(fit_y)
            or any(not math.isfinite(item) or item <= 0 for item in fit_x + fit_y)
            or any(right <= left for left, right in zip(fit_x, fit_x[1:]))
            or any(right < left for left, right in zip(fit_y, fit_y[1:]))
        ):
            return False
        epoch_seconds = [float(item) for item in projection.get("epoch_optimizer_seconds", [])]
        if len(epoch_seconds) != 5 or any(
            not math.isfinite(item) or item <= 0 for item in epoch_seconds
        ):
            return False
        mean_seconds = sum(epoch_seconds) / 5.0
        if not math.isclose(
            mean_seconds, float(projection.get("mean_epoch_optimizer_seconds", -1.0)),
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            return False
        if not math.isclose(
            mean_seconds, float(value.get("projected_full_epoch_optimizer_seconds", -1.0)),
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            return False
        expected_rate = dataset_count / mean_seconds
        if not math.isclose(
            expected_rate,
            float(value.get("projected_full_epoch_examples_per_second", -1.0)),
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            return False
        digest = str(value.get("epoch_length_profile_sha256", ""))
        return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def throughput_benchmark_config_matches(
    value: Mapping[str, Any], config: Mapping[str, Any], action: str,
    batch_size: int, measured_steps: int, warmup_steps: int,
) -> bool:
    training = config["training"]
    probe = config["throughput_probe"]
    expected = {
        "action": action,
        "device": str(probe["device"]),
        "batch_size": int(batch_size),
        "measured_steps": int(measured_steps),
        "warmup_steps": int(warmup_steps),
        "num_workers": int(training["num_workers"]),
        "seed": int(training["seed"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "grad_clip_norm": float(training["grad_clip_norm"]),
        "ema_decay": float(training["ema_decay"]),
        "diffusion_steps": int(training["diffusion_steps"]),
        "base_channels": int(training["base_channels"]),
        "cond_dim": int(training["cond_dim"]),
        "time_dim": int(training["time_dim"]),
        "n_blocks": int(training["n_blocks"]),
        "dropout": float(training["dropout"]),
        "keycode_vocab": int(training["keycode_vocab"]),
        "reference_cache_size": int(training["reference_cache_size"]),
        "amp": True,
        "optimizer": "AdamW",
        "profile_epochs": [0, 1, 2, 3, 4],
    }
    if value.get("benchmark_config") != expected:
        return False
    return value.get("benchmark_config_sha256") == canonical_sha256(expected)

STAGES = (
    "preflight",
    "corpus_audit",
    "e2e_smoke",
    "condition_preflight",
    "throughput_probe",
    "training_bootstrap",
    "training",
    "maps",
    "generation",
    "generation_audit",
    "detector_bundle",
    "detector_probes",
    "detector_pairs",
    "benchmark_merge",
    "benchmark_audit",
    "final_report",
)
SCRIPT_FLAGS = {
    "scripts/audit_training_corpus.py": ("--corpus-dir", "--split-json", "--output"),
    "scripts/train_trajectory_diffusion.py": (
        "--action", "--corpus-dir", "--output-dir", "--split-json", "--resume",
        "--epochs", "--diffusion-steps", "--keycode-vocab", "--device",
        "--amp-overflow-max-retries",
    ),
    "scripts/benchmark_training_throughput.py": (
        "--action", "--corpus-dir", "--split-json", "--output", "--device",
        "--batch-size", "--steps", "--warmup-steps", "--diffusion-steps",
        "--keycode-vocab",
    ),
    "scripts/migrate_v15_training_state.py": (
        "--config", "--selection", "--output",
    ),
    "scripts/run_v2_e2e_smoke.py": (
        "--corpus-dir", "--output-dir", "--split-json", "--device",
        "--users-per-pool", "--samples-per-user", "--optimizer-steps",
        "--reference-seed", "--generation-seed", "--confirm-quick-smoke",
    ),
    "scripts/preflight_all_condition_requests.py": (
        "--corpus-dir", "--split-json", "--output", "--reference-seed",
        "--generation-seed", "--samples-per-user-action", "--batch-size", "--workers",
    ),
    "scripts/generate_five_shot_trajectories.py": (
        "--corpus-dir", "--reference-registry-map", "--checkpoint-map",
        "--output-dir", "--num-shards", "--shard-id", "--confirm-formal-100k",
    ),
    "scripts/audit_five_shot_generation.py": (
        "--output-dir", "--corpus-dir", "--reference-registry-map", "--num-shards",
        "--condition-preflight",
    ),
    # The formal builder must consume generation archives directly.  The old
    # --fake-dir extractor-schema bridge is intentionally not accepted here.
    "scripts/build_trajectory_pad_bundle.py": (
        "--real-dir", "--fake-archive-dir", "--output-dir", "--fake-user-split",
        "--reference-registry-map", "--real-hash-seed",
    ),
    "scripts/probe_deep_batch_size.py": (
        "--dataset-dir", "--fake-user-split", "--action", "--detector", "--device",
        "--requested-batch-size", "--base-seed", "--real-hash-seed", "--output",
    ),
    "scripts/run_trajectory_pair.py": (
        "--dataset-dir", "--fake-user-split", "--output-root", "--action", "--family",
        "--detector", "--seed", "--bootstrap-replicates", "--epochs", "--patience",
        "--batch-probe-json",
    ),
    "scripts/merge_trajectory_pairs.py": ("--experiment-root",),
    "scripts/audit_trajectory_pair_merge.py": ("--experiment-root",),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def formal_source_snapshot(project: Path) -> Dict[str, Any]:
    """Hash every executable source file that can affect the formal run."""

    project = Path(project).resolve()
    files: Dict[str, str] = {}
    for root_name in FORMAL_SOURCE_ROOTS:
        root = project / root_name
        if not root.is_dir():
            raise FileNotFoundError("formal source root is missing: %s" % root)
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and path.suffix in (".py", ".sh")
                and "__pycache__" not in path.parts
            ):
                files[str(path.relative_to(project))] = sha256_file(path)
    for file_name in FORMAL_SOURCE_FILES:
        path = project / file_name
        if not path.is_file():
            raise FileNotFoundError("formal source file is missing: %s" % path)
        files[file_name] = sha256_file(path)
    if not files:
        raise RuntimeError("formal source snapshot is empty")
    return {
        "schema_version": "trajectory_formal_source_snapshot_v2",
        "roots": list(FORMAL_SOURCE_ROOTS),
        "root_files": list(FORMAL_SOURCE_FILES),
        "file_count": len(files),
        "files": files,
        "tree_sha256": canonical_sha256(files),
    }


def experiment_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash immutable experiment semantics, excluding launch authorization.

    ``formal_launch_authorized`` is an operational confirmation, not a model,
    data, metric, split, or resource-policy parameter.  A gate-only run must be
    able to bind durable state while it is false and later resume that exact
    state after an explicit false->true authorization change.  Every other
    config field remains part of the immutable identity.
    """

    identity = {key: value for key, value in config.items() if key not in OPERATIONAL_CONFIG_KEYS}
    return canonical_sha256(identity)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))
    directory_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def amp_overflow_events_valid(
    events: Any, completed_epoch: int, global_step: int, max_retries: int,
) -> bool:
    if not isinstance(events, list):
        return False
    per_batch: Dict[Tuple[int, int], int] = {}
    for event in events:
        if not isinstance(event, dict):
            return False
        key = (int(event.get("epoch_index", -1)), int(event.get("batch_index", -1)))
        if not (
            event.get("schema_version") == "trajectory_amp_same_batch_retry_v1"
            and key[0] == int(completed_epoch) - 1
            and key[1] >= 0
            and is_nonnegative_int(event.get("next_global_step"))
            and int(event["next_global_step"]) <= int(global_step)
            and is_nonnegative_int(event.get("retry_index"))
            and 1 <= int(event["retry_index"]) <= int(max_retries)
            and is_finite_positive(event.get("scale_before"))
            and is_finite_positive(event.get("scale_after"))
            and float(event["scale_after"]) < float(event["scale_before"])
            and event.get("same_batch") is True
            and event.get("pre_attempt_rng_restored") is True
        ):
            return False
        per_batch[key] = per_batch.get(key, 0) + 1
        if per_batch[key] > int(max_retries):
            return False
    return True


def is_finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def same_resolved_path(value: Any, expected: Path) -> bool:
    try:
        return Path(str(value)).expanduser().resolve() == expected.resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def bounded_e2e_optimizer_steps(requested: int, actual: int) -> bool:
    return int(requested) <= int(actual) <= 2 * int(requested)


def path_from(config: Mapping[str, Any], key: str) -> Path:
    return Path(str(config[key])).expanduser().resolve()


def _same_actions(value: Sequence[str]) -> bool:
    return tuple(value) == ACTIONS and len(set(value)) == len(ACTIONS)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "trajectory_formal_supervisor_config_v1":
        raise ValueError("unsupported supervisor config schema")
    if not _same_actions(config.get("actions", [])):
        raise ValueError("actions must be exactly %r in canonical order" % (ACTIONS,))
    gates = config.get("launch_gates", {})
    if int(gates.get("reference_seed", -1)) != 42 or int(gates.get("generation_seed", -1)) != 20260713:
        raise ValueError("launch gates must separate reference_seed=42 from generation_seed=20260713")
    if int(gates.get("samples_per_user_action", -1)) != 200:
        raise ValueError("condition launch gate must cover 200 requests/user/action")
    if int(gates.get("condition_workers", -1)) != 8:
        raise ValueError("formal condition preflight must use 8 deterministic fork workers")
    devices = tuple(config.get("devices", []))
    if devices != ("cuda:0", "cuda:1"):
        raise ValueError("formal config must expose exactly cuda:0 and cuda:1")
    action_device = config.get("action_device", {})
    if set(action_device) != set(ACTIONS) or set(action_device.values()) - set(devices):
        raise ValueError("action_device must map all five actions onto configured devices")
    training = config.get("training", {})
    if int(training.get("epochs", 0)) != 100:
        raise ValueError("formal training must be 100 epochs")
    if int(training.get("diffusion_steps", 0)) != 1000:
        raise ValueError("formal training diffusion schedule must be 1000 steps")
    if int(training.get("keycode_vocab", 0)) != 16384:
        raise ValueError("formal training must use the audited 16,384-token HMOG vocabulary")
    if int(training.get("seed", -1)) != 42:
        raise ValueError("formal training seed must remain 42")
    if int(training.get("amp_overflow_max_retries", -1)) != 4:
        raise ValueError("formal AMP same-batch recovery must allow exactly four retries")
    bootstrap = config.get("training_bootstrap", {})
    if (
        bootstrap.get("schema_version")
        != "trajectory_v15_to_v16_training_bootstrap_config_v1"
        or list(bootstrap.get("actions", {})) != ["tap", "scroll"]
        or not is_sha256(bootstrap.get("source_tree_sha256"))
        or any(
            not is_sha256(bootstrap.get("actions", {}).get(action, {}).get(
                "last_checkpoint_sha256"
            ))
            for action in ("tap", "scroll")
        )
    ):
        raise ValueError("formal v16 bootstrap identity is incomplete")
    probe = config.get("throughput_probe", {})
    if tuple(int(x) for x in probe.get("candidate_batch_sizes", [])) != (32, 64, 128, 256):
        raise ValueError("throughput candidates must be exactly 32/64/128/256")
    if int(probe.get("selected_measured_steps", 0)) != 100:
        raise ValueError("selected throughput batch must run 100 measured optimizer steps")
    if probe.get("selection_metric") != "projected_full_epoch_examples_per_second":
        raise ValueError(
            "throughput selection metric must cover the projected full epoch"
        )
    if not (60 <= float(probe.get("candidate_wall_time_limit_seconds", 0)) <= 600):
        raise ValueError("candidate throughput runtime limit must be 60..600 seconds")
    if float(probe.get("gpu_safety_margin_gib", 0)) < 4:
        raise ValueError("GPU safety margin is too small for the formal run")
    if probe.get("device") != "cuda:0":
        raise ValueError("training throughput must be compared serially on clean cuda:0")
    generation = config.get("generation", {})
    if int(generation.get("num_shards", 0)) != 2:
        raise ValueError("formal supervisor requires two generation shards")
    if int(generation.get("samples_per_user_action", 0)) != 200:
        raise ValueError("formal generation requires 200 fake/user/action")
    if int(generation.get("total_fake", 0)) != 100000 or int(generation.get("ddim_steps", 0)) != 50:
        raise ValueError("formal generation must be 100,000 fake with 50-step DDIM")
    if int(generation.get("seed", -1)) != 20260713:
        raise ValueError("formal generation seed must remain 20260713")
    if int(generation.get("batch_size", 0)) != 32:
        raise ValueError("formal generation and condition preflight batch size must remain 32")
    supervision = config.get("supervision", {})
    if int(supervision.get("poll_seconds", 0)) <= 0:
        raise ValueError("supervisor poll_seconds must be positive")
    if int(supervision.get("training_progress_stale_seconds", 0)) != 600:
        raise ValueError("formal worker progress stale threshold must remain 600 seconds")
    detector = config.get("detector", {})
    if tuple(detector.get("feature_detectors", [])) != ("linear_svm", "rbf_svm", "xgboost"):
        raise ValueError("formal Feature PAD detector set changed")
    if tuple(detector.get("deep_detectors", [])) != ("tcn", "transformer"):
        raise ValueError("formal Deep PAD detector set changed")
    if int(detector.get("bootstrap_replicates", 0)) != 500:
        raise ValueError("formal benchmark requires 500 user-level bootstrap replicates")
    if int(detector.get("epochs", 0)) != 40 or int(detector.get("patience", -1)) != 0:
        raise ValueError("formal Deep PAD must run all 40 epochs with patience=0")
    if int(detector.get("feature_cpu_concurrency", 0)) != 2:
        raise ValueError("formal Feature PAD CPU concurrency is fixed at two")
    if int(detector.get("deep_gpu_concurrency_per_device", 0)) != 1:
        raise ValueError("formal Deep PAD permits one long job per GPU")
    project = path_from(config, "project_root")
    formal_config = path_from(config, "formal_config_path")
    run_root = path_from(config, "run_root")
    corpus = path_from(config, "corpus_dir")
    gate_root = path_from(config, "gate_root")
    if run_root == corpus or str(run_root).startswith(str(corpus) + os.sep):
        raise ValueError("run_root cannot equal or live inside corpus_dir")
    if not str(run_root).startswith(str(project) + os.sep):
        raise ValueError("run_root must live inside the isolated project")
    if not str(gate_root).startswith(str(project) + os.sep) or gate_root == run_root:
        raise ValueError("gate_root must be a separate directory inside the isolated project")
    if not str(formal_config).startswith(str(project) + os.sep):
        raise ValueError("formal_config_path must live inside the isolated project")
    if path_from(bootstrap, "source_run_root") == run_root:
        raise ValueError("bootstrap source and target run roots must differ")


def paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    run = path_from(config, "run_root")
    gate_root = path_from(config, "gate_root")
    return {
        "project": path_from(config, "project_root"),
        "formal_config": path_from(config, "formal_config_path"),
        "python": path_from(config, "python"),
        "corpus": path_from(config, "corpus_dir"),
        "split": path_from(config, "split_json"),
        "run": run,
        "gate_root": gate_root,
        "e2e_smoke": gate_root / "v2_e2e_smoke",
        "condition_preflight": gate_root / "all_condition_requests.json",
        "logs": run / "logs",
        "state": run / "supervisor_status.json",
        "lock": run / "supervisor.lock",
        "stop": run / "STOP_REQUESTED",
        "command_manifest": run / "command_manifest.json",
        "corpus_audit": run / "audits" / "corpus_audit.json",
        "probe": run / "throughput_probe",
        "probe_selection": run / "manifests" / "throughput_selection.json",
        "training_bootstrap_receipt": (
            run / "manifests" / "v15_to_v16_training_bootstrap.json"
        ),
        "training": run / "training",
        "checkpoint_map": run / "manifests" / "checkpoint_map.json",
        "registry_map": run / "manifests" / "reference_registry_map.json",
        "generation": run / "generation",
        "bundle": run / "detector_bundle",
        "benchmark": run / "benchmark",
        "benchmark_merged": run / "benchmark" / "merged",
        "final_report": run / "FINAL_REPORT.md",
        "final_audit": run / "final_audit.json",
        "orphaned": run / "orphaned_attempts",
    }


def command_manifest(
    config: Mapping[str, Any], selected_batches: Optional[Mapping[str, int]] = None
) -> Dict[str, Any]:
    p = paths(config)
    py = str(p["python"])
    project = p["project"]
    split = str(p["split"])
    corpus = str(p["corpus"])
    training = config["training"]
    generation = config["generation"]
    detector = config["detector"]
    probe = config["throughput_probe"]
    gates = config["launch_gates"]
    source_code = formal_source_snapshot(project)
    train_commands = {}
    for action in ACTIONS:
        out = p["training"] / action
        train_commands[action] = [
            py, "-u", str(project / "scripts/train_trajectory_diffusion.py"),
            "--action", action, "--corpus-dir", corpus, "--output-dir", str(out),
            "--split-json", split, "--epochs", str(training["epochs"]),
            "--batch-size", str(
                selected_batches[action] if selected_batches is not None
                else "{PROBE_SELECTED_BATCH_SIZE_%s}" % action.upper()
            ),
            "--learning-rate", str(training["learning_rate"]),
            "--weight-decay", str(training["weight_decay"]),
            "--grad-clip-norm", str(training["grad_clip_norm"]),
            "--ema-decay", str(training["ema_decay"]),
            "--diffusion-steps", str(training["diffusion_steps"]),
            "--base-channels", str(training["base_channels"]),
            "--cond-dim", str(training["cond_dim"]), "--time-dim", str(training["time_dim"]),
            "--n-blocks", str(training["n_blocks"]), "--dropout", str(training["dropout"]),
            "--keycode-vocab", str(training["keycode_vocab"]),
            "--seed", str(training["seed"]), "--num-workers", str(training["num_workers"]),
            "--checkpoint-every-steps", str(training["checkpoint_every_steps"]),
            "--reference-cache-size", str(training["reference_cache_size"]),
            "--amp-overflow-max-retries", str(training["amp_overflow_max_retries"]),
            "--device", str(config["action_device"][action]),
        ]
    generation_commands = {}
    for shard in range(int(generation["num_shards"])):
        generation_commands[str(shard)] = [
            py, "-u", str(project / "scripts/generate_five_shot_trajectories.py"),
            "--corpus-dir", corpus, "--split-json", split,
            "--reference-registry-map", str(p["registry_map"]),
            "--checkpoint-map", str(p["checkpoint_map"]),
            "--output-dir", str(p["generation"]),
            "--device", str(config["devices"][shard]),
            "--batch-size", str(generation["batch_size"]), "--seed", str(generation["seed"]),
            "--num-shards", str(generation["num_shards"]), "--shard-id", str(shard),
            "--confirm-formal-100k",
        ]
    probe_commands = {}
    wrapper = project / "orchestration/probe_candidate.py"
    throughput_script = project / "scripts/benchmark_training_throughput.py"
    for action in ACTIONS:
        action_commands = {}
        for batch_size in probe["candidate_batch_sizes"]:
            output = p["probe"] / action / ("candidate_bs%03d.json" % int(batch_size))
            action_commands[str(batch_size)] = [
                py, "-u", str(wrapper), "--result", str(output),
                "--timeout-seconds", str(probe["candidate_wall_time_limit_seconds"]), "--",
                py, "-u", str(throughput_script), "--action", action,
                "--corpus-dir", corpus, "--split-json", split, "--output", str(output),
                "--device", str(probe["device"]), "--batch-size", str(batch_size),
                "--steps", str(probe["candidate_measured_steps"]),
                "--warmup-steps", str(probe["candidate_warmup_steps"]),
                "--num-workers", str(training["num_workers"]), "--seed", str(training["seed"]),
                "--learning-rate", str(training["learning_rate"]), "--weight-decay", str(training["weight_decay"]),
                "--grad-clip-norm", str(training["grad_clip_norm"]), "--ema-decay", str(training["ema_decay"]),
                "--diffusion-steps", str(training["diffusion_steps"]), "--base-channels", str(training["base_channels"]),
                "--cond-dim", str(training["cond_dim"]), "--time-dim", str(training["time_dim"]),
                "--n-blocks", str(training["n_blocks"]), "--dropout", str(training["dropout"]),
                "--keycode-vocab", str(training["keycode_vocab"]),
                "--reference-cache-size", str(training["reference_cache_size"]),
            ]
        probe_commands[action] = action_commands
    deep_probe_commands = {}
    pair_templates = {}
    deep_index = 0
    feature_index = 0
    for action in ACTIONS:
        for family, detector_names in (
            ("feature_pad", detector["feature_detectors"]),
            ("deep_pad", detector["deep_detectors"]),
        ):
            for detector_name in detector_names:
                identity = "%s/%s/%s" % (action, family, detector_name)
                if family == "deep_pad":
                    device = str(config["devices"][deep_index % len(config["devices"])])
                    deep_index += 1
                    probe_output = p["benchmark"] / "probes" / action / (detector_name + ".json")
                    deep_probe_commands[identity] = [
                        py, "-u", str(project / "scripts/probe_deep_batch_size.py"),
                        "--dataset-dir", str(p["bundle"]), "--fake-user-split", split,
                        "--action", action, "--detector", detector_name, "--device", device,
                        "--requested-batch-size", str(detector["requested_deep_batch_size"]),
                        "--base-seed", str(detector["seed"]),
                        "--real-hash-seed", str(detector["real_hash_seed"]),
                        "--model-params-json", "{}", "--output", str(probe_output),
                    ]
                    resource = device
                    batch_size = "{DEEP_PROBE_SELECTED_BATCH_SIZE_%s_%s}" % (
                        action.upper(), detector_name.upper()
                    )
                else:
                    device = None
                    probe_output = None
                    resource = "cpu:%d" % (feature_index % int(detector["feature_cpu_concurrency"]))
                    feature_index += 1
                    batch_size = str(detector["batch_size"])
                command = [
                    py, "-u", str(project / "scripts/run_trajectory_pair.py"),
                    "--dataset-dir", str(p["bundle"]), "--fake-user-split", split,
                    "--output-root", str(p["benchmark"]), "--action", action,
                    "--family", family, "--detector", detector_name,
                    "--seed", str(detector["seed"]),
                    "--real-hash-seed", str(detector["real_hash_seed"]),
                    "--bootstrap-replicates", str(detector["bootstrap_replicates"]),
                    "--epochs", str(detector["epochs"]), "--batch-size", batch_size,
                    "--learning-rate", "0.0003", "--weight-decay", "0.0001",
                    "--patience", str(detector["patience"]), "--num-workers", "0",
                    "--gradient-clip-norm", "5", "--model-params-json", "{}",
                ]
                if family == "deep_pad":
                    command.extend(("--device", str(device), "--batch-probe-json", str(probe_output)))
                pair_templates[identity] = {"resource": resource, "command": command}
    commands = {
        "corpus_audit": [
            py, "-u", str(project / "scripts/audit_training_corpus.py"),
            "--corpus-dir", corpus, "--split-json", split, "--output", str(p["corpus_audit"]),
        ],
        "e2e_smoke": [
            py, "-u", str(project / "scripts/run_v2_e2e_smoke.py"),
            "--corpus-dir", corpus, "--output-dir", str(p["e2e_smoke"]),
            "--split-json", split, "--device", str(gates["e2e_device"]),
            "--users-per-pool", str(gates["e2e_users_per_pool"]),
            "--samples-per-user", str(gates["e2e_samples_per_user"]),
            "--optimizer-steps", str(gates["e2e_optimizer_steps"]),
            "--reference-seed", str(gates["reference_seed"]),
            "--generation-seed", str(gates["generation_seed"]),
            "--confirm-quick-smoke",
        ],
        "condition_preflight": [
            py, "-u", str(project / "scripts/preflight_all_condition_requests.py"),
            "--corpus-dir", corpus, "--split-json", split,
            "--output", str(p["condition_preflight"]),
            "--reference-seed", str(gates["reference_seed"]),
            "--generation-seed", str(gates["generation_seed"]),
            "--samples-per-user-action", str(gates["samples_per_user_action"]),
            "--batch-size", str(generation["batch_size"]),
            "--workers", str(gates["condition_workers"]),
        ],
        "training_bootstrap": [
            py, "-u", str(project / "scripts/migrate_v15_training_state.py"),
            "--config", str(p["formal_config"]),
            "--selection", str(p["probe_selection"]),
            "--output", str(p["training_bootstrap_receipt"]),
        ],
        "throughput_probe_candidates": probe_commands,
        "training": train_commands,
        "generation": generation_commands,
        "generation_audit": [
            py, "-u", str(project / "scripts/audit_five_shot_generation.py"),
            "--output-dir", str(p["generation"]), "--corpus-dir", corpus,
            "--split-json", split, "--reference-registry-map", str(p["registry_map"]),
            "--num-shards", str(generation["num_shards"]),
            "--condition-preflight", str(p["condition_preflight"]),
        ],
        "detector_bundle": [
            py, "-u", str(project / "scripts/build_trajectory_pad_bundle.py"),
            "--real-dir", corpus, "--fake-archive-dir", str(p["generation"]),
            "--output-dir", str(p["bundle"]), "--fake-user-split", split,
            "--reference-registry-map", str(p["registry_map"]),
            "--real-hash-seed", str(detector["real_hash_seed"]),
        ],
        "detector_deep_probes": deep_probe_commands,
        "detector_pair_templates": pair_templates,
        "benchmark_merge": [
            py, "-u", str(project / "scripts/merge_trajectory_pairs.py"),
            "--experiment-root", str(p["benchmark"]),
        ],
        "benchmark_audit": [
            py, "-u", str(project / "scripts/audit_trajectory_pair_merge.py"),
            "--experiment-root", str(p["benchmark"]),
        ],
    }
    return {
        "schema_version": "trajectory_formal_command_manifest_v1",
        "config_sha256": experiment_config_sha256(config),
        "config_identity_excludes": sorted(OPERATIONAL_CONFIG_KEYS),
        "source_code": source_code,
        "training_commands_finalized": selected_batches is not None,
        "fixed_split_sha256": config["split_sha256"],
        "formal_invariants": {
            "actions": list(ACTIONS), "epochs_per_action": 100,
            "validation_fractions": [0.2, 0.4, 0.6, 0.8, 1.0],
            "training_diffusion_steps": 1000, "fixed_references": 5,
            "training_protocol": "trajectory_diffusion_strict_five_ref_v2",
            "runtime_determinism": dict(EXPECTED_RUNTIME_DETERMINISM),
            "runtime_determinism_sha256": STRICT_RUNTIME_DETERMINISM_SHA256,
            "keycode_vocab": int(training["keycode_vocab"]),
            "fake_per_user_action": 200, "total_fake": 100000,
            "ddim_steps": 50, "ddim_eta": 0.0, "selector_used": False,
            "feature_detectors": ["linear_svm", "rbf_svm", "xgboost"],
            "deep_detectors": ["tcn", "transformer"],
            "detector_pairs": 25,
            "detector_operating_rows": 50,
            "deep_epochs": 40,
            "deep_patience": 0,
            "feature_cpu_concurrency": 2,
            "deep_gpu_concurrency_per_device": 1,
            "training_batch_size_source": "real_train_optimizer_throughput_probe",
            "throughput_candidates": [32, 64, 128, 256],
            "selected_batch_measured_steps": 100,
            "reference_seed": 42,
            "training_seed": 42,
            "amp_overflow_max_retries": int(training["amp_overflow_max_retries"]),
            "amp_overflow_retry_same_batch_same_rng": True,
            "generation_seed": 20260713,
            "generation_batch_size": 32,
            "generation_archive_schema": [1, 5],
            "generation_shard_manifest_schema":
            "five_shot_generation_shard_manifest_v4",
            "generation_formal_audit_schema":
            "five_shot_generation_formal_audit_v4",
            "condition_request_seed_derivation":
            "stable_seed(base_seed,action,user_id,sample_index)",
            "ddim_noise_seed_derivation":
            "stable_seed(condition_request_seed_xor_0xDD1A50,action,user_id,sample_index)",
            "condition_requests": 100000,
            "keystroke_hard_timeline_requests": 20000,
            "condition_request_digest_schema": "trajectory_condition_request_canonical_v1",
            "condition_set_digest_schema": "trajectory_condition_request_set_v1",
        },
        "selected_batch_size_by_action": None if selected_batches is None else dict(selected_batches),
        "commands": commands,
    }


def check_cli_contract(config: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    p = paths(config)
    checks, blockers = [], []
    for relative, required_flags in SCRIPT_FLAGS.items():
        script = p["project"] / relative
        record = {"script": str(script), "exists": script.is_file(), "required_flags": list(required_flags)}
        if not script.is_file():
            blockers.append("missing script: %s" % script)
            record["passed"] = False
            checks.append(record)
            continue
        result = subprocess.run(
            [str(p["python"]), str(script), "--help"], cwd=str(p["project"]),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
        )
        missing = [flag for flag in required_flags if flag not in result.stdout]
        record.update({"returncode": result.returncode, "missing_flags": missing, "passed": result.returncode == 0 and not missing})
        if not record["passed"]:
            blockers.append("CLI contract mismatch: %s missing=%r rc=%d" % (relative, missing, result.returncode))
        checks.append(record)
    return checks, blockers


def preflight(config: Mapping[str, Any], require_runtime_inputs: bool) -> Dict[str, Any]:
    validate_config(config)
    p = paths(config)
    blockers = []
    checks, cli_blockers = check_cli_contract(config)
    blockers.extend(cli_blockers)
    if not p["python"].is_file():
        blockers.append("configured Python does not exist")
    if not p["formal_config"].is_file():
        blockers.append("formal_config_path does not exist")
    else:
        try:
            if read_json(p["formal_config"]) != dict(config):
                blockers.append("formal_config_path bytes do not describe the loaded config")
        except (OSError, TypeError, ValueError):
            blockers.append("formal_config_path is not a valid config object")
    if not p["split"].is_file():
        blockers.append("fixed split JSON is missing")
    elif sha256_file(p["split"]) != config["split_sha256"]:
        blockers.append("fixed split SHA-256 mismatch")
    bootstrap = config["training_bootstrap"]
    bootstrap_source = path_from(bootstrap, "source_run_root")
    source_state_path = bootstrap_source / "supervisor_status.json"
    if not source_state_path.is_file():
        blockers.append("v15 bootstrap source supervisor status is missing")
    else:
        try:
            source_state = read_json(source_state_path)
            if (
                source_state.get("status") != "failed"
                or source_state.get("source_tree_sha256")
                != bootstrap["source_tree_sha256"]
            ):
                blockers.append("v15 bootstrap source failed-run identity mismatch")
        except (OSError, TypeError, ValueError):
            blockers.append("v15 bootstrap source status is invalid")
    for action in ("tap", "scroll"):
        source_last = bootstrap_source / "training" / action / "last.pt"
        if not source_last.is_file():
            blockers.append("v15 bootstrap source last.pt missing: %s" % action)
        elif sha256_file(source_last) != bootstrap["actions"][action]["last_checkpoint_sha256"]:
            blockers.append("v15 bootstrap source last.pt SHA mismatch: %s" % action)
    corpus_files = [p["corpus"] / ("hmog_trajectory_%s.npz" % action) for action in ACTIONS]
    missing_corpus = [str(path) for path in corpus_files if not path.is_file()]
    if missing_corpus:
        blockers.append("corrected v2 corpus is not complete (%d action archives missing)" % len(missing_corpus))
    free_gib = shutil.disk_usage(str(p["project"])).free / float(1024 ** 3)
    minimum = float(config["supervision"]["minimum_free_disk_gib"])
    if free_gib < minimum:
        blockers.append("free disk %.2f GiB is below formal gate %.2f GiB" % (free_gib, minimum))
    if require_runtime_inputs and config.get("formal_launch_authorized") is not True:
        blockers.append("formal_launch_authorized=false; root confirmation is still required")
    authorized = config.get("formal_launch_authorized") is True
    report = {
        "passed": not blockers,
        "ready_for_gates": not blockers,
        "ready_to_launch": not blockers and authorized,
        "formal_launch_authorized": authorized,
        "runtime_inputs_required": bool(require_runtime_inputs),
        "blockers": blockers,
        "cli_checks": checks,
        "free_disk_gib": free_gib,
        "corpus_files_present": len(corpus_files) - len(missing_corpus),
        "config_sha256": experiment_config_sha256(config),
        "config_identity_excludes": sorted(OPERATIONAL_CONFIG_KEYS),
        "source_code": formal_source_snapshot(p["project"]),
    }
    if require_runtime_inputs and blockers:
        raise RuntimeError("preflight failed:\n- " + "\n- ".join(blockers))
    return report


def initial_state(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "trajectory_formal_supervisor_status_v1",
        "config_sha256": experiment_config_sha256(config),
        "config_identity_excludes": sorted(OPERATIONAL_CONFIG_KEYS),
        "source_tree_sha256": formal_source_snapshot(path_from(config, "project_root"))[
            "tree_sha256"
        ],
        "status": "pending", "current_stage": None,
        "created_unix_time": time.time(), "updated_unix_time": time.time(),
        "supervisor_pid": os.getpid(),
        "stages": {stage: {"status": "pending"} for stage in STAGES},
        "jobs": {}, "history": [],
    }


def gate_review_recorded(state: Mapping[str, Any]) -> bool:
    """Return whether a durable gates-only review point exists.

    This is intentionally only the structural, pre-construction check.  The
    Supervisor subsequently revalidates every gate and every recorded hash
    before it mutates formal state or starts throughput/training.
    """

    evidence = state.get("launch_gate_evidence", {})
    stages = state.get("stages", {})
    artifact_names = set(evidence.get("artifacts", {}))
    expected_artifacts = {"corpus_audit", "e2e_smoke", "condition_preflight"}
    if evidence.get("condition_preflight_reproduction_required") is True:
        expected_artifacts.add("condition_preflight_reproduction")
    return (
        state.get("config_sha256") == evidence.get("config_sha256")
        and state.get("source_tree_sha256")
        == evidence.get("source_code", {}).get("tree_sha256")
        and evidence.get("schema_version") == "trajectory_launch_gate_evidence_v1"
        and evidence.get("formal_launch_authorized_during_gates") is False
        and float(state.get("gates_completed_unix_time", 0.0)) > 0.0
        and all(
            stages.get(stage, {}).get("status") == "complete"
            for stage in ("corpus_audit", "e2e_smoke", "condition_preflight")
        )
        and artifact_names == expected_artifacts
    )


class Supervisor:
    def __init__(self, config: Mapping[str, Any], allow_failed_resume: bool = False) -> None:
        self.config = dict(config)
        self.p = paths(config)
        # Preserve a throughput-finalized training argv across gates-only or
        # supervisor restarts, but reconstruct every command from the current
        # immutable config rather than trusting arbitrary persisted argv.
        selected_batches = None
        if self.p["command_manifest"].is_file():
            try:
                prior_manifest = read_json(self.p["command_manifest"])
                prior_selected = prior_manifest.get("selected_batch_size_by_action")
                if (
                    prior_manifest.get("config_sha256") == experiment_config_sha256(config)
                    and prior_manifest.get("training_commands_finalized") is True
                    and isinstance(prior_selected, dict)
                    and set(prior_selected) == set(ACTIONS)
                    and all(int(value) in (32, 64, 128, 256) for value in prior_selected.values())
                ):
                    selected_batches = {
                        action: int(prior_selected[action]) for action in ACTIONS
                    }
            except (KeyError, OSError, TypeError, ValueError):
                selected_batches = None
        self.manifest = command_manifest(config, selected_batches)
        if selected_batches is not None and self.p["probe_selection"].is_file():
            self.manifest["throughput_selection"] = {
                "path": str(self.p["probe_selection"]),
                "sha256": sha256_file(self.p["probe_selection"]),
            }
        self.poll_seconds = int(config["supervision"]["poll_seconds"])
        self.allow_failed_resume = allow_failed_resume
        self.p["run"].mkdir(parents=True, exist_ok=True)
        self.p["logs"].mkdir(parents=True, exist_ok=True)
        self.lock_stream = self.p["lock"].open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_stream.close()
            raise RuntimeError("another formal supervisor owns %s" % self.p["lock"])
        try:
            if self.p["state"].is_file():
                self.state = read_json(self.p["state"])
                if self.state.get("config_sha256") != experiment_config_sha256(config):
                    raise ValueError("existing formal run is bound to a different config")
                if self.state.get("source_tree_sha256") != self.manifest.get(
                    "source_code", {}
                ).get("tree_sha256"):
                    raise ValueError(
                        "existing formal run is bound to different executable source bytes"
                    )
                if self.state.get("status") == "failed" and not allow_failed_resume:
                    raise RuntimeError("previous run failed; inspect status/log and pass --resume-failed after fixing the cause")
            else:
                self.state = initial_state(config)
            self.state["supervisor_pid"] = os.getpid()
            atomic_json(self.p["command_manifest"], self.manifest)
            self.save()
        except BaseException:
            self.lock_stream.close()
            raise

    def save(self) -> None:
        self.state["updated_unix_time"] = time.time()
        atomic_json(self.p["state"], self.state)

    def event(self, kind: str, **values: Any) -> None:
        event = {"unix_time": time.time(), "kind": kind}
        event.update(values)
        self.state.setdefault("history", []).append(event)
        self.state["history"] = self.state["history"][-200:]
        self.save()

    def set_stage(self, stage: str, status: str, **values: Any) -> None:
        item = self.state["stages"].setdefault(stage, {})
        item.update(values)
        item["status"] = status
        item["updated_unix_time"] = time.time()
        if status in ("running", "waiting_for_clean_gpu"):
            self.state["current_stage"] = stage
        elif self.state.get("current_stage") == stage and status in ("complete", "stopped"):
            self.state["current_stage"] = None
        self.save()

    def _record_failure(self, exc: BaseException) -> None:
        """Durably fail the active stage and supervisor exactly once."""

        stage = self.state.get("current_stage")
        trace = traceback.format_exc()
        failed_at = time.time()
        if stage in self.state.get("stages", {}):
            self.state["stages"][stage].update({
                "status": "failed",
                "error": str(exc),
                "traceback": trace,
                "failed_unix_time": failed_at,
                "updated_unix_time": failed_at,
            })
        self.state.update({
            "status": "failed",
            "error": str(exc),
            "traceback": trace,
            "failed_unix_time": failed_at,
            "current_stage": stage,
        })
        self.save()

    def _record_stopped(self, exc: BaseException) -> None:
        stage = self.state.get("current_stage")
        stopped_at = time.time()
        if stage in self.state.get("stages", {}):
            self.state["stages"][stage].update({
                "status": "stopped",
                "error": str(exc),
                "stopped_unix_time": stopped_at,
                "updated_unix_time": stopped_at,
            })
        self.state.update({
            "status": "stopped",
            "error": str(exc),
            "stopped_unix_time": stopped_at,
            "current_stage": stage,
        })
        self.save()

    def stop_requested(self) -> bool:
        return self.p["stop"].exists()

    def _completion_corpus(self) -> bool:
        if not self.p["corpus_audit"].is_file():
            return False
        try:
            value = read_json(self.p["corpus_audit"])
            rows = value.get("actions", {})
            if not (
                value.get("protocol") == "strict_five_action_trajectory_corpus_v1"
                and value.get("passed") is True
                and value.get("formal_no_sample_cap") is True
                and same_resolved_path(value.get("corpus_dir"), self.p["corpus"])
                and value.get("split") == self._expected_split_audit()
                and set(rows) == set(ACTIONS)
            ):
                return False
            totals = {"events": 0, "flat_rows": 0, "keys": 0}
            for action in ACTIONS:
                row = rows[action]
                source = row.get("source", {})
                counts = row.get("counts", {})
                corpus_path = self.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
                if not (
                    row.get("action") == action
                    and same_resolved_path(source.get("npz"), corpus_path)
                    and source.get("sha256") == sha256_file(corpus_path)
                    and int(source.get("size_bytes", -1)) == corpus_path.stat().st_size
                    and source.get("all_fields_allow_pickle_false") is True
                    and int(source.get("object_array_count", -1)) == 0
                    and row.get("split") == self._expected_split_audit()
                    and int(row.get("full_event_validation", {}).get("events", -1))
                    == int(counts.get("events", -2))
                    and row.get("reference_gate", {}).get("require_all_users") is True
                    and all(
                        not row.get("reference_gate", {}).get("users_with_fewer_than_six", {}).get(name)
                        for name in ("train", "val", "test")
                    )
                ):
                    return False
                for name in totals:
                    totals[name] += int(counts[name])
            return value.get("totals") == totals
        except Exception:
            return False

    def _current_corpus_hashes(self) -> Optional[Dict[str, str]]:
        result = {}
        for action in ACTIONS:
            path = self.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
            if not path.is_file():
                return None
            result[action] = sha256_file(path)
        return result

    def _expected_split_audit(self) -> Dict[str, Any]:
        actual_sha256 = sha256_file(self.p["split"])
        if actual_sha256 != self.config["split_sha256"]:
            raise ValueError("current split file SHA-256 differs from pinned formal split")
        source = read_json(self.p["split"])
        return {
            "path": str(self.p["split"]),
            "sha256": actual_sha256,
            "seed": 42,
            "counts": {"train": 70, "val": 10, "test": 20},
            "train_users": list(source["train_users"]),
            "val_users": list(source["val_users"]),
            "test_users": list(source["test_users"]),
        }

    def _load_canonical_registry(self, path: Path, action: str):
        """Recompute a registry digest and require exact action/user/split coverage."""

        from generation.protocol import ReferenceRegistry

        registry = ReferenceRegistry.load(
            str(Path(path).resolve()), self.config["split_sha256"]
        )
        split = self._expected_split_audit()
        expected = {
            (action, int(user_id), split_name)
            for split_name in ("train", "val", "test")
            for user_id in split[split_name + "_users"]
        }
        if set(registry.entries) != expected:
            raise ValueError(
                "%s reference registry does not have exact 100-user action coverage"
                % action
            )
        return registry

    def _e2e_smoke_complete(self) -> bool:
        path = self.p["e2e_smoke"] / "e2e_smoke.json"
        if not path.is_file():
            return False
        try:
            value = read_json(path)
            configuration = value.get("configuration", {})
            source = value.get("source", {})
            split = value.get("split", {})
            corpus_hashes = self._current_corpus_hashes()
            if corpus_hashes is None:
                return False
            gates = self.config["launch_gates"]
            expected_selected_users = 3 * int(gates["e2e_users_per_pool"])
            if not (
                value.get("schema_version") == "trajectory_finalized_v2_e2e_smoke_v2"
                and value.get("status") == "passed" and value.get("formal_result") is False
                and runtime_determinism_matches(value.get("runtime_determinism"))
                and value.get("runtime_determinism_sha256")
                == STRICT_RUNTIME_DETERMINISM_SHA256
                and same_resolved_path(source.get("root"), self.p["corpus"])
                and source.get("formal_audit_passed") is True
                and int(source.get("processed_users", -1)) == 100
                and split == self._expected_split_audit()
                and int(configuration.get("users_per_pool", -1)) == int(gates["e2e_users_per_pool"])
                and int(configuration.get("samples_per_user_action", -1)) == int(gates["e2e_samples_per_user"])
                and int(configuration.get("optimizer_steps_requested", -1)) == int(gates["e2e_optimizer_steps"])
                and configuration.get("device") == str(gates["e2e_device"])
                and int(configuration.get("reference_seed", -1)) == 42
                and int(configuration.get("training_seed", -1)) == 42
                and int(configuration.get("generation_seed", -1)) == 20260713
                and int(configuration.get("training_diffusion_steps", -1)) == 1000
                and int(configuration.get("ddim_inference_steps", -1)) == 50
                and value.get("all_training_loss_finite_and_decreased") is True
                and value.get("all_checkpoint_schedule_best_ema_gates_passed") is True
                and value.get("all_archive_adapter_paths_passed") is True
                and value.get("all_smoke_physical_validity_gates_passed") is True
                and value.get("all_formal_physical_gates_evaluated") is True
                and isinstance(value.get("all_formal_physical_gates_passed"), bool)
                and value.get("all_25_detector_interface_smokes_passed") is True
                and int(value.get("detector_pairs_completed", -1)) == 25
                and set(value.get("training", {})) == set(ACTIONS)
                and set(value.get("generation", {})) == set(ACTIONS)
                and set(value.get("detectors", {})) == set(ACTIONS)
                and set(source.get("files", {})) == set(ACTIONS)
            ):
                return False
            for name in ("manifest", "audit", "formal_audit"):
                source_path = self.p["corpus"] / (
                    "formal_audit/formal_data_audit.json" if name == "formal_audit"
                    else name + ".json"
                )
                if (
                    not same_resolved_path(source.get(name), source_path)
                    or not source_path.is_file()
                    or source.get(name + "_sha256") != sha256_file(source_path)
                ):
                    return False
            selected = value.get("selected_users", {})
            if set(selected) != {"train", "val", "test"} or any(
                len(selected[name]) != int(gates["e2e_users_per_pool"])
                or not set(int(user) for user in selected[name]).issubset(
                    set(split[name + "_users"])
                )
                for name in ("train", "val", "test")
            ):
                return False
            for action in ACTIONS:
                current_path = self.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
                source_file = source["files"][action]
                training = value["training"][action]
                generation = value["generation"][action]
                detectors = value["detectors"][action]
                if not (
                    same_resolved_path(source_file.get("path"), current_path)
                    and source_file.get("sha256") == corpus_hashes[action]
                    and training.get("action") == action and training.get("passed") is True
                    and training.get("corpus_sha256") == corpus_hashes[action]
                    and int(training.get("reference_seed", -1)) == 42
                    and int(training.get("training_seed", -1)) == 42
                    and bounded_e2e_optimizer_steps(
                        int(gates["e2e_optimizer_steps"]),
                        int(training.get("loss", {}).get("optimizer_steps", -1)),
                    )
                    and int(generation.get("n_users", -1)) == expected_selected_users
                    and int(generation.get("n_fake", -1)) == expected_selected_users * int(gates["e2e_samples_per_user"])
                    and int(generation.get("ddim_steps", -1)) == 50
                    and generation.get("selector_used") is False
                    and generation.get("smoke_physical_validity_gate_passed") is True
                    and generation.get("formal_physical_gate_evaluated") is True
                    and isinstance(generation.get("formal_physical_gate_passed"), bool)
                    and isinstance(generation.get("formal_physical_gate_failures"), list)
                    and bool(generation.get("formal_physical_gate_failures"))
                    == (generation.get("formal_physical_gate_passed") is False)
                    and generation.get("train_prior_contains_only_fixed_train_users") is True
                    and int(generation.get("denoiser_calls", -1)) == int(generation.get("expected_denoiser_calls", -2))
                    and detectors.get("action") == action and detectors.get("passed") is True
                    and int(detectors.get("detector_kind_count", -1)) == 5
                    and set(detectors.get("detectors", {})) == {
                        "linear_svm", "rbf_svm", "xgboost", "tcn", "transformer"
                    }
                ):
                    return False
            for relative, record in value.get("artifact_hashes", {}).items():
                artifact = (self.p["e2e_smoke"] / str(relative)).resolve()
                if self.p["e2e_smoke"] not in artifact.parents or not artifact.is_file():
                    return False
                if record.get("sha256") != sha256_file(artifact) or int(
                    record.get("size_bytes", -1)
                ) != artifact.stat().st_size:
                    return False
            return bool(value.get("artifact_hashes"))
        except (KeyError, OSError, TypeError, ValueError):
            return False

    def _condition_preflight_complete(self) -> bool:
        path = self.p["condition_preflight"]
        if not path.is_file():
            return False
        try:
            from generation.protocol import CONDITION_REQUEST_DIGEST_FIELDS
            from scripts.preflight_all_condition_requests import producer_source_identity

            value = read_json(path)
            per_action = value.get("per_action", {})
            registry_map = value.get("training_reference_registry_sha256_by_action", {})
            prior_map = value.get("train_prior_sha256_by_action", {})
            corpus_hashes = self._current_corpus_hashes()
            if corpus_hashes is None or not (
                value.get("schema_version") == "trajectory_all_condition_requests_preflight_v1"
                and value.get("status") == "passed" and value.get("formal_result") is False
                and value.get("producer_source")
                == producer_source_identity(self.p["project"])
                and int(value.get("worker_count", -1)) == 8
                and value.get("parallelization")
                == "fork_per_user_deterministic_parent_aggregation"
                and int(value.get("reference_seed", -1)) == 42
                and int(value.get("generation_seed", -1)) == 20260713
                and value.get("seed_roles_are_distinct") is True
                and same_resolved_path(value.get("split_json"), self.p["split"])
                and value.get("split_sha256") == self.config["split_sha256"]
                and int(value.get("samples_per_user_action", -1)) == 200
                and int(value.get("generation_batch_size", -1)) == 32
                and int(value.get("sampling_batches_per_user_action", -1)) == 7
                and int(value.get("sampling_batches_per_action", -1)) == 700
                and int(value.get("sampling_batches_total", -1)) == 3500
                and int(value.get("total_requests", -1)) == 100000
                and int(value.get("keystroke_hard_timeline_requests", -1)) == 20000
                and value.get("no_retries") is True and value.get("no_skips") is True
                and value.get("train_prior_only_fixed_train_users") is True
                and value.get("all_condition_source_code_eq_2") is True
                and int(value.get("condition_source_code_2_count", -1)) == 100000
                and value.get("all_fake_ids_globally_unique") is True
                and int(value.get("unique_fake_id_count", -1)) == 100000
                and value.get("all_condition_request_seeds_globally_unique") is True
                and int(value.get("unique_condition_request_seed_count", -1)) == 100000
                and value.get("all_ddim_noise_seeds_globally_unique") is True
                and int(value.get("unique_ddim_noise_seed_count", -1)) == 100000
                and value.get("condition_and_noise_seed_domains_disjoint") is True
                and value.get("condition_request_digest_schema") == "trajectory_condition_request_canonical_v1"
                and value.get("condition_set_digest_schema") == "trajectory_condition_request_set_v1"
                and value.get("condition_request_digest_fields")
                == list(CONDITION_REQUEST_DIGEST_FIELDS)
                and is_sha256(value.get("condition_set_sha256"))
                and set(per_action) == set(ACTIONS)
                and set(registry_map) == set(ACTIONS)
                and set(prior_map) == set(ACTIONS)
                and set(value.get("per_action_condition_set_sha256", {})) == set(ACTIONS)
            ):
                return False
            for action in ACTIONS:
                row = per_action[action]
                counts = row.get("counts", {})
                current_path = self.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
                expected_hard = 20000 if action == "keystroke" else 0
                if not (
                    same_resolved_path(row.get("corpus"), current_path)
                    and row.get("corpus_sha256") == corpus_hashes[action]
                    and is_sha256(row.get("training_reference_registry_sha256"))
                    and row.get("training_reference_registry_sha256") == row.get("generation_registry_compatibility_sha256")
                    and row.get("training_reference_registry_sha256") == registry_map[action]
                    and is_sha256(row.get("train_prior_sha256"))
                    and row.get("train_prior_sha256") == prior_map[action]
                    and is_sha256(row.get("condition_set_sha256"))
                    and row.get("condition_set_sha256") == value["per_action_condition_set_sha256"][action]
                    and int(counts.get("users", -1)) == 100
                    and int(counts.get("requests", -1)) == 20000
                    and int(counts.get("sampling_batches", -1)) == 700
                    and int(counts.get("condition_source_code_2", -1)) == 20000
                    and int(counts.get("unique_fake_ids", -1)) == 20000
                    and int(counts.get("unique_condition_request_seeds", -1)) == 20000
                    and int(counts.get("unique_ddim_noise_seeds", -1)) == 20000
                    and int(counts.get("hard_timeline_projections", -1)) == expected_hard
                ):
                    return False
            keycode = value.get("keycode", {})
            return (
                int(keycode.get("keycode_vocab", -1)) == 16384
                and keycode.get("per_key_and_event_counts_exact") is True
                and keycode.get("ellipsis_u2026_observed") is True
            )
        except (KeyError, OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _condition_semantic_signature(value: Mapping[str, Any]) -> Dict[str, Any]:
        global_fields = (
            "schema_version", "status", "formal_result", "reference_seed",
            "generation_seed", "seed_roles_are_distinct", "split_json", "split_sha256",
            "samples_per_user_action", "n_users", "generation_batch_size",
            "sampling_batches_per_user_action", "sampling_batches_per_action",
            "sampling_batches_total", "n_actions", "total_requests",
            "keystroke_hard_timeline_requests", "no_retries", "no_skips",
            "fixed_references_per_user_action", "train_prior_only_fixed_train_users",
            "all_condition_source_code_eq_2", "condition_source_code_2_count",
            "all_fake_ids_globally_unique", "unique_fake_id_count",
            "training_reference_registry_sha256_by_action",
            "train_prior_sha256_by_action", "condition_request_digest_schema",
            "condition_set_digest_schema", "condition_request_digest_fields",
            "condition_set_sha256", "per_action_condition_set_sha256", "keycode",
        )
        result = {field: value.get(field) for field in global_fields}
        result["per_action"] = {}
        # The reproduction receipt compares the condition semantics produced
        # before the parallel/source-identity patch.  New fail-closed seed
        # uniqueness evidence is validated by ``_condition_preflight_complete``
        # above, but is deliberately excluded from this legacy comparison.
        legacy_count_fields = (
            "users", "requests", "sampling_batches", "condition_source_code_2",
            "unique_fake_ids", "zero_flight_boundaries",
            "positive_flight_boundaries", "hard_timeline_projections",
            "projected_zero_intervals",
        )
        for action in ACTIONS:
            row = value.get("per_action", {}).get(action, {})
            result["per_action"][action] = {
                field: row.get(field)
                for field in (
                    "corpus", "corpus_sha256",
                    "training_reference_registry_sha256",
                    "generation_registry_compatibility_sha256",
                    "train_prior_sha256", "condition_set_sha256",
                    "minimum", "maximum",
                )
            }
            counts = row.get("counts", {})
            result["per_action"][action]["counts"] = {
                field: counts.get(field) for field in legacy_count_fields
            }
        return result

    def _condition_reproduction_receipt_path(self) -> Path:
        return self.p["gate_root"] / "condition_preflight_reproduction.json"

    def _write_condition_reproduction_receipt(self) -> None:
        baseline = self.state.get("pre_patch_condition_preflight")
        if not isinstance(baseline, dict):
            return
        old_path = Path(str(baseline.get("path", "")))
        new_path = self.p["condition_preflight"]
        if not old_path.is_file() or not self._condition_preflight_complete():
            raise RuntimeError("condition preflight reproduction inputs are incomplete")
        old_value = read_json(old_path)
        new_value = read_json(new_path)
        old_semantic = self._condition_semantic_signature(old_value)
        new_semantic = self._condition_semantic_signature(new_value)
        if old_semantic != new_semantic:
            raise RuntimeError(
                "post-patch condition preflight differs from pre-patch 100k semantic result"
            )
        receipt = {
            "schema_version": "trajectory_condition_preflight_reproduction_v1",
            "passed": True,
            "comparison_scope": (
                "global/per-action condition_set SHA, all counts, priors, registries, "
                "bounds, keycode and fixed protocol fields"
            ),
            "pre_patch": {
                "path": str(old_path.resolve()), "sha256": sha256_file(old_path),
            },
            "post_patch": {
                "path": str(new_path.resolve()), "sha256": sha256_file(new_path),
            },
            "semantic_sha256": canonical_sha256(new_semantic),
            "matched_exactly": True,
        }
        atomic_json(self._condition_reproduction_receipt_path(), receipt)

    def _condition_reproduction_complete(self) -> bool:
        baseline = self.state.get("pre_patch_condition_preflight")
        if not isinstance(baseline, dict):
            return True
        receipt_path = self._condition_reproduction_receipt_path()
        if not receipt_path.is_file() or not self._condition_preflight_complete():
            return False
        try:
            receipt = read_json(receipt_path)
            old_path = Path(str(baseline["path"]))
            current = read_json(self.p["condition_preflight"])
            return (
                old_path.is_file()
                and receipt.get("schema_version")
                == "trajectory_condition_preflight_reproduction_v1"
                and receipt.get("passed") is True
                and receipt.get("matched_exactly") is True
                and receipt.get("pre_patch", {}).get("sha256") == sha256_file(old_path)
                and receipt.get("post_patch", {}).get("sha256")
                == sha256_file(self.p["condition_preflight"])
                and receipt.get("semantic_sha256")
                == canonical_sha256(self._condition_semantic_signature(current))
            )
        except (KeyError, OSError, TypeError, ValueError):
            return False

    def _prepare_e2e_output(self) -> None:
        root = self.p["e2e_smoke"]
        if not root.exists() or self._e2e_smoke_complete():
            return
        command = list(self.manifest["commands"]["e2e_smoke"])
        if self._recorded_job_is_live("e2e_smoke", command):
            return
        destination = self.p["gate_root"] / "orphaned" / (
            "v2_e2e_smoke_" + time.strftime("%Y%m%d_%H%M%S") + "_%d" % os.getpid()
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        root.rename(destination)
        self.event("archived_incomplete_e2e_smoke", source=str(root), destination=str(destination))

    def _recorded_job_is_live(self, name: str, command: Sequence[str]) -> bool:
        prior = self.state.get("jobs", {}).get(name, {})
        return (
            prior.get("status") == "running"
            and self._pid_matches(int(prior.get("pid", -1)), command)
        )

    def _prepare_condition_output(self) -> None:
        source = self.p["condition_preflight"]
        if not source.is_file() or self._condition_preflight_complete():
            return
        command = list(self.manifest["commands"]["condition_preflight"])
        if self._recorded_job_is_live("condition_preflight", command):
            return
        destination = self.p["gate_root"] / "orphaned" / (
            "all_condition_requests_" + time.strftime("%Y%m%d_%H%M%S")
            + "_%d.json" % os.getpid()
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        try:
            archived = read_json(destination)
            if (
                not isinstance(self.state.get("pre_patch_condition_preflight"), dict)
                and archived.get("schema_version")
                == "trajectory_all_condition_requests_preflight_v1"
                and archived.get("status") == "passed"
                and int(archived.get("total_requests", -1)) == 100000
                and is_sha256(archived.get("condition_set_sha256"))
            ):
                self.state["pre_patch_condition_preflight"] = {
                    "path": str(destination.resolve()),
                    "sha256": sha256_file(destination),
                    "reason": "producer source changed during direct preflight; comparison required",
                }
                self.save()
        except (OSError, TypeError, ValueError):
            pass
        self.event("archived_stale_condition_preflight", source=str(source), destination=str(destination))

    def _prepare_bundle_output(self) -> None:
        root = self.p["bundle"]
        if not root.exists() or not any(root.iterdir()) or self._bundle_complete():
            return
        command = list(self.manifest["commands"]["detector_bundle"])
        if self._recorded_job_is_live("detector_bundle", command):
            return
        destination = self.p["orphaned"] / (
            "detector_bundle_" + time.strftime("%Y%m%d_%H%M%S") + "_%d" % os.getpid()
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        root.rename(destination)
        self.event("archived_incomplete_detector_bundle", source=str(root), destination=str(destination))

    def _run_launch_gates(self) -> None:
        if not self._completion_corpus():
            self.set_stage("corpus_audit", "running")
            self._wait_one(
                "corpus_audit", list(self.manifest["commands"]["corpus_audit"]),
                self._completion_corpus,
            )
        self.set_stage("corpus_audit", "complete")
        if not self._e2e_smoke_complete():
            device = str(self.config["launch_gates"]["e2e_device"])
            self._wait_for_clean_gpus((device,), "v2 end-to-end smoke", "e2e_smoke")
            self._prepare_e2e_output()
            self.set_stage("e2e_smoke", "running")
            self._wait_one(
                "e2e_smoke", list(self.manifest["commands"]["e2e_smoke"]),
                self._e2e_smoke_complete,
            )
        self.set_stage("e2e_smoke", "complete")
        if not self._condition_preflight_complete():
            self._prepare_condition_output()
            self.set_stage("condition_preflight", "running")
            self._wait_one(
                "condition_preflight", list(self.manifest["commands"]["condition_preflight"]),
                self._condition_preflight_complete,
            )
        self._write_condition_reproduction_receipt()
        if not self._condition_reproduction_complete():
            raise RuntimeError("condition preflight reproduction comparison is incomplete")
        self.set_stage("condition_preflight", "complete")

    def _current_launch_gate_evidence(self) -> Dict[str, Any]:
        """Build the exact, currently valid evidence reviewed before launch."""

        checks = {
            "corpus_audit": self._completion_corpus(),
            "e2e_smoke": self._e2e_smoke_complete(),
            "condition_preflight": self._condition_preflight_complete(),
            "condition_preflight_reproduction": self._condition_reproduction_complete(),
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise RuntimeError(
                "cannot publish launch-gate evidence; failed current gates: %s"
                % ", ".join(failed)
            )
        artifact_paths = {
            "corpus_audit": self.p["corpus_audit"],
            "e2e_smoke": self.p["e2e_smoke"] / "e2e_smoke.json",
            "condition_preflight": self.p["condition_preflight"],
        }
        reproduction_required = isinstance(
            self.state.get("pre_patch_condition_preflight"), dict
        )
        if reproduction_required:
            artifact_paths["condition_preflight_reproduction"] = (
                self._condition_reproduction_receipt_path()
            )
        corpus_hashes = self._current_corpus_hashes()
        if corpus_hashes is None:
            raise RuntimeError("five-action corpus disappeared after launch gates")
        return {
            "schema_version": "trajectory_launch_gate_evidence_v1",
            "formal_launch_authorized_during_gates": False,
            "condition_preflight_reproduction_required": reproduction_required,
            "config_sha256": experiment_config_sha256(self.config),
            "source_code": formal_source_snapshot(self.p["project"]),
            "split_sha256": self.config["split_sha256"],
            "corpus_sha256_by_action": corpus_hashes,
            "artifacts": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for name, path in artifact_paths.items()
            },
        }

    def _reviewed_launch_gates_are_current(self) -> bool:
        """Require the exact gates-only evidence that was explicitly reviewed."""

        if not gate_review_recorded(self.state):
            return False
        try:
            return self.state.get("launch_gate_evidence") == self._current_launch_gate_evidence()
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            return False

    def _probe_candidate_complete(self, action: str, batch_size: int) -> bool:
        path = self.p["probe"] / action / ("candidate_bs%03d.json" % int(batch_size))
        if not path.is_file():
            return False
        value = read_json(path)
        if value.get("expected_resource_failure") is True:
            command = list(
                self.manifest["commands"]["throughput_probe_candidates"][action][str(batch_size)]
            )
            inner = command[command.index("--") + 1 :]
            expected_command_sha = hashlib.sha256(
                json.dumps(inner, sort_keys=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            kind = value.get("failure_kind")
            common = (
                value.get("passed") is False
                and value.get("command_sha256") == expected_command_sha
            )
            if kind == "cuda_oom":
                return bool(
                    common
                    and value.get("schema_version")
                    == "trajectory_training_throughput_candidate_failure_v1"
                )
            if kind == "runtime_budget_exceeded":
                return bool(
                    common
                    and value.get("schema_version")
                    == "trajectory_training_throughput_candidate_failure_v2"
                    and float(value.get("timeout_seconds", -1))
                    == float(self.config["throughput_probe"]["candidate_wall_time_limit_seconds"])
                    and value.get("process_group_terminated") is True
                    and value.get("termination_signal") in ("SIGTERM", "SIGKILL")
                )
            return False
        probe = self.config["throughput_probe"]
        condition = read_json(self.p["condition_preflight"])
        return (
            value.get("schema_version") == "trajectory_training_throughput_v2"
            and value.get("passed") is True
            and value.get("uses_exact_formal_train_loader_and_model") is True
            and value.get("reads_validation_or_test_targets") is False
            and value.get("creates_or_updates_formal_checkpoint") is False
            and runtime_determinism_matches(value.get("runtime_determinism"))
            and value.get("action") == action
            and value.get("device") == self.config["throughput_probe"]["device"]
            and int(value.get("batch_size", -1)) == int(batch_size)
            and int(value.get("measured_optimizer_steps", -1)) == int(probe["candidate_measured_steps"])
            and int(value.get("diffusion_steps", -1)) == 1000
            and int(value.get("keycode_vocab", -1)) == 16384
            and value.get("longest_uncapped_batch_exercised") is True
            and int(value.get("worst_case_safety_optimizer_steps", -1)) == 1
            and value.get("exact_canonical_target_and_reference_lengths") is True
            and value.get("exact_target_and_reference_keycode_lengths") is True
            and value.get("projection_includes_worst_case_measurement") is True
            and value.get("optimizer_state_initialization_excluded_from_projection") is True
            and int(value.get("optimizer_state_initialization_steps", -1))
            == int(probe["candidate_warmup_steps"])
            and value.get("projection_has_extrapolation") is False
            and value.get("epoch_projection", {}).get("projection_has_extrapolation") is False
            and int(value.get("profile_epoch_count", -1)) == 5
            and int(value.get("projection_measurement_count", -1))
            == int(probe["candidate_measured_steps"]) + 1
            and float(value.get("worst_case_elapsed_seconds", 0.0)) > 0.0
            and float(value.get("projected_full_epoch_optimizer_seconds", 0.0)) > 0.0
            and float(value.get("projected_full_epoch_examples_per_second", 0.0)) > 0.0
            and throughput_projection_matches(
                value,
                int(probe["candidate_measured_steps"]),
                int(probe["candidate_warmup_steps"]),
            )
            and throughput_benchmark_config_matches(
                value, self.config, action, int(batch_size),
                int(probe["candidate_measured_steps"]),
                int(probe["candidate_warmup_steps"]),
            )
            and value.get("loss", {}).get("all_finite") is True
            and int(value.get("keycode_vocab", -1)) == 16384
            and value.get("split_sha256") == self.config["split_sha256"]
            and value.get("corpus_sha256") == sha256_file(
                self.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
            )
            and value.get("reference_registry_sha256")
            == condition["training_reference_registry_sha256_by_action"][action]
        )

    def _selected_probe_complete(self, action: str, batch_size: int) -> bool:
        path = self.p["probe"] / action / ("selected_bs%03d_100steps.json" % int(batch_size))
        if not path.is_file():
            return False
        value = read_json(path)
        condition = read_json(self.p["condition_preflight"])
        return (
            value.get("schema_version") == "trajectory_training_throughput_v2"
            and value.get("passed") is True
            and value.get("action") == action
            and int(value.get("batch_size", -1)) == int(batch_size)
            and int(value.get("measured_optimizer_steps", -1)) == 100
            and value.get("uses_exact_formal_train_loader_and_model") is True
            and value.get("reads_validation_or_test_targets") is False
            and value.get("creates_or_updates_formal_checkpoint") is False
            and runtime_determinism_matches(value.get("runtime_determinism"))
            and value.get("longest_uncapped_batch_exercised") is True
            and int(value.get("worst_case_safety_optimizer_steps", -1)) == 1
            and value.get("exact_canonical_target_and_reference_lengths") is True
            and value.get("exact_target_and_reference_keycode_lengths") is True
            and value.get("projection_includes_worst_case_measurement") is True
            and value.get("optimizer_state_initialization_excluded_from_projection") is True
            and int(value.get("optimizer_state_initialization_steps", -1))
            == int(self.config["throughput_probe"]["selected_warmup_steps"])
            and value.get("projection_has_extrapolation") is False
            and value.get("epoch_projection", {}).get("projection_has_extrapolation") is False
            and int(value.get("profile_epoch_count", -1)) == 5
            and int(value.get("projection_measurement_count", -1)) == 101
            and float(value.get("worst_case_elapsed_seconds", 0.0)) > 0.0
            and float(value.get("projected_full_epoch_optimizer_seconds", 0.0)) > 0.0
            and float(value.get("projected_full_epoch_examples_per_second", 0.0)) > 0.0
            and throughput_projection_matches(
                value, 100,
                int(self.config["throughput_probe"]["selected_warmup_steps"]),
            )
            and throughput_benchmark_config_matches(
                value, self.config, action, int(batch_size), 100,
                int(self.config["throughput_probe"]["selected_warmup_steps"]),
            )
            and value.get("loss", {}).get("all_finite") is True
            and value.get("split_sha256") == self.config["split_sha256"]
            and value.get("corpus_sha256") == sha256_file(
                self.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
            )
            and value.get("reference_registry_sha256")
            == condition["training_reference_registry_sha256_by_action"][action]
        )

    def _load_probe_selection(self) -> Optional[Dict[str, int]]:
        if not self.p["probe_selection"].is_file():
            return None
        try:
            value = read_json(self.p["probe_selection"])
            probe = self.config["throughput_probe"]
            metric = str(probe["selection_metric"])
            candidate_batches = [int(item) for item in probe["candidate_batch_sizes"]]
            selected = value.get("selected_batch_size_by_action", {})
            if not (
                value.get("schema_version") == "trajectory_formal_throughput_selection_v2"
                and value.get("passed") is True
                and value.get("selection_uses_validation_or_test") is False
                and value.get("changes_model_data_or_truncation") is False
                and value.get("selection_metric") == metric
                and value.get("tie_break") == "larger_stable_batch"
                and value.get("candidate_batch_sizes") == candidate_batches
                and int(value.get("candidate_measured_steps", -1))
                == int(probe["candidate_measured_steps"])
                and int(value.get("selected_measured_steps", -1)) == 100
                and float(value.get("candidate_wall_time_limit_seconds", -1))
                == float(probe["candidate_wall_time_limit_seconds"])
                and int(value.get("gpu_safety_margin_bytes", -1))
                == int(float(probe["gpu_safety_margin_gib"]) * 1024 ** 3)
                and value.get("split_sha256") == self.config["split_sha256"]
                and set(selected) == set(ACTIONS)
                and set(value.get("selected_results", {})) == set(ACTIONS)
                and set(value.get("candidates", {})) == set(ACTIONS)
            ):
                return None
            result = {action: int(selected[action]) for action in ACTIONS}
            if any(batch not in candidate_batches for batch in result.values()):
                return None
            condition = read_json(self.p["condition_preflight"])
            if value.get("corpus_sha256_by_action") != {
                action: sha256_file(
                    self.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
                )
                for action in ACTIONS
            }:
                return None
            if value.get("reference_registry_sha256_by_action") != condition.get(
                "training_reference_registry_sha256_by_action"
            ):
                return None
            memory = value.get("gpu_memory_before_probe", {})
            device = str(probe["device"])
            memory_limit = int(memory[device]["free_bytes"]) - int(
                value["gpu_safety_margin_bytes"]
            )
            if memory_limit <= 0:
                return None
            for action in ACTIONS:
                candidate_rows = value["candidates"][action]
                if (
                    not isinstance(candidate_rows, list)
                    or len(candidate_rows) != len(candidate_batches)
                    or {int(row.get("batch_size", -1)) for row in candidate_rows}
                    != set(candidate_batches)
                ):
                    return None
                stable: List[Tuple[float, int]] = []
                for row in candidate_rows:
                    batch_size = int(row["batch_size"])
                    candidate_path = self.p["probe"] / action / (
                        "candidate_bs%03d.json" % batch_size
                    )
                    if not (
                        same_resolved_path(row.get("path"), candidate_path)
                        and candidate_path.is_file()
                        and row.get("result_sha256") == sha256_file(candidate_path)
                        and self._probe_candidate_complete(action, batch_size)
                    ):
                        return None
                    candidate_value = read_json(candidate_path)
                    is_resource_failure = candidate_value.get("expected_resource_failure") is True
                    if bool(row.get("expected_resource_failure")) != is_resource_failure:
                        return None
                    if is_resource_failure:
                        continue
                    peak = int(candidate_value.get("cuda_peak_memory_reserved_bytes", -1))
                    safe = peak > 0 and peak <= memory_limit
                    if not (
                        int(row.get("peak_vram_reserved_bytes", -1)) == peak
                        and int(row.get("memory_limit_after_margin_bytes", -1)) == memory_limit
                        and row.get("memory_safety_passed") is safe
                        and float(row.get(metric, -1.0))
                        == float(candidate_value.get(metric, -2.0))
                        and float(row.get("projected_full_epoch_optimizer_seconds", -1.0))
                        == float(candidate_value.get("projected_full_epoch_optimizer_seconds", -2.0))
                        and float(row.get("worst_case_elapsed_seconds", -1.0))
                        == float(candidate_value.get("worst_case_elapsed_seconds", -2.0))
                    ):
                        return None
                    if safe:
                        stable.append((float(candidate_value[metric]), batch_size))
                if not stable or result[action] != max(
                    stable, key=lambda item: (item[0], item[1])
                )[1]:
                    return None
                selected_path = self.p["probe"] / action / (
                    "selected_bs%03d_100steps.json" % result[action]
                )
                selected_record = value["selected_results"][action]
                selected_value = read_json(selected_path)
                if not (
                    self._selected_probe_complete(action, result[action])
                    and same_resolved_path(selected_record.get("path"), selected_path)
                    and selected_record.get("sha256") == sha256_file(selected_path)
                    and int(selected_record.get("batch_size", -1)) == result[action]
                    and int(selected_record.get("measured_optimizer_steps", -1)) == 100
                    and float(selected_record.get(metric, -1.0))
                    == float(selected_value.get(metric, -2.0))
                    and float(selected_record.get("projected_full_epoch_optimizer_seconds", -1.0))
                    == float(selected_value.get("projected_full_epoch_optimizer_seconds", -2.0))
                    and float(selected_record.get("worst_case_elapsed_seconds", -1.0))
                    == float(selected_value.get("worst_case_elapsed_seconds", -2.0))
                    and int(selected_record.get("peak_vram_allocated_bytes", -1))
                    == int(selected_value.get("cuda_peak_memory_allocated_bytes", -2))
                    and int(selected_record.get("peak_vram_reserved_bytes", -1))
                    == int(selected_value.get("cuda_peak_memory_reserved_bytes", -2))
                ):
                    return None
            return result
        except (KeyError, OSError, TypeError, ValueError):
            return None

    def _gpu_memory_snapshot(self) -> Dict[str, Dict[str, int]]:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        )
        if result.returncode != 0:
            raise RuntimeError("nvidia-smi memory query failed: %s" % result.stderr.strip())
        parsed = {}
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6:
                raise ValueError("unexpected nvidia-smi row: %r" % line)
            parsed["cuda:" + fields[0]] = {
                "total_bytes": int(fields[1]) * 1024 * 1024,
                "free_bytes": int(fields[2]) * 1024 * 1024,
                "used_bytes": int(fields[3]) * 1024 * 1024,
                "utilization_percent": int(fields[4]),
                "temperature_c": int(fields[5]),
            }
        if set(self.config["devices"]) - set(parsed):
            raise RuntimeError("configured CUDA devices are not visible")
        return parsed

    def _assert_clean_gpu(self, device: str, snapshot: Mapping[str, Mapping[str, int]], context: str) -> None:
        values = snapshot[device]
        probe = self.config["throughput_probe"]
        used_limit = int(probe["clean_gpu_max_memory_used_mib"]) * 1024 * 1024
        if (
            int(values["utilization_percent"]) > int(probe["clean_gpu_max_utilization_percent"])
            or int(values["temperature_c"]) > int(probe["clean_gpu_max_temperature_c"])
            or int(values["used_bytes"]) > used_limit
        ):
            raise RuntimeError(
                "%s requires an uncontended %s; observed util=%d%% temp=%dC used=%.1fMiB"
                % (
                    context, device, int(values["utilization_percent"]),
                    int(values["temperature_c"]), int(values["used_bytes"]) / float(1024 ** 2),
                )
            )

    def _wait_for_clean_gpus(self, devices: Sequence[str], context: str, stage: str) -> Dict[str, Dict[str, int]]:
        """Wait for external GPU work to leave; never launch a child while busy."""

        started = time.time()
        while True:
            if self.stop_requested():
                raise KeyboardInterrupt("STOP_REQUESTED while waiting for clean GPU")
            try:
                snapshot = self._gpu_memory_snapshot()
            except Exception as exc:
                self.set_stage(
                    stage, "waiting_for_clean_gpu", gpu_wait_context=context,
                    gpu_wait_started_unix_time=started,
                    last_gpu_query_error="%s: %s" % (type(exc).__name__, exc),
                    heartbeat_unix_time=time.time(),
                )
                time.sleep(self.poll_seconds)
                continue
            busy = []
            for device in devices:
                try:
                    self._assert_clean_gpu(str(device), snapshot, context)
                except RuntimeError as exc:
                    busy.append(str(exc))
            if not busy:
                self.set_stage(
                    stage, "running", gpu_wait_context=context,
                    gpu_wait_started_unix_time=started,
                    gpu_wait_finished_unix_time=time.time(),
                    clean_gpu_snapshot=snapshot,
                )
                return snapshot
            self.set_stage(
                stage, "waiting_for_clean_gpu", gpu_wait_context=context,
                gpu_wait_started_unix_time=started,
                latest_gpu_snapshot=snapshot, busy_reasons=busy,
                heartbeat_unix_time=time.time(),
            )
            time.sleep(self.poll_seconds)

    def _assert_training_device_capacity(
        self, selected: Mapping[str, int],
        snapshot: Optional[Mapping[str, Mapping[str, int]]] = None,
    ) -> None:
        if snapshot is None:
            snapshot = self._gpu_memory_snapshot()
        margin = int(float(self.config["throughput_probe"]["gpu_safety_margin_gib"]) * 1024 ** 3)
        selection = read_json(self.p["probe_selection"])
        for action in ACTIONS:
            device = str(self.config["action_device"][action])
            peak = int(selection["selected_results"][action]["peak_vram_reserved_bytes"])
            if int(snapshot[device]["free_bytes"]) < peak + margin:
                raise RuntimeError(
                    "formal training %s lacks selected-probe VRAM+margin on %s" % (action, device)
                )

    @staticmethod
    def _replace_cli_value(command: List[str], flag: str, value: Any) -> None:
        if flag not in command:
            raise ValueError("missing flag %s" % flag)
        command[command.index(flag) + 1] = str(value)

    def _run_throughput_probe(self) -> Dict[str, int]:
        existing = self._load_probe_selection()
        if existing is not None:
            self.manifest = command_manifest(self.config, existing)
            self.manifest["throughput_selection"] = {
                "path": str(self.p["probe_selection"]),
                "sha256": sha256_file(self.p["probe_selection"]),
            }
            atomic_json(self.p["command_manifest"], self.manifest)
            self.set_stage("throughput_probe", "complete", skipped_existing=True, selected_batch_size_by_action=existing)
            return existing
        self.set_stage("throughput_probe", "running")
        probe_device = str(self.config["throughput_probe"]["device"])
        memory = self._wait_for_clean_gpus(
            (probe_device,), "training throughput probe", "throughput_probe"
        )
        candidate_jobs = {}
        for action in ACTIONS:
            device = probe_device
            for batch_size in self.config["throughput_probe"]["candidate_batch_sizes"]:
                name = "throughput_probe/%s/bs%d" % (action, int(batch_size))
                candidate_jobs[name] = (
                    device,
                    list(self.manifest["commands"]["throughput_probe_candidates"][action][str(batch_size)]),
                    lambda action=action, batch_size=int(batch_size): self._probe_candidate_complete(action, batch_size),
                )
        self._run_parallel("throughput_probe", candidate_jobs, finalize_stage=False)

        margin = int(float(self.config["throughput_probe"]["gpu_safety_margin_gib"]) * 1024 ** 3)
        metric = str(self.config["throughput_probe"]["selection_metric"])
        selected, candidate_report = {}, {}
        for action in ACTIONS:
            device = probe_device
            memory_limit = int(memory[device]["free_bytes"]) - margin
            stable = []
            candidate_report[action] = []
            for batch_size in self.config["throughput_probe"]["candidate_batch_sizes"]:
                path = self.p["probe"] / action / ("candidate_bs%03d.json" % int(batch_size))
                value = read_json(path)
                record = {
                    "batch_size": int(batch_size), "path": str(path),
                    "result_sha256": sha256_file(path), "expected_resource_failure": bool(value.get("expected_resource_failure", False)),
                }
                if value.get("passed") is True:
                    peak = int(value.get("cuda_peak_memory_reserved_bytes", 0))
                    safe = peak > 0 and peak <= memory_limit
                    record.update({
                        "peak_vram_reserved_bytes": peak,
                        "memory_limit_after_margin_bytes": memory_limit,
                        "memory_safety_passed": safe,
                        metric: float(value[metric]),
                        "projected_full_epoch_optimizer_seconds": float(
                            value["projected_full_epoch_optimizer_seconds"]
                        ),
                        "worst_case_elapsed_seconds": float(
                            value["worst_case_elapsed_seconds"]
                        ),
                    })
                    if safe:
                        stable.append((float(value[metric]), int(batch_size)))
                candidate_report[action].append(record)
            if not stable:
                raise RuntimeError("no OOM-free, safety-margin-compliant batch candidate for %s on %s" % (action, device))
            # Primary criterion is projected full-train-epoch throughput with
            # canonical target/ref padding and the measured worst case.  A tie
            # is resolved toward the larger stable batch, never by validation
            # or test quality.
            selected[action] = max(stable, key=lambda item: (item[0], item[1]))[1]

        final_jobs = {}
        for action in ACTIONS:
            batch_size = selected[action]
            candidate = list(self.manifest["commands"]["throughput_probe_candidates"][action][str(batch_size)])
            separator = candidate.index("--")
            command = candidate[separator + 1:]
            output = self.p["probe"] / action / ("selected_bs%03d_100steps.json" % batch_size)
            self._replace_cli_value(command, "--output", output)
            self._replace_cli_value(command, "--steps", 100)
            self._replace_cli_value(command, "--warmup-steps", self.config["throughput_probe"]["selected_warmup_steps"])
            final_jobs["throughput_probe/%s/selected_100steps" % action] = (
                probe_device, command,
                lambda action=action, batch_size=batch_size: self._selected_probe_complete(action, batch_size),
            )
        self._run_parallel("throughput_probe", final_jobs, finalize_stage=False)

        selected_results = {}
        corpus_hashes = {}
        registry_hashes = {}
        for action in ACTIONS:
            output = self.p["probe"] / action / ("selected_bs%03d_100steps.json" % selected[action])
            value = read_json(output)
            selected_results[action] = {
                "path": str(output), "sha256": sha256_file(output),
                "batch_size": selected[action], metric: float(value[metric]),
                "projected_full_epoch_optimizer_seconds": float(
                    value["projected_full_epoch_optimizer_seconds"]
                ),
                "worst_case_elapsed_seconds": float(value["worst_case_elapsed_seconds"]),
                "peak_vram_allocated_bytes": int(value["cuda_peak_memory_allocated_bytes"]),
                "peak_vram_reserved_bytes": int(value["cuda_peak_memory_reserved_bytes"]),
                "measured_optimizer_steps": int(value["measured_optimizer_steps"]),
            }
            corpus_hashes[action] = value["corpus_sha256"]
            registry_hashes[action] = value["reference_registry_sha256"]
        selection = {
            "schema_version": "trajectory_formal_throughput_selection_v2", "passed": True,
            "selection_uses_validation_or_test": False,
            "changes_model_data_or_truncation": False,
            "selection_metric": metric,
            "tie_break": "larger_stable_batch",
            "candidate_batch_sizes": [32, 64, 128, 256],
            "candidate_measured_steps": int(self.config["throughput_probe"]["candidate_measured_steps"]),
            "selected_measured_steps": 100,
            "candidate_wall_time_limit_seconds": float(
                self.config["throughput_probe"]["candidate_wall_time_limit_seconds"]
            ),
            "gpu_safety_margin_bytes": margin,
            "gpu_memory_before_probe": memory,
            "split_sha256": self.config["split_sha256"],
            "corpus_sha256_by_action": corpus_hashes,
            "reference_registry_sha256_by_action": registry_hashes,
            "selected_batch_size_by_action": selected,
            "selected_results": selected_results,
            "candidates": candidate_report,
        }
        atomic_json(self.p["probe_selection"], selection)
        self.manifest = command_manifest(self.config, selected)
        self.manifest["throughput_selection"] = {
            "path": str(self.p["probe_selection"]), "sha256": sha256_file(self.p["probe_selection"]),
        }
        atomic_json(self.p["command_manifest"], self.manifest)
        self.set_stage("throughput_probe", "complete", selected_batch_size_by_action=selected)
        return selected

    def _training_bootstrap_complete(self) -> bool:
        """Verify the v15->v16 migration receipt and its durable lineage.

        Before resume, the migrated last.pt must still have its receipt hash.
        Once training has opened that file, the final run manifest must retain
        the exact migrated hash as ``resume_checkpoint_sha256`` instead.
        """

        public = self.p["training_bootstrap_receipt"]
        internal = self.p["training"] / ".v15_to_v16_bootstrap_receipt.json"
        if not public.is_file() or not internal.is_file():
            return False
        try:
            value = read_json(public)
            if value != read_json(internal):
                return False
            bootstrap = self.config["training_bootstrap"]
            selection_sha = sha256_file(self.p["probe_selection"])
            source_root = path_from(bootstrap, "source_run_root")
            source_state = read_json(source_root / "supervisor_status.json")
            if not (
                value.get("schema_version")
                == "trajectory_v15_to_v16_training_bootstrap_v1"
                and value.get("passed") is True
                and same_resolved_path(value.get("source_run_root"), source_root)
                and value.get("source_tree_sha256")
                == bootstrap["source_tree_sha256"]
                and same_resolved_path(value.get("target_run_root"), self.p["run"])
                and value.get("config_sha256")
                == experiment_config_sha256(self.config)
                and same_resolved_path(value.get("selection_path"), self.p["probe_selection"])
                and value.get("selection_sha256") == selection_sha
                and source_state.get("status") == "failed"
                and source_state.get("source_tree_sha256")
                == bootstrap["source_tree_sha256"]
                and [row.get("action") for row in value.get("actions", [])]
                == ["tap", "scroll"]
            ):
                return False
            protected_names = {"model", "ema", "optimizer", "amp_scaler", "rng_state"}
            for row in value["actions"]:
                action = row["action"]
                source_last = source_root / "training" / action / "last.pt"
                target_root = self.p["training"] / action
                expected_source_sha = bootstrap["actions"][action][
                    "last_checkpoint_sha256"
                ]
                checkpoint_receipts = row.get("checkpoint_receipts", {})
                last_receipt = checkpoint_receipts.get("last.pt", {})
                run = read_json(target_root / "run_manifest.json")
                target_last = target_root / "last.pt"
                target_last_is_migrated = (
                    target_last.is_file()
                    and sha256_file(target_last) == last_receipt.get("target_sha256")
                )
                resumed_from_migrated = (
                    run.get("resume_checkpoint_sha256")
                    == last_receipt.get("target_sha256")
                    and run.get("status") in ("running", "complete")
                )
                if not (
                    row.get("source_last_checkpoint_sha256") == expected_source_sha
                    and source_last.is_file()
                    and sha256_file(source_last) == expected_source_sha
                    and row.get("target_last_checkpoint_sha256")
                    == last_receipt.get("target_sha256")
                    and is_sha256(row.get("expected_config_sha256"))
                    and canonical_sha256(run.get("config"))
                    == row.get("expected_config_sha256")
                    and (target_last_is_migrated or resumed_from_migrated)
                    and set(checkpoint_receipts) >= {"last.pt"}
                ):
                    return False
                for filename, receipt in checkpoint_receipts.items():
                    source_path = Path(str(receipt.get("source_path"))).resolve()
                    target_path = Path(str(receipt.get("target_path"))).resolve()
                    protected = receipt.get("protected_content_sha256", {})
                    if not (
                        source_path.is_file()
                        and sha256_file(source_path) == receipt.get("source_sha256")
                        and target_path.parent == target_root.resolve()
                        and target_path.name == filename
                        and is_sha256(receipt.get("target_sha256"))
                        and set(protected) == protected_names
                        and all(is_sha256(item) for item in protected.values())
                    ):
                        return False
                    # A resumed last.pt is expected to change.  Every immutable
                    # migrated best checkpoint must retain its exact hash.
                    if filename != "last.pt" and (
                        not target_path.is_file()
                        or sha256_file(target_path) != receipt["target_sha256"]
                    ):
                        return False
                progress = last_receipt.get("progress", {})
                if not (
                    all(
                        is_nonnegative_int(progress.get(name))
                        for name in (
                            "epoch_index", "next_batch_in_epoch",
                            "examples_seen_in_epoch", "global_step",
                            "amp_overflow_retries_total",
                        )
                    )
                    and int(progress["amp_overflow_retries_total"]) == 0
                ):
                    return False
            return True
        except (KeyError, OSError, TypeError, ValueError):
            return False

    def _training_complete(self, action: str) -> bool:
        root = self.p["training"] / action
        manifest_path = root / "run_manifest.json"
        metrics_path = root / "metrics.jsonl"
        best_manifest_path = root / "best_manifest.json"
        last_state_path = root / "last_state.json"
        progress_path = root / "training_progress.json"
        if not all(path.is_file() for path in (
            manifest_path, metrics_path, best_manifest_path, root / "last.pt",
            last_state_path, progress_path, root / "reference_registry.json",
        )):
            return False
        try:
            import torch

            run = read_json(manifest_path)
            cfg = run.get("config", {})
            selection = read_json(self.p["probe_selection"])
            selected = selection.get("selected_batch_size_by_action", {})
            expected_cfg = {
                "action": action,
                "corpus_npz": str((self.p["corpus"] / ("hmog_trajectory_%s.npz" % action)).resolve()),
                "split_json": str(self.p["split"]),
                "output_dir": str(root.resolve()),
                "epochs": 100,
                "batch_size": int(selected[action]),
                "learning_rate": float(self.config["training"]["learning_rate"]),
                "weight_decay": float(self.config["training"]["weight_decay"]),
                "grad_clip_norm": float(self.config["training"]["grad_clip_norm"]),
                "ema_decay": float(self.config["training"]["ema_decay"]),
                "diffusion_steps": 1000,
                "base_channels": int(self.config["training"]["base_channels"]),
                "cond_dim": int(self.config["training"]["cond_dim"]),
                "time_dim": int(self.config["training"]["time_dim"]),
                "n_blocks": int(self.config["training"]["n_blocks"]),
                "dropout": float(self.config["training"]["dropout"]),
                "keycode_vocab": 16384,
                "seed": 42,
                "num_workers": int(self.config["training"]["num_workers"]),
                "amp": True,
                "checkpoint_every_steps": int(self.config["training"]["checkpoint_every_steps"]),
                "reference_cache_size": int(self.config["training"]["reference_cache_size"]),
                "device": str(self.config["action_device"][action]),
                "amp_overflow_max_retries": int(
                    self.config["training"]["amp_overflow_max_retries"]
                ),
                "allow_non_gaussian_terminal_for_test": False,
            }
            registry = read_json(root / "reference_registry.json")
            canonical_registry = self._load_canonical_registry(
                root / "reference_registry.json", action
            )
            registry_sha = canonical_registry.registry_sha256
            corpus_path = self.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
            corpus_sha = sha256_file(corpus_path)
            extraction_manifest = self.p["corpus"] / "manifest.json"
            source = run.get("source", {})
            source_matches = (
                same_resolved_path(source.get("corpus_npz"), corpus_path)
                and source.get("corpus_sha256") == corpus_sha
                and source.get("action") == action
                and same_resolved_path(source.get("split_json"), self.p["split"])
                and source.get("split_sha256") == self.config["split_sha256"]
                and int(source.get("split_seed", -1)) == 42
                and source.get("reference_registry_sha256") == registry_sha
                and same_resolved_path(source.get("extraction_manifest"), extraction_manifest)
                and source.get("extraction_manifest_sha256") == sha256_file(extraction_manifest)
            )
            if not (
                run.get("protocol_version")
                == "trajectory_diffusion_strict_five_ref_v2"
                and run.get("status") == "complete" and run.get("action") == action
                and cfg == expected_cfg and source_matches
                and run.get("validation_completed_epochs") == [20, 40, 60, 80, 100]
                and run.get("validation_fractions") == [0.2, 0.4, 0.6, 0.8, 1.0]
                and run.get("full_corpus_no_sample_cap") is True
                and run.get("drop_last") is False and run.get("truncation") is False
                and run.get("amp_effective") is True
                and run.get("numeric_recovery_policy")
                == numeric_recovery_policy(self.config)
                and runtime_determinism_matches(run.get("runtime_determinism"))
                and registry.get("registry_sha256") == registry_sha
                and registry.get("action") == action and int(registry.get("seed", -1)) == 42
                and same_resolved_path(registry.get("corpus_npz"), corpus_path)
                and registry.get("corpus_sha256") == corpus_sha
                and registry.get("split_sha256") == self.config["split_sha256"]
                and int(registry.get("references_per_group", -1)) == 5
                and is_sha256(registry_sha)
            ):
                return False
            counts = run.get("counts", {})
            if not (
                is_nonnegative_int(counts.get("train")) and int(counts["train"]) > 0
                and is_nonnegative_int(counts.get("val")) and int(counts["val"]) > 0
                and is_nonnegative_int(counts.get("test_reserved"))
                and int(counts["test_reserved"]) > 0
            ):
                return False
            validations, epochs = [], []
            train_rows: Dict[int, Dict[str, Any]] = {}
            validation_rows: Dict[int, Dict[str, Any]] = {}
            identities = set()
            with metrics_path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    identity = (row.get("type"), int(row.get("completed_epoch", -1)))
                    if identity in identities:
                        return False
                    identities.add(identity)
                    if row.get("type") == "validation":
                        validations.append(identity[1])
                        validation_rows[identity[1]] = row
                        if not (
                            row.get("full_validation_split") is True
                            and row.get("ema_weights") is True
                            and is_finite_positive(row.get("val_loss"))
                            and is_nonnegative_int(row.get("n_examples"))
                            and int(row["n_examples"]) == int(counts["val"])
                            and is_nonnegative_int(row.get("n_batches"))
                            and int(row["n_batches"]) > 0
                            and is_finite_positive(row.get("valid_feature_count"))
                            and isinstance(row.get("fraction"), (int, float))
                            and not isinstance(row.get("fraction"), bool)
                            and math.isfinite(float(row["fraction"]))
                            and math.isclose(
                                float(row["fraction"]), identity[1] / 100.0,
                                rel_tol=0.0, abs_tol=1e-12,
                            )
                            and is_nonnegative_int(row.get("global_step"))
                        ):
                            return False
                    elif row.get("type") == "train_epoch":
                        epochs.append(identity[1])
                        train_rows[identity[1]] = row
                        if not (
                            row.get("full_train_split_consumed") is True
                            and is_finite_positive(row.get("loss"))
                            and is_nonnegative_int(row.get("global_step"))
                            and int(row["global_step"]) > 0
                            and is_nonnegative_int(row.get("batches_total_in_epoch"))
                            and int(row["batches_total_in_epoch"]) > 0
                            and is_nonnegative_int(row.get("examples_total_in_epoch"))
                            and int(row["examples_total_in_epoch"]) == int(counts["train"])
                            and is_finite_positive(row.get("valid_feature_count_total"))
                            and is_nonnegative_int(row.get("amp_overflow_retries"))
                            and isinstance(row.get("amp_overflow_events"), list)
                            and len(row["amp_overflow_events"])
                            == int(row["amp_overflow_retries"])
                            and amp_overflow_events_valid(
                                row["amp_overflow_events"], identity[1],
                                int(row["global_step"]),
                                int(self.config["training"]["amp_overflow_max_retries"]),
                            )
                        ):
                            return False
                    else:
                        return False
            if validations != [20, 40, 60, 80, 100] or epochs != list(range(1, 101)):
                return False
            train_steps = [int(train_rows[epoch]["global_step"]) for epoch in epochs]
            total_amp_overflow_retries = sum(
                int(train_rows[epoch]["amp_overflow_retries"])
                for epoch in epochs
            )
            if int(run.get("amp_overflow_retries_total", -1)) != total_amp_overflow_retries:
                return False
            if any(
                later <= earlier
                for earlier, later in zip(train_steps, train_steps[1:])
            ):
                return False
            if any(
                int(validation_rows[epoch]["global_step"])
                != int(train_rows[epoch]["global_step"])
                for epoch in validations
            ):
                return False
            best_manifest = read_json(best_manifest_path)
            best = best_manifest.get("best", {})
            history = best_manifest.get("history", [])
            expected_improvement_epochs: List[int] = []
            running_best = float("inf")
            for epoch in validations:
                value = float(validation_rows[epoch]["val_loss"])
                if value < running_best:
                    running_best = value
                    expected_improvement_epochs.append(epoch)
            best_path = Path(str(best.get("path", "")))
            if not best_path.is_absolute():
                best_path = (root / best_path).resolve()
            if not (
                best_manifest.get("protocol_version")
                == "trajectory_diffusion_strict_five_ref_v2"
                and best_manifest.get("selection_split") == "val"
                and best_manifest.get("test_used_for_selection") is False
                and best_manifest.get("selection_metric")
                == "full_val_masked_epsilon_mse_ema"
                and best_manifest.get("lower_is_better") is True
                and best_manifest.get("numeric_recovery_policy")
                == numeric_recovery_policy(self.config)
                and best_manifest.get("checkpoint_role") == "validation_selected_best"
                and best_manifest.get("inference_weights") == "ema.shadow"
                and history and best == history[-1]
                and [int(row.get("completed_epoch", -1)) for row in history]
                == expected_improvement_epochs
                and int(best.get("completed_epoch", -1)) in (20, 40, 60, 80, 100)
                and math.isclose(
                    float(best.get("val_loss", float("nan"))), running_best,
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                and best.get("source_sha256") == corpus_sha
                and best.get("split_sha256") == self.config["split_sha256"]
                and best.get("reference_registry_sha256") == registry_sha
                and best.get("checkpoint_role") == "validation_selected_best"
                and best.get("inference_weights") == "ema.shadow"
                and best_path.is_file() and best.get("checkpoint_sha256") == sha256_file(best_path)
                and best_manifest.get("source") == source
                and all(
                    is_sha256(row.get("checkpoint_sha256"))
                    and Path(str(row.get("path"))).is_file()
                    and row.get("checkpoint_sha256") == sha256_file(Path(str(row["path"])))
                    and row.get("checkpoint_role") == "validation_selected_best"
                    and row.get("inference_weights") == "ema.shadow"
                    and row.get("source_sha256") == corpus_sha
                    and row.get("split_sha256") == self.config["split_sha256"]
                    and row.get("reference_registry_sha256") == registry_sha
                    and int(row.get("completed_epoch", -1)) in validation_rows
                    and math.isclose(
                        float(row.get("val_loss", float("nan"))),
                        float(validation_rows[int(row["completed_epoch"])]["val_loss"]),
                        rel_tol=1e-12, abs_tol=1e-12,
                    )
                    and int(row.get("global_step", -1))
                    == int(train_rows[int(row["completed_epoch"])]["global_step"])
                    for row in history
                )
                and len({str(row.get("path")) for row in history}) == len(history)
            ):
                return False
            last_path = root / "last.pt"
            if not (
                same_resolved_path(run.get("best_checkpoint"), best_path)
                and same_resolved_path(run.get("last_checkpoint"), last_path)
            ):
                return False
            best_checkpoint = torch.load(str(best_path), map_location="cpu")
            last_checkpoint = torch.load(str(last_path), map_location="cpu")
            expected_model_config = {
                "action": action,
                "diffusion_steps": 1000,
                "base_channels": int(expected_cfg["base_channels"]),
                "cond_dim": int(expected_cfg["cond_dim"]),
                "time_dim": int(expected_cfg["time_dim"]),
                "n_blocks": int(expected_cfg["n_blocks"]),
                "dropout": float(expected_cfg["dropout"]),
                "keycode_vocab": 16384,
            }
            for checkpoint in (best_checkpoint, last_checkpoint):
                model_state = checkpoint.get("model")
                ema_state = checkpoint.get("ema", {}).get("shadow")
                if not (
                    checkpoint.get("protocol_version")
                    == "trajectory_diffusion_strict_five_ref_v2"
                    and runtime_determinism_matches(
                        checkpoint.get("runtime_determinism")
                    )
                    and checkpoint.get("checkpoint_role") == "training_state_with_raw_model_and_ema"
                    and checkpoint.get("inference_weights_for_validation_selected_best") == "ema.shadow"
                    and checkpoint.get("config") == expected_cfg
                    and checkpoint.get("model_config") == expected_model_config
                    and checkpoint.get("source") == source
                    and isinstance(model_state, dict) and model_state
                    and isinstance(ema_state, dict) and ema_state
                    and set(model_state) == set(ema_state)
                    and float(checkpoint["ema"].get("decay", -1)) == float(expected_cfg["ema_decay"])
                    and checkpoint.get("diffusion_schedule", {}).get("diffusion_steps") == 1000
                    and checkpoint.get("diffusion_schedule", {}).get("terminal_gaussian_gate_passed") is True
                    and checkpoint.get("numeric_recovery_policy")
                    == numeric_recovery_policy(self.config)
                    and all(
                        isinstance(model_state[name], torch.Tensor)
                        and isinstance(ema_state[name], torch.Tensor)
                        and model_state[name].shape == ema_state[name].shape
                        and bool(torch.isfinite(model_state[name]).all().item())
                        and bool(torch.isfinite(ema_state[name]).all().item())
                        for name in model_state
                    )
                ):
                    return False
            best_progress = best_checkpoint.get("progress", {})
            last_progress = last_checkpoint.get("progress", {})
            if not (
                is_nonnegative_int(best_progress.get("epoch_index"))
                and int(best_progress["epoch_index"]) == int(best["completed_epoch"])
                and is_nonnegative_int(best_progress.get("global_step"))
                and int(best_progress["global_step"]) == int(best["global_step"])
                and is_finite_positive(best_progress.get("best_val_loss"))
                and math.isclose(
                    float(best_progress["best_val_loss"]), float(best["val_loss"]),
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                and isinstance(best_progress.get("last_validation"), dict)
                and int(best_progress["last_validation"].get("completed_epoch", -1))
                == int(best["completed_epoch"])
                and math.isclose(
                    float(best_progress["last_validation"].get("val_loss", float("nan"))),
                    float(best["val_loss"]), rel_tol=1e-12, abs_tol=1e-12,
                )
                and isinstance(last_checkpoint.get("optimizer"), dict)
                and last_checkpoint["optimizer"]
                and isinstance(last_checkpoint.get("amp_scaler"), dict)
                and last_checkpoint["amp_scaler"]
                and isinstance(last_checkpoint.get("rng_state"), dict)
                and set(("python", "numpy", "torch_cpu", "torch_cuda")).issubset(
                    last_checkpoint["rng_state"]
                )
            ):
                return False
            last_state = read_json(last_state_path)
            worker_progress = read_json(progress_path)
            expected_last_progress = {
                "epoch_index": int(last_progress.get("epoch_index", -1)),
                "next_batch_in_epoch": int(last_progress.get("next_batch_in_epoch", -1)),
                "examples_seen_in_epoch": int(last_progress.get("examples_seen_in_epoch", -1)),
                "global_step": int(last_progress.get("global_step", -1)),
            }
            sidecars_match = (
                last_state.get("schema_version") == "trajectory_last_state_v1"
                and last_state.get("protocol_version")
                == "trajectory_diffusion_strict_five_ref_v2"
                and last_state.get("action") == action
                and same_resolved_path(last_state.get("checkpoint_path"), last_path)
                and last_state.get("checkpoint_sha256") == sha256_file(last_path)
                and int(last_state.get("checkpoint_size_bytes", -1))
                == int(last_path.stat().st_size)
                and last_state.get("progress") == expected_last_progress
                and last_state.get("source") == source
                and last_state.get("config_sha256") == canonical_sha256(expected_cfg)
                and worker_progress.get("schema_version")
                == "trajectory_training_progress_v1"
                and worker_progress.get("protocol_version")
                == "trajectory_diffusion_strict_five_ref_v2"
                and worker_progress.get("run_instance_id")
                == last_state.get("run_instance_id")
                and worker_progress.get("action") == action
                and worker_progress.get("source") == source
                and worker_progress.get("config_sha256") == canonical_sha256(expected_cfg)
                and worker_progress.get("device") == expected_cfg["device"]
                and is_nonnegative_int(
                    worker_progress.get("amp_overflow_retries_total")
                )
                and worker_progress.get("phase") == "complete"
                and int(worker_progress.get("epoch_index", -1)) == 100
                and int(worker_progress.get("next_batch_in_epoch", -1)) == 0
                and int(worker_progress.get("examples_seen_in_epoch", -1)) == 0
                and int(worker_progress.get("global_step", -1))
                == int(last_progress.get("global_step", -2))
                and int(worker_progress.get("last_successful_step", -1))
                == int(last_progress.get("global_step", -2))
                and is_nonnegative_int(worker_progress.get("heartbeat_sequence"))
                and int(worker_progress["heartbeat_sequence"]) > 0
                and is_finite_positive(worker_progress.get("last_loss"))
                and is_finite_nonnegative(worker_progress.get("grad_norm"))
                and is_finite_positive(last_progress.get("last_step_loss"))
                and is_finite_nonnegative(last_progress.get("last_grad_norm"))
                and int(last_progress.get("amp_overflow_retries_total", -1))
                == total_amp_overflow_retries
                and last_progress.get("epoch_amp_overflow_events") == []
                and int(worker_progress.get("amp_overflow_retries_total", -1))
                == total_amp_overflow_retries
                and math.isclose(
                    float(worker_progress["last_loss"]),
                    float(last_progress["last_step_loss"]),
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                and math.isclose(
                    float(worker_progress["grad_norm"]),
                    float(last_progress["last_grad_norm"]),
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                and is_finite_positive(worker_progress.get("started_unix_time"))
                and is_finite_positive(worker_progress.get("updated_unix_time"))
                and is_finite_positive(
                    worker_progress.get("last_successful_progress_unix_time")
                )
            )
            return (
                int(best_checkpoint.get("progress", {}).get("epoch_index", -1))
                == int(best["completed_epoch"])
                and int(last_checkpoint.get("progress", {}).get("epoch_index", -1)) == 100
                and int(last_checkpoint.get("progress", {}).get("next_batch_in_epoch", -1)) == 0
                and int(last_checkpoint.get("progress", {}).get("examples_seen_in_epoch", -1)) == 0
                and int(last_checkpoint.get("progress", {}).get("global_step", -1))
                == train_steps[-1]
                and is_finite_positive(last_checkpoint.get("progress", {}).get("best_val_loss"))
                and math.isclose(
                    float(last_checkpoint["progress"]["best_val_loss"]), running_best,
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                and int(run.get("global_step", -1)) == train_steps[-1]
                and is_finite_positive(run.get("best_val_loss"))
                and math.isclose(
                    float(run["best_val_loss"]), running_best,
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                and last_path.stat().st_size > 0
                and sidecars_match
            )
        except Exception:
            return False

    def _generation_shard_complete(self, shard: int) -> bool:
        total = int(self.config["generation"]["num_shards"])
        path = self.p["generation"] / ("generation_manifest_shard_%03d_of_%03d.json" % (shard, total))
        if not path.is_file():
            return False
        try:
            value = read_json(path)
            checkpoint_map = read_json(self.p["checkpoint_map"])
            registry_map = read_json(self.p["registry_map"])
            expected_checkpoints = {
                action: sha256_file(Path(checkpoint_map[action])) for action in ACTIONS
            }
            expected_registries = {
                action: self._load_canonical_registry(
                    Path(registry_map[action]), action
                ).registry_sha256
                for action in ACTIONS
            }
            split = read_json(self.p["split"])
            split_by_user = {
                int(user_id): split_name
                for split_name in ("train", "val", "test")
                for user_id in split[split_name + "_users"]
            }
            expected_unit_keys = {
                (action, user_id)
                for action_index, action in enumerate(ACTIONS)
                for user_id in range(100)
                if (action_index * 100 + user_id) % total == shard
            }
            expected_units = len(expected_unit_keys)
            expected_fake = expected_units * 200
            results = value.get("results", [])
            observed_unit_keys = {
                (str(row.get("action")), int(row.get("user_id", -1)))
                for row in results
            }
            expected_paths = {
                (action, user_id): (
                    self.p["generation"] / "shards"
                    / ("shard_%03d_of_%03d" % (shard, total))
                    / action / ("user_%03d.npz" % user_id)
                ).resolve()
                for action, user_id in expected_unit_keys
            }
            return (
                value.get("schema_version") == "five_shot_generation_shard_manifest_v4"
                and value.get("formal") is True and value.get("selector_used") is False
                and runtime_determinism_matches(value.get("runtime_determinism"))
                and value.get("runtime_determinism_sha256")
                == STRICT_RUNTIME_DETERMINISM_SHA256
                and int(value.get("generation_base_seed", -1)) == 20260713
                and int(value.get("generation_batch_size", -1)) == 32
                and value.get("condition_request_seed_derivation")
                == "stable_seed(base_seed,action,user_id,sample_index)"
                and value.get("ddim_noise_seed_derivation")
                == "stable_seed(condition_request_seed_xor_0xDD1A50,action,user_id,sample_index)"
                and value.get("condition_request_digest_schema")
                == "trajectory_condition_request_canonical_v1"
                and value.get("condition_set_digest_schema")
                == "trajectory_condition_request_set_v1"
                and is_sha256(value.get("condition_set_sha256"))
                and set(value.get("per_action_condition_set_sha256", {})) == set(ACTIONS)
                and all(
                    is_sha256(digest)
                    for digest in value.get("per_action_condition_set_sha256", {}).values()
                )
                and int(value.get("shard_id", -1)) == shard
                and int(value.get("num_shards", -1)) == total
                and int(value.get("planned_fake", -1)) == expected_fake
                and int(value.get("completed_fake", -1)) == expected_fake
                and int(value.get("completed_units", -1)) == expected_units
                and int(value.get("planned_units", -1)) == expected_units
                and int(value.get("ddim_steps", -1)) == 50
                and float(value.get("eta", -1)) == 0.0
                and int(value.get("fixed_refs_per_user_action", -1)) == 5
                and value.get("fixed_split_sha256") == self.config["split_sha256"]
                and value.get("checkpoint_sha256_by_action") == expected_checkpoints
                and value.get("reference_registry_sha256_by_action") == expected_registries
                and int(value.get("condition_request_seed_recomputed_count", -1))
                == expected_fake
                and int(value.get("ddim_noise_seed_recomputed_count", -1))
                == expected_fake
                and int(value.get("unique_condition_request_seed_count", -1))
                == expected_fake
                and int(value.get("unique_ddim_noise_seed_count", -1))
                == expected_fake
                and value.get("condition_and_noise_seed_domains_disjoint") is True
                and int(value.get("condition_request_replay_count", -1)) == expected_fake
                and len(results) == expected_units
                and len(observed_unit_keys) == expected_units
                and observed_unit_keys == expected_unit_keys
                and all(
                    row.get("passed") is True
                    and runtime_determinism_matches(row.get("runtime_determinism"))
                    and row.get("runtime_determinism_sha256")
                    == STRICT_RUNTIME_DETERMINISM_SHA256
                    and row.get("status") in ("generated", "resumed")
                    and int(row.get("generation_base_seed", -1)) == 20260713
                    and int(row.get("generation_batch_size", -1)) == 32
                    and int(row.get("n_fake", -1)) == 200
                    and int(row.get("ddim_steps", -1)) == 50
                    and float(row.get("ddim_eta", float("nan"))) == 0.0
                    and int(row.get("training_diffusion_steps", -1)) == 1000
                    and row.get("condition_request_seed_derivation")
                    == "stable_seed(base_seed,action,user_id,sample_index)"
                    and row.get("ddim_noise_seed_derivation")
                    == "stable_seed(condition_request_seed_xor_0xDD1A50,action,user_id,sample_index)"
                    and int(row.get("condition_request_seed_recomputed_count", -1))
                    == 200
                    and int(row.get("ddim_noise_seed_recomputed_count", -1)) == 200
                    and int(row.get("unique_condition_request_seed_count", -1)) == 200
                    and int(row.get("unique_ddim_noise_seed_count", -1)) == 200
                    and row.get("condition_and_noise_seed_domains_disjoint") is True
                    and int(row.get("condition_request_replay_count", -1)) == 200
                    and row.get("neural_ddim") is True
                    and row.get("selector_used") is False
                    and int(row.get("batch_size", -1)) == 32
                    and row.get("split") == split_by_user[int(row["user_id"])]
                    and row.get("fixed_split_sha256") == self.config["split_sha256"]
                    and row.get("checkpoint_sha256") == expected_checkpoints[row["action"]]
                    and row.get("reference_registry_sha256") == expected_registries[row["action"]]
                    and int(row.get("exact_replay_count", -1)) == 0
                    and int(row.get("exact_metadata_copy_count", -1)) == 0
                    and int(row.get("complete_key_sequence_copy_count", -1)) == 0
                    and is_sha256(row.get("condition_set_sha256"))
                    and same_resolved_path(
                        row.get("path"),
                        expected_paths[(str(row["action"]), int(row["user_id"]))],
                    )
                    and same_resolved_path(
                        row.get("output_path"),
                        expected_paths[(str(row["action"]), int(row["user_id"]))],
                    )
                    and expected_paths[(str(row["action"]), int(row["user_id"]))].is_file()
                    and expected_paths[(str(row["action"]), int(row["user_id"]))]
                    .with_suffix(".audit.json").is_file()
                    and read_json(
                        expected_paths[(str(row["action"]), int(row["user_id"]))]
                        .with_suffix(".audit.json")
                    ) == row
                    for row in results
                )
            )
        except Exception:
            return False

    def _generation_audit_complete(self) -> bool:
        path = self.p["generation"] / "formal_generation_audit.json"
        if not path.is_file() or not self.p["condition_preflight"].is_file():
            return False
        try:
            value = read_json(path)
            preflight = read_json(self.p["condition_preflight"])
            checkpoint_map = read_json(self.p["checkpoint_map"])
            registry_map = read_json(self.p["registry_map"])
            expected_checkpoints = {
                action: sha256_file(Path(checkpoint_map[action])) for action in ACTIONS
            }
            expected_registries = {
                action: self._load_canonical_registry(
                    Path(registry_map[action]), action
                ).registry_sha256
                for action in ACTIONS
            }
            manifest_hashes = {
                str(shard): sha256_file(
                    self.p["generation"]
                    / ("generation_manifest_shard_%03d_of_%03d.json" % (shard, 2))
                )
                for shard in range(2)
            }
            archive_hash_path = self.p["generation"] / "generation_archive_file_hashes.json"
            archive_paths = sorted(
                self.p["generation"].glob("shards/shard_*_of_*/*/user_*.npz")
            )
            expected_archive_hashes = {
                str(archive.relative_to(self.p["generation"])): sha256_file(archive)
                for archive in archive_paths
            }
            reports = value.get("unit_reports", [])
            split = read_json(self.p["split"])
            split_by_user = {
                int(user_id): split_name
                for split_name in ("train", "val", "test")
                for user_id in split[split_name + "_users"]
            }
            expected_unit_keys = {
                (action, user_id) for action in ACTIONS for user_id in range(100)
            }
            observed_unit_keys = {
                (str(row.get("action")), int(row.get("user_id", -1)))
                for row in reports
            }
            expected_paths = {
                (action, user_id): (
                    self.p["generation"] / "shards"
                    / ("shard_%03d_of_002" % ((ACTIONS.index(action) * 100 + user_id) % 2))
                    / action / ("user_%03d.npz" % user_id)
                ).resolve()
                for action, user_id in expected_unit_keys
            }
            return (
                value.get("schema_version") == "five_shot_generation_formal_audit_v4"
                and value.get("passed") is True and value.get("formal") is True
                and runtime_determinism_matches(value.get("runtime_determinism"))
                and value.get("runtime_determinism_sha256")
                == STRICT_RUNTIME_DETERMINISM_SHA256
                and int(value.get("generation_base_seed", -1)) == 20260713
                and int(value.get("generation_batch_size", -1)) == 32
                and value.get("condition_request_seed_derivation")
                == "stable_seed(base_seed,action,user_id,sample_index)"
                and value.get("ddim_noise_seed_derivation")
                == "stable_seed(condition_request_seed_xor_0xDD1A50,action,user_id,sample_index)"
                and int(value.get("condition_request_seed_recomputed_count", -1))
                == 100000
                and int(value.get("ddim_noise_seed_recomputed_count", -1)) == 100000
                and int(value.get("unique_condition_request_seed_count", -1)) == 100000
                and int(value.get("unique_ddim_noise_seed_count", -1)) == 100000
                and value.get("condition_and_noise_seed_domains_disjoint") is True
                and int(value.get("condition_request_replay_count", -1)) == 100000
                and value.get("condition_request_digest_schema")
                == preflight.get("condition_request_digest_schema")
                and value.get("condition_set_digest_schema")
                == preflight.get("condition_set_digest_schema")
                and value.get("condition_set_sha256") == preflight.get("condition_set_sha256")
                and value.get("per_action_condition_set_sha256")
                == preflight.get("per_action_condition_set_sha256")
                and same_resolved_path(value.get("condition_preflight"), self.p["condition_preflight"])
                and value.get("condition_preflight_sha256") == sha256_file(self.p["condition_preflight"])
                and value.get("generation_manifest_sha256_by_shard") == manifest_hashes
                and int(value.get("generation_archive_file_count", -1)) == 500
                and same_resolved_path(
                    value.get("generation_archive_file_hashes"), archive_hash_path
                )
                and archive_hash_path.is_file()
                and value.get("generation_archive_file_hashes_sha256")
                == sha256_file(archive_hash_path)
                and read_json(archive_hash_path) == expected_archive_hashes
                and len(expected_archive_hashes) == 500
                and value.get("selector_used") is False
                and int(value.get("n_units", -1)) == 500
                and int(value.get("n_fake", -1)) == 100000
                and int(value.get("ddim_steps", -1)) == 50
                and float(value.get("eta", float("nan"))) == 0.0
                and int(value.get("training_diffusion_steps", -1)) == 1000
                and int(value.get("fixed_refs_per_user_action", -1)) == 5
                and value.get("checkpoint_sha256_by_action") == expected_checkpoints
                and value.get("split_counts_per_action")
                == {"train": 14000, "val": 2000, "test": 4000}
                and int(value.get("exact_replay_total", -1)) == 0
                and int(value.get("exact_metadata_copy_total", -1)) == 0
                and int(value.get("complete_key_sequence_copy_total", -1)) == 0
                and len(reports) == 500
                and len(observed_unit_keys) == 500
                and observed_unit_keys == expected_unit_keys
                and all(
                    row.get("passed") is True
                    and runtime_determinism_matches(row.get("runtime_determinism"))
                    and row.get("runtime_determinism_sha256")
                    == STRICT_RUNTIME_DETERMINISM_SHA256
                    and int(row.get("n_fake", -1)) == 200
                    and int(row.get("ddim_steps", -1)) == 50
                    and float(row.get("ddim_eta", float("nan"))) == 0.0
                    and int(row.get("training_diffusion_steps", -1)) == 1000
                    and int(row.get("generation_base_seed", -1)) == 20260713
                    and int(row.get("generation_batch_size", -1)) == 32
                    and row.get("condition_request_seed_derivation")
                    == "stable_seed(base_seed,action,user_id,sample_index)"
                    and row.get("ddim_noise_seed_derivation")
                    == "stable_seed(condition_request_seed_xor_0xDD1A50,action,user_id,sample_index)"
                    and int(row.get("condition_request_seed_recomputed_count", -1))
                    == 200
                    and int(row.get("ddim_noise_seed_recomputed_count", -1)) == 200
                    and int(row.get("unique_condition_request_seed_count", -1)) == 200
                    and int(row.get("unique_ddim_noise_seed_count", -1)) == 200
                    and row.get("condition_and_noise_seed_domains_disjoint") is True
                    and int(row.get("condition_request_replay_count", -1)) == 200
                    and row.get("split") == split_by_user[int(row["user_id"])]
                    and row.get("fixed_split_sha256") == self.config["split_sha256"]
                    and row.get("checkpoint_sha256") == expected_checkpoints[row["action"]]
                    and row.get("reference_registry_sha256") == expected_registries[row["action"]]
                    and int(row.get("exact_replay_count", -1)) == 0
                    and int(row.get("exact_metadata_copy_count", -1)) == 0
                    and int(row.get("complete_key_sequence_copy_count", -1)) == 0
                    and is_sha256(row.get("condition_set_sha256"))
                    and same_resolved_path(
                        row.get("path"),
                        expected_paths[(str(row["action"]), int(row["user_id"]))],
                    )
                    for row in reports
                )
            )
        except Exception:
            return False

    def _bundle_complete(self) -> bool:
        manifest_path = self.p["bundle"] / "bundle_manifest.json"
        split_audit_path = self.p["bundle"] / "split_audit.json"
        if not manifest_path.is_file() or not split_audit_path.is_file():
            return False
        try:
            value = read_json(manifest_path)
            split_audit = read_json(split_audit_path)
            registry_map = read_json(self.p["registry_map"])
            expected_registries = {
                action: self._load_canonical_registry(
                    Path(registry_map[action]), action
                ).registry_sha256
                for action in ACTIONS
            }
            outputs = value.get("outputs", {})
            per_action = value.get("per_action", {})
            overlaps = value.get("reference_overlap_with_detector_real_event_pools", {})
            sources = value.get("sources", [])
            if not (
                value.get("schema_version") == "trajectory_pad_bundle_manifest_v2"
                and value.get("status") == "complete"
                and same_resolved_path(value.get("fake_user_split"), self.p["split"])
                and value.get("fake_user_split_sha256") == self.config["split_sha256"]
                and same_resolved_path(value.get("fake_archive_dir"), self.p["generation"])
                and int(value.get("fake_archive_file_count", -1)) == 500
                and same_resolved_path(value.get("reference_registry_map"), self.p["registry_map"])
                and value.get("reference_registry_map_sha256") == sha256_file(self.p["registry_map"])
                and value.get("reference_registry_sha256_by_action") == expected_registries
                and int(value.get("real_hash_seed", -1)) == int(self.config["detector"]["real_hash_seed"])
                and same_resolved_path(value.get("split_audit"), split_audit_path)
                and split_audit.get("schema_version") == "trajectory_detector_split_by_action_v1"
                and set(split_audit.get("per_action", {})) == set(ACTIONS)
                and set(outputs) == set(ACTIONS) and set(per_action) == set(ACTIONS)
                and set(overlaps) == set(ACTIONS)
                and len(sources) == 10
            ):
                return False
            archive_hash_path = Path(str(value.get("fake_archive_file_hashes", "")))
            generation_archive_hash_path = (
                self.p["generation"] / "generation_archive_file_hashes.json"
            )
            archive_paths = sorted(self.p["generation"].glob("shards/shard_*_of_*/*/user_*.npz"))
            expected_archive_hashes = {
                str(path.relative_to(self.p["generation"])): sha256_file(path)
                for path in archive_paths
            }
            if not (
                same_resolved_path(archive_hash_path, self.p["bundle"] / "fake_archive_file_hashes.json")
                and archive_hash_path.is_file()
                and value.get("fake_archive_file_hashes_sha256") == sha256_file(archive_hash_path)
                and read_json(archive_hash_path) == expected_archive_hashes
                and generation_archive_hash_path.is_file()
                and read_json(generation_archive_hash_path) == expected_archive_hashes
                and len(expected_archive_hashes) == 500
            ):
                return False
            for action in ACTIONS:
                output = outputs[action]
                output_path = self.p["bundle"] / (action + ".npz")
                audit = split_audit["per_action"][action]
                split_users = read_json(self.p["split"])
                expected_fake_users = {
                    pool: list(split_users[pool + "_users"])
                    for pool in ("train", "val", "test")
                }
                expected_fake_counts = {"train": 14000, "val": 2000, "test": 4000}
                real_counts = audit.get("counts", {}).get("real", {})
                fake_counts = audit.get("counts", {}).get("fake", {})
                real_group_counts = audit.get("real_complete_event_group_counts", {})
                if not (
                    same_resolved_path(output.get("path"), output_path)
                    and output_path.is_file()
                    and output.get("sha256") == sha256_file(output_path)
                    and int(per_action[action].get("fake", -1)) == 20000
                    and int(output.get("n", -1))
                    == int(per_action[action].get("real", -1)) + 20000
                    and audit.get("schema_version") == "trajectory_detector_split_v1"
                    and audit.get("fake_policy") == "fixed_disjoint_users_70_10_20"
                    and audit.get("real_policy")
                    == "sha256_ranked_complete_event_group_per_user_action_60_20_20"
                    and int(audit.get("real_hash_seed", -1))
                    == int(self.config["detector"]["real_hash_seed"])
                    and audit.get("fake_sample_counts") == expected_fake_counts
                    and fake_counts == expected_fake_counts
                    and sum(int(x) for x in real_counts.values())
                    == int(per_action[action].get("real", -1))
                    and set(real_counts) == {"train", "val", "test"}
                    and all(int(real_counts[pool]) > 0 for pool in ("train", "val", "test"))
                    and set(real_group_counts) == {"train", "val", "test"}
                    and all(int(real_group_counts[pool]) > 0 for pool in ("train", "val", "test"))
                    and audit.get("user_counts", {}).get("real")
                    == {"train": 100, "val": 100, "test": 100}
                    and audit.get("user_counts", {}).get("fake")
                    == {"train": 70, "val": 10, "test": 20}
                    and audit.get("fake_users") == expected_fake_users
                    and int(overlaps[action].get("n_reference_events", -1)) == 500
                ):
                    return False
                real_sources = [row for row in sources if row.get("action") == action and row.get("label") == "real"]
                fake_sources = [row for row in sources if row.get("action") == action and row.get("label") == "fake"]
                real_path = self.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
                if not (
                    len(real_sources) == 1 and len(fake_sources) == 1
                    and same_resolved_path(real_sources[0].get("path"), real_path)
                    and real_sources[0].get("sha256") == sha256_file(real_path)
                    and int(real_sources[0].get("n", -1)) == int(per_action[action]["real"])
                    and same_resolved_path(fake_sources[0].get("path"), self.p["generation"])
                    and int(fake_sources[0].get("n", -1)) == 20000
                ):
                    return False
            return True
        except Exception:
            return False

    def _deep_probe_complete(self, identity: str) -> bool:
        action, family, detector = identity.split("/")
        if family != "deep_pad":
            return False
        command = self.manifest["commands"]["detector_deep_probes"][identity]
        output = Path(command[command.index("--output") + 1])
        if not output.is_file():
            return False
        value = read_json(output)
        dataset = self.p["bundle"] / (action + ".npz")
        from detectors.pair_runner import stable_pair_seed
        expected_seed = stable_pair_seed(
            int(self.config["detector"]["seed"]), action, "deep_pad", detector
        )
        return (
            value.get("schema_version") == "trajectory_deep_batch_probe_v1"
            and value.get("status") == "passed"
            and value.get("action") == action and value.get("detector") == detector
            and value.get("device") == command[command.index("--device") + 1]
            and int(value.get("seed", -1)) == expected_seed
            and value.get("model_params") == {}
            and same_resolved_path(value.get("dataset_file"), dataset)
            and value.get("dataset_sha256") == sha256_file(dataset)
            and same_resolved_path(value.get("fake_user_split"), self.p["split"])
            and value.get("fake_user_split_sha256") == self.config["split_sha256"]
            and int(value.get("requested_batch_size", -1))
            == int(self.config["detector"]["requested_deep_batch_size"])
            and 1 <= int(value.get("selected_batch_size", 0)) <= int(self.config["detector"]["requested_deep_batch_size"])
            and value.get("truncation") is False and value.get("resampling") is False
            and int(value.get("longest_observed_train_event_length", 0)) > 0
            and value.get("probe")
            == "one_full_forward_backward_optimizer_step_on_repeated_longest_event"
        )

    def _pair_complete(self, identity: str) -> bool:
        action, family, detector = identity.split("/")
        path = self.p["benchmark"] / "pairs" / action / family / detector / "pair_manifest.json"
        if not path.is_file():
            return False
        try:
            from detectors.pair_runner import (
                PAIR_SCHEMA,
                _build_deep_pair_input_identity,
                _config_digest,
                audit_protocol_result,
                stable_pair_seed,
            )

            value = read_json(path)
            config = value.get("config", {})
            dataset = self.p["bundle"] / (action + ".npz")
            pair_seed = stable_pair_seed(
                int(self.config["detector"]["seed"]), action, family, detector
            )
            command = self.manifest.get("commands", {}).get("detector_pairs", {}).get(identity, {}).get("command")
            if not command:
                return False
            batch_size = int(command[command.index("--batch-size") + 1])
            deep_train = {
                "epochs": 40,
                "batch_size": batch_size,
                "learning_rate": float(command[command.index("--learning-rate") + 1]),
                "weight_decay": float(command[command.index("--weight-decay") + 1]),
                "patience": 0,
                "num_workers": int(command[command.index("--num-workers") + 1]),
                "seed": int(pair_seed),
                "bootstrap_replicates": 500,
                "gradient_clip_norm": float(command[command.index("--gradient-clip-norm") + 1]),
            }
            batch_probe = None
            if family == "deep_pad":
                probe_path = Path(command[command.index("--batch-probe-json") + 1])
                probe = read_json(probe_path)
                batch_probe = {
                    "path": str(probe_path.resolve()),
                    "sha256": sha256_file(probe_path),
                    "selected_batch_size": int(probe["selected_batch_size"]),
                    "longest_observed_train_event_length": int(
                        probe["longest_observed_train_event_length"]
                    ),
                    "truncation": False,
                    "resampling": False,
                }
            expected_config = {
                "action": action,
                "family": family,
                "detector": detector,
                "seed": int(pair_seed),
                "base_seed": int(self.config["detector"]["seed"]),
                "seed_policy": "sha256(base_seed|action|family|detector)_uint32",
                "formal_protocol": True,
                "real_hash_seed": int(self.config["detector"]["real_hash_seed"]),
                "feature_bootstrap_replicates": 500,
                "deep_train": deep_train,
                "feature_model_params": {},
                "deep_model_params": {},
                "batch_probe": batch_probe,
            }
            expected_config_sha = _config_digest(expected_config)
            expected_deep_identity = (
                _build_deep_pair_input_identity(
                    dataset_file=dataset,
                    dataset_sha256=sha256_file(dataset),
                    fake_user_split=self.p["split"],
                    fake_user_split_sha256=self.config["split_sha256"],
                    real_hash_seed=int(self.config["detector"]["real_hash_seed"]),
                    action=action,
                    detector=detector,
                    pair_config=expected_config,
                    pair_config_sha256=expected_config_sha,
                )
                if family == "deep_pad" else None
            )
            plot_path = self.p["benchmark"] / "pairs" / action / family / detector / "test_fa_frr.png"
            result_dir = self.p["benchmark"] / "pairs" / action / family / detector / "result"
            audited = audit_protocol_result(
                result_dir,
                action=action,
                family=family,
                detector=detector,
                expected_bootstrap_replicates=500,
                dataset_file=dataset,
                fake_user_split=self.p["split"],
                real_hash_seed=int(self.config["detector"]["real_hash_seed"]),
                expected_dataset_sha256=sha256_file(dataset),
                expected_fake_user_split_sha256=self.config["split_sha256"],
                expected_bootstrap_seed=int(pair_seed) + (
                    31 if family == "feature_pad" else 17
                ),
                expected_deep_run_identity=expected_deep_identity,
            )
            deep_training = audited.get("deep_training_audit", {})
            return (
                value.get("schema_version") == PAIR_SCHEMA
                and value.get("status") == "complete"
                and (value.get("action"), value.get("family"), value.get("detector"))
                == (action, family, detector)
                and same_resolved_path(value.get("dataset_file"), dataset)
                and value.get("dataset_sha256") == sha256_file(dataset)
                and same_resolved_path(value.get("fake_user_split"), self.p["split"])
                and value.get("fake_user_split_sha256") == self.config["split_sha256"]
                and config == expected_config
                and value.get("config_sha256") == expected_config_sha
                and same_resolved_path(value.get("result_dir"), result_dir)
                and result_dir.is_dir()
                and same_resolved_path(value.get("plot"), plot_path)
                and plot_path.is_file() and value.get("plot_sha256") == sha256_file(plot_path)
                and value.get("artifact_hashes") == audited.get("artifact_hashes")
                and value.get("dataset_relink_audit")
                == audited.get("dataset_relink_audit")
                and value.get("operating_rows") == audited.get("rows")
                and len(value.get("operating_rows", [])) == 2
                and set(row.get("operating_point") for row in value["operating_rows"])
                == {"eer", "val_frr_le_5pct"}
                and value.get("split_audit", {}).get("fake_sample_counts")
                == {"train": 14000, "val": 2000, "test": 4000}
                and (
                    family != "deep_pad"
                    or (
                        int(audited.get("summary", {}).get("last_epoch", -1)) == 40
                        and int(deep_training.get("history_epoch_count", -1)) == 40
                        and int(deep_training.get("history_last_epoch", -1)) == 40
                    )
                )
            )
        except Exception:
            return False

    def _detector_pair_jobs(self) -> Dict[str, Tuple[str, List[str], Any]]:
        jobs = {}
        finalized = {}
        for identity, template in self.manifest["commands"]["detector_pair_templates"].items():
            command = list(template["command"])
            action, family, detector = identity.split("/")
            if family == "deep_pad":
                probe_path = Path(command[command.index("--batch-probe-json") + 1])
                probe = read_json(probe_path)
                self._replace_cli_value(command, "--batch-size", int(probe["selected_batch_size"]))
            jobs["detector_pair/" + identity] = (
                str(template["resource"]), command,
                lambda identity=identity: self._pair_complete(identity),
            )
            finalized[identity] = {"resource": str(template["resource"]), "command": command}
        if len(jobs) != 25:
            raise AssertionError("formal detector job graph must contain exactly 25 pairs")
        self.manifest["commands"]["detector_pairs"] = finalized
        self.manifest["detector_pair_commands_finalized"] = True
        atomic_json(self.p["command_manifest"], self.manifest)
        return jobs

    def _merged_benchmark_complete(self) -> bool:
        path = self.p["benchmark_merged"] / "benchmark_manifest.json"
        if not path.is_file():
            return False
        try:
            value = read_json(path)
            pair_hashes = {}
            for identity in self.manifest["commands"]["detector_pair_templates"]:
                action, family, detector = identity.split("/")
                pair_path = self.p["benchmark"] / "pairs" / action / family / detector / "pair_manifest.json"
                pair_hashes[identity] = sha256_file(pair_path)
            dataset_hashes = {
                action: sha256_file(self.p["bundle"] / (action + ".npz"))
                for action in ACTIONS
            }
            outputs = value.get("outputs", {})
            output_hashes = value.get("output_sha256", {})
            summary_root = self.p["benchmark_merged"] / "summaries"
            expected_summary_hashes = {
                str(path.relative_to(self.p["benchmark_merged"])): sha256_file(path)
                for path in sorted(
                    path for path in summary_root.rglob("*") if path.is_file()
                )
            }
            return (
                value.get("schema_version") == "trajectory_pad_25pair_merge_v1"
                and value.get("status") == "complete" and value.get("formal_protocol") is True
                and int(value.get("n_pairs", -1)) == 25
                and int(value.get("n_operating_rows", -1)) == 50
                and int(value.get("n_macro_rows", -1)) == 10
                and int(value.get("plot_count", -1)) == 25
                and value.get("actions") == list(ACTIONS)
                and value.get("feature_detectors") == ["linear_svm", "rbf_svm", "xgboost"]
                and value.get("deep_detectors") == ["tcn", "transformer"]
                and value.get("fake_user_split_sha256") == self.config["split_sha256"]
                and same_resolved_path(value.get("fake_user_split"), self.p["split"])
                and value.get("dataset_sha256_by_action") == dataset_hashes
                and value.get("pair_manifest_sha256") == pair_hashes
                and set(outputs) == {
                    "per_action", "macro", "macro_markdown", "report", "plots",
                    "by_action", "by_detector",
                }
                and all(
                    same_resolved_path(outputs.get(name), self.p["benchmark_merged"] / filename)
                    and (self.p["benchmark_merged"] / filename).is_file()
                    and output_hashes.get(name) == sha256_file(self.p["benchmark_merged"] / filename)
                    for name, filename in (
                        ("per_action", "per_action_detector.csv"),
                        ("macro", "macro_by_detector.csv"),
                        ("macro_markdown", "macro_by_detector.md"),
                        ("report", "benchmark_report.md"),
                    )
                )
                and same_resolved_path(
                    outputs.get("plots"), self.p["benchmark_merged"] / "plots"
                )
                and same_resolved_path(
                    outputs.get("by_action"), summary_root / "by_action"
                )
                and same_resolved_path(
                    outputs.get("by_detector"), summary_root / "by_detector"
                )
                and len(expected_summary_hashes) == 20
                and value.get("summary_artifact_sha256") == expected_summary_hashes
            )
        except Exception:
            return False

    def _merged_benchmark_audit_complete(self) -> bool:
        path = self.p["benchmark_merged"] / "benchmark_audit.json"
        if not path.is_file():
            return False
        try:
            value = read_json(path)
            manifest = self.p["benchmark_merged"] / "benchmark_manifest.json"
            return (
                value.get("schema_version") == "trajectory_pad_25pair_independent_audit_v1"
                and value.get("status") == "passed" and value.get("formal_protocol") is True
                and same_resolved_path(value.get("experiment_root"), self.p["benchmark"])
                and same_resolved_path(value.get("merged_manifest"), manifest)
                and value.get("merged_manifest_sha256") == sha256_file(manifest)
                and int(value.get("n_reaudited_pairs", -1)) == 25
                and int(value.get("n_recomputed_operating_rows", -1)) == 50
                and int(value.get("n_recomputed_macro_rows", -1)) == 10
                and int(value.get("n_verified_plots", -1)) == 25
                and value.get("threshold_selection_pool") == "validation_only"
                and int(value.get("formal_epochs", -1)) == 40
                and int(value.get("formal_patience", -1)) == 0
                and int(value.get("formal_bootstrap_replicates", -1)) == 500
                and self._merged_benchmark_complete()
            )
        except Exception:
            return False

    def _json_gate(self, path: Path, **equals: Any) -> bool:
        if not path.is_file():
            return False
        value = read_json(path)
        return all(value.get(key) == expected for key, expected in equals.items())

    def _pid_matches(self, pid: int, command: Sequence[str]) -> bool:
        try:
            raw = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (OSError, IOError):
            return False
        anchors = [Path(command[2]).name if len(command) > 2 else Path(command[0]).name]
        for flag in ("--output-dir", "--result-dir"):
            if flag in command:
                anchors.append(command[command.index(flag) + 1])
        return all(anchor in raw for anchor in anchors)

    def _observe_training_progress(self, job: str) -> None:
        """Fail closed when a live training PID stops making useful progress."""

        if not job.startswith("training/"):
            return
        action = job.split("/", 1)[1]
        if action not in ACTIONS:
            raise ValueError("unknown training action in job name: %s" % job)
        now = time.time()
        job_state = self.state["jobs"][job]
        started_raw = job_state.get("started_unix_time")
        if (
            not isinstance(started_raw, (int, float)) or isinstance(started_raw, bool)
            or not math.isfinite(float(started_raw)) or float(started_raw) > now + 60.0
        ):
            raise RuntimeError("%s has invalid started_unix_time" % job)
        started = float(started_raw)
        stale_seconds = float(
            self.config.get("supervision", {}).get(
                "training_progress_stale_seconds", 600.0
            )
        )
        progress_path = self.p["training"] / action / "training_progress.json"
        if not progress_path.is_file():
            if now - started > stale_seconds:
                raise RuntimeError(
                    "%s worker has no training_progress.json after %.0f seconds"
                    % (job, stale_seconds)
                )
            job_state["worker_progress"] = {
                "path": str(progress_path), "status": "awaiting_initial_publish",
                "observed_unix_time": now,
            }
            return
        value = read_json(progress_path)
        stat = progress_path.stat()
        if float(stat.st_mtime) <= started and int(value.get("pid", -1)) != int(
            job_state.get("pid", -2)
        ):
            age = max(0.0, now - started)
            job_state["worker_progress"] = {
                "path": str(progress_path),
                "status": "awaiting_new_run_publish",
                "old_run_instance_id": value.get("run_instance_id"),
                "observed_unix_time": now,
                "wait_age_seconds": age,
            }
            if age > stale_seconds:
                raise RuntimeError(
                    "%s resumed worker did not replace old progress within %.0f seconds"
                    % (job, stale_seconds)
                )
            return
        required = {
            "schema_version": "trajectory_training_progress_v1",
            "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
            "action": action,
            "pid": int(job_state.get("pid", -1)),
        }
        if any(value.get(key) != expected for key, expected in required.items()):
            raise RuntimeError("%s worker progress identity mismatch" % job)
        if not isinstance(value.get("run_instance_id"), str) or not value["run_instance_id"]:
            raise RuntimeError("%s worker progress lacks run_instance_id" % job)
        run_manifest_path = self.p["training"] / action / "run_manifest.json"
        if not run_manifest_path.is_file():
            raise RuntimeError("%s published progress before a durable run manifest" % job)
        run_manifest = read_json(run_manifest_path)
        if (
            value.get("config_sha256")
            != canonical_sha256(run_manifest.get("config"))
            or value.get("source") != run_manifest.get("source")
            or value.get("device") != str(self.config["action_device"][action])
        ):
            raise RuntimeError("%s worker progress config/source/device mismatch" % job)
        allowed_phases = {"init", "train", "validation", "checkpoint_commit", "complete"}
        phase = value.get("phase")
        if phase not in allowed_phases:
            raise RuntimeError("%s worker progress phase is invalid" % job)
        integer_fields = (
            "epoch_index", "next_batch_in_epoch", "global_step",
            "examples_seen_in_epoch", "last_successful_step",
            "heartbeat_sequence",
        )
        if not all(is_nonnegative_int(value.get(name)) for name in integer_fields):
            raise RuntimeError("%s worker progress counters are invalid" % job)
        if (
            int(value["heartbeat_sequence"]) <= 0
            or int(value["epoch_index"]) > int(self.config["training"]["epochs"])
            or int(value["last_successful_step"]) > int(value["global_step"])
        ):
            raise RuntimeError("%s worker progress counters are inconsistent" % job)
        validation_index = value.get("validation_batch_index")
        validation_total = value.get("validation_batches_total")
        if not (
            (validation_index is None and validation_total is None)
            or (
                is_nonnegative_int(validation_index)
                and is_nonnegative_int(validation_total)
                and int(validation_index) > 0
                and int(validation_total) > 0
                and int(validation_index) <= int(validation_total)
            )
        ):
            raise RuntimeError("%s worker validation progress is invalid" % job)
        last_loss = value.get("last_loss")
        grad_norm = value.get("grad_norm")
        if phase == "init":
            if last_loss is not None or grad_norm is not None:
                raise RuntimeError("%s init progress unexpectedly contains optimization scalars" % job)
        elif not (
            is_finite_positive(last_loss)
            and is_finite_nonnegative(grad_norm)
            and int(value["last_successful_step"]) == int(value["global_step"])
        ):
            raise RuntimeError("%s worker loss/gradient progress is invalid" % job)
        if phase == "complete" and not (
            int(value["epoch_index"]) == int(self.config["training"]["epochs"])
            and int(value["next_batch_in_epoch"]) == 0
            and int(value["examples_seen_in_epoch"]) == 0
        ):
            raise RuntimeError("%s complete progress counters are invalid" % job)
        updated = value.get("updated_unix_time")
        successful = value.get("last_successful_progress_unix_time")
        worker_started = value.get("started_unix_time")
        if (
            not isinstance(updated, (int, float)) or isinstance(updated, bool)
            or not math.isfinite(float(updated)) or float(updated) > now + 60.0
            or not math.isfinite(float(stat.st_mtime)) or float(stat.st_mtime) > now + 60.0
            or not isinstance(worker_started, (int, float)) or isinstance(worker_started, bool)
            or not math.isfinite(float(worker_started))
            or float(worker_started) < started - 60.0
            or float(worker_started) > float(updated)
            or abs(float(stat.st_mtime) - float(updated)) > 60.0
        ):
            raise RuntimeError("%s worker progress timestamp is invalid" % job)
        freshness_time = float(updated)
        if successful is not None:
            if (
                not isinstance(successful, (int, float)) or isinstance(successful, bool)
                or not math.isfinite(float(successful)) or float(successful) > now + 60.0
            ):
                raise RuntimeError("%s worker successful-progress timestamp is invalid" % job)
            freshness_time = float(successful)
            if not float(worker_started) <= float(successful) <= float(updated):
                raise RuntimeError("%s worker progress timestamp ordering is invalid" % job)
        elif phase != "init":
            raise RuntimeError("%s non-init progress has no successful progress time" % job)
        signature_fields = (
            "run_instance_id", "phase", "epoch_index", "next_batch_in_epoch",
            "global_step", "examples_seen_in_epoch", "last_successful_step",
            "validation_batch_index", "validation_batches_total",
            "last_successful_progress_unix_time", "heartbeat_sequence",
            "last_loss", "grad_norm",
        )
        signature = canonical_sha256({key: value.get(key) for key in signature_fields})
        prior = job_state.get("worker_progress")
        if (
            isinstance(prior, dict)
            and prior.get("status") == "fresh"
            and prior.get("run_instance_id") == value["run_instance_id"]
        ):
            if (
                int(value["heartbeat_sequence"]) <= int(prior.get("heartbeat_sequence", -1))
                and prior.get("signature") != signature
            ):
                raise RuntimeError("%s worker heartbeat sequence did not advance" % job)
            if (
                int(value["global_step"]) < int(prior.get("global_step", -1))
                or int(value["epoch_index"]) < int(prior.get("epoch_index", -1))
                or int(value["last_successful_step"])
                < int(prior.get("last_successful_step", -1))
                or (
                    int(value["epoch_index"]) == int(prior.get("epoch_index", -1))
                    and int(value["next_batch_in_epoch"])
                    < int(prior.get("next_batch_in_epoch", -1))
                )
            ):
                raise RuntimeError("%s worker progress moved backwards" % job)
        if not isinstance(prior, dict) or prior.get("signature") != signature:
            changed_at = now
        else:
            changed_at = float(prior.get("last_change_observed_unix_time", now))
        file_age = max(0.0, now - float(stat.st_mtime))
        progress_age = max(0.0, now - freshness_time)
        unchanged_age = max(0.0, now - changed_at)
        job_state["worker_progress"] = {
            "path": str(progress_path),
            "status": "fresh",
            "signature": signature,
            "run_instance_id": value["run_instance_id"],
            "phase": value.get("phase"),
            "epoch_index": value.get("epoch_index"),
            "next_batch_in_epoch": value.get("next_batch_in_epoch"),
            "global_step": value.get("global_step"),
            "last_successful_step": value.get("last_successful_step"),
            "heartbeat_sequence": value.get("heartbeat_sequence"),
            "last_loss": value.get("last_loss"),
            "grad_norm": value.get("grad_norm"),
            "last_change_observed_unix_time": changed_at,
            "observed_unix_time": now,
            "file_age_seconds": file_age,
            "progress_age_seconds": progress_age,
            "unchanged_age_seconds": unchanged_age,
        }
        if max(file_age, progress_age, unchanged_age) > stale_seconds:
            job_state["worker_progress"]["status"] = "stalled"
            raise RuntimeError(
                "%s worker progress stalled: file=%.1fs progress=%.1fs unchanged=%.1fs"
                % (job, file_age, progress_age, unchanged_age)
            )

    def _launch(self, job: str, command: List[str]) -> subprocess.Popen:
        log_path = self.p["logs"] / (job.replace("/", "__") + ".log")
        log = log_path.open("a", buffering=1, encoding="utf-8")
        log.write("\n[%s] COMMAND %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), json.dumps(command)))
        process = subprocess.Popen(
            command, cwd=str(self.p["project"]), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        self.state["jobs"][job] = {
            "status": "running", "pid": process.pid, "command": command,
            "command_sha256": canonical_sha256(command), "log": str(log_path),
            "started_unix_time": time.time(),
        }
        self.save()
        return process

    def _wait_one(self, job: str, command: List[str], completion) -> None:
        prior = self.state.get("jobs", {}).get(job, {})
        if prior.get("status") == "running" and self._pid_matches(int(prior.get("pid", -1)), command):
            while self._pid_matches(int(prior["pid"]), command):
                if self.stop_requested():
                    raise KeyboardInterrupt("STOP_REQUESTED")
                # Keep both the supervisor and the detached-child observation
                # fresh after a frontend/supervisor restart.
                prior["heartbeat_unix_time"] = time.time()
                prior["supervisor_observed_unix_time"] = time.time()
                self.save()
                time.sleep(self.poll_seconds)
            if completion():
                prior["status"] = "complete"
                prior["finished_unix_time"] = time.time()
                self.save()
                return
        process = self._launch(job, command)
        while process.poll() is None:
            if self.stop_requested():
                os.killpg(process.pid, signal.SIGTERM)
                raise KeyboardInterrupt("STOP_REQUESTED")
            self.state["jobs"][job]["heartbeat_unix_time"] = time.time()
            self.save()
            time.sleep(self.poll_seconds)
        code = int(process.returncode)
        self.state["jobs"][job].update({"returncode": code, "finished_unix_time": time.time()})
        if code != 0 or not completion():
            self.state["jobs"][job]["status"] = "failed"
            self.save()
            raise RuntimeError("job %s failed gate (returncode=%d); see %s" % (job, code, self.state["jobs"][job]["log"]))
        self.state["jobs"][job]["status"] = "complete"
        self.save()

    def _training_command(self, action: str) -> List[str]:
        command = list(self.manifest["commands"]["training"][action])
        root = self.p["training"] / action
        last = root / "last.pt"
        journal = root / "epoch_commit.json"
        if last.is_file():
            command.extend(("--resume", str(last)))
            return command
        if journal.is_file():
            # Recovery reconciles the staged transaction before opening this
            # eventual last.pt path; it may not exist at process start.
            command.extend(("--resume", str(last)))
            return command
        if any(root.glob(".epoch_*_next.pt.pending")):
            command.extend(("--resume", str(last)))
            return command
        prior = self.state.get("jobs", {}).get("training/" + action, {})
        if prior.get("status") == "running" and self._pid_matches(int(prior.get("pid", -1)), command):
            # The detached child can still be inside its first epoch.  It is
            # not safe to rename a live output directory merely because the
            # first durable checkpoint has not been reached yet.
            return command
        has_precheckpoint_artifacts = root.is_dir() and any(root.iterdir())
        if has_precheckpoint_artifacts:
            if not self.config["supervision"].get("archive_precheckpoint_training_attempts", False):
                raise RuntimeError("%s has partial artifacts but no resumable last.pt" % root)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            destination = self.p["orphaned"] / (action + "_precheckpoint_" + stamp)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            root.rename(destination)
            self.event("archived_precheckpoint_attempt", action=action, source=str(root), destination=str(destination))
        return command

    def _run_parallel(
        self, stage: str, jobs: Mapping[str, Tuple[str, List[str], Any]],
        finalize_stage: bool = True,
    ) -> None:
        """Run at most one job/device; stop siblings on a current-process failure."""
        self.set_stage(stage, "running")
        pending = {name: value for name, value in jobs.items() if not value[2]()}
        active: Dict[str, Tuple[str, subprocess.Popen, List[str], Any]] = {}
        try:
            while pending or active:
                if self.stop_requested():
                    raise KeyboardInterrupt("STOP_REQUESTED")
                for name in list(pending):
                    device, command, completion = pending[name]
                    if device in active:
                        continue
                    prior = self.state.get("jobs", {}).get(name, {})
                    if prior.get("status") == "running" and self._pid_matches(int(prior.get("pid", -1)), command):
                        # An orphaned job from a dead supervisor still owns its
                        # device.  Observe it without launching a duplicate.
                        active[device] = (name, None, command, completion)  # type: ignore
                    else:
                        process = self._launch(name, command)
                        active[device] = (name, process, command, completion)
                    del pending[name]
                time.sleep(self.poll_seconds)
                for device, (name, process, command, completion) in list(active.items()):
                    if process is None:
                        if self._pid_matches(int(self.state["jobs"][name]["pid"]), command):
                            self.state["jobs"][name]["heartbeat_unix_time"] = time.time()
                            self.state["jobs"][name]["supervisor_observed_unix_time"] = time.time()
                            if stage == "training":
                                self._observe_training_progress(name)
                            self.save()
                            continue
                        code = None
                    else:
                        code = process.poll()
                        if code is None:
                            self.state["jobs"][name]["heartbeat_unix_time"] = time.time()
                            self.state["jobs"][name]["supervisor_observed_unix_time"] = time.time()
                            if stage == "training":
                                self._observe_training_progress(name)
                            continue
                    if completion():
                        self.state["jobs"][name].update({"status": "complete", "returncode": code, "finished_unix_time": time.time()})
                        del active[device]
                        self.save()
                    elif process is None:
                        # Likely supervisor/power interruption.  Requeue using
                        # durable last/unit archives rather than guessing exit.
                        self.state["jobs"][name]["status"] = "interrupted"
                        pending[name] = (device, self._training_command(name.split("/")[-1]) if stage == "training" else command, completion)
                        del active[device]
                        self.save()
                    else:
                        self.state["jobs"][name].update({"status": "failed", "returncode": int(code), "finished_unix_time": time.time()})
                        self.save()
                        raise RuntimeError("parallel job %s failed (returncode=%s); see %s" % (name, code, self.state["jobs"][name]["log"]))
                self.save()
        except BaseException:
            for _device, (name, process, command, _completion) in active.items():
                pid = (
                    int(process.pid) if process is not None
                    else int(self.state.get("jobs", {}).get(name, {}).get("pid", -1))
                )
                owns_live_process = (
                    process is not None and process.poll() is None
                ) or (
                    process is None and self._pid_matches(pid, command)
                )
                if owns_live_process and pid > 0:
                    try:
                        os.killpg(pid, signal.SIGTERM)
                        self.state["jobs"][name]["termination_requested_unix_time"] = time.time()
                    except OSError:
                        pass
            self.save()
            raise
        if finalize_stage:
            self.set_stage(stage, "complete")

    def _write_maps(self) -> None:
        checkpoints, registries = {}, {}
        condition_gate = read_json(self.p["condition_preflight"])
        expected_registry_hashes = condition_gate[
            "training_reference_registry_sha256_by_action"
        ]
        for action in ACTIONS:
            root = self.p["training"] / action
            if not self._training_complete(action):
                raise RuntimeError("cannot build maps before completed training: %s" % action)
            best = read_json(root / "best_manifest.json")["best"]
            checkpoint = Path(str(best["path"]))
            if not checkpoint.is_absolute():
                checkpoint = (root / checkpoint).resolve()
            checkpoints[action] = str(checkpoint)
            registry_path = (root / "reference_registry.json").resolve()
            registry_payload = read_json(registry_path)
            canonical_registry = self._load_canonical_registry(registry_path, action)
            if (
                registry_payload.get("registry_sha256")
                != canonical_registry.registry_sha256
                or canonical_registry.registry_sha256
                != expected_registry_hashes.get(action)
            ):
                raise ValueError(
                    "%s training registry does not match seed-42 condition preflight" % action
                )
            registries[action] = str(registry_path)
        atomic_json(self.p["checkpoint_map"], checkpoints)
        atomic_json(self.p["registry_map"], registries)

    def _maps_complete(self) -> bool:
        """Bind the published maps to each action's current selected best/registry."""

        if not self.p["checkpoint_map"].is_file() or not self.p["registry_map"].is_file():
            return False
        try:
            checkpoints = read_json(self.p["checkpoint_map"])
            registries = read_json(self.p["registry_map"])
            if set(checkpoints) != set(ACTIONS) or set(registries) != set(ACTIONS):
                return False
            for action in ACTIONS:
                root = self.p["training"] / action
                best = read_json(root / "best_manifest.json")["best"]
                best_path = Path(str(best["path"]))
                if not best_path.is_absolute():
                    best_path = (root / best_path).resolve()
                registry_path = (root / "reference_registry.json").resolve()
                canonical_registry = self._load_canonical_registry(
                    registry_path, action
                )
                if not (
                    self._training_complete(action)
                    and same_resolved_path(checkpoints[action], best_path)
                    and same_resolved_path(registries[action], registry_path)
                    and sha256_file(Path(checkpoints[action])) == best["checkpoint_sha256"]
                    and canonical_registry.registry_sha256
                    == read_json(registry_path).get("registry_sha256")
                ):
                    return False
            return True
        except Exception:
            return False

    def _command_manifest_complete(self) -> bool:
        if not self.p["command_manifest"].is_file():
            return False
        try:
            value = read_json(self.p["command_manifest"])
            selected = self._load_probe_selection()
            throughput = value.get("throughput_selection", {})
            current_source = formal_source_snapshot(self.p["project"])
            return (
                value == self.manifest
                and value.get("config_sha256") == experiment_config_sha256(self.config)
                and value.get("source_code") == current_source
                and value.get("source_code", {}).get("tree_sha256")
                == self.state.get("source_tree_sha256")
                and value.get("training_commands_finalized") is True
                and value.get("selected_batch_size_by_action") == selected
                and same_resolved_path(
                    throughput.get("path"), self.p["probe_selection"]
                )
                and throughput.get("sha256") == sha256_file(self.p["probe_selection"])
                and value.get("detector_pair_commands_finalized") is True
                and set(value.get("commands", {}).get("detector_pairs", {}))
                == set(value.get("commands", {}).get("detector_pair_templates", {}))
                and len(value.get("commands", {}).get("detector_pairs", {})) == 25
            )
        except Exception:
            return False

    def _final_completion_failures(self) -> List[str]:
        """Revalidate the complete artifact closure immediately before PASS."""

        failures: List[str] = []

        def check(name: str, callback) -> None:
            try:
                passed = bool(callback())
            except Exception:
                passed = False
            if not passed:
                failures.append(name)

        check("corpus_audit", self._completion_corpus)
        check("e2e_smoke", self._e2e_smoke_complete)
        check("condition_preflight", self._condition_preflight_complete)
        check("reviewed_launch_gate_evidence", self._reviewed_launch_gates_are_current)
        check("throughput_selection", lambda: self._load_probe_selection() is not None)
        check("training_bootstrap", self._training_bootstrap_complete)
        check("command_manifest", self._command_manifest_complete)
        for action in ACTIONS:
            check("training/%s" % action, lambda action=action: self._training_complete(action))
        check("checkpoint_registry_maps", self._maps_complete)
        for shard in range(int(self.config["generation"]["num_shards"])):
            check(
                "generation/shard_%d" % shard,
                lambda shard=shard: self._generation_shard_complete(shard),
            )
        check("generation_audit", self._generation_audit_complete)
        check("detector_bundle", self._bundle_complete)
        deep_identities = set(self.manifest["commands"]["detector_deep_probes"])
        if len(deep_identities) != 10:
            failures.append("detector_probe_graph")
        for identity in sorted(deep_identities):
            check(
                "detector_probe/%s" % identity,
                lambda identity=identity: self._deep_probe_complete(identity),
            )
        pair_identities = set(self.manifest["commands"]["detector_pair_templates"])
        finalized_pairs = set(
            self.manifest.get("commands", {}).get("detector_pairs", {})
        )
        if len(pair_identities) != 25 or finalized_pairs != pair_identities:
            failures.append("detector_pair_graph")
        for identity in sorted(pair_identities):
            check(
                "detector_pair/%s" % identity,
                lambda identity=identity: self._pair_complete(identity),
            )
        check("benchmark_merge", self._merged_benchmark_complete)
        check("benchmark_audit", self._merged_benchmark_audit_complete)
        return failures

    def _final_artifact_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Hash the receipts that transitively bind every formal artifact."""

        artifacts: Dict[str, Path] = {
            "formal_split": self.p["split"],
            "corpus_audit": self.p["corpus_audit"],
            "corpus_supplemental_provenance": (
                self.p["corpus"] / "formal_audit" / "supplemental_provenance.json"
            ),
            "e2e_smoke": self.p["e2e_smoke"] / "e2e_smoke.json",
            "condition_preflight": self.p["condition_preflight"],
            "throughput_selection": self.p["probe_selection"],
            "training_bootstrap": self.p["training_bootstrap_receipt"],
            "training_bootstrap_internal": (
                self.p["training"] / ".v15_to_v16_bootstrap_receipt.json"
            ),
            "command_manifest": self.p["command_manifest"],
            "checkpoint_map": self.p["checkpoint_map"],
            "registry_map": self.p["registry_map"],
            "generation_audit": self.p["generation"] / "formal_generation_audit.json",
            "generation_archive_hashes": self.p["generation"] / "generation_archive_file_hashes.json",
            "bundle_manifest": self.p["bundle"] / "bundle_manifest.json",
            "bundle_split_audit": self.p["bundle"] / "split_audit.json",
            "bundle_archive_hashes": self.p["bundle"] / "fake_archive_file_hashes.json",
            "benchmark_manifest": self.p["benchmark_merged"] / "benchmark_manifest.json",
            "benchmark_audit": self.p["benchmark_merged"] / "benchmark_audit.json",
            "benchmark_report": self.p["benchmark_merged"] / "benchmark_report.md",
            "benchmark_per_action_detector": (
                self.p["benchmark_merged"] / "per_action_detector.csv"
            ),
            "benchmark_macro_by_detector": (
                self.p["benchmark_merged"] / "macro_by_detector.csv"
            ),
        }
        for action in ACTIONS:
            root = self.p["training"] / action
            best = read_json(root / "best_manifest.json")["best"]
            best_path = Path(str(best["path"]))
            if not best_path.is_absolute():
                best_path = (root / best_path).resolve()
            artifacts.update({
                "training/%s/run_manifest" % action: root / "run_manifest.json",
                "training/%s/metrics" % action: root / "metrics.jsonl",
                "training/%s/best_manifest" % action: root / "best_manifest.json",
                "training/%s/best_checkpoint" % action: best_path,
                "training/%s/last_checkpoint" % action: root / "last.pt",
                "training/%s/last_state" % action: root / "last_state.json",
                "training/%s/training_progress" % action: root / "training_progress.json",
                "training/%s/reference_registry" % action: root / "reference_registry.json",
                "bundle/%s" % action: self.p["bundle"] / (action + ".npz"),
            })
        for shard in range(int(self.config["generation"]["num_shards"])):
            artifacts["generation/shard_%d_manifest" % shard] = (
                self.p["generation"]
                / ("generation_manifest_shard_%03d_of_%03d.json" % (
                    shard, int(self.config["generation"]["num_shards"])
                ))
            )
        for identity, command in self.manifest["commands"]["detector_deep_probes"].items():
            artifacts["detector_probe/%s" % identity] = Path(
                command[command.index("--output") + 1]
            )
        for identity in self.manifest["commands"]["detector_pair_templates"]:
            action, family, detector = identity.split("/")
            artifacts["detector_pair/%s" % identity] = (
                self.p["benchmark"] / "pairs" / action / family / detector
                / "pair_manifest.json"
            )
        return {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in sorted(artifacts.items())
        }

    def _archive_existing_final_receipt(self) -> None:
        existing = [
            path for path in (self.p["final_report"], self.p["final_audit"])
            if path.exists()
        ]
        if not existing:
            return
        destination = self.p["orphaned"] / (
            "prior_final_" + time.strftime("%Y%m%d_%H%M%S") + "_%d" % os.getpid()
        )
        destination.mkdir(parents=True, exist_ok=False)
        for path in existing:
            os.replace(str(path), str(destination / path.name))
        self.event(
            "archived_prior_final_receipt",
            destination=str(destination),
            files=[path.name for path in existing],
        )

    def _write_final(self) -> None:
        failures = self._final_completion_failures()
        if failures:
            raise RuntimeError(
                "final artifact closure failed: %s" % ", ".join(failures)
            )
        generation = read_json(self.p["generation"] / "formal_generation_audit.json")
        benchmark = read_json(self.p["benchmark_merged"] / "benchmark_audit.json")
        split_audit = read_json(self.p["bundle"] / "split_audit.json")
        if (
            generation.get("passed") is not True
            or generation.get("formal") is not True
            or generation.get("n_fake") != 100000
            or generation.get("n_units") != 500
            or generation.get("selector_used") is not False
            or benchmark.get("schema_version")
            != "trajectory_pad_25pair_independent_audit_v1"
            or benchmark.get("status") != "passed"
            or benchmark.get("formal_protocol") is not True
        ):
            raise RuntimeError("final audit invariant failed")
        if not split_audit.get("per_action") or set(split_audit["per_action"]) != set(ACTIONS):
            raise RuntimeError("detector split audit does not contain five actions")
        provenance_path = self.p["corpus"] / "formal_audit" / "supplemental_provenance.json"
        provenance = read_json(provenance_path)
        if (
            provenance.get("schema_version")
            != "hmog_trajectory_extraction_supplemental_provenance_v1"
            or provenance.get("formal_audit_passed") is not True
            or provenance.get("manifest_mutated_after_extraction") is not False
            or provenance.get("extractor", {}).get("exact_launch_source_bytes_archived")
            is not False
            or not is_sha256(provenance.get("input_archive", {}).get("sha256"))
        ):
            raise RuntimeError("supplemental corpus provenance is missing or inconsistent")
        metrics_path = self.p["benchmark_merged"] / "per_action_detector.csv"
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            metric_rows = list(csv.DictReader(stream))
        required_metric_fields = {
            "action", "detector_family", "detector", "operating_point",
            "test_fa", "test_frr", "test_auc",
            "test_fa_ci95_low", "test_fa_ci95_high",
            "test_frr_ci95_low", "test_frr_ci95_high",
            "test_auc_ci95_low", "test_auc_ci95_high",
        }
        if len(metric_rows) != 50 or any(
            not required_metric_fields.issubset(row) for row in metric_rows
        ):
            raise RuntimeError("formal per-action detector table is not the exact 50-row schema")
        for row in metric_rows:
            if row["action"] not in ACTIONS or row["detector_family"] not in {
                "feature_pad", "deep_pad"
            } or row["operating_point"] not in {"eer", "val_frr_le_5pct"}:
                raise RuntimeError("formal detector metric identity is invalid")
            for field in required_metric_fields - {
                "action", "detector_family", "detector", "operating_point"
            }:
                value = float(row[field])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise RuntimeError("formal detector metric is outside [0,1]")
        snapshot = self._final_artifact_snapshot()
        self._archive_existing_final_receipt()
        rows = [
            "# 五动作轨迹生成正式流水线报告", "", "- 结论：PASS", "- 生成器：五个独立 5-reference diffusion",
            "- 训练：每 action 100 epochs；20/40/60/80/100% 完整 validation；best EMA + last",
            "- 生成：schema 1.5；strict runtime digest；batch=32；100 users × 5 actions × 200 = 100,000；50-step DDIM；ConditionRequest、共享 EventPlan 与双 seed 逐条审计；无 selector",
            "- 检测：3 Feature PAD + 2 Deep PAD；validation-only 阈值；完整 test FA/FRR/AUC/曲线；user-level bootstrap",
            "", "## 权威产物", "", "- corpus audit: `%s`" % self.p["corpus_audit"],
            "- generation audit: `%s`" % (self.p["generation"] / "formal_generation_audit.json"),
            "- detector split audit: `%s`" % (self.p["bundle"] / "split_audit.json"),
            "- benchmark report: `%s`" % (self.p["benchmark_merged"] / "benchmark_report.md"),
            "- benchmark audit: `%s`" % (self.p["benchmark_merged"] / "benchmark_audit.json"),
            "- end-to-end audit: `%s`" % self.p["final_audit"],
            "- supplemental corpus provenance: `%s`" % provenance_path,
            "", "## Corpus provenance limitation", "",
            "- 原始 HMOG zip SHA-256：`%s`" % provenance["input_archive"]["sha256"],
            "- extractor 启动前记录 source SHA-256：`%s`"
            % provenance["extractor"]["launch_source_sha256_recorded_before_start"],
            "- exact launch extractor source bytes archived: **false**。运行中仅发生文档编辑；"
            "all-five archive 不受后续 subset fix 影响，但现有证据不能证明比该记录更强的源码逐字节归档。",
            "", "## Complete 50-row PAD results", "",
            "下表逐 action / family / detector / operating point 报告 test FA、FRR、AUC 与 user-level 95% CI；没有省略 detector。",
            "",
            "| Action | Family | Detector | Operating point | FA [95% CI] | FRR [95% CI] | AUC [95% CI] |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
        for row in metric_rows:
            rows.append(
                "| {action} | {detector_family} | {detector} | {operating_point} | "
                "{test_fa:.6f} [{test_fa_ci95_low:.6f}, {test_fa_ci95_high:.6f}] | "
                "{test_frr:.6f} [{test_frr_ci95_low:.6f}, {test_frr_ci95_high:.6f}] | "
                "{test_auc:.6f} [{test_auc_ci95_low:.6f}, {test_auc_ci95_high:.6f}] |".format(
                    **{
                        **row,
                        **{
                            field: float(row[field])
                            for field in required_metric_fields - {
                                "action", "detector_family", "detector", "operating_point"
                            }
                        },
                    }
                )
            )
        rows.append("")
        temporary = self.p["final_report"].with_suffix(".md.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write("\n".join(rows))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(self.p["final_report"]))
        directory_fd = os.open(
            str(self.p["final_report"].parent), os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        final = {
            "schema_version": "trajectory_formal_end_to_end_audit_v2",
            "passed": True,
            "commit_marker": True,
            "config_sha256": experiment_config_sha256(self.config),
            "config_identity_excludes": sorted(OPERATIONAL_CONFIG_KEYS),
            "source_code": formal_source_snapshot(self.p["project"]),
            "split_sha256": self.config["split_sha256"],
            "launch_gate_evidence": self.state["launch_gate_evidence"],
            "training_complete_actions": list(ACTIONS),
            "generation": {
                "n_fake": generation["n_fake"],
                "n_units": generation["n_units"],
                "selector_used": generation["selector_used"],
                "condition_set_sha256": generation["condition_set_sha256"],
                "runtime_determinism": generation["runtime_determinism"],
                "runtime_determinism_sha256": generation[
                    "runtime_determinism_sha256"
                ],
            },
            "benchmark": benchmark,
            "invariants": self.manifest["formal_invariants"],
            "artifact_snapshot": snapshot,
            "report": {
                "path": str(self.p["final_report"].resolve()),
                "sha256": sha256_file(self.p["final_report"]),
                "size_bytes": self.p["final_report"].stat().st_size,
            },
        }
        # This is the last commit marker.  A power loss before this atomic
        # replace leaves at most an uncommitted report, never a PASS receipt.
        atomic_json(self.p["final_audit"], final)

    def _final_complete(self) -> bool:
        if not self.p["final_audit"].is_file() or not self.p["final_report"].is_file():
            return False
        try:
            value = read_json(self.p["final_audit"])
            report = value.get("report", {})
            generation = read_json(
                self.p["generation"] / "formal_generation_audit.json"
            )
            benchmark = read_json(self.p["benchmark_merged"] / "benchmark_audit.json")
            expected_generation = {
                "n_fake": generation["n_fake"],
                "n_units": generation["n_units"],
                "selector_used": generation["selector_used"],
                "condition_set_sha256": generation["condition_set_sha256"],
                "runtime_determinism": generation["runtime_determinism"],
                "runtime_determinism_sha256": generation[
                    "runtime_determinism_sha256"
                ],
            }
            return (
                value.get("schema_version") == "trajectory_formal_end_to_end_audit_v2"
                and value.get("passed") is True
                and value.get("commit_marker") is True
                and value.get("config_sha256") == experiment_config_sha256(self.config)
                and value.get("config_identity_excludes")
                == sorted(OPERATIONAL_CONFIG_KEYS)
                and value.get("source_code") == formal_source_snapshot(self.p["project"])
                and value.get("source_code", {}).get("tree_sha256")
                == self.state.get("source_tree_sha256")
                and value.get("split_sha256") == self.config["split_sha256"]
                and value.get("launch_gate_evidence") == self.state.get("launch_gate_evidence")
                and value.get("training_complete_actions") == list(ACTIONS)
                and value.get("generation") == expected_generation
                and value.get("benchmark") == benchmark
                and value.get("invariants") == self.manifest.get("formal_invariants")
                and same_resolved_path(report.get("path"), self.p["final_report"])
                and report.get("sha256") == sha256_file(self.p["final_report"])
                and int(report.get("size_bytes", -1)) == self.p["final_report"].stat().st_size
                and not self._final_completion_failures()
                and value.get("artifact_snapshot") == self._final_artifact_snapshot()
            )
        except Exception:
            return False

    def run(self) -> None:
        # Authorization is deliberately outside the failure transaction: a
        # mistaken non-gates invocation while authorization is false must not
        # poison a completed gate state or require --resume-failed.
        if self.config.get("formal_launch_authorized") is not True:
            raise PermissionError(
                "formal launch is not authorized; run --gates-only or set "
                "formal_launch_authorized=true after reviewing the gates"
            )
        if not self._reviewed_launch_gates_are_current():
            raise PermissionError(
                "formal launch requires a completed --gates-only run whose exact "
                "corpus/E2E/ConditionRequest hashes are still current; rerun gates-only, "
                "review launch_gate_evidence, then authorize"
            )
        self.state["status"] = "running"
        self.state["formal_launch_authorized_at_start"] = True
        self.save()
        try:
            self.set_stage("preflight", "running")
            report = preflight(self.config, require_runtime_inputs=True)
            self.set_stage("preflight", "complete", report=report)

            self._run_launch_gates()

            self._run_throughput_probe()
            self.set_stage("training_bootstrap", "running")
            if not self._training_bootstrap_complete():
                self._wait_one(
                    "training_bootstrap",
                    list(self.manifest["commands"]["training_bootstrap"]),
                    self._training_bootstrap_complete,
                )
            self.set_stage("training_bootstrap", "complete")
            training_snapshot = self._wait_for_clean_gpus(
                tuple(str(device) for device in self.config["devices"]),
                "formal training launch", "training",
            )
            selected_training_batches = self._load_probe_selection()
            if selected_training_batches is None:
                raise RuntimeError("throughput selection disappeared before formal training")
            self._assert_training_device_capacity(selected_training_batches, training_snapshot)

            training_jobs = {}
            for action in ACTIONS:
                training_jobs["training/" + action] = (
                    str(self.config["action_device"][action]), self._training_command(action),
                    lambda action=action: self._training_complete(action),
                )
            self._run_parallel("training", training_jobs)

            self.set_stage("maps", "running")
            self._write_maps()
            self.set_stage("maps", "complete")

            self.set_stage("generation", "running", phase="build_shard_commands")
            generation_jobs = {}
            for shard in range(int(self.config["generation"]["num_shards"])):
                generation_jobs["generation/shard_%d" % shard] = (
                    str(self.config["devices"][shard]), list(self.manifest["commands"]["generation"][str(shard)]),
                    lambda shard=shard: self._generation_shard_complete(shard),
                )
            self._run_parallel("generation", generation_jobs)

            generation_audit = self.p["generation"] / "formal_generation_audit.json"
            self.set_stage("generation_audit", "running")
            if not self._generation_audit_complete():
                self._wait_one(
                    "generation_audit",
                    list(self.manifest["commands"]["generation_audit"]),
                    self._generation_audit_complete,
                )
            self.set_stage("generation_audit", "complete")

            bundle_manifest = self.p["bundle"] / "bundle_manifest.json"
            self.set_stage("detector_bundle", "running")
            if not self._bundle_complete():
                self._prepare_bundle_output()
                self._wait_one(
                    "detector_bundle", list(self.manifest["commands"]["detector_bundle"]),
                    self._bundle_complete,
                )
            self.set_stage("detector_bundle", "complete")

            self._wait_for_clean_gpus(
                tuple(str(device) for device in self.config["devices"]),
                "Deep PAD longest-event probe", "detector_probes",
            )
            deep_probe_jobs = {}
            for identity, command in self.manifest["commands"]["detector_deep_probes"].items():
                device = command[command.index("--device") + 1]
                deep_probe_jobs["detector_probe/" + identity] = (
                    device, list(command),
                    lambda identity=identity: self._deep_probe_complete(identity),
                )
            if len(deep_probe_jobs) != 10:
                raise AssertionError("formal benchmark requires exactly 10 Deep PAD probes")
            self._run_parallel("detector_probes", deep_probe_jobs)

            self.set_stage("detector_pairs", "running", phase="finalize_pair_commands")
            pair_jobs = self._detector_pair_jobs()
            self._run_parallel("detector_pairs", pair_jobs)

            self.set_stage("benchmark_merge", "running")
            if not self._merged_benchmark_complete():
                self._wait_one(
                    "benchmark_merge", list(self.manifest["commands"]["benchmark_merge"]),
                    self._merged_benchmark_complete,
                )
            self.set_stage("benchmark_merge", "complete")

            self.set_stage("benchmark_audit", "running")
            if not self._merged_benchmark_audit_complete():
                self._wait_one(
                    "benchmark_audit", list(self.manifest["commands"]["benchmark_audit"]),
                    self._merged_benchmark_audit_complete,
                )
            self.set_stage("benchmark_audit", "complete")

            self.set_stage("final_report", "running")
            self._write_final()
            if not self._final_complete():
                raise RuntimeError("final commit marker failed post-publication verification")
            self.set_stage("final_report", "complete")
            self.state.update({"status": "complete", "current_stage": None, "completed_unix_time": time.time()})
            self.save()
        except KeyboardInterrupt as exc:
            self._record_stopped(exc)
            raise
        except BaseException as exc:
            self._record_failure(exc)
            raise

    def run_gates_only(self) -> None:
        """Run only audited smoke/metadata gates; never enter formal training."""
        if self.config.get("formal_launch_authorized") is not False:
            raise PermissionError(
                "--gates-only requires formal_launch_authorized=false; review must precede the false->true transition"
            )
        self.state["status"] = "running_gates_only"
        self.save()
        try:
            self.set_stage("preflight", "running", gates_only=True)
            report = preflight(self.config, require_runtime_inputs=False)
            cli_failures = [item for item in report["cli_checks"] if not item["passed"]]
            if (
                report.get("passed") is not True
                or report.get("ready_for_gates") is not True
                or cli_failures or report["corpus_files_present"] != 5
            ):
                raise RuntimeError("launch-gate preflight is not ready")
            self.set_stage("preflight", "complete", report=report, gates_only=True)
            self._run_launch_gates()
            evidence = self._current_launch_gate_evidence()
            self.state.update({
                "status": "gates_complete_awaiting_formal_authorization",
                "current_stage": None,
                "gates_completed_unix_time": time.time(),
                "launch_gate_evidence": evidence,
            })
            self.save()
        except KeyboardInterrupt as exc:
            self._record_stopped(exc)
            raise
        except BaseException as exc:
            self._record_failure(exc)
            raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervise the strict 5-action/100k formal trajectory pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Check CLI/path contracts and print commands; run no formal stage.")
    parser.add_argument("--gates-only", action="store_true", help="Run corpus/E2E/100k-condition gates only; never start formal training.")
    parser.add_argument("--status", action="store_true", help="Print durable status and exit.")
    parser.add_argument("--request-stop", action="store_true", help="Request a cooperative stop between/polling commands.")
    parser.add_argument("--resume-failed", action="store_true", help="Retry only after the recorded failure cause was fixed.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = read_json(args.config.resolve())
    validate_config(config)
    p = paths(config)
    if args.status:
        if not p["state"].is_file():
            print(json.dumps({"status": "not_started", "run_root": str(p["run"])}, indent=2))
        else:
            print(json.dumps(read_json(p["state"]), indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if args.request_stop:
        p["run"].mkdir(parents=True, exist_ok=True)
        p["stop"].write_text("requested_unix_time=%.6f\n" % time.time(), encoding="utf-8")
        print(str(p["stop"]))
        return 0
    if args.dry_run:
        report = preflight(config, require_runtime_inputs=False)
        output = {"formal_work_started": False, "preflight": report, "command_manifest": command_manifest(config)}
        print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
        # A dry run is useful even while extraction is pending, but a CLI
        # mismatch is a code integration failure and returns nonzero.
        return 0 if all(item["passed"] for item in report["cli_checks"]) else 2
    if not args.gates_only and config.get("formal_launch_authorized") is not True:
        raise PermissionError(
            "formal launch is not authorized; only --gates-only is allowed until "
            "formal_launch_authorized=true is explicitly recorded"
        )
    if args.gates_only and config.get("formal_launch_authorized") is not False:
        raise PermissionError(
            "--gates-only requires formal_launch_authorized=false before constructing formal state"
        )
    if not args.gates_only:
        if not p["state"].is_file() or not gate_review_recorded(read_json(p["state"])):
            raise PermissionError(
                "formal launch requires a durable completed --gates-only review point; "
                "the formal Supervisor was not constructed"
            )
    if p["stop"].exists():
        if not args.resume_failed:
            raise RuntimeError("STOP_REQUESTED exists; remove it deliberately before resume")
        p["stop"].unlink()
    supervisor = Supervisor(config, allow_failed_resume=args.resume_failed)
    if args.gates_only:
        supervisor.run_gates_only()
    else:
        supervisor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
