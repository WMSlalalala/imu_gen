#!/usr/bin/env python3
"""Write a small public-safe recovery snapshot for the next agent."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_RUN = REPO_ROOT / (
    "trajectory_humanization_full_20260722_v16_numeric_recovery/results/"
    "formal_eventplan_v16_numeric_recovery_100epoch_100k_20260722"
)
TOTAL_RUN = REPO_ROOT / (
    "trajectory_estimator_pack_20260721/results/"
    "formal_paired_total_eventplan_v16_20260722"
)
ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def git_text(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def compact_action(name: str, health: dict[str, Any]) -> dict[str, Any]:
    action = health.get("actions", {}).get(name, {})
    metrics = action.get("metrics", {})
    worker = action.get("worker_progress", {})
    integrity = action.get("last_checkpoint_integrity", {})
    return {
        "status": action.get("status"),
        "completed_epoch": action.get("completed_epoch"),
        "total_epochs": action.get("total_epochs"),
        "device": action.get("configured_device"),
        "error_count": action.get("error_count"),
        "warning_count": action.get("warning_count"),
        "latest_train_loss": metrics.get("train_loss", {}).get("latest"),
        "latest_validation_loss": metrics.get("validation_loss", {}).get("latest"),
        "selected_best_epoch": action.get("best", {}).get("selected_epoch"),
        "worker_pid": action.get("process", {}).get("pid"),
        "worker_phase": worker.get("phase"),
        "worker_epoch_index": worker.get("epoch_index"),
        "worker_global_step": worker.get("global_step"),
        "worker_next_batch": worker.get("next_batch_in_epoch"),
        "worker_heartbeat_age_seconds": worker.get("heartbeat_age_seconds"),
        "last_checkpoint_verified": integrity.get("verified"),
        "last_checkpoint_sha256": integrity.get("checkpoint_sha256"),
    }


def main() -> int:
    now = time.time()
    local_time = dt.datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds")
    health = load_json(TRAJECTORY_RUN / "training_health.json") or {}
    trajectory_supervisor = load_json(TRAJECTORY_RUN / "supervisor_status.json") or {}
    total_supervisor = load_json(TOTAL_RUN / "supervisor_status.json") or {}

    snapshot = {
        "schema_version": "agent-handoff-cache-v1",
        "generated_unix_time": now,
        "generated_local_time": local_time,
        "purpose": "Lightweight recovery state; not a substitute for formal manifests or audits.",
        "repository": {
            "web_url": "https://github.com/WMSlalalala/imu_gen",
            "origin": git_text("remote", "get-url", "origin"),
            "local_head_before_this_sync": git_text("rev-parse", "HEAD"),
            "branch": git_text("branch", "--show-current"),
        },
        "active_goal": (
            "Finish and audit five-action trajectory training, 100k generation, "
            "25 trajectory PAD detectors, duration analyses, and 5 IMU+trajectory total detectors."
        ),
        "authoritative_documents": [
            "trajectory_estimator_pack_20260721/docs/IMU与轨迹交付状态及问题清单.md",
            "trajectory_estimator_pack_20260721/docs/轨迹生成方法与HMOG标准数据集测试说明.md",
        ],
        "local_artifact_paths": {
            "trajectory_run": str(TRAJECTORY_RUN),
            "total_detector_run": str(TOTAL_RUN),
            "formal_corpus": str(
                REPO_ROOT
                / "trajectory_humanization_full_20260713/results/trajectories_full_v2"
            ),
            "imu_release": str(
                REPO_ROOT
                / "android_duration_time_fixed_20260720/imu_release_20260721"
            ),
        },
        "trajectory_supervisor": {
            "status": trajectory_supervisor.get("status"),
            "current_stage": trajectory_supervisor.get("current_stage"),
            "pid": trajectory_supervisor.get("supervisor_pid"),
            "updated_unix_time": trajectory_supervisor.get("updated_unix_time"),
        },
        "health": {
            "overall_status": health.get("overall_status"),
            "error_count": health.get("error_count"),
            "warning_count": health.get("warning_count"),
            "checked_unix_time": health.get("checked_unix_time"),
            "gpus": health.get("gpu", {}).get("gpus", []),
        },
        "training": {name: compact_action(name, health) for name in ACTIONS},
        "total_detector_supervisor": {
            "status": total_supervisor.get("status"),
            "current_stage": total_supervisor.get("current_stage"),
            "pid": total_supervisor.get("supervisor_pid"),
            "updated_unix_time": total_supervisor.get("updated_unix_time"),
            "jobs": total_supervisor.get("jobs"),
        },
        "resume_rules": [
            "Read both authoritative Chinese documents before taking action.",
            "Treat results, checkpoints, manifests, and caches as local artifacts; verify current bytes and SHA before reuse.",
            "Do not modify the frozen v16 source/config or estimator-pack source/config.",
            "Do not mark the overall task complete until every formal artifact and audit passes.",
            "Record every detected or unresolved problem in the authoritative issue document.",
        ],
        "monitoring_policy": {
            "training_health_interval_seconds": 3600,
            "github_handoff_sync_interval_seconds": 1800,
            "note": "Formal supervisors remain continuous; the separate read-only health summary is sampled hourly.",
        },
        "publication_policy": {
            "tracked": "Source, tests, Chinese documents, protocols, and this lightweight handoff cache.",
            "local_only": "HMOG data, results, NPZ/NPY, runtime caches, checkpoints, logs, and secrets.",
            "local_only_recovery": "Use paths plus formal count/schema/identity/SHA manifests; never infer completion from this cache alone.",
        },
    }

    target_dir = REPO_ROOT / "agent_handoff"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "latest_state.json"
    fd, temporary_name = tempfile.mkstemp(prefix=".latest_state.", suffix=".tmp", dir=target_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(f"updated {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
