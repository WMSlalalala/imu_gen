#!/usr/bin/env python3
"""Durable post-trajectory supervisor for exact paired total detection."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from estimator.trajectory_duration_report import (
    TRAJECTORY_DURATION_REPORT_SCHEMA,
    validate_trajectory_duration_report,
)
from estimator.total_detector_audit import (
    TOTAL_DETECTOR_REAUDIT_SCHEMA,
    validate_total_detector_outputs,
)
from estimator.trajectory_release import validate_trajectory_estimator_release
from estimator.runtime_benchmark import validate_trajectory_latency_report


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_snapshot(root: Path) -> Dict[str, Any]:
    files = sorted(
        path for directory in ("estimator", "runtime", "scripts", "orchestration", "tests")
        for path in (root / directory).rglob("*.py")
        if path.name != "formal_total_supervisor.pyc"
    )
    hashes = {str(path.relative_to(root)): sha256(path) for path in files}
    digest = hashlib.sha256()
    for name, value in hashes.items():
        digest.update((name + "\0" + value + "\n").encode("utf-8"))
    return {"files": hashes, "tree_sha256": digest.hexdigest()}


def artifact_snapshot(root: Path) -> Dict[str, Any]:
    files = sorted(path for path in Path(root).rglob("*") if path.is_file())
    records = {
        str(path.relative_to(root)): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in files
    }
    digest = hashlib.sha256()
    for name, record in records.items():
        digest.update((name + "\0" + record["sha256"] + "\0" + str(record["size_bytes"]) + "\n").encode("utf-8"))
    return {"files": records, "tree_sha256": digest.hexdigest(), "file_count": len(records)}


class Supervisor:
    def __init__(self, config_path: Path, *, resume_failed: bool = False):
        self.config_path = Path(config_path).resolve()
        self.config = read_json(self.config_path)
        if self.config.get("schema_version") != "formal_paired_total_config_v1":
            raise ValueError("formal total config schema mismatch")
        self.root = Path(self.config["pack_root"]).resolve()
        self.trajectory = Path(self.config["trajectory_run_root"]).resolve()
        self.output = Path(self.config["output_root"]).resolve()
        self.python = Path(self.config["python"]).resolve()
        self.state_path = self.output / "supervisor_status.json"
        self.lock_path = self.output / "supervisor.lock"
        self.logs = self.output / "logs"
        self.output.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.lock_stream = self.lock_path.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another formal total supervisor holds the lock") from exc
        snapshot = source_snapshot(self.root)
        if self.state_path.is_file():
            self.state = read_json(self.state_path)
            if self.state.get("config_sha256") != sha256(self.config_path):
                raise ValueError("formal total config changed after supervisor initialization")
            if self.state.get("source_tree_sha256") != snapshot["tree_sha256"]:
                raise ValueError("formal total source changed after supervisor initialization")
            if self.state.get("status") == "failed" and not resume_failed:
                raise RuntimeError("prior formal total run failed; use --resume-failed after fixing its cause")
        else:
            self.state = {
                "schema_version": "formal_paired_total_supervisor_state_v1",
                "status": "initialized", "current_stage": None,
                "created_unix_time": time.time(), "config": str(self.config_path),
                "config_sha256": sha256(self.config_path), "source_code": snapshot,
                "source_tree_sha256": snapshot["tree_sha256"], "stages": {}, "jobs": {},
            }
        self.state.update({"supervisor_pid": os.getpid(), "updated_unix_time": time.time()})
        self.save()

    def save(self) -> None:
        self.state["updated_unix_time"] = time.time()
        atomic_json(self.state_path, self.state)

    def stage(self, name: str, status: str, **extra: Any) -> None:
        self.state["current_stage"] = name
        self.state["status"] = "running"
        record = self.state.setdefault("stages", {}).setdefault(name, {})
        record.update({"status": status, "updated_unix_time": time.time(), **extra})
        self.save()

    def _assert_source(self) -> None:
        observed = source_snapshot(self.root)
        if observed["tree_sha256"] != self.state["source_tree_sha256"]:
            raise RuntimeError("formal total source tree changed while supervisor was waiting/running")

    def _run(self, name: str, command: Sequence[str]) -> None:
        self._assert_source()
        log = self.logs / (name.replace("/", "__") + ".log")
        with log.open("a", encoding="utf-8") as stream:
            stream.write("\n[%s] COMMAND %s\n" % (time.strftime("%F %T"), json.dumps(list(command))))
            stream.flush()
            process = subprocess.Popen(
                list(command), cwd=str(self.root), stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.state["jobs"][name] = {
                "status": "running", "pid": process.pid, "command": list(command),
                "command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest(),
                "log": str(log), "started_unix_time": time.time(),
            }
            self.save()
            while process.poll() is None:
                self.state["jobs"][name]["heartbeat_unix_time"] = time.time()
                self.save()
                time.sleep(20)
            code = int(process.returncode)
        self.state["jobs"][name].update({
            "status": "complete" if code == 0 else "failed", "returncode": code,
            "finished_unix_time": time.time(),
        })
        self.save()
        if code != 0:
            raise RuntimeError("job %s failed with return code %d; see %s" % (name, code, log))

    def _run_parallel(self, jobs: Mapping[str, Sequence[str]]) -> None:
        self._assert_source()
        processes = {}
        streams = {}
        try:
            for name, command in jobs.items():
                log = self.logs / (name.replace("/", "__") + ".log")
                stream = log.open("a", encoding="utf-8")
                stream.write("\n[%s] COMMAND %s\n" % (time.strftime("%F %T"), json.dumps(list(command))))
                stream.flush()
                process = subprocess.Popen(
                    list(command), cwd=str(self.root), stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes[name] = process
                streams[name] = stream
                self.state["jobs"][name] = {
                    "status": "running", "pid": process.pid, "command": list(command),
                    "log": str(log), "started_unix_time": time.time(),
                }
            self.save()
            while any(process.poll() is None for process in processes.values()):
                for name, process in processes.items():
                    if process.poll() is None:
                        self.state["jobs"][name]["heartbeat_unix_time"] = time.time()
                self.save()
                time.sleep(20)
            failures = []
            for name, process in processes.items():
                code = int(process.returncode)
                self.state["jobs"][name].update({
                    "status": "complete" if code == 0 else "failed", "returncode": code,
                    "finished_unix_time": time.time(),
                })
                if code:
                    failures.append("%s=%d" % (name, code))
            self.save()
            if failures:
                raise RuntimeError("parallel jobs failed: %s" % ", ".join(failures))
        finally:
            for stream in streams.values():
                stream.close()

    def _wait_for_trajectory(self) -> None:
        self.stage("wait_trajectory", "waiting")
        status_path = self.trajectory / "supervisor_status.json"
        audit_path = self.trajectory / "final_audit.json"
        while True:
            self._assert_source()
            observed = read_json(status_path) if status_path.is_file() else {"status": "not_started"}
            self.state["stages"]["wait_trajectory"].update({
                "trajectory_status": observed.get("status"),
                "trajectory_stage": observed.get("current_stage"),
                "heartbeat_unix_time": time.time(),
            })
            self.save()
            if observed.get("status") == "failed":
                raise RuntimeError("upstream formal trajectory supervisor failed")
            if observed.get("status") == "complete" and audit_path.is_file():
                audit = read_json(audit_path)
                if (
                    audit.get("schema_version") != "trajectory_formal_end_to_end_audit_v2"
                    or audit.get("passed") is not True or audit.get("commit_marker") is not True
                    or audit.get("generation", {}).get("n_fake") != 100_000
                ):
                    raise RuntimeError("upstream trajectory final audit is not a valid formal PASS")
                self.state["stages"]["wait_trajectory"].update({
                    "status": "complete", "trajectory_final_audit": str(audit_path),
                    "trajectory_final_audit_sha256": sha256(audit_path),
                })
                self.save()
                return
            time.sleep(30)

    def _gpu_processes(self) -> List[str]:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("nvidia-smi failed while checking clean GPUs")
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _wait_clean_gpus(self) -> None:
        self.stage("wait_clean_gpus", "waiting")
        while True:
            processes = self._gpu_processes()
            self.state["stages"]["wait_clean_gpus"].update({
                "observed_compute_processes": processes, "heartbeat_unix_time": time.time(),
            })
            self.save()
            if not processes:
                self.state["stages"]["wait_clean_gpus"]["status"] = "complete"
                self.save()
                return
            time.sleep(30)

    def _cmd(self, script: str, *args: Any) -> List[str]:
        return [str(self.python), "-u", str(self.root / "scripts" / script), *(str(value) for value in args)]

    def _single_action(self, action: str) -> None:
        split = Path(self.config["split_json"])
        pair_root = Path(self.config["real_pair_root"])
        consistency_root = Path(self.config["real_consistency_root"])
        real_imu_root = Path(self.config["real_imu_root"])
        fake_imu = self.output / "paired_fake_imu"
        trajectory_bundle = self.trajectory / "detector_bundle" / (action + ".npz")
        detector_root = self.trajectory / "benchmark" / "pairs"
        action_root = self.output / "actions" / action
        components = action_root / "components"
        components.mkdir(parents=True, exist_ok=True)

        feature_table = action_root / "paired_imu_features.npz"
        self._run("imu_features/" + action, self._cmd(
            "build_paired_imu_features.py", "--action", action,
            "--pair-index", pair_root / ("real_pair_index_%s.npz" % action),
            "--real-imu-source", real_imu_root / ("hmog_%s.npz" % action),
            "--fake-imu-root", fake_imu, "--pad-detectors-root", self.config["pad_detectors_root"],
            "--output", feature_table, "--manifest", action_root / "paired_imu_features_manifest.json",
            "--batch-size", self.config["feature_batch_size"],
        ))
        imu_component = components / "imu_base_val_test.npz"
        self._run("imu_scorers/" + action, self._cmd(
            "train_paired_imu_scorers.py", "--action", action,
            "--feature-table", feature_table, "--output-component", imu_component,
            "--artifact", action_root / "paired_imu_scorers.joblib",
            "--manifest", action_root / "paired_imu_scorers_manifest.json", "--seed", self.config["seed"],
        ))
        trajectory_component = components / "trajectory_base_val_test.npz"
        self._run("trajectory_scores/" + action, self._cmd(
            "build_trajectory_score_component.py", "--action", action,
            "--bundle", trajectory_bundle, "--detector-root", detector_root,
            "--real-pair-index", pair_root / ("real_pair_index_%s.npz" % action),
            "--fake-imu-root", fake_imu, "--output", trajectory_component,
            "--manifest", action_root / "trajectory_score_component_manifest.json",
        ))
        consistency = components / "consistency_all_base_pools.npz"
        self._run("merge_consistency/" + action, self._cmd(
            "merge_detector_components.py", "--component", "consistency", "--inputs",
            consistency_root / ("real_consistency_%s.npz" % action),
            self.output / "fake_consistency" / ("fake_consistency_%s.npz" % action),
            "--output", consistency, "--manifest", action_root / "consistency_merge_manifest.json",
        ))
        meta_paths = {}
        for component, source in (
            ("imu", imu_component), ("trajectory", trajectory_component), ("consistency", consistency)
        ):
            target = components / (component + "_meta.npz")
            self._run("meta_%s/%s" % (component, action), self._cmd(
                "remap_component_meta_pools.py", "--component", component, "--input", source,
                "--split-json", split, "--output", target,
                "--manifest", action_root / (component + "_meta_manifest.json"),
            ))
            meta_paths[component] = target
        paired = action_root / "paired_total_table.npz"
        self._run("paired_table/" + action, self._cmd(
            "build_paired_detector_table.py", "--imu-table", meta_paths["imu"],
            "--trajectory-table", meta_paths["trajectory"],
            "--consistency-table", meta_paths["consistency"], "--output", paired,
            "--manifest", action_root / "paired_total_table_manifest.json",
        ))
        self._run("total_detector/" + action, self._cmd(
            "train_total_detector.py", "--dataset", paired, "--action", action,
            "--output-dir", action_root / "total_detector",
            "--seed", self.config["seed"], "--bootstrap-replicates", self.config["bootstrap_replicates"],
            "--duration-bins", self.config["duration_bins"],
        ))
        self._run("total_latency/" + action, self._cmd(
            "benchmark_total_detector_runtime.py",
            "--artifact", action_root / "total_detector" / "total_detector.joblib",
            "--dataset", paired,
            "--action", action,
            "--output", action_root / "total_detector" / "runtime_latency.json",
            "--iterations", self.config["total_latency_iterations"],
            "--warmup-iterations", 20,
        ))

    def _build_trajectory_duration_reports(self) -> None:
        detector_root = self.trajectory / "benchmark" / "pairs"
        output_root = self.output / "trajectory_duration"
        jobs = {}
        for action in ACTIONS:
            jobs["trajectory_duration/" + action] = self._cmd(
                "build_trajectory_duration_report.py",
                "--action", action,
                "--bundle", self.trajectory / "detector_bundle" / (action + ".npz"),
                "--detector-root", detector_root,
                "--output", output_root / (action + ".json"),
                "--duration-bins", self.config["duration_bins"],
            )
        self._run_parallel(jobs)

    def _build_trajectory_estimator_release(self) -> None:
        self._run("trajectory_estimator_release", self._cmd(
            "build_formal_trajectory_estimator_release.py",
            "--bundle-dir", self.trajectory / "detector_bundle",
            "--detector-root", self.trajectory / "benchmark" / "pairs",
            "--output-dir", self.output / "trajectory_estimator_release",
            "--base-seed", self.config["trajectory_detector_base_seed"],
        ))

    def _benchmark_trajectory_estimator_release(self) -> None:
        self._run("trajectory_estimator_latency", self._cmd(
            "benchmark_trajectory_estimator_runtime.py",
            "--manifest", self.output / "trajectory_estimator_release" / "estimator_manifest.json",
            "--bundle-dir", self.trajectory / "detector_bundle",
            "--output", self.output / "trajectory_estimator_release" / "runtime_latency.json",
            "--device", self.config["devices"][0],
            "--iterations-per-action", self.config["trajectory_latency_iterations_per_action"],
            "--warmup-per-action", 2,
        ))

    def _write_final(self) -> None:
        tests_log = self.logs / "final_tests.log"
        self._run("final_tests", [str(self.python), "-m", "unittest", "discover", "-s", "tests", "-v"])
        paired_audit = self.output / "paired_fake_imu_audit_final.json"
        self._run("paired_fake_imu_reaudit", self._cmd(
            "audit_paired_fake_imu.py", "--trajectory-archive-root", self.trajectory / "generation",
            "--fake-imu-root", self.output / "paired_fake_imu",
            "--trajectory-generation-audit", self.trajectory / "generation" / "formal_generation_audit.json",
            "--output", paired_audit, "--samples-per-user", 200, "--confirm-formal-100k-paired",
        ))
        paired_audit_value = read_json(paired_audit)
        if paired_audit_value.get("passed") is not True or paired_audit_value.get("total_events") != 100_000:
            raise RuntimeError("final paired fake IMU re-audit is not a 100k PASS")
        kinematics_path = self.output / "trajectory_kinematics_audit.json"
        kinematics = read_json(kinematics_path)
        if (
            kinematics.get("schema_version") != "trajectory_kinematics_human_likeness_audit_v1"
            or kinematics.get("passed") is not True or kinematics.get("violations") != []
            or set(kinematics.get("actions", {})) != set(ACTIONS)
        ):
            raise RuntimeError("final trajectory kinematics audit is not a five-action PASS")
        trajectory_duration_receipts = {}
        detector_root = self.trajectory / "benchmark" / "pairs"
        for action in ACTIONS:
            duration_path = self.output / "trajectory_duration" / (action + ".json")
            duration_report = validate_trajectory_duration_report(
                duration_path,
                expected_action=action,
                expected_bundle=self.trajectory / "detector_bundle" / (action + ".npz"),
                expected_detector_root=detector_root,
                expected_bins=self.config["duration_bins"],
            )
            if (
                duration_report.get("schema_version") != TRAJECTORY_DURATION_REPORT_SCHEMA
                or duration_report.get("detector_count") != 5
            ):
                raise RuntimeError("%s base trajectory duration report is incomplete" % action)
            trajectory_duration_receipts[action] = {
                "path": str(duration_path),
                "sha256": sha256(duration_path),
                "detector_count": duration_report["detector_count"],
                "bin_spec": duration_report["bin_spec"],
            }
        if sum(value["detector_count"] for value in trajectory_duration_receipts.values()) != 25:
            raise RuntimeError("base trajectory duration report closure is not exactly 25 detectors")
        trajectory_release_manifest = self.output / "trajectory_estimator_release" / "estimator_manifest.json"
        trajectory_release = validate_trajectory_estimator_release(
            trajectory_release_manifest,
            expected_bundle_dir=self.trajectory / "detector_bundle",
            expected_detector_root=detector_root,
        )
        if trajectory_release.get("detector_count") != 25 or trajectory_release.get("formal_result") is not True:
            raise RuntimeError("formal trajectory runtime release is not a 25-detector closure")
        trajectory_latency_path = self.output / "trajectory_estimator_release" / "runtime_latency.json"
        trajectory_latency = validate_trajectory_latency_report(
            trajectory_latency_path,
            manifest_path=trajectory_release_manifest,
            bundle_dir=self.trajectory / "detector_bundle",
            expected_actions=ACTIONS,
            expected_iterations=self.config["trajectory_latency_iterations_per_action"],
            expected_detectors_per_action=5,
            expected_device=self.config["devices"][0],
            expected_load_deep=True,
            expected_warmup_iterations=2,
        )
        actions = {}
        required_total_files = {
            "total_detector.joblib", "summary.json", "score_dump.npz", "curves.npz",
            "bootstrap_summary.json", "bootstrap_replicates.npz",
            "duration_stratified_metrics.json", "training_manifest.json", "runtime_latency.json",
        }
        for action in ACTIONS:
            root = self.output / "actions" / action
            total_root = root / "total_detector"
            observed_total_files = {path.name for path in total_root.iterdir() if path.is_file()}
            if not required_total_files.issubset(observed_total_files):
                raise RuntimeError("%s total detector is missing required formal artifacts" % action)
            training_path = total_root / "training_manifest.json"
            training = read_json(training_path)
            if (
                training.get("schema_version") != "total_detector_training_manifest_v1"
                or training.get("status") != "complete" or training.get("action") != action
                or training.get("bootstrap_replicates") != 500
                or training.get("normalization_fit_pool") != "train_only"
                or training.get("threshold_selection_pool") != "validation_only"
                or training.get("test_role") != "fixed_threshold_reporting_only"
                or sha256(Path(training["dataset"])) != training.get("dataset_sha256")
            ):
                raise RuntimeError("%s total detector training manifest protocol/hash mismatch" % action)
            summary = read_json(total_root / "summary.json")
            if (
                summary.get("schema_version") != "imu_trajectory_total_detector_artifact_v1"
                or summary.get("action") != action or summary.get("requires_consistency") is not True
                or summary.get("normalization_fit_pool") != "train_only"
                or summary.get("threshold_selection_pool") != "validation_only"
                or set(summary.get("test_metrics", {})) != {"eer", "val_frr_le_5pct"}
            ):
                raise RuntimeError("%s total detector summary protocol mismatch" % action)
            for pool_name in ("validation_metrics", "test_metrics"):
                for point in ("eer", "val_frr_le_5pct"):
                    metric = summary[pool_name][point]
                    for name in ("fa", "frr", "auc"):
                        value = float(metric[name])
                        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                            raise RuntimeError("%s %s/%s/%s metric is invalid" % (action, pool_name, point, name))
            total_reaudit = validate_total_detector_outputs(
                total_root,
                dataset_path=root / "paired_total_table.npz",
                expected_action=action,
                expected_bootstrap_replicates=self.config["bootstrap_replicates"],
                expected_duration_bins=self.config["duration_bins"],
                require_runtime_latency=True,
                expected_latency_iterations=self.config["total_latency_iterations"],
            )
            if (
                total_reaudit.get("schema_version") != TOTAL_DETECTOR_REAUDIT_SCHEMA
                or total_reaudit.get("passed") is not True
                or total_reaudit.get("action") != action
                or total_reaudit.get("model_inference_exact") is not True
                or total_reaudit.get("bootstrap_recomputed_exact") is not True
                or total_reaudit.get("duration_report_recomputed_exact") is not True
                or total_reaudit.get("runtime_latency_validated") is not True
            ):
                raise RuntimeError("%s total detector independent re-audit failed" % action)
            total_reaudit_path = total_root / "formal_reaudit.json"
            atomic_json(total_reaudit_path, total_reaudit)
            expected_manifests = {
                "paired_total_table_manifest.json": "paired_detector_table_build_manifest_v1",
                "paired_imu_scorers_manifest.json": "paired_imu_scorer_training_manifest_v1",
                "trajectory_score_component_manifest.json": "paired_trajectory_score_component_manifest_v1",
                "consistency_merge_manifest.json": "detector_component_merge_manifest_v1",
                "imu_meta_manifest.json": "paired_total_detector_meta_pool_v1",
                "trajectory_meta_manifest.json": "paired_total_detector_meta_pool_v1",
                "consistency_meta_manifest.json": "paired_total_detector_meta_pool_v1",
            }
            manifest_receipts = {}
            for filename, schema in expected_manifests.items():
                path = root / filename
                value = read_json(path)
                if value.get("schema_version") != schema or value.get("status") != "complete":
                    raise RuntimeError("%s formal manifest is incomplete: %s" % (action, filename))
                manifest_receipts[filename] = sha256(path)
            snapshot = artifact_snapshot(root)
            if snapshot["file_count"] < 25:
                raise RuntimeError("%s formal action artifact closure is unexpectedly small" % action)
            actions[action] = {
                "training_manifest": str(training_path), "training_manifest_sha256": sha256(training_path),
                "summary": summary, "paired_table_sha256": sha256(root / "paired_total_table.npz"),
                "base_trajectory_duration_report": trajectory_duration_receipts[action],
                "total_detector_reaudit": str(total_reaudit_path),
                "total_detector_reaudit_sha256": sha256(total_reaudit_path),
                "manifest_receipts": manifest_receipts, "artifact_snapshot": snapshot,
            }
        report_path = self.output / "FINAL_REPORT.md"
        lines = [
            "# IMU + 轨迹综合检测器正式报告", "", "- 结论：PASS",
            "- 五动作 shared EventPlan paired fake：100,000",
            "- Level-1：5 个 paired IMU scorer + 每动作 5 个 trajectory PAD scorer",
            "- 基础轨迹时长指标：5 动作 × 5 检测器 = 25 组，train-only 分箱、全局 validation 固定阈值",
            "- 轨迹运行时封装：15 个 feature artifact + 10 个 deep best checkpoint，五动作实际 latency benchmark",
            "- Level-2：排除 base train；base validation 6/4 划分 meta train/validation；base test 固定汇报",
            "- 阈值：meta validation only；test fixed threshold；500 次 user bootstrap", "",
            "- 最终复核：模型推理、score identity、curve、bootstrap、时长指标全部从 paired dataset 独立重算", "",
            "## 动作结果", "",
        ]
        for action in ACTIONS:
            summary = actions[action]["summary"]
            lines.extend(["### %s" % action, "", "```json", json.dumps(summary, indent=2, sort_keys=True), "```", ""])
        report_path.write_text("\n".join(lines), encoding="utf-8")
        final = {
            "schema_version": "formal_paired_total_end_to_end_audit_v1", "passed": True,
            "commit_marker": True, "config_sha256": self.state["config_sha256"],
            "source_tree_sha256": self.state["source_tree_sha256"],
            "trajectory_final_audit": str(self.trajectory / "final_audit.json"),
            "trajectory_final_audit_sha256": sha256(self.trajectory / "final_audit.json"),
            "paired_fake_imu_audit": str(paired_audit), "paired_fake_imu_audit_sha256": sha256(paired_audit),
            "kinematics_audit": str(kinematics_path),
            "kinematics_audit_sha256": sha256(kinematics_path),
            "base_trajectory_duration_reports": trajectory_duration_receipts,
            "base_trajectory_duration_detector_count": 25,
            "trajectory_estimator_release": str(trajectory_release_manifest),
            "trajectory_estimator_release_sha256": sha256(trajectory_release_manifest),
            "trajectory_estimator_latency": str(trajectory_latency_path),
            "trajectory_estimator_latency_sha256": sha256(trajectory_latency_path),
            "trajectory_estimator_latency_summary": trajectory_latency,
            "actions": actions, "tests_log": str(tests_log),
            "report": str(report_path), "report_sha256": sha256(report_path),
        }
        atomic_json(self.output / "final_audit.json", final)

    def run(self) -> None:
        try:
            self._wait_for_trajectory()
            self.stage("trajectory_duration", "running")
            self._build_trajectory_duration_reports()
            self.stage("trajectory_duration", "complete")
            self.stage("trajectory_estimator_release", "running")
            self._build_trajectory_estimator_release()
            self.stage("trajectory_estimator_release", "complete")
            self._wait_clean_gpus()
            self.stage("trajectory_estimator_latency", "running")
            self._benchmark_trajectory_estimator_release()
            self.stage("trajectory_estimator_latency", "complete")
            self.stage("paired_fake_imu", "running")
            jobs = {}
            for shard, device in enumerate(self.config["devices"]):
                jobs["paired_fake_imu/shard_%d" % shard] = self._cmd(
                    "generate_paired_fake_imu.py", "--trajectory-archive-root", self.trajectory / "generation",
                    "--output-dir", self.output / "paired_fake_imu", "--device", device,
                    "--samples-per-user", 200, "--num-shards", self.config["paired_imu_shards"],
                    "--shard-id", shard, "--confirm-formal-100k-paired",
                )
            self._run_parallel(jobs)
            self.stage("paired_fake_imu", "complete")
            self.stage("paired_fake_imu_audit", "running")
            self._run("paired_fake_imu_audit", self._cmd(
                "audit_paired_fake_imu.py", "--trajectory-archive-root", self.trajectory / "generation",
                "--fake-imu-root", self.output / "paired_fake_imu",
                "--trajectory-generation-audit", self.trajectory / "generation" / "formal_generation_audit.json",
                "--output", self.output / "paired_fake_imu_audit.json",
                "--samples-per-user", 200, "--confirm-formal-100k-paired",
            ))
            self.stage("paired_fake_imu_audit", "complete")
            self.stage("fake_consistency", "running")
            self._run("fake_consistency", self._cmd(
                "build_fake_consistency_components.py", "--trajectory-archive-root", self.trajectory / "generation",
                "--fake-imu-root", self.output / "paired_fake_imu", "--split-json", self.config["split_json"],
                "--output-dir", self.output / "fake_consistency", "--samples-per-user", 200,
                "--confirm-formal-100k-paired",
            ))
            self.stage("fake_consistency", "complete")
            self.stage("trajectory_kinematics", "running")
            self._run("trajectory_kinematics", self._cmd(
                "audit_trajectory_kinematics.py", "--bundle-dir", self.trajectory / "detector_bundle",
                "--output", self.output / "trajectory_kinematics_audit.json", "--fail-on-violation",
            ))
            self.stage("trajectory_kinematics", "complete")
            for action in ACTIONS:
                self.stage("total_action_" + action, "running")
                self._single_action(action)
                self.stage("total_action_" + action, "complete")
            self.stage("final_audit", "running")
            self._write_final()
            self.stage("final_audit", "complete")
            self.state.update({
                "status": "complete", "current_stage": None, "completed_unix_time": time.time(),
                "final_audit": str(self.output / "final_audit.json"),
            })
            self.save()
        except BaseException as exc:
            self.state.update({
                "status": "failed", "failed_unix_time": time.time(),
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            self.save()
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-failed", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    state = Path(config["output_root"]) / "supervisor_status.json"
    if args.status:
        print(json.dumps(read_json(state) if state.is_file() else {"status": "not_started"}, indent=2, sort_keys=True))
        return
    Supervisor(args.config, resume_failed=args.resume_failed).run()


if __name__ == "__main__":
    main()
