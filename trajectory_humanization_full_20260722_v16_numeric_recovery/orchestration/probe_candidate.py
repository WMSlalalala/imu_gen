#!/usr/bin/env python3
"""Run one throughput candidate with auditable resource/runtime failures."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("missing candidate command")
    if args.result.exists():
        value = json.loads(args.result.read_text(encoding="utf-8"))
        if value.get("passed") is True or value.get("expected_resource_failure") is True:
            print("existing audited candidate result: %s" % args.result)
            return 0
        raise FileExistsError("unrecognized existing candidate result: %s" % args.result)
    started = time.time()
    if args.timeout_seconds < 0:
        raise ValueError("timeout-seconds cannot be negative")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(
            timeout=(float(args.timeout_seconds) if args.timeout_seconds > 0 else None)
        )
    except subprocess.TimeoutExpired as error:
        partial = error.stdout or b""
        if isinstance(partial, str):
            partial = partial.encode("utf-8", "replace")
        os.killpg(process.pid, signal.SIGTERM)
        termination_signal = "SIGTERM"
        try:
            remainder, _ = process.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            termination_signal = "SIGKILL"
            remainder, _ = process.communicate()
        # TimeoutExpired.stdout may already contain the prefix also returned
        # by communicate; avoid duplicating it in the durable log.
        output = remainder or partial
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        atomic_json(args.result, {
            "schema_version": "trajectory_training_throughput_candidate_failure_v2",
            "passed": False,
            "expected_resource_failure": True,
            "failure_kind": "runtime_budget_exceeded",
            "timeout_seconds": float(args.timeout_seconds),
            "elapsed_seconds": time.time() - started,
            "process_group_terminated": process.poll() is not None,
            "termination_signal": termination_signal,
            "post_termination_wait_seconds": 10.0,
            "command_sha256": __import__("hashlib").sha256(
                json.dumps(command, sort_keys=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        })
        print("recorded expected candidate runtime-budget failure: %s" % args.result)
        return 0
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()
    if process.returncode == 0:
        if not args.result.is_file():
            raise RuntimeError("candidate returned zero without publishing result JSON")
        return 0
    text = output.decode("utf-8", "replace").lower()
    oom_tokens = (
        "cuda out of memory", "out of memory", "cublas_status_alloc_failed",
        "cuda error: memory allocation", "hip out of memory",
    )
    if not any(token in text for token in oom_tokens):
        return int(process.returncode)
    atomic_json(args.result, {
        "schema_version": "trajectory_training_throughput_candidate_failure_v1",
        "passed": False,
        "expected_resource_failure": True,
        "failure_kind": "cuda_oom",
        "returncode": int(process.returncode),
        "elapsed_seconds": time.time() - started,
        "command_sha256": __import__("hashlib").sha256(
            json.dumps(command, sort_keys=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    })
    print("recorded expected CUDA OOM candidate failure: %s" % args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
