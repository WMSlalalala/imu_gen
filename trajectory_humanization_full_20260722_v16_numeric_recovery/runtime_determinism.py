"""One fail-closed determinism contract shared by training and generation.

This module intentionally sets ``CUBLAS_WORKSPACE_CONFIG`` before importing
PyTorch.  Entry points that need CUDA determinism must import this module
before their first ``torch`` import, then call :func:`seed_everything`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Dict, Mapping


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

EXPECTED_RUNTIME_DETERMINISM: Dict[str, Any] = {
    "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
    "deterministic_algorithms_enabled": True,
    "deterministic_algorithms_warn_only": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
}


def seed_everything(seed: int) -> None:
    """Seed all RNGs and enable strict deterministic PyTorch execution."""

    import numpy as np
    import torch

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def runtime_determinism_audit() -> Dict[str, Any]:
    """Return the exact typed runtime contract persisted with artifacts."""

    import torch

    warn_only = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
    )
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(warn_only()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def runtime_determinism_matches(value: Any) -> bool:
    """Accept only the exact contract, including key set and Python types."""

    if not isinstance(value, dict):
        return False
    if set(value) != set(EXPECTED_RUNTIME_DETERMINISM):
        return False
    return all(
        type(value[key]) is type(expected) and value[key] == expected
        for key, expected in EXPECTED_RUNTIME_DETERMINISM.items()
    )


def require_strict_runtime_determinism() -> Dict[str, Any]:
    """Return the current audit or fail before publishing any artifact."""

    value = runtime_determinism_audit()
    if not runtime_determinism_matches(value):
        raise RuntimeError("runtime does not satisfy the strict determinism contract: %r" % value)
    return value


def runtime_determinism_sha256(value: Mapping[str, Any]) -> str:
    """Canonical digest used by numeric-only NPZ generation archives."""

    if not runtime_determinism_matches(value):
        raise ValueError("cannot digest a non-strict runtime determinism contract")
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


STRICT_RUNTIME_DETERMINISM_SHA256 = runtime_determinism_sha256(
    EXPECTED_RUNTIME_DETERMINISM
)
