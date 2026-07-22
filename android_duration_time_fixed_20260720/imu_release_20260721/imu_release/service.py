"""One fail-closed facade for the formal IMU online and cache backends."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
RELEASE_SCHEMA = "audited_imu_release_result_v1"
DEFAULT_FORMAL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_ROOT = DEFAULT_FORMAL_ROOT.parent
DEFAULT_CACHE_ROOT = DEFAULT_FORMAL_ROOT / "user_cache_runtime_active_len2"
DEFAULT_QUERY_SOURCE = (
    DEFAULT_FINAL_ROOT / "android_user_cache_xytime_full_20260710" / "scripts" / "query_user_cache.py"
)
DEFAULT_PHYSICAL_ROOT = DEFAULT_FINAL_ROOT / "android_physical_layer_20260709"


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def _load_query_module(source: Path):
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError("canonical cache query source is missing: %s" % path)
    name = "_audited_imu_release_query_user_cache"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load cache query source: %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_result(result: Dict[str, Any], action: str) -> Dict[str, Any]:
    if str(result.get("action")) != action:
        raise ValueError("IMU result action mismatch")
    active = np.ascontiguousarray(result.get("active_imu"), dtype=np.float32)
    if active.ndim != 2 or active.shape[1] != 6 or active.shape[0] < 1:
        raise ValueError("active_imu must have shape [N,6]")
    if not np.all(np.isfinite(active)):
        raise ValueError("active_imu contains non-finite values")
    relative = np.asarray(result.get("relative_timestamps_ns"), dtype=np.int64)
    if relative.shape != (active.shape[0],) or relative[0] != 0 or np.any(np.diff(relative) <= 0):
        raise ValueError("relative IMU timeline is invalid")
    if not np.isclose(float(result.get("hz", np.nan)), 100.0, rtol=0.0, atol=1.0e-6):
        raise ValueError("formal IMU output must be 100 Hz")
    result["active_imu"] = active
    result["release_schema_version"] = RELEASE_SCHEMA
    return result


class IMUReleaseService:
    """Expose an audited cache query or true online diffusion backend.

    ``cache`` is low latency and can match duration/orientation/XY/n_letters.
    It cannot invent unsupported EventPlan conditions.  Use ``online`` for
    exact text/n_keys, pinch span and per-event diffusion noise conditions.
    """

    def __init__(
        self,
        *,
        mode: str = "cache",
        formal_root: Path = DEFAULT_FORMAL_ROOT,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        query_source: Path = DEFAULT_QUERY_SOURCE,
        physical_root: Path = DEFAULT_PHYSICAL_ROOT,
        seed: int = 42,
        device: Optional[str] = None,
        split: str = "test",
        verify_formal_audits: bool = True,
    ) -> None:
        if mode not in ("cache", "online"):
            raise ValueError("mode must be cache or online")
        self.mode = mode
        self.formal_root = Path(formal_root)
        self.cache_root = Path(cache_root)
        self.query_source = Path(query_source)
        self.physical_root = Path(physical_root)
        self.seed = int(seed)
        self.device = device
        self.split = str(split)
        self._lock = threading.RLock()
        self._cache_by_user: Dict[int, Any] = {}
        self._online = None
        self.audit = self._verify_audits() if verify_formal_audits else {"verified": False}

    def _verify_audits(self) -> Dict[str, Any]:
        status_path = self.formal_root / "results" / "formal_pipeline_status.txt"
        lines = [line.strip() for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines or not lines[-1].startswith("complete "):
            raise RuntimeError("IMU formal pipeline latest status is not complete")
        runtime = _read_json(self.formal_root / "results" / "runtime_active_len_cache_audit.json")
        if runtime.get("status") != "pass" or runtime.get("strict_formal_protocol") is not True:
            raise RuntimeError("IMU runtime cache audit did not pass strict formal protocol")
        protocol = runtime.get("protocol", {})
        observed = runtime.get("observed", {})
        expected = int(protocol.get("expected_total", -1))
        found = int(observed.get("found_npz", -2))
        checked = int(observed.get("checked_npz", -3))
        if (expected, found, checked) != (138200, 138200, 138200):
            raise RuntimeError("IMU runtime cache audit does not cover exactly 138,200 files")
        formal = _read_json(
            self.formal_root / "formal_eval_200" / "formal_results" / "formal_result_audit.json"
        )
        if formal.get("passed") is not True or int(formal.get("formal_rows", -1)) != 90:
            raise RuntimeError("IMU formal detector audit did not pass")
        return {
            "verified": True,
            "pipeline_status": lines[-1],
            "runtime_cache_files": checked,
            "formal_detector_rows": int(formal["formal_rows"]),
            "cache_schema_revision": formal.get("cache_schema_revision"),
        }

    def _cache(self, user_id: int):
        user = int(user_id)
        if user not in self._cache_by_user:
            module = _load_query_module(self.query_source)
            self._cache_by_user[user] = module.UserCache(self.cache_root, user, seed=self.seed + user)
        return self._cache_by_user[user]

    def _online_layer(self):
        if self._online is None:
            root_text = str(self.physical_root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            from android_imu_layer import AndroidIMUDiffusionLayer  # type: ignore

            self._online = AndroidIMUDiffusionLayer(
                seed=self.seed, device=self.device, split=self.split,
                protocol="fewshot_adv", method="diffusion",
            )
        return self._online

    def generate(self, action: str, **conditions: Any) -> Dict[str, Any]:
        action = str(action)
        if action not in ACTIONS:
            raise ValueError("unsupported action: %s" % action)
        with self._lock:
            if self.mode == "online":
                forbidden = {
                    "match_mode", "duration_tolerance_ms", "xy_tolerance_px"
                } & set(conditions)
                if forbidden:
                    raise ValueError("online backend does not accept cache matching fields: %s" % sorted(forbidden))
                result = self._online_layer().generate(action, **conditions)
                result["release_backend"] = "online_five_shot_diffusion"
                return _validate_result(result, action)

            unsupported = {
                "pinch_start_span", "pinch_end_span", "text", "n_keys", "sample_steps"
            } & {name for name, value in conditions.items() if value is not None}
            if unsupported:
                raise ValueError(
                    "cache backend cannot represent %s; use mode=online so EventPlan conditions are not ignored"
                    % sorted(unsupported)
                )
            if conditions.get("user_id") is None:
                raise ValueError("cache backend requires explicit user_id")
            allowed = {
                "duration_ms", "xy_start", "xy_end", "orientation_id", "n_letters",
                "match_mode", "duration_tolerance_ms", "xy_tolerance_px", "active_len",
                "start_time_ns",
            }
            unknown = set(conditions) - allowed - {
                "user_id", "noise_seed", "pinch_start_span", "pinch_end_span", "text",
                "n_keys", "sample_steps",
            }
            if unknown:
                raise ValueError("unknown cache conditions: %s" % sorted(unknown))
            kwargs = {name: conditions[name] for name in allowed if name in conditions}
            cache = self._cache(int(conditions["user_id"]))
            noise_seed = conditions.get("noise_seed")
            previous_rng = cache._rng
            if noise_seed is not None:
                if isinstance(noise_seed, bool) or int(noise_seed) != noise_seed or int(noise_seed) < 0:
                    raise ValueError("noise_seed must be a non-negative integer")
                cache._rng = random.Random(int(noise_seed))
            try:
                result = cache.query(action, **kwargs)
            finally:
                cache._rng = previous_rng
            result["release_backend"] = "audited_runtime_cache"
            result["selection_seed"] = None if noise_seed is None else int(noise_seed)
            return _validate_result(result, action)

    def health(self) -> Dict[str, Any]:
        return {
            "schema_version": "audited_imu_release_health_v1",
            "mode": self.mode,
            "actions": list(ACTIONS),
            "audit": dict(self.audit),
            "cache_root": str(self.cache_root),
            "online_source_root": str(self.physical_root),
            "loaded_users": sorted(self._cache_by_user),
            "online_loaded": self._online is not None,
        }


__all__ = ["ACTIONS", "IMUReleaseService", "RELEASE_SCHEMA"]
