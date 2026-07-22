"""Five-shot trajectory diffusion exposed through a shared EventPlan API."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


TRAJECTORY_PROJECT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
)
if str(TRAJECTORY_PROJECT) not in sys.path:
    sys.path.insert(0, str(TRAJECTORY_PROJECT))

from generation.android import record_from_generated  # noqa: E402
from generation.batching import build_sampling_batch  # noqa: E402
from generation.corpus import load_action_corpus  # noqa: E402
from generation.event_plan import (  # noqa: E402
    EventPlan,
    bind_explicit_event_conditions,
)
from generation.pad_export import record_from_android_trajectory  # noqa: E402
from generation.protocol import (  # noqa: E402
    ACTIONS,
    FixedUserSplit,
    ReferenceConditionPolicy,
    ReferenceRegistry,
    TrainGlobalPrior,
    canonical_condition_request_sha256,
)
from generation.sampler import load_model_checkpoint, sample_ddim_seeded_batch  # noqa: E402


DEFAULT_CORPUS_DIR = TRAJECTORY_PROJECT / "results" / "trajectories_full_v2"
DEFAULT_SPLIT_JSON = Path(
    "/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json"
)


def _read_map(value: Any, name: str) -> Dict[str, str]:
    if isinstance(value, Mapping):
        result = {str(key): str(path) for key, path in value.items()}
    else:
        path = Path(value)
        result = {str(key): str(item) for key, item in json.loads(path.read_text(encoding="utf-8")).items()}
    if set(result) != set(ACTIONS):
        raise ValueError("%s must map exactly the five actions" % name)
    return result


def _text_keycodes(text: str) -> Tuple[int, ...]:
    values = tuple(ord(character) for character in str(text))
    if not values:
        raise ValueError("keystroke text must not be empty")
    return values


@dataclass
class _ActionRuntime:
    action: str
    corpus: Any
    prior: TrainGlobalPrior
    registry: ReferenceRegistry
    model: torch.nn.Module
    checkpoint_sha256: str
    load_wall_ms: float


class TrajectoryDiffusionLayer:
    """Runtime trajectory generator with exactly five references per user/action.

    A call has two explicit phases:

    1. :meth:`resolve_plan` combines the five immutable references, the
       train-user-only prior and caller conditions into a persisted EventPlan.
    2. :meth:`generate_plan` samples trajectory diffusion from that exact plan.

    The plan can be sent independently to the already-completed IMU generator;
    both outputs then carry the same ``sample_id`` and ``plan_sha256``.
    """

    def __init__(
        self,
        *,
        checkpoint_map: Any,
        reference_registry_map: Any,
        corpus_dir: Path = DEFAULT_CORPUS_DIR,
        split_json: Path = DEFAULT_SPLIT_JSON,
        device: Optional[str] = None,
        base_seed: int = 20260721,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.split = FixedUserSplit.load(str(split_json), require_formal=True)
        self.checkpoint_map = _read_map(checkpoint_map, "checkpoint_map")
        self.registry_map = _read_map(reference_registry_map, "reference_registry_map")
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.base_seed = int(base_seed)
        self._runtime: Dict[str, _ActionRuntime] = {}

    def _get_runtime(self, action: str) -> Tuple[_ActionRuntime, bool]:
        if action not in ACTIONS:
            raise ValueError("unsupported action %r" % action)
        if action in self._runtime:
            return self._runtime[action], False
        started = time.perf_counter()
        corpus = load_action_corpus(
            str(self.corpus_dir / ("hmog_trajectory_%s.npz" % action)),
            action,
            self.split,
            user_ids=set(self.split.all_users),
            strict=True,
        )
        prior = TrainGlobalPrior.fit(action, corpus, self.split.train_users)
        registry = ReferenceRegistry.load(
            self.registry_map[action], self.split.source_sha256
        )
        model, checkpoint_digest = load_model_checkpoint(
            self.checkpoint_map[action], action, self.device,
            expected_registry_sha256=registry.registry_sha256,
            expected_split_sha256=self.split.source_sha256,
        )
        runtime = _ActionRuntime(
            action=action, corpus=corpus, prior=prior, registry=registry,
            model=model.eval(), checkpoint_sha256=checkpoint_digest,
            load_wall_ms=(time.perf_counter() - started) * 1000.0,
        )
        self._runtime[action] = runtime
        return runtime, True

    def resolve_plan(
        self,
        *,
        action: str,
        user_id: int,
        sample_index: int,
        sample_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        start_time_ns: Optional[int] = None,
        orientation_id: Optional[int] = None,
        start_xy: Optional[Sequence[Sequence[float]]] = None,
        end_xy: Optional[Sequence[Sequence[float]]] = None,
        pointer_start_offset_ms: Optional[Sequence[float]] = None,
        pointer_end_offset_ms: Optional[Sequence[float]] = None,
        text: Optional[str] = None,
        keycodes: Optional[Sequence[int]] = None,
        n_letters: Optional[int] = None,
    ) -> EventPlan:
        runtime, _ = self._get_runtime(action)
        user = int(user_id)
        split = self.split.split_for_user(user)
        refs = runtime.registry.resolve(runtime.corpus, action, user, split)
        explicit_keys = None
        explicit_letters = None
        resolved_text = None
        if action == "keystroke":
            if text is not None and keycodes is not None:
                raise ValueError("use either text or keycodes, not both")
            if text is not None:
                resolved_text = str(text)
                explicit_keys = _text_keycodes(resolved_text)
                explicit_letters = sum(character.isalpha() for character in resolved_text)
                if n_letters is not None and int(n_letters) != explicit_letters:
                    raise ValueError("n_letters contradicts text")
            elif keycodes is not None:
                explicit_keys = tuple(int(value) for value in keycodes)
                explicit_letters = int(n_letters) if n_letters is not None else sum(
                    65 <= value <= 90 or 97 <= value <= 122 for value in explicit_keys
                )
            elif n_letters is not None:
                raise ValueError("n_letters requires text or keycodes")
        elif text is not None or keycodes is not None or n_letters is not None:
            raise ValueError("text/keycodes/n_letters are keystroke-only conditions")
        base = ReferenceConditionPolicy(runtime.prior).sample(
            action, user, split, int(sample_index), self.base_seed, refs,
            explicit_keycodes=explicit_keys,
            explicit_n_letters=explicit_letters,
            explicit_orientation_id=orientation_id,
        )
        bound = bind_explicit_event_conditions(
            base, runtime.prior, refs,
            duration_ms=duration_ms,
            start_xy=start_xy,
            end_xy=end_xy,
            pointer_start_offset_ms=pointer_start_offset_ms,
            pointer_end_offset_ms=pointer_end_offset_ms,
        )
        if sample_id is None:
            digest = canonical_condition_request_sha256(bound)[:16]
            sample_id = "paired:%s:u%03d:i%06d:%s" % (
                action, user, int(sample_index), digest
            )
        return EventPlan.from_condition_request(
            bound, sample_id=str(sample_id), start_time_ns=start_time_ns,
            text=resolved_text,
        )

    def generate_plan(self, plan: EventPlan) -> Dict[str, Any]:
        started = time.perf_counter()
        runtime, loaded_now = self._get_runtime(plan.action)
        if runtime.prior.digest != plan.train_prior_digest:
            raise ValueError("event plan train-prior digest does not match runtime")
        if self.split.split_for_user(plan.user_id) != plan.split:
            raise ValueError("event plan user split does not match fixed split")
        refs = runtime.registry.resolve(runtime.corpus, plan.action, plan.user_id, plan.split)
        request = plan.to_condition_request()
        batch = build_sampling_batch([request], [refs], self.device)
        sample_started = time.perf_counter()
        output = sample_ddim_seeded_batch(
            runtime.model, batch, [int(plan.trajectory_noise_seed)], inference_steps=50
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        sample_wall_ms = (time.perf_counter() - sample_started) * 1000.0
        android = record_from_generated(output, batch, 0, request)
        raw_record, feature_vector = record_from_android_trajectory(
            android, sample_id=plan.sample_id, pool=plan.split
        )
        relative_ns = np.rint(np.asarray(android.android_t_ms, np.float64) * 1.0e6).astype(np.int64)
        absolute_ns = None
        if plan.start_time_ns is not None:
            absolute_ns = relative_ns + int(plan.start_time_ns)
        result = {
            "sample_id": plan.sample_id,
            "event_plan_sha256": plan.plan_sha256,
            "event_plan": plan.to_dict(),
            "action": plan.action,
            "user_id": int(plan.user_id),
            "split": plan.split,
            "duration_ms": float(plan.duration_ms),
            "orientation_id": int(plan.orientation_id),
            "relative_timestamps_ns": relative_ns,
            "timestamps_ns": absolute_ns,
            "x": np.asarray(android.android_x, np.float32),
            "y": np.asarray(android.android_y, np.float32),
            "pressure": np.asarray(android.android_pressure, np.float32),
            "size": np.asarray(android.android_size, np.float32),
            "pointer_id": np.asarray(android.android_pointer_id, np.int8),
            "phase": np.asarray(android.android_phase, np.int8),
            "android_action": np.asarray(android.android_action, np.int16),
            "key_index": np.asarray(android.android_key_index, np.int16),
            "keycode": np.asarray(android.android_keycode, np.int32),
            "frame_index": np.asarray(android.android_frame_index, np.int32),
            "raw_detector_record": raw_record,
            "feature_vector": feature_vector,
            "metadata": {
                "generator": "trajectory_diffusion",
                "checkpoint_sha256": runtime.checkpoint_sha256,
                "five_shot_reference_ids": list(plan.reference_ids),
                "condition_source_code": int(plan.condition_source_code),
                "model_loaded_this_call": bool(loaded_now),
                "model_load_ms": float(runtime.load_wall_ms if loaded_now else 0.0),
                "sampling_ms": float(sample_wall_ms),
                "generation_wall_ms": float((time.perf_counter() - started) * 1000.0),
            },
        }
        return result

    def generate(self, **conditions: Any) -> Dict[str, Any]:
        plan = self.resolve_plan(**conditions)
        return self.generate_plan(plan)

    def tap(self, *, user_id: int, sample_index: int, x: float, y: float, duration_ms: float,
            orientation_id: int, **kwargs: Any) -> Dict[str, Any]:
        return self.generate(
            action="tap", user_id=user_id, sample_index=sample_index,
            duration_ms=duration_ms, orientation_id=orientation_id,
            start_xy=(x, y), end_xy=(x, y), **kwargs
        )

    def scroll(self, *, user_id: int, sample_index: int, x0: float, y0: float, x1: float,
               y1: float, duration_ms: float, orientation_id: int, **kwargs: Any) -> Dict[str, Any]:
        return self.generate(
            action="scroll", user_id=user_id, sample_index=sample_index,
            duration_ms=duration_ms, orientation_id=orientation_id,
            start_xy=(x0, y0), end_xy=(x1, y1), **kwargs
        )

    def swipe(self, *, user_id: int, sample_index: int, x0: float, y0: float, x1: float,
              y1: float, duration_ms: float, orientation_id: int, **kwargs: Any) -> Dict[str, Any]:
        return self.generate(
            action="swipe", user_id=user_id, sample_index=sample_index,
            duration_ms=duration_ms, orientation_id=orientation_id,
            start_xy=(x0, y0), end_xy=(x1, y1), **kwargs
        )

    def pinch(
        self, *, user_id: int, sample_index: int,
        start_xy: Sequence[Sequence[float]], end_xy: Sequence[Sequence[float]],
        duration_ms: float, orientation_id: int, **kwargs: Any
    ) -> Dict[str, Any]:
        return self.generate(
            action="pinch", user_id=user_id, sample_index=sample_index,
            duration_ms=duration_ms, orientation_id=orientation_id,
            start_xy=start_xy, end_xy=end_xy, **kwargs
        )

    def type_text(
        self, *, user_id: int, sample_index: int, text: str,
        duration_ms: float, orientation_id: int, **kwargs: Any
    ) -> Dict[str, Any]:
        return self.generate(
            action="keystroke", user_id=user_id, sample_index=sample_index,
            text=text, duration_ms=duration_ms, orientation_id=orientation_id,
            **kwargs
        )


__all__ = ["TrajectoryDiffusionLayer", "DEFAULT_CORPUS_DIR", "DEFAULT_SPLIT_JSON"]
