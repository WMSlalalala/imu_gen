"""Requirement-by-requirement audit for a generated trajectory unit."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from runtime_determinism import (
    EXPECTED_RUNTIME_DETERMINISM,
    STRICT_RUNTIME_DETERMINISM_SHA256,
)
from trajectory.data import CanonicalTrajectory, keystroke_zero_flight_flags
from trajectory.features import is_hmog_ascii_letter_keycode
from training.corpus import canonical_sample_sha256

from .archive import SCHEMA_VERSION
from .event_plan import EventPlan
from .android import (
    ACTION_DOWN, ACTION_MOVE, ACTION_POINTER_DOWN, ACTION_POINTER_UP,
    ACTION_UP, PHASE_DOWN, PHASE_MOVE, PHASE_UP, TYPE_B_NO_TRACKING_UPDATE,
)
from .protocol import (
    ACTION_TO_ID, CONDITION_REQUEST_DIGEST_SCHEMA, CONDITION_SET_DIGEST_SCHEMA,
    ID_TO_SPLIT, ORIENTATION_IDS, FixedUserSplit,
    ReferenceConditionPolicy, ReferenceRegistry, TrainGlobalPrior,
    canonical_condition_request_digest, condition_request_set_sha256,
    ddim_noise_seed, make_fake_id, numeric_sample_id, stable_seed,
)


def _hex_digest(array: np.ndarray) -> str:
    return bytes(np.asarray(array, np.uint8).tolist()).hex()


def _reference_flat(ref: CanonicalTrajectory) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.concatenate(ref.pointer_features, axis=0).astype(np.float32)
    contact = np.concatenate(ref.pointer_contact_masks, axis=0).astype(np.uint8)
    event = np.concatenate(ref.pointer_event_ids, axis=0).astype(np.int16)
    return features, contact, event


def _metadata_equal(a, ref: CanonicalTrajectory) -> bool:
    lengths = tuple(len(x) for x in ref.pointer_features) + tuple(0 for _ in range(2 - len(ref.pointer_features)))
    return (
        abs(float(a["duration_ms"]) - ref.duration_ms) <= 1e-7
        and tuple(int(x) for x in a["lengths"]) == lengths
        and np.array_equal(a["start_xy"], ref.start_xy)
        and np.array_equal(a["end_xy"], ref.end_xy)
        and np.array_equal(a["pointer_start_offset_ms"], ref.pointer_start_offset_ms)
        and np.array_equal(a["pointer_end_offset_ms"], ref.pointer_end_offset_ms)
        and int(a["orientation_id"]) == ref.orientation_id
        and int(a["n_keys"]) == ref.n_keys
        and int(a["n_letters"]) == ref.n_letters
    )


def audit_generated_unit(
    path: str,
    real_pool: Sequence[CanonicalTrajectory],
    fixed_split: FixedUserSplit,
    reference_registry: ReferenceRegistry,
    train_prior: TrainGlobalPrior,
    expected_count: int,
    expected_base_seed: int,
    expected_generation_batch_size: int,
    expected_ddim_steps: int = 50,
    max_aggregate_clip_rate: float = 0.05,
    max_event_clip_rate: float = 0.25,
    max_alpha_bar_final: float = 0.001,
) -> Dict[str, object]:
    """Fail closed on provenance, replay, geometry and Android lifecycle."""
    source = Path(path)
    with np.load(str(source), allow_pickle=False) as a:
        required_provenance = {
            "runtime_determinism_sha256", "ddim_eta_scalar",
            "event_plan_sha256",
        }
        if not required_provenance.issubset(a.files):
            raise ValueError("generated archive lacks runtime/DDIM provenance")
        if tuple(int(value) for value in a["schema_version"].tolist()) != SCHEMA_VERSION:
            raise ValueError("unsupported generated archive schema")
        archived_runtime_digest = _hex_digest(a["runtime_determinism_sha256"])
        if archived_runtime_digest != STRICT_RUNTIME_DETERMINISM_SHA256:
            raise ValueError("generated archive runtime determinism digest mismatch")
        eta_array = np.asarray(a["ddim_eta_scalar"])
        if (
            eta_array.shape != ()
            or eta_array.dtype != np.dtype(np.float32)
            or not np.isfinite(float(eta_array))
            or float(eta_array) != 0.0
        ):
            raise ValueError("generated archive violates deterministic DDIM eta=0")
        archived_eta = float(eta_array)
        if any("selector" in name.lower() for name in a.files):
            raise ValueError("selector fields are forbidden in formal generation output")
        for name in a.files:
            if a[name].dtype.kind not in "biufc":
                raise ValueError("non-numeric archive field %s" % name)
        n = int(a["fake_id"].size)
        if n != expected_count or int(a["ddim_steps_scalar"]) != expected_ddim_steps:
            raise ValueError("sample count/DDIM steps mismatch")
        training_steps = int(a["training_diffusion_steps_scalar"])
        alpha_bar_final = float(a["alpha_bar_final_scalar"])
        if training_steps < expected_ddim_steps or not 0.0 < alpha_bar_final < 1.0:
            raise ValueError("invalid checkpoint diffusion schedule metadata")
        if alpha_bar_final > max_alpha_bar_final:
            raise ValueError(
                "checkpoint terminal alpha_bar %.8f is too high for a pure-Gaussian DDIM start" % alpha_bar_final
            )
        if len(set(a["fake_id"].tolist())) != n:
            raise ValueError("duplicate fake ids")
        action_id = int(a["action_id_scalar"])
        action = next((name for name, value in ACTION_TO_ID.items() if value == action_id), None)
        if action is None or action != train_prior.action:
            raise ValueError("action/prior mismatch")
        user_values = np.unique(a["user_id"])
        split_values = np.unique(a["split_id"])
        if user_values.size != 1 or split_values.size != 1:
            raise ValueError("unit mixes users/splits")
        user_id = int(user_values[0])
        split = ID_TO_SPLIT[int(split_values[0])]
        if fixed_split.split_for_user(user_id) != split:
            raise ValueError("unit contradicts fixed user split")
        archived_base_seed = int(a["generation_base_seed_scalar"])
        if archived_base_seed != int(expected_base_seed):
            raise ValueError("generation base seed mismatch")
        archived_batch_size = int(a["generation_batch_size_scalar"])
        if archived_batch_size != int(expected_generation_batch_size):
            raise ValueError("generation batch size mismatch")
        sample_indices = np.asarray(a["sample_index"], np.int64)
        expected_indices = np.arange(n, dtype=np.int64)
        if not np.array_equal(sample_indices, expected_indices):
            raise ValueError("unit sample_index must be exactly 0..N-1 in archive order")
        expected_seeds = np.asarray([
            stable_seed(expected_base_seed, action, user_id, int(sample_index))
            for sample_index in expected_indices.tolist()
        ], np.int64)
        if not np.array_equal(np.asarray(a["seed"], np.int64), expected_seeds):
            raise ValueError("per-sample seed is not stable_seed(base_seed,action,user,sample_index)")
        expected_noise_seeds = np.asarray([
            ddim_noise_seed(
                int(expected_seeds[index]), action, user_id, int(sample_index)
            )
            for index, sample_index in enumerate(expected_indices.tolist())
        ], np.int64)
        if not np.array_equal(
            np.asarray(a["ddim_noise_seed"], np.int64), expected_noise_seeds
        ):
            raise ValueError(
                "DDIM noise seed is not the audited domain-separated derivation"
            )
        if (
            np.unique(expected_seeds).size != n
            or np.unique(expected_noise_seeds).size != n
            or np.intersect1d(expected_seeds, expected_noise_seeds).size
        ):
            raise ValueError(
                "ConditionRequest/DDIM seed domains are not unique and disjoint"
            )
        expected_fake_ids = np.asarray([
            make_fake_id(action, user_id, int(sample_index))
            for sample_index in expected_indices.tolist()
        ], np.int64)
        if not np.array_equal(np.asarray(a["fake_id"], np.int64), expected_fake_ids):
            raise ValueError("fake_id does not match action/user/sample_index")
        if a["condition_request_sha256"].shape != (n, 32):
            raise ValueError("condition request digest array must have shape [N,32]")
        if a["event_plan_sha256"].shape != (n, 32):
            raise ValueError("event plan digest array must have shape [N,32]")
        if _hex_digest(a["fixed_split_sha256"]) != fixed_split.source_sha256:
            raise ValueError("fixed split digest mismatch")
        if _hex_digest(a["reference_registry_sha256"]) != reference_registry.registry_sha256:
            raise ValueError("reference registry digest mismatch")
        if _hex_digest(a["train_prior_sha256"]) != train_prior.digest:
            raise ValueError("train prior digest mismatch")
        if not set(train_prior.source_user_ids.tolist()).issubset(set(fixed_split.train_users)):
            raise ValueError("train prior includes validation/test users")

        registry_ids = reference_registry.entries[(action, user_id, split)]
        if a["reference_event_ids"].shape != (n, 5) or not np.all(a["reference_event_ids"] == np.asarray(registry_ids)[None, :]):
            raise ValueError("five refs are not fixed to the training registry for all samples")
        if np.any(a["condition_source_code"] != 2):
            raise ValueError("condition source is not refs+train shrinkage prior")
        if any(int(x) not in registry_ids for x in a["condition_carrier_ref_id"]):
            raise ValueError("discrete carrier is outside the five refs")
        real_index = {numeric_sample_id(x.sample_id): x for x in real_pool}
        if len(real_index) != len(real_pool):
            raise ValueError("real pool has duplicate ids")
        refs = []
        for ref_id in registry_ids:
            if ref_id not in real_index:
                raise ValueError("registry ref missing from audit pool")
            ref = real_index[ref_id]
            if not ref.is_real or ref.action != action or ref.user_id != user_id or ref.split != split:
                raise ValueError("reference user/action/split leakage")
            refs.append(ref)
        expected_reference_hashes = np.stack([
            np.frombuffer(bytes.fromhex(canonical_sample_sha256(ref)), dtype=np.uint8)
            for ref in refs
        ])
        if a["reference_canonical_sha256"].shape != (n, 5, 32) or not np.all(
            a["reference_canonical_sha256"] == expected_reference_hashes[None, :, :]
        ):
            raise ValueError("generation reference canonical tensors differ from training SHA-256")
        condition_policy = ReferenceConditionPolicy(train_prior)

        for offset_name, flat_name in (("trajectory_offsets", "flat_trajectory_t_ms"), ("android_offsets", "flat_android_t_ms")):
            offsets = a[offset_name]
            if offsets.shape != (n + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) <= 0) or offsets[-1] != a[flat_name].size:
                raise ValueError("invalid %s" % offset_name)
        if a["key_offsets"].shape != (n + 1,) or a["key_offsets"][0] != 0 or np.any(np.diff(a["key_offsets"]) < 0):
            raise ValueError("invalid key_offsets")
        if a["key_offsets"][-1] != a["keycodes"].size:
            raise ValueError("key offset/code mismatch")
        if (
            "key_flight_offsets" not in a
            or "flat_zero_flight_after_key" not in a
            or "zero_flight_probability" not in a
        ):
            raise ValueError("archive lacks zero-flight condition provenance")
        key_flight_offsets = a["key_flight_offsets"]
        if (
            key_flight_offsets.shape != (n + 1,)
            or key_flight_offsets[0] != 0
            or np.any(np.diff(key_flight_offsets) < 0)
            or key_flight_offsets[-1] != a["flat_zero_flight_after_key"].size
        ):
            raise ValueError("invalid key_flight_offsets")
        expected_flight_counts = np.maximum(a["n_keys"].astype(np.int64) - 1, 0)
        if not np.array_equal(np.diff(key_flight_offsets), expected_flight_counts):
            raise ValueError("zero-flight condition count contradicts n_keys")
        if np.any(~np.isin(a["flat_zero_flight_after_key"], [0, 1])):
            raise ValueError("zero-flight conditions are not binary")
        expected_zero_probability = (
            ReferenceConditionPolicy.zero_flight_probability(refs, train_prior)
            if action == "keystroke"
            else 0.0
        )
        if not np.allclose(
            a["zero_flight_probability"], expected_zero_probability,
            rtol=0.0, atol=1.0e-15,
        ):
            raise ValueError("archived zero-flight probability is not refs+train-prior derived")

        exact_replay = 0
        metadata_copy = 0
        exact_key_sequence_copy = 0
        unique_key_sequences = set()
        key_endpoint_source_counts = {1: 0, 2: 0, 3: 0}
        key_endpoint_fallback_count = 0
        key_endpoint_zero_token_count = 0
        endpoint_errors: List[float] = []
        duration_errors: List[float] = []
        lifecycle_failures = 0
        zero_flight_boundary_count = 0
        positive_flight_boundary_count = 0
        condition_request_replay_count = 0
        for i in range(n):
            expected_request = condition_policy.sample(
                action, user_id, split, int(sample_indices[i]), expected_base_seed, refs
            )
            expected_request_digest = np.frombuffer(
                canonical_condition_request_digest(expected_request), dtype=np.uint8
            )
            if not np.array_equal(a["condition_request_sha256"][i], expected_request_digest):
                raise ValueError("archived ConditionRequest digest differs from deterministic replay")
            expected_plan_digest = np.frombuffer(bytes.fromhex(
                EventPlan.from_condition_request(
                    expected_request,
                    sample_id=str(expected_request.fake_id),
                    start_time_ns=None,
                    text=None,
                ).plan_sha256
            ), dtype=np.uint8)
            if not np.array_equal(a["event_plan_sha256"][i], expected_plan_digest):
                raise ValueError(
                    "archived EventPlan digest differs from deterministic replay"
                )
            scalar_fields = (
                ("fake_id", expected_request.fake_id),
                ("sample_index", expected_request.sample_index),
                ("seed", expected_request.seed),
                ("condition_carrier_ref_id", expected_request.carrier_ref_id),
                ("orientation_id", expected_request.orientation_id),
                ("n_keys", expected_request.n_keys),
                ("n_letters", expected_request.n_letters),
                ("condition_source_code", expected_request.condition_source_code),
            )
            if any(int(a[name][i]) != int(expected) for name, expected in scalar_fields):
                raise ValueError("archived scalar condition request differs from deterministic replay")
            float_scalar_fields = (
                ("duration_ms", expected_request.duration_ms, np.float32),
                ("zero_flight_probability", expected_request.zero_flight_probability, np.float64),
            )
            if any(
                np.asarray(a[name][i], dtype=dtype).item()
                != np.asarray(expected, dtype=dtype).item()
                for name, expected, dtype in float_scalar_fields
            ):
                raise ValueError("archived floating condition request differs from deterministic replay")
            array_fields = (
                ("lengths", expected_request.lengths, np.int32),
                ("pointer_start_offset_ms", expected_request.pointer_start_offset_ms, np.float32),
                ("pointer_end_offset_ms", expected_request.pointer_end_offset_ms, np.float32),
                ("start_xy", expected_request.start_xy, np.float32),
                ("end_xy", expected_request.end_xy, np.float32),
                ("pinch_span", expected_request.pinch_span, np.float32),
                ("pinch_angle", expected_request.pinch_angle, np.float32),
                ("key_endpoint_source_code", expected_request.key_endpoint_source_code, np.int8),
                ("screen_min_xy", expected_request.screen_min_xy, np.float32),
                ("screen_max_xy", expected_request.screen_max_xy, np.float32),
            )
            if any(
                not np.array_equal(np.asarray(a[name][i], dtype=dtype), np.asarray(expected, dtype=dtype))
                for name, expected, dtype in array_fields
            ):
                raise ValueError("archived array condition request differs from deterministic replay")
            ks, ke = (int(value) for value in a["key_offsets"][i:i + 2])
            fs, fe = (int(value) for value in key_flight_offsets[i:i + 2])
            if not np.array_equal(
                np.asarray(a["keycodes"][ks:ke], np.int32),
                np.asarray(expected_request.keycodes, np.int32),
            ) or not np.array_equal(
                np.asarray(a["flat_zero_flight_after_key"][fs:fe], np.uint8),
                np.asarray(expected_request.zero_flight_after_key, np.uint8),
            ):
                raise ValueError("archived variable-length condition request differs from deterministic replay")
            condition_request_replay_count += 1
            if not float(np.min(train_prior.duration_ms)) - 1e-3 <= float(a["duration_ms"][i]) <= float(np.max(train_prior.duration_ms)) + 1e-3:
                raise ValueError("duration escaped train-only prior range")
            if abs(float(a["duration_ms"][i]) - round(float(a["duration_ms"][i]))) > 1e-6:
                raise ValueError("generated duration is not on the HMOG integer-ms lattice")
            orientation_id = int(a["orientation_id"][i])
            orientation_index = ORIENTATION_IDS.index(orientation_id)
            if train_prior.screen_orientation_observed[orientation_index] == 0:
                raise ValueError("generated orientation has no train-only screen observations")
            if not np.array_equal(a["screen_min_xy"][i], train_prior.screen_min_xy_by_orientation[orientation_index]) or not np.array_equal(
                a["screen_max_xy"][i], train_prior.screen_max_xy_by_orientation[orientation_index]
            ):
                raise ValueError("screen bounds do not match the train-only orientation prior")
            trajectory_start, trajectory_end = (int(x) for x in a["trajectory_offsets"][i : i + 2])
            ts = a["flat_trajectory_t_ms"][trajectory_start:trajectory_end]
            if np.any(np.abs(ts - np.rint(ts)) > 1e-6):
                raise ValueError("generated trajectory timestamps expose a fractional-ms fake cue")
            features = a["flat_trajectory_features"][trajectory_start:trajectory_end]
            contact = a["flat_trajectory_contact_mask"][trajectory_start:trajectory_end]
            event_ids = a["flat_trajectory_event_id"][trajectory_start:trajectory_end]
            local_pointer_offsets = a["trajectory_pointer_offsets"][i]
            if local_pointer_offsets.shape != (3,) or local_pointer_offsets[0] != 0 or local_pointer_offsets[-1] != trajectory_end - trajectory_start:
                raise ValueError("invalid per-event pointer offsets")
            active_pointers = 2 if action == "pinch" else 1
            if action == "pinch" and float(np.max(a["pointer_start_offset_ms"][i, :2])) >= float(
                np.min(a["pointer_end_offset_ms"][i, :2])
            ):
                raise ValueError("pinch pointers do not overlap on the global union timeline")
            for pointer_id in range(active_pointers):
                ps, pe = int(local_pointer_offsets[pointer_id]), int(local_pointer_offsets[pointer_id + 1])
                pointer_t = ts[ps:pe]
                if pointer_t.size != int(a["lengths"][i, pointer_id]) or pointer_t.size < 2:
                    raise ValueError("generated pointer length/timeline mismatch")
                pointer_dt = np.diff(pointer_t)
                if not np.array_equal(
                    contact[ps:pe].astype(np.bool_), expected_request.contact_masks[pointer_id]
                ) or not np.array_equal(
                    event_ids[ps:pe].astype(np.int64), expected_request.event_ids[pointer_id]
                ):
                    raise ValueError("generated topology differs from deterministic condition request")
                if action == "keystroke":
                    pointer_contact = contact[ps:pe].astype(bool)
                    pointer_events = event_ids[ps:pe].astype(np.int64)
                    allowed_zero = (
                        pointer_contact[:-1]
                        & pointer_contact[1:]
                        & (pointer_events[:-1] >= 0)
                        & (pointer_events[1:] == pointer_events[:-1] + 1)
                    )
                    if np.any(pointer_dt[allowed_zero] != 0) or np.any(
                        pointer_dt[~allowed_zero] < 1.0 - 1e-6
                    ):
                        raise ValueError("generated keystroke timeline contradicts zero/positive flight topology")
                elif np.any(pointer_dt < 1.0 - 1e-6):
                    raise ValueError("integer-ms pointer timeline is not strictly increasing")
                duration_errors.extend([
                    abs(float(pointer_t[0]) - float(a["pointer_start_offset_ms"][i, pointer_id])),
                    abs(float(pointer_t[-1]) - float(a["pointer_end_offset_ms"][i, pointer_id])),
                ])
            if abs(float(np.min(ts)) - 0.0) > 1e-3 or abs(float(np.max(ts)) - float(a["duration_ms"][i])) > 1e-3:
                raise ValueError("trajectory does not span requested global duration")

            row_start, row_end = (int(x) for x in a["android_offsets"][i : i + 2])
            t = a["flat_android_t_ms"][row_start:row_end]
            if np.any(np.abs(t - np.rint(t)) > 1e-6):
                raise ValueError("Android timestamps expose a fractional-ms fake cue")
            x = a["flat_android_x"][row_start:row_end]
            y = a["flat_android_y"][row_start:row_end]
            pointer = a["flat_android_pointer_id"][row_start:row_end]
            phase = a["flat_android_phase"][row_start:row_end]
            actions = a["flat_android_action"][row_start:row_end]
            key_index = a["flat_android_key_index"][row_start:row_end]
            type_b = a["flat_android_type_b_tracking_value"][row_start:row_end]
            if int(a["contact_point_count"][i]) != row_end - row_start:
                raise ValueError("contact point count contradicts Android rows")
            if not 0 <= int(a["clipped_point_count"][i]) <= int(a["contact_point_count"][i]):
                raise ValueError("invalid clipped point count")
            expected_clip_rate = float(a["clipped_point_count"][i]) / max(int(a["contact_point_count"][i]), 1)
            if abs(float(a["clipped_point_rate"][i]) - expected_clip_rate) > 1e-6:
                raise ValueError("clipped point rate/count mismatch")
            if not np.all(np.isfinite(np.stack([t, x, y], axis=-1))) or np.any(a["flat_android_pressure"][row_start:row_end] < 0) or np.any(a["flat_android_pressure"][row_start:row_end] > 1):
                raise ValueError("invalid Android numeric/pressure values")
            low, high = a["screen_min_xy"][i], a["screen_max_xy"][i]
            if np.any(x < low[0] - 1e-4) or np.any(x > high[0] + 1e-4) or np.any(y < low[1] - 1e-4) or np.any(y > high[1] + 1e-4):
                raise ValueError("contact escaped train-only orientation screen bounds")
            if np.any((phase == PHASE_DOWN) & (type_b < 0)) or np.any((phase == PHASE_MOVE) & (type_b != TYPE_B_NO_TRACKING_UPDATE)) or np.any((phase == PHASE_UP) & (type_b != -1)):
                raise ValueError("invalid Type-B tracking lifecycle values")

            if action == "keystroke":
                ks, ke = (int(x) for x in a["key_offsets"][i : i + 2])
                keys = a["keycodes"][ks:ke]
                fs, fe = (int(x) for x in key_flight_offsets[i : i + 2])
                archived_zero = a["flat_zero_flight_after_key"][fs:fe].astype(bool)
                if keys.size != int(a["n_keys"][i]) or len(set(key_index.tolist()) - {-1}) != keys.size:
                    raise ValueError("keystroke key/contact count mismatch")
                inferred_letters = int(sum(
                    is_hmog_ascii_letter_keycode(int(value)) for value in keys
                ))
                if inferred_letters != int(a["n_letters"][i]):
                    raise ValueError("generated n_letters contradicts HMOG ASCII letter tokens")
                if any(np.array_equal(keys, np.asarray(ref.keycodes, keys.dtype)) for ref in refs):
                    exact_key_sequence_copy += 1
                unique_key_sequences.add(tuple(int(value) for value in keys.tolist()))
                source_codes = np.asarray(a["key_endpoint_source_code"][i], np.int64)
                if source_codes.shape != (2,) or np.any(~np.isin(source_codes, [1, 2, 3])):
                    raise ValueError("keystroke endpoint provenance code is missing")
                endpoint_tokens = (int(keys[0]), int(keys[-1]))
                for endpoint, (token, source_code) in enumerate(zip(endpoint_tokens, source_codes.tolist())):
                    exact_prior = np.any(
                        (train_prior.key_position_orientation == orientation_id)
                        & (train_prior.key_position_token == token)
                    )
                    exact_ref = any(
                        ref.orientation_id == orientation_id and np.any(ref.keycodes == token)
                        for ref in refs
                    )
                    expected_source = 1 if exact_prior and exact_ref else 2 if exact_prior else 3
                    if int(source_code) != expected_source:
                        raise ValueError("first/last key endpoint provenance contradicts available sources")
                    key_endpoint_source_counts[int(source_code)] += 1
                    key_endpoint_fallback_count += int(source_code == 3)
                    key_endpoint_zero_token_count += int(token == 0)
                for event_id in range(keys.size):
                    positions = np.flatnonzero(key_index == event_id)
                    if positions.size < 2 or phase[positions[0]] != PHASE_DOWN or phase[positions[-1]] != PHASE_UP or np.any(phase[positions[1:-1]] != PHASE_MOVE):
                        lifecycle_failures += 1
                    if np.any(np.diff(t[positions]) < 1.0 - 1e-6):
                        raise ValueError("generated key contact is not strictly increasing")
                    if event_id + 1 < keys.size:
                        next_positions = np.flatnonzero(key_index == event_id + 1)
                        flight = float(t[next_positions[0]] - t[positions[-1]])
                        if bool(archived_zero[event_id]):
                            if flight != 0.0:
                                raise ValueError("zero-flight condition did not produce same-ms UP/DOWN")
                            zero_flight_boundary_count += 1
                        else:
                            if flight < 1.0 - 1e-6:
                                raise ValueError("positive-flight condition did not produce positive time")
                            positive_flight_boundary_count += 1
                topology_zero = keystroke_zero_flight_flags(
                    contact[local_pointer_offsets[0]:local_pointer_offsets[1]],
                    event_ids[local_pointer_offsets[0]:local_pointer_offsets[1]],
                    int(keys.size),
                )
                if not np.array_equal(topology_zero, archived_zero):
                    raise ValueError("archive zero-flight flags contradict generated topology")
                contact_positions = np.flatnonzero(pointer == 0)
                endpoint_errors.extend([
                    float(np.linalg.norm(np.asarray([x[contact_positions[0]], y[contact_positions[0]]]) - a["start_xy"][i, 0])),
                    float(np.linalg.norm(np.asarray([x[contact_positions[-1]], y[contact_positions[-1]]]) - a["end_xy"][i, 0])),
                ])
            else:
                for pointer_id in range(active_pointers):
                    positions = np.flatnonzero(pointer == pointer_id)
                    if positions.size < 2 or phase[positions[0]] != PHASE_DOWN or phase[positions[-1]] != PHASE_UP or np.any(phase[positions[1:-1]] != PHASE_MOVE):
                        lifecycle_failures += 1
                    endpoint_errors.extend([
                        float(np.linalg.norm(np.asarray([x[positions[0]], y[positions[0]]]) - a["start_xy"][i, pointer_id])),
                        float(np.linalg.norm(np.asarray([x[positions[-1]], y[positions[-1]]]) - a["end_xy"][i, pointer_id])),
                    ])
            masked_actions = actions & 0xFF
            if action == "pinch":
                counts = {code: int(np.sum(masked_actions == code)) for code in (ACTION_DOWN, ACTION_POINTER_DOWN, ACTION_POINTER_UP, ACTION_UP)}
                if any(counts[code] != 1 for code in counts) or set(pointer.tolist()) != {0, 1}:
                    lifecycle_failures += 1
            elif action != "keystroke":
                if int(np.sum(masked_actions == ACTION_DOWN)) != 1 or int(np.sum(masked_actions == ACTION_UP)) != 1:
                    lifecycle_failures += 1

            generated_tuple = (features, contact, event_ids)
            if any(all(np.array_equal(left, right) for left, right in zip(generated_tuple, _reference_flat(ref))) for ref in refs):
                exact_replay += 1
            metadata_row = {name: a[name][i] for name in (
                "duration_ms", "lengths", "start_xy", "end_xy", "pointer_start_offset_ms",
                "pointer_end_offset_ms", "orientation_id", "n_keys", "n_letters",
            )}
            if any(_metadata_equal(metadata_row, ref) for ref in refs):
                metadata_copy += 1

        if lifecycle_failures:
            raise ValueError("Android lifecycle failures: %d" % lifecycle_failures)
        if exact_replay:
            raise ValueError("exact neural output/reference replay detected: %d" % exact_replay)
        if metadata_copy:
            raise ValueError("complete metadata copy detected: %d" % metadata_copy)
        if exact_key_sequence_copy:
            raise ValueError("exact five-shot key-sequence copy detected: %d" % exact_key_sequence_copy)
        if action != "keystroke":
            if "key_endpoint_source_code" not in a or np.any(a["key_endpoint_source_code"] != 0):
                raise ValueError("non-keystroke carries key endpoint provenance")
        aggregate_clip = float(np.sum(a["clipped_point_count"]) / max(int(np.sum(a["contact_point_count"])), 1))
        maximum_event_clip = float(np.max(a["clipped_point_rate"]))
        if aggregate_clip > max_aggregate_clip_rate or maximum_event_clip > max_event_clip_rate:
            raise ValueError(
                "screen clipping rate is abnormal: aggregate %.6f (limit %.6f), event max %.6f (limit %.6f)"
                % (aggregate_clip, max_aggregate_clip_rate, maximum_event_clip, max_event_clip_rate)
            )
        if endpoint_errors and max(endpoint_errors) > 1e-3:
            raise ValueError("endpoint hard constraint error %.6g" % max(endpoint_errors))
        if duration_errors and max(duration_errors) > 1e-3:
            raise ValueError("pointer lifetime hard constraint error %.6g" % max(duration_errors))

        result: Dict[str, object] = {
            "passed": True,
            "path": str(source.resolve()),
            "action": action,
            "user_id": user_id,
            "split": split,
            "n_fake": n,
            "ddim_steps": int(a["ddim_steps_scalar"]),
            "ddim_eta": archived_eta,
            "training_diffusion_steps": training_steps,
            "alpha_bar_final": alpha_bar_final,
            "generation_base_seed": archived_base_seed,
            "generation_batch_size": archived_batch_size,
            "runtime_determinism": dict(EXPECTED_RUNTIME_DETERMINISM),
            "runtime_determinism_sha256": archived_runtime_digest,
            "condition_request_seed_recomputed_count": n,
            "ddim_noise_seed_recomputed_count": n,
            "unique_condition_request_seed_count": n,
            "unique_ddim_noise_seed_count": n,
            "condition_and_noise_seed_domains_disjoint": True,
            "condition_request_seed_derivation":
            "stable_seed(base_seed,action,user_id,sample_index)",
            "ddim_noise_seed_derivation":
            "stable_seed(condition_request_seed_xor_0xDD1A50,action,user_id,sample_index)",
            "condition_request_replay_count": condition_request_replay_count,
            "condition_set_sha256": condition_request_set_sha256(zip(
                np.asarray(a["fake_id"], np.int64).tolist(),
                [bytes(row) for row in np.asarray(a["condition_request_sha256"], np.uint8)],
            )),
            "condition_request_digest_schema": CONDITION_REQUEST_DIGEST_SCHEMA,
            "condition_set_digest_schema": CONDITION_SET_DIGEST_SCHEMA,
            "checkpoint_sha256": _hex_digest(a["checkpoint_sha256"]),
            "fixed_split_sha256": _hex_digest(a["fixed_split_sha256"]),
            "reference_registry_sha256": _hex_digest(a["reference_registry_sha256"]),
            "train_prior_sha256": _hex_digest(a["train_prior_sha256"]),
            "fixed_reference_ids": [int(x) for x in registry_ids],
            "unique_reference_rows": int(np.unique(a["reference_event_ids"], axis=0).shape[0]),
            "exact_replay_count": exact_replay,
            "exact_metadata_copy_count": metadata_copy,
            "exact_key_sequence_copy_count": exact_key_sequence_copy,
            "complete_key_sequence_copy_count": exact_key_sequence_copy,
            "unique_key_sequence_count": len(unique_key_sequences),
            "key_endpoint_source_counts": {str(key): int(value) for key, value in key_endpoint_source_counts.items()},
            "key_endpoint_orientation_fallback_count": int(key_endpoint_fallback_count),
            "key_endpoint_zero_token_count": int(key_endpoint_zero_token_count),
            "zero_flight_probability": float(expected_zero_probability),
            "zero_flight_boundary_count": int(zero_flight_boundary_count),
            "positive_flight_boundary_count": int(positive_flight_boundary_count),
            "endpoint_error_max_px": 0.0 if not endpoint_errors else float(max(endpoint_errors)),
            "pointer_time_error_max_ms": 0.0 if not duration_errors else float(max(duration_errors)),
            "aggregate_clipped_point_rate": aggregate_clip,
            "max_event_clipped_point_rate": maximum_event_clip,
            "train_prior_source_users_subset_of_train": True,
            "selector_used": False,
        }
    return result
