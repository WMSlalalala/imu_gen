"""Clean-room trajectory feature extraction.

The 24 single-pointer feature names follow the public feature list in the AHB
paper.  The numerical definitions and all edge-case handling in this module
were implemented independently from that public description; no third-party
benchmark implementation is imported or copied.

Coordinates are expressed in input coordinate units (normally screen pixels),
timestamps in milliseconds, velocity in coordinate-units/second, acceleration
in coordinate-units/second**2, and angles in radians.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


EPS = 1.0e-12

# Persisted Feature PAD inputs must bind their numerical definitions, not only
# their column names.  v2 corrects three Table-6 semantics present in the first
# clean-room draft: signed dv/dt acceleration, signed deviation percentiles,
# and the median of the first five acceleration samples.  HMOG supplies a real
# UP coordinate, so this schema uses that observed endpoint rather than AHB's
# synthetic/dummy vanishing point convention.
TRAJECTORY_FEATURE_SCHEMA_VERSION = "trajectory_features_v2_ahb_table6_hmog_real_up"

# The paper defines the feature as the signed deviation whose magnitude is
# largest.  The audited public repository at commit
# 8924fa3e687af6f264d3b91a2d2f48faf8adfd8c currently returns max(abs(devs))
# before its signed implementation, making that implementation unreachable.
# We intentionally follow the paper and bind the choice to a machine-readable
# policy string so it cannot be mistaken for exact public-code reproduction.
MAX_DEVIATION_POLICY = "paper_signed_value_at_argmax_absolute_deviation"

# One shared, lossless keycode vocabulary is used by the diffusion model,
# generated-archive adapter, Feature PAD and Deep PAD.  HMOG contains the rare
# non-letter value 8230, so the former compact/Android-sized vocabularies are
# not sufficient.  Canonical raw-negative sentinels use token 0; every audited
# non-negative value through 16383 keeps its own identity.
KEYCODE_VOCAB_SIZE = 16384
KEYCODE_TOKEN_MAX = KEYCODE_VOCAB_SIZE - 1


# Public AHB-style order.  Keep this tuple stable: persisted models rely on it.
SINGLE_FINGER_FEATURE_NAMES: Tuple[str, ...] = (
    "duration",
    "startX",
    "startY",
    "endX",
    "endY",
    "displacement",
    "meanResultantLength",
    "direction",
    "v20",
    "v50",
    "v80",
    "a20",
    "a50",
    "a80",
    "v_last3_median",
    "maxDevSigned",
    "dev20",
    "dev50",
    "dev80",
    "avgDirection",
    "length",
    "ratio_end_to_length",
    "speed",
    "acc_first5pct_median",
)


PINCH_EXTRA_FEATURE_NAMES: Tuple[str, ...] = (
    "startSpan",
    "endSpan",
    "spanDelta",
    "absSpanDelta",
    "minSpan",
    "maxSpan",
    "meanSpan",
    "stdSpan",
    "spanPathLength",
    "spanRate20",
    "spanRate50",
    "spanRate80",
    "meanAbsSpanRate",
    "startAngle",
    "endAngle",
    "angleDelta",
    "anglePathLength",
    "angularSpeed20",
    "angularSpeed50",
    "angularSpeed80",
    "meanAbsAngularSpeed",
    "finger1Length",
    "finger2Length",
    "fingerLengthRatio",
    "fingerSpeedCorrelation",
)

PINCH_FEATURE_NAMES: Tuple[str, ...] = tuple(
    "center_" + name for name in SINGLE_FINGER_FEATURE_NAMES
) + PINCH_EXTRA_FEATURE_NAMES


KEYSTROKE_FEATURE_NAMES: Tuple[str, ...] = (
    "nKeys",
    "nLetters",
    "nUnique",
    "uniqueKeyRatio",
    "correctionRatio",
    "duration",
    "keyRate",
    "hold20",
    "hold50",
    "hold80",
    "holdMean",
    "holdStd",
    "flight20",
    "flight50",
    "flight80",
    "flightMean",
    "flightStd",
    "overlapFraction",
    "downDown20",
    "downDown50",
    "downDown80",
    "downDownMean",
    "downDownStd",
    "burstCount",
    "meanBurstSize",
    "maxBurstSize",
    "pauseFraction",
    "transitionDistance20",
    "transitionDistance50",
    "transitionDistance80",
    "transitionDistanceMean",
    "spatialPathLength",
    "hasUpTimes",
    "hasKeyXY",
)


def _as_finite_points(points: np.ndarray, name: str = "points") -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("%s must have shape [N, 2]" % name)
    if not np.all(np.isfinite(array)):
        raise ValueError("%s must contain only finite coordinates" % name)
    return array


def _as_finite_times(times_ms: np.ndarray, expected: int) -> np.ndarray:
    times = np.asarray(times_ms, dtype=np.float64)
    if times.ndim != 1 or len(times) != int(expected):
        raise ValueError("times_ms must have shape [N] matching the points")
    if not np.all(np.isfinite(times)):
        raise ValueError("times_ms must contain only finite values")
    if len(times) > 1 and np.any(np.diff(times) < 0.0):
        raise ValueError("timestamps must be monotonic non-decreasing")
    return times


def _last_at_duplicate_timestamps(
    values: np.ndarray, times_ms: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Coalesce equal consecutive timestamps using last-event-wins.

    Non-decreasing input guarantees all equal timestamps form one contiguous
    group.  Spatially repeated points at *different* timestamps are retained,
    because their dwell time contributes a legitimate zero-speed segment.
    """

    if len(times_ms) <= 1:
        return values.copy(), times_ms.copy()
    keep = np.concatenate((times_ms[1:] != times_ms[:-1], np.asarray([True])))
    return values[keep].copy(), times_ms[keep].copy()


def sanitize_timed_points(
    points: np.ndarray, times_ms: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a single-pointer trace and coalesce duplicate timestamps."""

    xy = _as_finite_points(points)
    times = _as_finite_times(times_ms, len(xy))
    if len(xy) == 0:
        raise ValueError("a trajectory must contain at least one point")
    return _last_at_duplicate_timestamps(xy, times)


def _percentile(values: np.ndarray, percentile: float) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0:
        return 0.0
    return float(np.percentile(array, percentile))


def _mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.mean(array)) if len(array) else 0.0


def _std(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.std(array)) if len(array) else 0.0


def _trajectory_primitives(
    points: np.ndarray, times_ms: np.ndarray
) -> Dict[str, np.ndarray]:
    if len(points) <= 1:
        empty_scalar = np.zeros((0,), dtype=np.float64)
        return {
            "dt_s": empty_scalar,
            "delta": np.zeros((0, 2), dtype=np.float64),
            "segment_length": empty_scalar,
            "velocity_vector": np.zeros((0, 2), dtype=np.float64),
            "speed": empty_scalar,
            "acceleration": empty_scalar,
        }

    dt_s = np.diff(times_ms) / 1000.0
    # Duplicate timestamps have already been coalesced, so this is a hard
    # internal invariant rather than a divide-by-zero workaround.
    if np.any(dt_s <= 0.0):
        raise RuntimeError("sanitized timestamps must be strictly increasing")
    delta = np.diff(points, axis=0)
    segment_length = np.linalg.norm(delta, axis=1)
    velocity_vector = delta / dt_s[:, None]
    speed = segment_length / dt_s

    if len(speed) >= 2:
        # AHB Table 6 F17--F19 use signed scalar-speed acceleration:
        # a_i = (v_i - v_{i-1}) / dt_i.  Direction changes at constant speed
        # therefore have zero tangential acceleration; deceleration is negative.
        acceleration = np.diff(speed) / dt_s[1:]
    else:
        acceleration = np.zeros((0,), dtype=np.float64)
    return {
        "dt_s": dt_s,
        "delta": delta,
        "segment_length": segment_length,
        "velocity_vector": velocity_vector,
        "speed": speed,
        "acceleration": acceleration,
    }


def _finite_feature_vector(values: Sequence[float], expected: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (int(expected),):
        raise RuntimeError("feature vector has an unexpected dimension")
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("feature extraction produced a non-finite value")
    return vector


def extract_single_finger_features(
    points: np.ndarray, times_ms: np.ndarray
) -> np.ndarray:
    """Return the fixed-order 24-dimensional AHB-style feature vector.

    Equal timestamps are coalesced with last-event-wins.  Equal coordinates at
    different timestamps are preserved as zero-speed dwell segments.  One-point
    traces are supported and yield finite zero-valued dynamics.
    """

    xy, times = sanitize_timed_points(points, times_ms)
    primitive = _trajectory_primitives(xy, times)
    segment_length = primitive["segment_length"]
    speeds = primitive["speed"]
    acceleration = primitive["acceleration"]

    start = xy[0]
    end = xy[-1]
    chord = end - start
    displacement = float(np.linalg.norm(chord))
    length = float(np.sum(segment_length))
    duration_ms = float(times[-1] - times[0]) if len(times) > 1 else 0.0

    moving = segment_length > EPS
    moving_delta = primitive["delta"][moving]
    if len(moving_delta):
        local_angles = np.arctan2(moving_delta[:, 1], moving_delta[:, 0])
        mean_cos = float(np.mean(np.cos(local_angles)))
        mean_sin = float(np.mean(np.sin(local_angles)))
        mean_resultant_length = float(np.hypot(mean_cos, mean_sin))
        average_direction = float(np.arctan2(mean_sin, mean_cos))
    else:
        mean_resultant_length = 0.0
        average_direction = 0.0

    if displacement > EPS:
        direction = float(np.arctan2(chord[1], chord[0]))
        relative = xy - start[None, :]
        signed_deviation = (
            chord[0] * relative[:, 1] - chord[1] * relative[:, 0]
        ) / displacement
        max_index = int(np.argmax(np.abs(signed_deviation)))
        max_deviation_signed = float(signed_deviation[max_index])
        deviation_values = signed_deviation
    else:
        # A point-to-point reference line is undefined for a zero chord.  Zero
        # is a neutral finite sentinel and is kept distinct from radial travel,
        # which remains represented by `length`.
        direction = 0.0
        max_deviation_signed = 0.0
        deviation_values = np.zeros((len(xy),), dtype=np.float64)

    efficiency = displacement / length if length > EPS else 0.0
    average_speed = length / (duration_ms / 1000.0) if duration_ms > 0.0 else 0.0
    if len(acceleration):
        # The persisted column retains AHB's legacy name
        # ``acc_first5pct_median``, but the published/benchmark implementation
        # takes the first five acceleration samples, not five percent.
        initial_acceleration = float(np.median(acceleration[:5]))
    else:
        initial_acceleration = 0.0

    values = (
        duration_ms,
        float(start[0]),
        float(start[1]),
        float(end[0]),
        float(end[1]),
        displacement,
        mean_resultant_length,
        direction,
        _percentile(speeds, 20.0),
        _percentile(speeds, 50.0),
        _percentile(speeds, 80.0),
        _percentile(acceleration, 20.0),
        _percentile(acceleration, 50.0),
        _percentile(acceleration, 80.0),
        _percentile(speeds[-3:], 50.0),
        max_deviation_signed,
        _percentile(deviation_values, 20.0),
        _percentile(deviation_values, 50.0),
        _percentile(deviation_values, 80.0),
        average_direction,
        length,
        efficiency,
        average_speed,
        initial_acceleration,
    )
    return _finite_feature_vector(values, len(SINGLE_FINGER_FEATURE_NAMES))


def extract_single_finger_feature_dict(
    points: np.ndarray, times_ms: np.ndarray
) -> Dict[str, float]:
    vector = extract_single_finger_features(points, times_ms)
    return dict(zip(SINGLE_FINGER_FEATURE_NAMES, vector.tolist()))


def _filled_unwrapped_angles(vectors: np.ndarray, spans: np.ndarray) -> np.ndarray:
    valid = spans > EPS
    if not np.any(valid):
        return np.zeros((len(spans),), dtype=np.float64)
    indices = np.arange(len(spans), dtype=np.float64)
    valid_indices = indices[valid]
    valid_angles = np.unwrap(np.arctan2(vectors[valid, 1], vectors[valid, 0]))
    # Interpolation also gives deterministic leading/trailing nearest fills.
    return np.interp(indices, valid_indices, valid_angles)


def extract_pinch_features(
    finger1_points: np.ndarray,
    finger2_points: np.ndarray,
    times_ms: np.ndarray,
) -> np.ndarray:
    """Extract center-path AHB features plus two-finger pinch dynamics.

    Both fingers must be sampled on the same timestamp sequence.  Duplicate
    timestamps are coalesced jointly, preserving the last complete pointer
    pair.  Temporary coincident fingers are handled by interpolating the
    unwrapped orientation from valid-span samples.
    """

    first = _as_finite_points(finger1_points, "finger1_points")
    second = _as_finite_points(finger2_points, "finger2_points")
    if len(first) != len(second):
        raise ValueError("both pinch fingers must contain the same number of points")
    times = _as_finite_times(times_ms, len(first))
    if len(first) == 0:
        raise ValueError("a pinch trajectory must contain at least one pointer pair")

    pair = np.stack((first, second), axis=1)
    pair, times = _last_at_duplicate_timestamps(pair, times)
    first = pair[:, 0, :]
    second = pair[:, 1, :]
    center = 0.5 * (first + second)
    center_features = extract_single_finger_features(center, times)

    vector = second - first
    span = np.linalg.norm(vector, axis=1)
    angles = _filled_unwrapped_angles(vector, span)
    primitive_first = _trajectory_primitives(first, times)
    primitive_second = _trajectory_primitives(second, times)

    if len(times) > 1:
        dt_s = np.diff(times) / 1000.0
        span_rate = np.diff(span) / dt_s
        angular_speed = np.diff(angles) / dt_s
    else:
        span_rate = np.zeros((0,), dtype=np.float64)
        angular_speed = np.zeros((0,), dtype=np.float64)

    first_length = float(np.sum(primitive_first["segment_length"]))
    second_length = float(np.sum(primitive_second["segment_length"]))
    longer_length = max(first_length, second_length)
    finger_length_ratio = (
        min(first_length, second_length) / longer_length
        if longer_length > EPS
        else 0.0
    )
    first_speed = primitive_first["speed"]
    second_speed = primitive_second["speed"]
    if (
        len(first_speed) >= 2
        and float(np.std(first_speed)) > EPS
        and float(np.std(second_speed)) > EPS
    ):
        speed_correlation = float(np.corrcoef(first_speed, second_speed)[0, 1])
        speed_correlation = float(np.clip(speed_correlation, -1.0, 1.0))
    else:
        speed_correlation = 0.0

    extras = (
        float(span[0]),
        float(span[-1]),
        float(span[-1] - span[0]),
        float(abs(span[-1] - span[0])),
        float(np.min(span)),
        float(np.max(span)),
        float(np.mean(span)),
        float(np.std(span)),
        float(np.sum(np.abs(np.diff(span)))),
        _percentile(span_rate, 20.0),
        _percentile(span_rate, 50.0),
        _percentile(span_rate, 80.0),
        _mean(np.abs(span_rate)),
        float(angles[0]),
        float(angles[-1]),
        float(angles[-1] - angles[0]),
        float(np.sum(np.abs(np.diff(angles)))),
        _percentile(angular_speed, 20.0),
        _percentile(angular_speed, 50.0),
        _percentile(angular_speed, 80.0),
        _mean(np.abs(angular_speed)),
        first_length,
        second_length,
        finger_length_ratio,
        speed_correlation,
    )
    values = np.concatenate((center_features, np.asarray(extras, dtype=np.float64)))
    return _finite_feature_vector(values, len(PINCH_FEATURE_NAMES))


def extract_pinch_feature_dict(
    finger1_points: np.ndarray,
    finger2_points: np.ndarray,
    times_ms: np.ndarray,
) -> Dict[str, float]:
    vector = extract_pinch_features(finger1_points, finger2_points, times_ms)
    return dict(zip(PINCH_FEATURE_NAMES, vector.tolist()))


def is_hmog_ascii_letter_keycode(keycode: int) -> bool:
    """Authoritative HMOG KeyPress.csv ASCII letter predicate."""

    value = int(keycode)
    return 65 <= value <= 90 or 97 <= value <= 122


def canonical_keycode_feature_token(keycode: int) -> str:
    """Map a canonical HMOG keycode to the feature-level symbolic token.

    HMOG letter keys are ASCII (A-Z/a-z), not Android KEYCODE_A..Z 29..54.
    Non-letters retain identity through an explicit prefix; canonical token 0
    (all raw negative sentinels) is therefore non-alphabetic.
    """

    value = int(keycode)
    if value < 0:
        raise ValueError("canonical feature keycode must be non-negative")
    if is_hmog_ascii_letter_keycode(value):
        return chr(value)
    return "keycode_%d" % value


def _normalise_keys(keys: Iterable[object]) -> Tuple[str, ...]:
    if isinstance(keys, str):
        return tuple(keys)
    return tuple(str(key) for key in keys)


def _correction_key(token: str) -> bool:
    return token.strip().lower() in {
        "backspace",
        "delete",
        "del",
        "<bs>",
        "keycode_del",
        "\b",
    }


def extract_keystroke_features(
    keys: Iterable[object],
    down_times_ms: np.ndarray,
    up_times_ms: Optional[np.ndarray] = None,
    key_points: Optional[np.ndarray] = None,
    pause_threshold_ms: float = 500.0,
) -> np.ndarray:
    """Extract timing, burst, correction, and optional key-position features.

    Keystrokes are discrete events: equal DOWN timestamps are retained rather
    than coalesced.  DOWN timestamps must be non-decreasing.  Each UP must be
    no earlier than its matching DOWN; overlapping keys are valid and appear as
    negative flight time.  Missing UP or XY streams produce zero-valued feature
    groups together with explicit availability flags.
    """

    tokens = _normalise_keys(keys)
    down = np.asarray(down_times_ms, dtype=np.float64)
    if down.ndim != 1 or len(down) != len(tokens):
        raise ValueError("down_times_ms must match the number of keys")
    if not np.all(np.isfinite(down)):
        raise ValueError("down_times_ms must contain only finite values")
    if len(down) > 1 and np.any(np.diff(down) < 0.0):
        raise ValueError("key DOWN timestamps must be monotonic non-decreasing")
    if not np.isfinite(float(pause_threshold_ms)) or pause_threshold_ms < 0.0:
        raise ValueError("pause_threshold_ms must be finite and non-negative")

    has_up = up_times_ms is not None
    if has_up:
        up = np.asarray(up_times_ms, dtype=np.float64)
        if up.ndim != 1 or len(up) != len(tokens):
            raise ValueError("up_times_ms must match the number of keys")
        if not np.all(np.isfinite(up)):
            raise ValueError("up_times_ms must contain only finite values")
        if np.any(up < down):
            raise ValueError("every key UP timestamp must be >= its DOWN timestamp")
        hold = up - down
        flight = down[1:] - up[:-1] if len(down) > 1 else np.zeros((0,))
    else:
        up = np.zeros((0,), dtype=np.float64)
        hold = np.zeros((0,), dtype=np.float64)
        flight = np.zeros((0,), dtype=np.float64)

    down_down = np.diff(down) if len(down) > 1 else np.zeros((0,))
    if len(tokens) == 0:
        duration_ms = 0.0
    elif has_up:
        duration_ms = float(np.max(up) - down[0])
    else:
        duration_ms = float(down[-1] - down[0])
    key_rate = len(tokens) / (duration_ms / 1000.0) if duration_ms > 0.0 else 0.0

    lower_tokens = tuple(token.lower() for token in tokens)
    n_letters = sum(len(token) == 1 and token.isalpha() for token in tokens)
    n_unique = len(set(lower_tokens))
    correction_count = sum(_correction_key(token) for token in tokens)
    unique_ratio = n_unique / float(len(tokens)) if tokens else 0.0
    correction_ratio = correction_count / float(len(tokens)) if tokens else 0.0

    if tokens:
        pause_mask = down_down > float(pause_threshold_ms)
        burst_count = 1 + int(np.sum(pause_mask))
        boundaries = np.flatnonzero(pause_mask) + 1
        burst_sizes = np.diff(
            np.concatenate((np.asarray([0]), boundaries, np.asarray([len(tokens)])))
        )
        mean_burst_size = float(np.mean(burst_sizes))
        max_burst_size = float(np.max(burst_sizes))
    else:
        pause_mask = np.zeros((0,), dtype=bool)
        burst_count = 0
        mean_burst_size = 0.0
        max_burst_size = 0.0
    pause_fraction = _mean(pause_mask.astype(np.float64))

    has_xy = key_points is not None
    if has_xy:
        xy = _as_finite_points(np.asarray(key_points), "key_points")
        if len(xy) != len(tokens):
            raise ValueError("key_points must match the number of keys")
        transition_distance = (
            np.linalg.norm(np.diff(xy, axis=0), axis=1)
            if len(xy) > 1
            else np.zeros((0,), dtype=np.float64)
        )
    else:
        transition_distance = np.zeros((0,), dtype=np.float64)

    values = (
        float(len(tokens)),
        float(n_letters),
        float(n_unique),
        unique_ratio,
        correction_ratio,
        duration_ms,
        key_rate,
        _percentile(hold, 20.0),
        _percentile(hold, 50.0),
        _percentile(hold, 80.0),
        _mean(hold),
        _std(hold),
        _percentile(flight, 20.0),
        _percentile(flight, 50.0),
        _percentile(flight, 80.0),
        _mean(flight),
        _std(flight),
        _mean((flight < 0.0).astype(np.float64)),
        _percentile(down_down, 20.0),
        _percentile(down_down, 50.0),
        _percentile(down_down, 80.0),
        _mean(down_down),
        _std(down_down),
        float(burst_count),
        mean_burst_size,
        max_burst_size,
        pause_fraction,
        _percentile(transition_distance, 20.0),
        _percentile(transition_distance, 50.0),
        _percentile(transition_distance, 80.0),
        _mean(transition_distance),
        float(np.sum(transition_distance)),
        float(bool(has_up)),
        float(bool(has_xy)),
    )
    return _finite_feature_vector(values, len(KEYSTROKE_FEATURE_NAMES))


def extract_keystroke_feature_dict(
    keys: Iterable[object],
    down_times_ms: np.ndarray,
    up_times_ms: Optional[np.ndarray] = None,
    key_points: Optional[np.ndarray] = None,
    pause_threshold_ms: float = 500.0,
) -> Dict[str, float]:
    vector = extract_keystroke_features(
        keys,
        down_times_ms,
        up_times_ms=up_times_ms,
        key_points=key_points,
        pause_threshold_ms=pause_threshold_ms,
    )
    return dict(zip(KEYSTROKE_FEATURE_NAMES, vector.tolist()))
