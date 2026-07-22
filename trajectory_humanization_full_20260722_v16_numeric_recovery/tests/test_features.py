import unittest

import numpy as np

from trajectory.features import (
    KEYSTROKE_FEATURE_NAMES,
    PINCH_FEATURE_NAMES,
    SINGLE_FINGER_FEATURE_NAMES,
    MAX_DEVIATION_POLICY,
    canonical_keycode_feature_token,
    is_hmog_ascii_letter_keycode,
    extract_keystroke_feature_dict,
    extract_keystroke_features,
    extract_pinch_feature_dict,
    extract_pinch_features,
    extract_single_finger_feature_dict,
    extract_single_finger_features,
    sanitize_timed_points,
)


class SingleFingerFeatureTests(unittest.TestCase):
    def test_public_dimension_and_constant_speed_line(self):
        points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        times = np.asarray([0.0, 100.0, 200.0])
        vector = extract_single_finger_features(points, times)
        features = extract_single_finger_feature_dict(points, times)

        self.assertEqual(len(SINGLE_FINGER_FEATURE_NAMES), 24)
        self.assertEqual(vector.shape, (24,))
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertAlmostEqual(features["duration"], 200.0)
        self.assertAlmostEqual(features["displacement"], 2.0)
        self.assertAlmostEqual(features["length"], 2.0)
        self.assertAlmostEqual(features["ratio_end_to_length"], 1.0)
        self.assertAlmostEqual(features["meanResultantLength"], 1.0)
        self.assertAlmostEqual(features["direction"], 0.0)
        self.assertAlmostEqual(features["avgDirection"], 0.0)
        self.assertAlmostEqual(features["v20"], 10.0)
        self.assertAlmostEqual(features["v50"], 10.0)
        self.assertAlmostEqual(features["v80"], 10.0)
        self.assertAlmostEqual(features["speed"], 10.0)
        self.assertAlmostEqual(features["a50"], 0.0)
        self.assertAlmostEqual(features["maxDevSigned"], 0.0)

    def test_duplicate_timestamp_uses_last_complete_event(self):
        points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        times = np.asarray([0.0, 0.0, 100.0])
        clean_points, clean_times = sanitize_timed_points(points, times)
        features = extract_single_finger_feature_dict(points, times)

        np.testing.assert_array_equal(clean_points, [[1.0, 0.0], [2.0, 0.0]])
        np.testing.assert_array_equal(clean_times, [0.0, 100.0])
        self.assertAlmostEqual(features["startX"], 1.0)
        self.assertAlmostEqual(features["duration"], 100.0)
        self.assertAlmostEqual(features["speed"], 10.0)

    def test_repeated_spatial_point_preserves_dwell_time(self):
        points = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        times = np.asarray([0.0, 100.0, 200.0])
        features = extract_single_finger_feature_dict(points, times)

        self.assertAlmostEqual(features["duration"], 200.0)
        self.assertAlmostEqual(features["length"], 1.0)
        self.assertAlmostEqual(features["v50"], 5.0)
        self.assertAlmostEqual(features["speed"], 5.0)
        self.assertAlmostEqual(features["meanResultantLength"], 1.0)

    def test_signed_curve_deviation(self):
        features = extract_single_finger_feature_dict(
            np.asarray([
                [0.0, 0.0], [1.0, -3.0], [2.0, -1.0],
                [3.0, 2.0], [4.0, 0.0],
            ]),
            np.asarray([0.0, 100.0, 200.0, 300.0, 400.0]),
        )
        # Signed perpendicular values are [0,-3,-1,2,0].  Percentiles must
        # not silently use absolute distances.
        self.assertAlmostEqual(features["maxDevSigned"], -3.0)
        self.assertAlmostEqual(features["dev20"], -1.4)
        self.assertAlmostEqual(features["dev50"], 0.0)
        self.assertAlmostEqual(features["dev80"], 0.4)
        self.assertEqual(
            MAX_DEVIATION_POLICY,
            "paper_signed_value_at_argmax_absolute_deviation",
        )
        # The audited public repository's first executable return is
        # max(abs(devs)), despite its signed docstring and unreachable signed
        # code below it.  This asymmetric trace proves that our paper policy
        # (-3) is deliberately not that repository's actual result (+3).
        public_repository_actual = float(
            np.max(np.abs(np.asarray([0.0, -3.0, -1.0, 2.0, 0.0])))
        )
        self.assertAlmostEqual(public_repository_actual, 3.0)
        self.assertNotEqual(features["maxDevSigned"], public_repository_actual)

    def test_table6_signed_speed_acceleration_and_first_five_samples(self):
        # At 100 ms intervals the segment speeds are
        # [70,50,40,40,50,70,100] px/s, hence signed accelerations are
        # [-200,-100,0,100,200,300] px/s^2.
        points = np.stack(
            (
                np.asarray([0.0, 7.0, 12.0, 16.0, 20.0, 25.0, 32.0, 42.0]),
                np.zeros(8),
            ),
            axis=1,
        )
        features = extract_single_finger_feature_dict(
            points, np.arange(8, dtype=np.float64) * 100.0
        )
        self.assertAlmostEqual(features["a20"], -100.0)
        self.assertAlmostEqual(features["a50"], 50.0)
        self.assertAlmostEqual(features["a80"], 200.0)
        # First five acceleration samples have median 0; the former erroneous
        # first-5-percent rule would have returned the first sample, -200.
        self.assertAlmostEqual(features["acc_first5pct_median"], 0.0)

    def test_one_point_and_stationary_trace_are_finite(self):
        one = extract_single_finger_feature_dict(
            np.asarray([[7.0, 9.0]]), np.asarray([15.0])
        )
        self.assertAlmostEqual(one["startX"], 7.0)
        self.assertAlmostEqual(one["endY"], 9.0)
        self.assertAlmostEqual(one["duration"], 0.0)
        self.assertAlmostEqual(one["length"], 0.0)
        self.assertTrue(np.all(np.isfinite(list(one.values()))))

        stationary = extract_single_finger_features(
            np.asarray([[1.0, 2.0], [1.0, 2.0]]),
            np.asarray([0.0, 100.0]),
        )
        self.assertTrue(np.all(np.isfinite(stationary)))

    def test_bad_time_and_nonfinite_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "monotonic"):
            extract_single_finger_features(
                np.asarray([[0.0, 0.0], [1.0, 0.0]]),
                np.asarray([2.0, 1.0]),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            extract_single_finger_features(
                np.asarray([[0.0, 0.0], [np.nan, 0.0]]),
                np.asarray([0.0, 1.0]),
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            extract_single_finger_features(np.zeros((0, 2)), np.zeros((0,)))


class PinchFeatureTests(unittest.TestCase):
    def test_symmetric_expansion(self):
        first = np.asarray([[-1.0, 0.0], [-2.0, 0.0]])
        second = np.asarray([[1.0, 0.0], [2.0, 0.0]])
        times = np.asarray([0.0, 100.0])
        vector = extract_pinch_features(first, second, times)
        features = extract_pinch_feature_dict(first, second, times)

        self.assertEqual(vector.shape, (len(PINCH_FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertAlmostEqual(features["center_displacement"], 0.0)
        self.assertAlmostEqual(features["startSpan"], 2.0)
        self.assertAlmostEqual(features["endSpan"], 4.0)
        self.assertAlmostEqual(features["spanDelta"], 2.0)
        self.assertAlmostEqual(features["spanRate50"], 20.0)
        self.assertAlmostEqual(features["finger1Length"], 1.0)
        self.assertAlmostEqual(features["finger2Length"], 1.0)
        self.assertAlmostEqual(features["fingerLengthRatio"], 1.0)

    def test_duplicate_time_is_coalesced_jointly(self):
        first = np.asarray([[0.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
        second = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        features = extract_pinch_feature_dict(
            first, second, np.asarray([0.0, 0.0, 100.0])
        )
        self.assertAlmostEqual(features["startSpan"], 2.0)
        self.assertAlmostEqual(features["endSpan"], 4.0)
        self.assertAlmostEqual(features["center_duration"], 100.0)

    def test_coincident_fingers_and_short_pinch_are_finite(self):
        vector = extract_pinch_features(
            np.asarray([[3.0, 4.0]]),
            np.asarray([[3.0, 4.0]]),
            np.asarray([10.0]),
        )
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertEqual(vector.shape, (len(PINCH_FEATURE_NAMES),))

    def test_pinch_shape_and_time_validation(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            extract_pinch_features(
                np.zeros((2, 2)), np.zeros((1, 2)), np.asarray([0.0, 1.0])
            )
        with self.assertRaisesRegex(ValueError, "monotonic"):
            extract_pinch_features(
                np.zeros((2, 2)), np.ones((2, 2)), np.asarray([2.0, 1.0])
            )


class KeystrokeFeatureTests(unittest.TestCase):
    def test_hmog_ascii_letter_codebook_boundaries_and_nonletters(self):
        letter_codes = (65, 90, 97, 122)
        nonletter_codes = (0, 64, 91, 96, 123)
        tokens = [canonical_keycode_feature_token(value) for value in letter_codes + nonletter_codes]
        self.assertEqual(tokens[:4], ["A", "Z", "a", "z"])
        self.assertTrue(all(is_hmog_ascii_letter_keycode(value) for value in letter_codes))
        self.assertTrue(all(not is_hmog_ascii_letter_keycode(value) for value in nonletter_codes))
        self.assertTrue(all(not token.isalpha() for token in tokens[4:]))
        self.assertEqual(canonical_keycode_feature_token(8230), "keycode_8230")
        features = extract_keystroke_feature_dict(
            tokens,
            np.arange(len(tokens), dtype=np.float64) * 100.0,
            np.arange(len(tokens), dtype=np.float64) * 100.0 + 40.0,
        )
        self.assertEqual(features["nLetters"], 4.0)

    def test_timing_bursts_correction_and_spatial_features(self):
        keys = ["a", "b", "BACKSPACE", "c"]
        down = np.asarray([0.0, 100.0, 800.0, 900.0])
        up = np.asarray([50.0, 160.0, 850.0, 970.0])
        points = np.asarray([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0], [6.0, 8.0]])
        vector = extract_keystroke_features(keys, down, up, points)
        features = extract_keystroke_feature_dict(keys, down, up, points)

        self.assertEqual(vector.shape, (len(KEYSTROKE_FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertAlmostEqual(features["nKeys"], 4.0)
        self.assertAlmostEqual(features["nLetters"], 3.0)
        self.assertAlmostEqual(features["correctionRatio"], 0.25)
        self.assertAlmostEqual(features["duration"], 970.0)
        self.assertAlmostEqual(features["holdMean"], 57.5)
        self.assertAlmostEqual(features["flight50"], 50.0)
        self.assertAlmostEqual(features["burstCount"], 2.0)
        self.assertAlmostEqual(features["pauseFraction"], 1.0 / 3.0)
        self.assertAlmostEqual(features["spatialPathLength"], 10.0)
        self.assertAlmostEqual(features["hasUpTimes"], 1.0)
        self.assertAlmostEqual(features["hasKeyXY"], 1.0)

    def test_equal_down_timestamps_remain_distinct(self):
        features = extract_keystroke_feature_dict(
            ["a", "b"], np.asarray([10.0, 10.0])
        )
        self.assertAlmostEqual(features["nKeys"], 2.0)
        self.assertAlmostEqual(features["downDown50"], 0.0)

    def test_one_key_without_optional_streams_is_finite(self):
        vector = extract_keystroke_features(["a"], np.asarray([100.0]))
        features = dict(zip(KEYSTROKE_FEATURE_NAMES, vector))
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertAlmostEqual(features["nKeys"], 1.0)
        self.assertAlmostEqual(features["hasUpTimes"], 0.0)
        self.assertAlmostEqual(features["hasKeyXY"], 0.0)

    def test_empty_sequence_is_finite(self):
        vector = extract_keystroke_features([], np.asarray([]))
        self.assertEqual(vector.shape, (len(KEYSTROKE_FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertAlmostEqual(float(np.sum(np.abs(vector))), 0.0)

    def test_invalid_key_times_fail(self):
        with self.assertRaisesRegex(ValueError, "monotonic"):
            extract_keystroke_features(["a", "b"], np.asarray([2.0, 1.0]))
        with self.assertRaisesRegex(ValueError, ">="):
            extract_keystroke_features(
                ["a"], np.asarray([10.0]), np.asarray([9.0])
            )
        with self.assertRaisesRegex(ValueError, "match"):
            extract_keystroke_features(
                ["a", "b"], np.asarray([0.0, 1.0]), key_points=np.zeros((1, 2))
            )


if __name__ == "__main__":
    unittest.main()
