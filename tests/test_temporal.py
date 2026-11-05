import unittest
import numpy as np
from possession.temporal import decode
from possession.features import interpolate_short_gaps, tactical_prior
from possession.run import predict


class TemporalTests(unittest.TestCase):
    def test_occlusion_preserves_possession(self):
        score = np.zeros((100, 3))
        score[:, 2] = -0.5
        score[:30, 0] = 2
        score[70:, 0] = 2
        np.testing.assert_array_equal(decode(score), np.zeros(100))

    def test_touch_is_not_control(self):
        score = np.tile([1.0, -1.0, -2.0], (100, 1))
        score[49] = [-1, 2, -2]
        np.testing.assert_array_equal(decode(score), np.zeros(100))

    def test_turnover_and_stoppage(self):
        score = np.repeat([[2., -2, -2], [-2, 2, -2], [-2, -2, 2]], 30, axis=0)
        np.testing.assert_array_equal(decode(score), np.repeat([0, 1, 2], 30))

    def test_short_gap_only(self):
        xy = np.array([[0., 0], [np.nan, np.nan], [2, 0], [np.nan, np.nan],
                       [np.nan, np.nan], [np.nan, np.nan], [6, 0]])
        out = interpolate_short_gaps(xy, 1, 45, 25)
        np.testing.assert_allclose(out[1], [1, 0])
        self.assertTrue(np.isnan(out[3:6]).all())

    def test_impossible_speed_not_interpolated(self):
        xy = np.array([[0., 0], [np.nan, np.nan], [100, 0]])
        self.assertTrue(np.isnan(interpolate_short_gaps(xy, 3, 45, 25)[1]).all())

    def test_defending_team_is_inverted_and_unknown_ignored(self):
        events = [dict(code="DEFENDING IN MIDDLE THIRD", start_frame=0, end_frame=5,
                       tags={"Team": "Barcelona"}),
                  dict(code="BUILD UP", start_frame=5, end_frame=10, tags={"Team": "N/A"})]
        tag, _ = tactical_prior(events, 10)
        np.testing.assert_array_equal(tag[:5], -np.ones(5))
        np.testing.assert_array_equal(tag[5:], np.zeros(5))

    def test_halves_do_not_bleed_and_frames_preserved(self):
        score = np.tile([2., -2, -2], (103, 1))
        score[61:] = [-2, 2, -2]
        config = dict(step_frames=5, minimum_state_steps=[3, 3, 4], switch_cost=3.)
        labels = predict(score, config, [(1, 3, 42), (2, 61, 100)])
        self.assertEqual(len(labels), 103)
        self.assertTrue((labels[3:43] == "H").all())
        self.assertTrue((labels[61:101] == "A").all())
        self.assertTrue((labels[43:61] == "X").all())


if __name__ == "__main__":
    unittest.main()
