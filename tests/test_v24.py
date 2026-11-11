"""Mechanism checks for the camera-view fusion. Synthetic inputs only; no match data."""
import unittest
import numpy as np
from possession import v24_camera_fusion as v24


class CameraFusionTest(unittest.TestCase):
    def test_run_position_measures_distance_to_both_edges(self):
        flag = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
        since, until, previous, following = v24.run_position(flag)
        self.assertEqual(since.tolist(), [0, 1, 2, 0, 1, 0])
        self.assertEqual(until.tolist(), [2, 1, 0, 1, 0, 0])
        self.assertEqual(previous[3:5].tolist(), [3, 3])
        self.assertEqual(following[:3].tolist(), [2, 2, 2])
        self.assertEqual(previous[0], v24.CAP)

    def test_one_offset_per_half_from_alignment_runs(self):
        runs = [dict(source_first=100, offset_frames=-50, accepted_frames=10),
                dict(source_first=300, offset_frames=-51, accepted_frames=2),
                dict(source_first=900, offset_frames=-400, accepted_frames=5)]
        blocks = [(1, 100, 500), (2, 1200, 1500)]
        self.assertEqual(v24.half_offsets(runs, blocks), {1: -50, 2: -400})
        with self.assertRaises(ValueError):
            v24.half_offsets(runs[:2], blocks)

    def test_fusion_is_neutral_at_the_prior_and_keeps_team_ratio(self):
        p = np.array([[0.6, 0.2, 0.2], [0.1, 0.1, 0.8]])
        prior = np.array([0.4, 0.4])
        same = v24.fuse(p, prior.copy(), prior)
        np.testing.assert_allclose(same, p, atol=1e-6)
        more_dead = v24.fuse(p, np.array([0.9, 0.9]), prior)
        self.assertTrue((more_dead[:, 2] > p[:, 2]).all())
        np.testing.assert_allclose(more_dead[:, 0] / more_dead[:, 1], p[:, 0] / p[:, 1], rtol=1e-6)
        np.testing.assert_allclose(more_dead.sum(axis=1), 1.0, atol=1e-9)

    def test_confident_teacher_survives_a_contrary_camera_view(self):
        p = np.array([[0.97, 0.02, 0.01]])
        fused = v24.fuse(p, np.array([0.99]), np.array([0.45]))
        self.assertLess(fused[0, 2], 0.5)

    def test_frames_before_the_published_kickoff(self):
        kickoffs = sorted(v24.KICKOFF_VIDEO_FRAMES.values())
        blocks = [(1, kickoffs[0], kickoffs[0] + 100), (2, kickoffs[1] - 10, kickoffs[1] + 10)]
        frames = np.array([kickoffs[0], kickoffs[1] - 10, kickoffs[1] - 1, kickoffs[1]])
        self.assertEqual(v24.before_published_kickoff(frames, blocks).tolist(),
                         [False, True, True, False])

    def test_centred_rate_is_nan_across_missing_camera(self):
        x = np.arange(10, dtype=float)
        x[5] = np.nan
        rate = v24.centred_rate(x, 2)
        self.assertTrue(np.isnan(rate[[0, 4, 6, 9]]).all())
        self.assertAlmostEqual(rate[2], v24.FPS)


if __name__ == "__main__":
    unittest.main()
