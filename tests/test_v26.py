"""Mechanism checks for the video ball linker. Synthetic candidates only; no match data."""
import unittest
import numpy as np
from possession import video_ball as vb


def moving_ball(length, start=(100.0, 100.0), step=(6.0, 2.0)):
    return np.array(start) + np.arange(length)[:, None] * np.array(step)


class VideoBallLinkTest(unittest.TestCase):
    def test_follows_a_consistent_ball_and_ignores_a_stronger_flicker(self):
        truth = moving_ball(30)
        candidates, scores = [], []
        for t in range(30):
            c, s = [truth[t]], [0.8]
            if t in (7, 19):                      # isolated, confident false alarm far away
                c.append([900.0, 500.0]); s.append(0.95)
            candidates.append(np.array(c)); scores.append(np.array(s))
        track = vb.link_run(candidates, scores, (None, None))
        np.testing.assert_allclose(track, truth)

    def test_short_isolated_detections_are_not_a_ball(self):
        candidates = [np.zeros((0, 2))] * 40
        scores = [np.zeros(0)] * 40
        for t in (10, 11, 12):
            candidates[t] = np.array([[300.0, 300.0]]); scores[t] = np.array([0.9])
        track = vb.link_run(candidates, scores, (None, None))
        self.assertTrue(np.isnan(track).all())

    def test_weak_candidates_alone_do_not_start_a_track(self):
        truth = moving_ball(40)
        track = vb.link_run([truth[t:t + 1] for t in range(40)], [np.array([0.3])] * 40, (None, None))
        self.assertTrue(np.isnan(track).all())

    def test_anchor_keeps_a_short_run_next_to_the_known_ball(self):
        truth = moving_ball(4)
        candidates = [truth[t:t + 1] for t in range(4)] + [np.zeros((0, 2))] * 20
        scores = [np.array([0.9])] * 4 + [np.zeros(0)] * 20
        anchored = vb.link_run(candidates, scores, (truth[0] - (6.0, 2.0), None))
        self.assertTrue(np.isfinite(anchored[:4]).all())
        self.assertTrue(np.isnan(anchored[4:]).all())

    def test_impossible_jump_is_not_linked(self):
        a, b = moving_ball(10), moving_ball(10, start=(1500.0, 800.0))
        candidates = [a[t:t + 1] for t in range(10)] + [b[t:t + 1] for t in range(10)]
        track = vb.link_run(candidates, [np.array([0.9])] * 20, (None, None))
        seen = np.isfinite(track[:, 0])
        self.assertGreaterEqual(seen.sum(), 19)             # two tracks; at most the seam is dropped
        steps = np.linalg.norm(np.diff(track, axis=0), axis=1)
        self.assertTrue((steps[np.isfinite(steps)] <= vb.MAX_STEP_PX).all())

    def test_bridge_fills_only_short_interior_holes(self):
        track = moving_ball(60).astype(float)
        track[10:15] = np.nan
        track[30:50] = np.nan
        out = vb.bridge(track.copy())
        np.testing.assert_allclose(out[10:15], moving_ball(60)[10:15])
        self.assertTrue(np.isnan(out[30:50]).all())

    def test_held_out_blocks_are_one_in_eight(self):
        frames = np.arange(vb.HELD_BLOCK * vb.HELD_EVERY)
        self.assertAlmostEqual(vb.held_out(frames).mean(), 1 / vb.HELD_EVERY)


if __name__ == "__main__":
    unittest.main()
