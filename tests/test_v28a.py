import unittest
import numpy as np
from possession import v27_core as c
from possession import v28a_core as k28


def scene(n=600, fps=25):
    """Home player 0 carries the ball for 8 s, then it flies to away player 0 who keeps it."""
    t = np.arange(n) / fps
    players = np.zeros((n, 4, 2), dtype=np.float32)
    team = np.array([0, 0, 1, 1])
    players[:, 0] = np.column_stack([20 + 2.0 * t, np.full(n, 30.0)])       # home carrier, 2 m/s
    players[:, 1] = [10.0, 10.0]
    players[:, 2] = [70.0, 30.0]                                            # away receiver
    players[:, 3] = [90.0, 60.0]
    ball = np.full((n, 2), np.nan)
    carry = t < 8
    ball[carry] = players[carry, 0] + [0.5, 0.0]
    start = ball[carry][-1]
    flight = (t >= 8) & (t < 10)
    ball[flight] = start + ((t[flight] - 8) / 2)[:, None] * (np.array([70.0, 30.0]) - start)
    ball[t >= 10] = [70.3, 30.0]
    return ball, players, team


class V28aCore(unittest.TestCase):
    def setUp(self):
        self.ball, self.players, self.team = scene()
        self.s = k28.signals(self.ball, self.players, self.team, (105.0, 68.0))

    def test_signals_follow_the_scene(self):
        s = self.s
        self.assertTrue(s["seen"].all())
        self.assertAlmostEqual(float(s["distance"][100, 0]), 0.5, places=4)
        self.assertGreater(float(s["distance"][100, 1]), 30)
        self.assertAlmostEqual(float(s["speed"][100]), 2.0, places=3)
        self.assertLess(float(s["relative"][100, 0]), 0.01)                 # ball moves with the carrier
        self.assertGreater(float(np.nanmax(s["speed"][200:250])), 15)       # the pass
        self.assertGreaterEqual(float(np.nansum(s["event"][190:260])), 1)   # struck and received

    def test_physical_features_separate_control_from_flight(self):
        row = dict(zip(k28.PHYSICAL, k28.physical_features(50, 150, 0, self.s)))        # home carries
        self.assertEqual(row["seen_in"], 1.0)
        self.assertEqual(row["near_own_share"], 1.0)
        self.assertAlmostEqual(row["own_close_s"], 4.0, places=2)
        self.assertEqual(row["own_carry_share"], 1.0)
        self.assertEqual(row["opp_carry_share"], 0.0)
        wrong = dict(zip(k28.PHYSICAL, k28.physical_features(50, 150, 1, self.s)))      # same frames, called away
        self.assertEqual(wrong["near_own_share"], 0.0)
        self.assertEqual(wrong["own_close_s"], 0.0)
        self.assertEqual(wrong["own_carry_share"], 0.0)
        flight = dict(zip(k28.PHYSICAL, k28.physical_features(205, 245, 1, self.s)))    # ball in the air
        self.assertGreater(flight["free_share"], 0.9)
        self.assertGreater(flight["straight"], 0.99)
        self.assertLess(flight["line_dev_m"], 0.01)

    def test_missing_ball_gives_nan_not_numbers(self):
        ball = self.ball.copy()
        ball[40:160] = np.nan
        s = k28.signals(ball, self.players, self.team, (105.0, 68.0))
        row = dict(zip(k28.PHYSICAL, k28.physical_features(50, 150, 0, s)))
        self.assertEqual(row["seen_in"], 0.0)
        for name in ("near_own_share", "d_own_med", "own_close_s", "speed_mean", "path_m", "touch_own"):
            self.assertTrue(np.isnan(row[name]), name)

    def test_candidate_features_extend_v27_without_changing_it(self):
        rng = np.random.default_rng(0)
        n = 600
        codes = np.zeros(n, dtype=np.int8)
        codes[100:130] = 1
        codes[300:400] = 2
        codes[400:440] = 1
        blocks = [(1, 0, n - 1)]
        segs = c.segments(codes, blocks)
        p = rng.dirichlet(np.ones(3), n)
        edge = rng.random(n)
        cands = c.candidates(segs, 200)
        self.assertEqual([action for _, action, _ in cands], ["absorb", "to_previous", "to_dead", "to_next", "to_dead"])
        x = k28.candidate_features(segs, cands, p, edge, self.s)
        self.assertEqual(x.shape, (5, len(k28.FEATURES)))
        self.assertEqual(len(set(k28.FEATURES)), len(k28.FEATURES))
        self.assertTrue(np.array_equal(x[:, :len(c.FEATURES)], c.candidate_features(segs, cands, p, edge)))
        row = dict(zip(k28.FEATURES, x[0]))                                 # A inside H: absorb
        self.assertAlmostEqual(row["since_dead_s"], 4.0)                    # live since the start of the half
        self.assertAlmostEqual(row["to_dead_s"], (300 - 130) / 25)
        self.assertAlmostEqual(row["merged_s"], 300 / 25)                   # H 0..300 after the merge
        self.assertEqual(row["prev_live"], 1.0)
        row = dict(zip(k28.FEATURES, x[3]))                                 # D-A-H: to_next
        self.assertEqual(row["since_dead_s"], 0.0)
        self.assertEqual(row["prev_live"], 0.0)
        self.assertAlmostEqual(row["merged_s"], (n - 400) / 25)


class V28aLab(unittest.TestCase):
    def test_models_score_and_blend(self):
        from possession import v28a_lab as lab
        rng = np.random.default_rng(1)
        x = rng.normal(size=(400, len(k28.FEATURES)))
        x[rng.random(x.shape) < 0.1] = np.nan
        signal = np.nan_to_num(x[:, 0])
        gain = 0.002 * signal + rng.normal(0, 0.0005, 400)
        purity = np.clip(signal, -1, 1)
        for kind in ("gbm", "gbm_mono", "gbc", "logit", "ridge", "blend", "blend_mono"):
            model = lab.Model(kind).fit(x, gain, purity)
            score = model.predict(x)
            self.assertEqual(score.shape, (400,))
            self.assertTrue(np.isfinite(score).all(), kind)
            if kind != "gbm_mono":                                           # duration_s is constrained to -1
                self.assertGreater(np.corrcoef(score, signal)[0, 1], 0.5, kind)
        self.assertEqual(sorted(sum(lab.BLOCKS.values(), [])), list(range(len(k28.FEATURES))))

    def test_live_only_blanks_physics_next_to_a_stoppage(self):
        model = k28.Model("gbm", live_only=True)
        x = np.ones((2, len(k28.FEATURES)))
        x[1, k28.FEATURES.index("is_absorb")] = 0
        view = model._view(x)
        first = len(c.FEATURES)
        physical = slice(first, first + len(k28.PHYSICAL))
        self.assertTrue(np.isfinite(view[0]).all())
        self.assertTrue(np.isnan(view[1, physical]).all())
        self.assertTrue(np.isfinite(np.delete(view[1], np.arange(first, first + len(k28.PHYSICAL)))).all())
        self.assertTrue(np.isfinite(x).all())                                # the input is not modified

    def test_adapter_only_removes_short_possessions_and_is_deterministic(self):
        import pickle
        from types import SimpleNamespace
        from possession import v28a_verifier as verifier
        from possession.v14_resolution import decode_fine
        rng = np.random.default_rng(3)
        n = 3000
        frames = np.arange(0, n, 5)
        logits = np.cumsum(rng.normal(0, 0.7, (len(frames), 3)), axis=0)
        logits -= logits.mean(1, keepdims=True)
        p = (np.exp(logits) / np.exp(logits).sum(1, keepdims=True)).astype("float32")
        edges = rng.random(len(frames)).astype("float32")
        blocks = [(1, 0, 1499), (2, 1600, n - 5)]
        ball = np.cumsum(rng.normal(0, 0.3, (n, 2)), axis=0) + [50, 30]
        ball[rng.random(n) < 0.2] = np.nan
        players = rng.uniform(0, 60, (n, 6, 2)).astype("float32")
        signals = k28.signals(ball, players, np.array([0, 0, 0, 1, 1, 1]), (105.0, 68.0))
        x = rng.normal(size=(300, len(k28.FEATURES)))
        model = k28.Model("blend", live_only=True).fit(x, 0.001 * x[:, 0], x[:, 0])
        model = pickle.loads(pickle.dumps(model))                            # as a saved verifier
        saved = dict(model=model, max_frames=200, variant="test")
        after, before, report = verifier.physical_adapter(p, edges, frames, blocks, n, signals, verifier=saved)
        again, _, _ = verifier.physical_adapter(p, edges, frames, blocks, n, signals, verifier=saved)
        self.assertTrue(np.array_equal(after, again))
        match = SimpleNamespace(ball=np.empty((n, 2)), blocks=blocks)
        self.assertTrue(np.array_equal(before, decode_fine(p, edges, frames, match, "probability")))
        self.assertTrue(np.array_equal(after == 3, before == 3))
        for a, z, k in c.segments(before, blocks):
            if k == 2 or z - a > 200:                                        # stoppages and long possessions stay
                self.assertTrue((after[a:z] == k).all())
        self.assertEqual(report["graded_frames_changed"], int((after != before).sum()))
        self.assertLessEqual(report["after"]["spells"], report["before"]["spells"])

    def test_purity_gap_and_oracle_pass(self):
        from possession import v28a_lab as lab
        n = 1000
        truth = np.zeros(n, dtype=np.int8)
        truth[600:] = 1
        codes = truth.copy()
        codes[200:230] = 1                                                   # spurious flip inside H
        blocks = [(1, 0, n - 1)]
        segs = c.segments(codes, blocks)
        cands = c.candidates(segs, 200)
        gap = lab.purity_gaps(segs, cands, {r: truth for r in c.REFERENCES})
        self.assertTrue(np.allclose(gap, 1.0))
        gt = c.spells_of(c.segments(truth, blocks))
        new, applied = lab.oracle_pass(segs, gt)
        self.assertEqual(applied, 1)
        self.assertTrue(np.array_equal(c.to_codes(new, n), truth))


if __name__ == "__main__":
    unittest.main()
