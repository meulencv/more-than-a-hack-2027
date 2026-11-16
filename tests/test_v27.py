import unittest
from types import SimpleNamespace
import numpy as np
import evaluation.possession_metrics as pm
from possession import v27_core as c
from possession.v14_resolution import decode_fine


def synthetic(seed=0, n=1200):
    rng = np.random.default_rng(seed)
    frames = np.arange(0, n, 5)
    logits = np.cumsum(rng.normal(0, 0.6, (len(frames), 3)), axis=0)
    logits -= logits.mean(1, keepdims=True)
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return p.astype("float32"), rng.random(len(frames)).astype("float32"), frames, [(1, 0, 499), (2, 600, n - 5)]


class V27Core(unittest.TestCase):
    def test_default_decoder_is_decode_fine(self):
        for seed in range(3):
            p, e, frames, blocks = synthetic(seed)
            match = SimpleNamespace(ball=np.empty((1200, 2)), blocks=blocks)
            expected = decode_fine(p, e, frames, match, "probability")
            got = c.decode(c.interpolate(p, e, frames, blocks), 1200, [c.DEFAULT, dict(c.DEFAULT, c_ha=9.0)])
            self.assertTrue(np.array_equal(expected, got[0]))

    def test_team_redecode_with_baseline_costs_changes_nothing_and_keeps_stoppages(self):
        p, e, frames, blocks = synthetic(4)
        inter = c.interpolate(p, e, frames, blocks)
        base = c.decode(inter, 1200)[0]
        again = c.redecode_teams(inter, base, [(c.DEFAULT["c_ha"], c.DEFAULT["gamma"]), (20.0, 0.0)])
        self.assertTrue(np.array_equal(again[0], base))
        self.assertTrue(np.array_equal(again[1] == 2, base == 2))
        self.assertLessEqual(len(c.spells_of(c.segments(again[1], blocks))), len(c.spells_of(c.segments(base, blocks))))

    def test_fast_matching_equals_official_metric(self):
        rng = np.random.default_rng(1)
        for _ in range(5):
            def labels():
                lengths = rng.integers(1, 60, 80)
                return np.repeat(rng.integers(0, 3, 80), lengths)[:1500].astype(np.int8)
            gt, pred = labels(), labels()
            n = min(len(gt), len(pred))
            gt, pred = gt[:n].copy(), pred[:n].copy()
            gt[700:760] = 3
            pred[700:760] = 3
            blocks = [(1, 0, 699), (2, 760, n - 1)]
            official = pm.spell_scores(gt, pred, pm.valid_runs(gt))["mean_f1"]
            fast = c.spell_f1(c.spells_of(c.segments(gt, blocks)), c.spells_of(c.segments(pred, blocks)))
            self.assertAlmostEqual(official, fast, places=12)

    def test_candidates_and_relabel(self):
        codes = np.array([0] * 10 + [1] * 3 + [0] * 10 + [2] * 5 + [1] * 2 + [2] * 5 + [0] * 4 + [1] * 6 + [2] * 3, dtype=np.int8)
        blocks = [(1, 0, len(codes) - 1)]
        segs = c.segments(codes, blocks)
        found = {(segs[i][0], action) for i, action, _ in c.candidates(segs, 5)}
        self.assertEqual(found, {(10, "absorb"), (28, "to_dead"), (35, "to_next"), (35, "to_dead")})
        merged = c.relabel(segs, {1: 0, 4: 2})
        self.assertEqual(merged[0], (0, 23, 0))
        self.assertEqual(merged[1], (23, 35, 2))
        self.assertTrue(np.array_equal(c.to_codes(segs, len(codes)), codes))

    def test_apply_scored_never_removes_neighbours_together_and_respects_threshold(self):
        codes = np.array([0] * 20 + [1] * 3 + [0] * 3 + [1] * 20, dtype=np.int8)
        segs = c.segments(codes, [(1, 0, len(codes) - 1)])
        same, applied = c.apply_scored(segs, lambda s, cands: np.full(len(cands), -1.0), 0.0, 10)
        self.assertEqual((same, applied), (segs, []))
        new, applied = c.apply_scored(segs, lambda s, cands: np.ones(len(cands)), 0.0, 10, passes=1)
        self.assertEqual(len(applied), 1)
        self.assertEqual(len(new), 2)

    def test_features_are_finite_and_label_free(self):
        p, e, frames, blocks = synthetic(2)
        inter = c.interpolate(p, e, frames, blocks)
        codes = c.decode(inter, 1200)[0]
        dense_p, dense_e = c.dense(inter, 1200)
        segs = c.segments(codes, blocks)
        cands = c.candidates(segs, 200)
        x = c.candidate_features(segs, cands, dense_p, dense_e)
        self.assertEqual(x.shape, (len(cands), len(c.FEATURES)))
        self.assertTrue(np.isfinite(x).all())


if __name__ == "__main__":
    unittest.main()
