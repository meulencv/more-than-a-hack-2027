import unittest
import numpy as np
from possession import v27_core as c
from possession import v31a_insert as ins
from possession import v31a_lab as lab
from possession import v31a_tta as tta

H, A, D = 0, 1, 2
FPS = c.FPS


def codes_of(plan):
    return np.concatenate([np.full(int(round(s * FPS)), k, dtype=np.int8) for s, k in plan])


class CompromiseTarget(unittest.TestCase):
    def test_short_sandwiched_possession_is_absorbed(self):
        alternative = codes_of([(10, H), (1.6, A), (10, H), (5, D), (10, A)])
        m = lab.m_reference(alternative, [(1, 0, len(alternative) - 1)])
        self.assertEqual(c.segments(m, [(1, 0, len(m) - 1)]), [(0, 540, H), (540, 665, D), (665, len(m), A)])

    def test_two_second_possession_and_possession_next_to_a_stoppage_are_kept(self):
        alternative = codes_of([(10, H), (2, A), (10, H), (1, A), (5, D), (1, H), (10, A)])
        m = lab.m_reference(alternative, [(1, 0, len(alternative) - 1)])
        self.assertTrue(np.array_equal(m, alternative))

    def test_ungraded_frames_stay_ungraded_and_threshold_is_the_argument(self):
        alternative = np.concatenate([np.full(50, -1, dtype=np.int8), codes_of([(10, H), (2.5, A), (10, H)])])
        blocks = [(1, 50, len(alternative) - 1)]
        self.assertTrue(np.array_equal(lab.m_reference(alternative, blocks, 2.0), alternative))
        m = lab.m_reference(alternative, blocks, 3.0)
        self.assertTrue((m[:50] == -1).all() and (m[50:] == H).all())

    def test_verifier_targets_are_weights_over_the_gain_columns(self):
        d = np.arange(2 * len(lab.RNAMES), dtype=float).reshape(2, len(lab.RNAMES))
        self.assertTrue(np.allclose(lab.target_of(d, "w0.50"), 0.5 * d[:, 0] + 0.5 * d[:, 1]))
        self.assertTrue(np.allclose(lab.target_of(d, "M"), d[:, lab.RNAMES.index("M")]))
        for weights in list(lab.TARGETS.values()) + list(lab.STACK_TARGETS.values()):
            self.assertAlmostEqual(sum(weights.values()), 1.0)
            self.assertTrue(set(weights) <= set(lab.RNAMES))

    def test_gains_match_the_official_metric(self):
        reference = codes_of([(10, H), (10, A), (4, D), (10, H)])
        predicted = codes_of([(4, H), (1, A), (5, H), (10, A), (4, D), (10, H)])
        blocks = [(1, 0, len(reference) - 1)]
        gt = {r: c.spells_of(c.segments(reference, blocks)) for r in lab.RNAMES}
        segs = c.segments(predicted, blocks)
        cands = c.candidates(segs, lab.MAX_FRAMES)
        self.assertEqual(cands, [(1, "absorb", H), (2, "absorb", A)])
        delta = lab.gains(segs, cands, gt)
        before = c.official(reference, predicted)["spell_f1"]
        after = c.official(reference, c.to_codes(c.relabel(segs, {1: H}), len(reference)))["spell_f1"]
        self.assertAlmostEqual(delta[0, 0], after - before)
        self.assertGreater(delta[0, 0], 0)


class Insertions(unittest.TestCase):
    def setUp(self):
        self.segs = [(0, 500, H), (500, 600, D), (600, 1000, A)]
        liberal = np.zeros((2, 1000), dtype=np.int8)
        liberal[:, 500:600] = D
        liberal[:, 600:] = A
        liberal[0, 200:260] = A                 # found by the first level
        liberal[1, 190:270] = A                 # the same possession, wider, at the second level
        liberal[1, 700:705] = H                 # too short
        liberal[1, 601:660] = H                 # touches the host's start
        liberal[1, 800:850] = H
        self.liberal = liberal

    def test_candidates_are_inside_one_host_and_deduplicated(self):
        items = ins.insertions(self.segs, self.liberal)
        self.assertEqual(items, [(0, 200, 260, A, 0), (2, 800, 850, H, 1)])

    def test_inserted_segmentation_is_contiguous(self):
        items = ins.insertions(self.segs, self.liberal)
        new = ins.inserted(self.segs, items)
        self.assertEqual(new, [(0, 200, H), (200, 260, A), (260, 500, H), (500, 600, D), (600, 800, A), (800, 850, H), (850, 1000, A)])
        self.assertEqual(ins.inserted(self.segs, []), self.segs)

    def test_features_and_gains_shapes(self):
        items = ins.insertions(self.segs, self.liberal)
        p = np.full((1000, 3), 0.2)
        x = ins.features(self.segs, items, p, np.zeros(1000))
        self.assertEqual(x.shape, (2, len(c.FEATURES) + len(ins.EXTRA)))
        self.assertAlmostEqual(x[0, c.FEATURES.index("duration_s")], 60 / FPS)
        self.assertEqual(x[0, len(c.FEATURES):].tolist(), [0.0, 200 / FPS, 240 / FPS, 500 / FPS])
        reference = c.to_codes(ins.inserted(self.segs, items[:1]), 1000)
        gt = {r: c.spells_of(c.segments(reference, [(1, 0, 999)])) for r in lab.RNAMES}
        delta = ins.gains(self.segs, items, gt)
        self.assertGreater(delta[0, 0], 0)
        self.assertLess(delta[1, 0], 0)
        new, n = ins.greedy_oracle(self.segs, items, gt["consistent"])
        self.assertEqual((n, c.to_codes(new, 1000).tolist()), (1, reference.tolist()))


class Mirror(unittest.TestCase):
    def test_record_flip_changes_only_y_columns_and_is_an_involution(self):
        rng = np.random.default_rng(0)
        r = dict(nodes=rng.normal(size=(5, 4, 14)).astype("float16"), ball_tokens=rng.normal(size=(5, 7)).astype("float32"),
                 direct=rng.normal(size=(5, 3)), frames=np.arange(5))
        f = tta.flip_record(r)
        keep = [j for j in range(14) if j not in tta.Y_NODES]
        self.assertTrue(np.array_equal(f["nodes"][:, :, keep], r["nodes"][:, :, keep]))
        self.assertTrue(np.array_equal(f["nodes"][:, :, tta.Y_NODES], -r["nodes"][:, :, tta.Y_NODES]))
        self.assertTrue(np.array_equal(f["ball_tokens"][:, [0, 2, 4, 5, 6]], r["ball_tokens"][:, [0, 2, 4, 5, 6]]))
        self.assertIs(f["direct"], r["direct"])
        back = tta.flip_record(f)
        self.assertTrue(np.array_equal(back["nodes"], r["nodes"]) and np.array_equal(back["ball_tokens"], r["ball_tokens"]))


if __name__ == "__main__":
    unittest.main()
