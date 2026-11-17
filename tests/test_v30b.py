import unittest
from types import SimpleNamespace
import numpy as np
from possession import v27_core as c
from possession import v30b_budget as b
from possession import v30b_refine as refine
from possession import v30b_stack as stack

H, A, D = 0, 1, 2


def codes_of(plan):
    return np.concatenate([np.full(n, k, dtype=np.int8) for n, k in plan])


def condition(reference_plan, predicted_plan):
    reference, predicted = codes_of(reference_plan), codes_of(predicted_plan)
    n = len(reference)
    blocks = [(1, 0, n - 1)]
    k = SimpleNamespace(n=n, blocks=blocks, references={"consistent": reference}, pred=predicted)
    k.pred_segs = c.segments(predicted, blocks)
    k.ref_segs = {"consistent": c.segments(reference, blocks)}
    k.gt = {"consistent": c.spells_of(k.ref_segs["consistent"])}
    k.p = np.full((n, 3), 0.1)
    k.p[np.arange(n), predicted] = 0.8
    return k


class Boundaries(unittest.TestCase):
    def test_kinds(self):
        segs = c.segments(codes_of([(50, H), (50, D), (50, A), (50, H)]), [(1, 0, 199)])
        self.assertEqual([(f, kind) for f, kind, _, _ in b.boundaries(segs)], [(50, "onset"), (100, "restart"), (150, "change")])

    def test_pairs_are_one_to_one_and_typed(self):
        reference = [(100, "onset", H, D), (200, "restart", D, A)]
        predicted = [(90, "onset", H, D), (110, "onset", H, D), (205, "restart", D, H)]
        self.assertEqual(b.pair_boundaries(reference, predicted, 25), [(0, 0)])            # tie -> first; restart labels differ
        self.assertEqual(sorted(b.pair_boundaries(reference, predicted, 25, exact=False)), [(0, 0), (1, 2)])

    def test_nearest_fill(self):
        values = np.array([9, 0, 9, 9, 1, 9])
        self.assertEqual(b.nearest_fill(values, values < 2).tolist(), [0, 0, 0, 1, 1, 1])


class Oracles(unittest.TestCase):
    def test_snap_moves_only_boundaries_within_tolerance(self):
        k = condition([(100, H), (100, D), (100, A)], [(110, H), (150, D), (40, A)])
        near = b.snap(k, "consistent", 25)
        self.assertEqual(c.segments(near, k.blocks), [(0, 100, H), (100, 260, D), (260, 300, A)])
        both = b.snap(k, "consistent", 75)
        self.assertTrue(np.array_equal(both, k.references["consistent"]))

    def test_snap_one_kind(self):
        k = condition([(100, H), (100, D), (100, A)], [(110, H), (80, D), (110, A)])
        only = b.snap(k, "consistent", 50, ("restart",))
        self.assertEqual(c.segments(only, k.blocks), [(0, 110, H), (110, 200, D), (200, 300, A)])

    def test_reference_team_and_reference_bounds(self):
        k = condition([(100, H), (100, A), (100, D)], [(140, H), (60, A), (100, D)])
        self.assertTrue(np.array_equal(b.ref_team(k, "consistent"), k.references["consistent"]))
        self.assertTrue(np.array_equal(b.ref_bounds(k, "consistent"), k.references["consistent"]))
        wrong = condition([(100, H), (100, A), (100, D)], [(100, H), (100, H), (100, D)])
        self.assertEqual(c.segments(b.ref_bounds(wrong, "consistent"), wrong.blocks), [(0, 200, H), (200, 300, D)])
        self.assertTrue(np.array_equal(b.ref_label(k, "consistent"), k.pred))

    def test_causes(self):
        cases = {
            "lost_in_stoppage": ([(100, D), (100, H), (100, D)], [(300, D)]),
            "wrong_team": ([(100, D), (100, H), (100, D)], [(100, D), (100, A), (100, D)]),
            "split": ([(50, D), (200, H), (50, D)], [(50, D), (90, H), (20, A), (90, H), (50, D)]),
            "merged": ([(100, H), (50, A), (100, H), (50, D)], [(250, H), (50, D)]),
            "shift_stoppage_side": ([(100, H), (200, D)], [(250, H), (50, D)]),
            "shift_change_side": ([(100, H), (200, A)], [(250, H), (50, A)]),
        }
        for cause, (reference, predicted) in cases.items():
            k = condition(reference, predicted)
            gt, pred = k.gt["consistent"], c.spells_of(k.pred_segs)
            g, p = c.matching(gt, pred)[0.5]
            found = [b.classify(spell, pred, k.references["consistent"], k.pred, p) for i, spell in enumerate(gt) if not g[i]]
            self.assertIn(cause, found, cause)

    def test_latency_sign(self):
        k = condition([(100, H), (100, D), (100, A)], [(110, H), (80, D), (110, A)])
        found = b.latencies(k.ref_segs["consistent"], k.pred_segs, k.blocks)
        self.assertEqual(found["onset"][0], [0.4])                 # predicted 10 frames late
        self.assertEqual(found["restart"][0], [-0.4])

    def test_jitter_keeps_order_and_repair_only_touches_live(self):
        segs = c.segments(codes_of([(5, H), (3, A), (4, D), (6, H)]), [(1, 0, 17)])
        for seed in range(20):
            out = b.jittered(segs, 30.0, np.random.default_rng(seed), 18)
            self.assertEqual([code for _, _, code in c.segments(out, [(1, 0, 17)])], [H, A, D, H])
        reference = codes_of([(100, H), (100, D), (100, A)])
        seen = np.ones(300, dtype=bool)
        seen[10:90] = False            # 80-frame blank inside live play
        seen[150:300] = False
        fixed = b.repaired(reference, seen, minimum=75)
        self.assertTrue((fixed[10:90] == D).all() and (fixed[:10] == H).all() and (fixed[90:100] == H).all())
        self.assertTrue((fixed[200:] == D).all())


class Refiner(unittest.TestCase):
    def setUp(self):
        n = 400
        predicted = codes_of([(150, H), (100, D), (150, A)])
        self.k = SimpleNamespace(n=n, blocks=[(1, 0, n - 1)], edge=np.zeros(n))
        self.k.p = np.full((n, 3), 0.1)
        self.k.p[np.arange(n), predicted] = 0.8
        nan = np.full(n, np.nan)
        self.k.signals = dict(seen=np.zeros(n, dtype=bool), speed=nan, touchline=nan, goalline=nan, event=nan,
                              distance=np.full((n, 2), np.nan))
        self.segs = c.segments(predicted, self.k.blocks)

    def test_constant_shift_and_limits(self):
        moved = refine.apply(self.k, self.segs, dict(onset=-20.0, restart=5.0, change=0.0), "const")
        self.assertEqual(moved, [(0, 130, H), (130, 255, D), (255, 400, A)])
        clipped = refine.apply(self.k, self.segs, dict(onset=75.0, restart=-75.0, change=0.0), "const")
        self.assertEqual([code for _, _, code in clipped], [H, D, A])
        self.assertTrue(all(z - a >= 3 for a, z, _ in clipped))

    def test_feature_width(self):
        rows = refine.rows_of(self.k, self.segs, self.segs)
        self.assertEqual(rows["onset"][0].shape, (1, len(refine.FEATURES)))
        self.assertEqual(rows["onset"][1].tolist(), [0.0])
        self.assertEqual(len(rows["change"][2]), 0)


class Stacker(unittest.TestCase):
    def test_mirror_symmetry_of_the_design(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(3), size=200)
        edge = rng.random(200)
        even, odd = stack.half_features(p, edge)
        even2, odd2 = stack.half_features(p[:, [1, 0, 2]], edge)
        self.assertTrue(np.allclose(even, even2, equal_nan=True))
        self.assertTrue(np.allclose(odd, -odd2, equal_nan=True))

    def test_rolling_and_lag(self):
        x = np.arange(6, dtype=float)
        self.assertTrue(np.allclose(stack._lag(x, 2)[:4], [2, 3, 4, 5]) and np.isnan(stack._lag(x, 2)[4:]).all())
        self.assertTrue(np.allclose(stack._rolling(x, 2, -1)[2:], [0.5, 1.5, 2.5, 3.5]))
        self.assertTrue(np.allclose(stack._rolling(x, 3, 0)[1:5], [1, 2, 3, 4]))


if __name__ == "__main__":
    unittest.main()
