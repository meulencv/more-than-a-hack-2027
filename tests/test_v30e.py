import unittest
import numpy as np
from possession import v27_core as c
from possession import v30e_features as F
from possession import v30e_retime as rt

BLOCKS = [(1, 0, 399), (2, 500, 899)]


def codes_from(runs, n=1000):
    codes = np.full(n, 3, dtype=np.int8)
    for a, z, k in runs:
        codes[a:z] = k
    return codes


def example():
    return codes_from([(0, 100, 0), (100, 160, 2), (160, 200, 1), (200, 215, 0), (215, 300, 2), (300, 400, 0),
                       (500, 520, 2), (520, 700, 1), (700, 760, 2), (760, 900, 0)])


class Retime(unittest.TestCase):
    def test_constant_moves_only_stoppage_boundaries_and_keeps_every_possession(self):
        codes = example()
        for k, r in ((10, 0), (40, 0), (0, 10), (25, 10), (400, 400)):
            out, applied = rt.constant(codes, BLOCKS, k, r)
            before = c.spells_of(c.segments(codes, BLOCKS))
            after = c.spells_of(c.segments(out, BLOCKS))
            self.assertEqual([s[2] for s in before], [s[2] for s in after])
            self.assertTrue(all(z - a >= min(rt.MIN_KEEP, zb - ab) for (a, z, _), (ab, zb, _) in zip(after, before)))
            self.assertTrue(all(a >= ab and z <= zb + r for (a, z, _), (ab, zb, _) in zip(after, before)))
            self.assertTrue(np.array_equal(out == 3, codes == 3))
            changed = out != codes
            self.assertTrue(((out[changed] == 2) | (codes[changed] == 2)).all())

    def test_constant_shift_values(self):
        out, applied = rt.constant(example(), BLOCKS, 10, 5)
        segs = c.segments(out, BLOCKS)
        self.assertIn((90, 165, 2), segs)                      # onset 100 -> 90, restart 160 -> 165
        self.assertIn((212, 305, 2), segs)                     # the 15-frame possession keeps MIN_KEEP frames
        self.assertIn((500, 525, 2), segs)                     # no onset move at the start of a half
        self.assertEqual(len([a for a in applied if a[1] == "onset"]), 3)
        self.assertEqual(len([a for a in applied if a[1] == "restart"]), 4)

    def test_zero_shift_is_identity(self):
        codes = example()
        self.assertTrue(np.array_equal(rt.constant(codes, BLOCKS, 0, 0)[0], codes))

    def test_hysteresis_goes_back_while_dead_probability_stays_high(self):
        codes = example()
        p = np.zeros(len(codes))
        p[80:100] = 0.3
        p[60:80] = 0.05
        p[690:700] = 0.3
        out, _ = rt.hysteresis(codes, BLOCKS, p, 0.2)
        segs = c.segments(out, BLOCKS)
        self.assertIn((80, 160, 2), segs)
        self.assertIn((690, 760, 2), segs)
        self.assertIn((215, 300, 2), segs)                     # P(D) below theta right before: unchanged
        self.assertTrue(np.array_equal(rt.hysteresis(codes, BLOCKS, p, 0.5)[0], codes))

    def test_move_never_crosses_the_previous_possession(self):
        codes = example()
        segs = c.segments(codes, BLOCKS)
        i = next(j for j, s in enumerate(segs) if s == (215, 300, 2))
        out, applied = rt.move(codes, segs, onsets={i: 0})
        self.assertEqual(applied, [(i, "onset", 215, 200 + rt.MIN_KEEP)])
        self.assertEqual(int((out[200:215] == 0).sum()), rt.MIN_KEEP)


class Features(unittest.TestCase):
    def test_dual_tone_needs_both_tones(self):
        rng = np.random.default_rng(0)
        base = rng.normal(-50, 1, (4, 65)).astype(np.float16)
        both, single, shifted, far = base.copy(), base.copy(), base.copy(), base.copy()
        both[1, 26] = both[1, 45] = -20
        single[1, 26] = -20
        shifted[1, 23] = shifted[1, 42] = -20                  # the pair moved together by three bins
        far[1, 10] = far[1, 60] = -20                          # two tones, wrong pitches
        self.assertGreater(F.dual_tone(both)[1], 25)
        self.assertGreater(F.dual_tone(shifted)[1], 25)
        self.assertLess(F.dual_tone(single)[1], 8)
        self.assertLess(F.dual_tone(far)[1], 8)
        self.assertLess(F.dual_tone(both)[0], 8)

    def test_events_merge_and_minimum_length(self):
        x = np.zeros(100)
        x[10:14] = 20
        x[17:20] = 20            # gap of 3 < 8: one event
        x[50:52] = 20            # two frames: too short
        x[80:90] = 15
        events = F.events_of(x, 12.0, 8, 3)
        self.assertEqual([(a, z) for a, z, _, _ in events], [(10, 20), (80, 90)])
        self.assertEqual(events[0][3], 7)

    def test_since_until(self):
        since, until = F.since_until([10, 40], 60)
        self.assertEqual(since[10], 0)
        self.assertEqual(since[25], 15)
        self.assertEqual(until[25], 15)
        self.assertEqual(since[5], F.CAP)
        self.assertEqual(until[50], F.CAP)

    def test_broadcast_frame_belongs_to_one_half(self):
        values = np.arange(100, dtype=np.float32)
        out = F.to_tactical(values, [(1, 10, 49), (2, 60, 99)], {1: -10, 2: -40}, 120)
        self.assertTrue(np.array_equal(out[10:50], np.arange(0, 40)))
        self.assertTrue(np.isnan(out[60:80]).all())            # those broadcast frames were used by half 1
        self.assertTrue(np.array_equal(out[80:100], np.arange(40, 60)))
        self.assertTrue(np.isnan(out[:10]).all() and np.isnan(out[100:]).all())

    def test_rolling_mean_ignores_missing(self):
        x = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan])
        m = F.rolling_mean(x, 3)
        self.assertEqual(m[1], 2.0)
        self.assertTrue(np.isnan(m[5]))

    def test_run_length(self):
        self.assertEqual(F.run_length(np.array([1, 1, 0, 1, 1, 1], dtype=bool)).tolist(), [1, 2, 0, 1, 2, 3])


class Candidate(unittest.TestCase):
    def setUp(self):
        from possession import v30e_candidate as cand
        self.cand = cand
        n = 2000
        self.ball = dict(b_seen=np.ones(n, np.float32), b_speed=np.full(n, 1.0, np.float32), b_outside_m=np.full(n, -10.0, np.float32))
        self.kin = dict(k_speed_mean=np.full(n, 3.0, np.float32))

    def test_whistle_needs_agreeing_evidence(self):
        ok, _ = self.cand.whistle_agrees(1000, 1010, self.ball, self.kin)        # stoppage within 1 s
        self.assertTrue(ok)
        ok, why = self.cand.whistle_agrees(1000, 1080, self.ball, self.kin)      # ball barely moves
        self.assertTrue(ok)
        self.ball["b_speed"][1020:1060] = 15.0                                   # ball played, players keep running
        ok, why = self.cand.whistle_agrees(1000, 1080, self.ball, self.kin)
        self.assertFalse(ok)
        self.kin["k_speed_mean"][1025:1050] = 1.5                                # ... but the players stop
        ok, why = self.cand.whistle_agrees(1000, 1080, self.ball, self.kin)
        self.assertTrue(ok)
        self.assertIn("slow", why)

    def test_whistle_far_from_the_stoppage_needs_a_quiet_ball(self):
        ok, _ = self.cand.whistle_agrees(1000, 1200, self.ball, self.kin)        # 8 s, ball never played
        self.assertTrue(ok)
        self.ball["b_speed"][1100:1120] = 12.0
        self.kin["k_speed_mean"][1025:1050] = 1.0                                # slowing down is not enough beyond 5 s
        ok, why = self.cand.whistle_agrees(1000, 1200, self.ball, self.kin)
        self.assertFalse(ok)
        self.assertIn("5 s", why)

    def test_physical_support_of_a_vetoed_stoppage(self):
        none = np.array([], dtype=np.int64)
        moving = dict(self.ball, b_speed=np.full(2000, 8.0, np.float32))
        self.assertEqual(self.cand.physical_support(1000, 1080, none, moving), [])
        self.assertEqual(self.cand.physical_support(1000, 1080, none, self.ball), ["ball at rest"])
        self.assertEqual(self.cand.physical_support(1000, 1080, np.array([950]), moving), ["whistle"])
        self.assertEqual(self.cand.physical_support(1000, 1080, np.array([900]), moving), [])       # 4 s before: too early
        outside = dict(moving, b_outside_m=np.full(2000, -10.0, np.float32))
        outside["b_outside_m"][1010:1014] = 2.0
        self.assertEqual(self.cand.physical_support(1000, 1080, none, outside), ["ball beyond the lines"])
        unseen = dict(self.ball, b_seen=np.zeros(2000, np.float32))
        self.assertEqual(self.cand.physical_support(1000, 1080, none, unseen), [])

    def test_restore_touches_only_supported_rows(self):
        frames = np.arange(0, 400, 5)
        fused = np.tile(np.array([0.6, 0.3, 0.1]), (len(frames), 1))
        rows = [dict(start=100, stop=150, support=["whistle"]), dict(start=200, stop=260, support=[])]
        out = self.cand.restore(fused, frames, rows)
        inside = (frames >= 100) & (frames < 150)
        self.assertTrue((out[inside, 2] > 0.999).all())
        self.assertTrue(np.array_equal(out[~inside], fused[~inside]))

    def test_crossing_rejected_when_the_ball_comes_back(self):
        self.assertTrue(self.cand.crossing_agrees(1000, 1040, dict(self.ball, b_outside_m=np.full(2000, 1.0, np.float32))))
        self.assertFalse(self.cand.crossing_agrees(1000, 1040, self.ball))


if __name__ == "__main__":
    unittest.main()
