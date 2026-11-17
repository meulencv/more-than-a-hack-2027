"""V31c: label-blind ball views and the compromise target M (no challenge labels)."""
import inspect
import unittest
import numpy as np
from possession import v31c_data, v31c_eval, v31c_train
from possession.sources import Match


def toy_match(n=600):
    t = np.arange(n, dtype=np.float32)
    ball = np.column_stack((10 + 0.05 * t, 30 + 0 * t)).astype(np.float32)
    ball[100:200] = np.nan
    ball[300:340] = np.nan
    ball[400:404] = np.nan
    ball[590:] = np.nan
    players = np.stack([np.column_stack((20 + 0.01 * t, 20 + 0 * t)), np.column_stack((60 - 0.01 * t, 40 + 0 * t))], axis=1).astype(np.float32)
    return Match("toy", players, np.array([0, 1], dtype=np.int8), ball, (105.0, 68.0), [(1, 0, n - 1)])


class V31cTest(unittest.TestCase):
    def test_blind_fill_has_no_label_argument_and_only_fills_gaps(self):
        self.assertTrue(list(inspect.signature(v31c_data.blind_fill).parameters) == ["match", "seed"])
        self.assertTrue(list(inspect.signature(v31c_data.random_removal).parameters)[:3] == ["match", "seed", "share_range"])
        match = toy_match()
        seen = np.isfinite(match.ball).all(axis=1)
        a, report = v31c_data.blind_fill(match, 5)
        b, _ = v31c_data.blind_fill(match, 5)
        self.assertTrue(np.array_equal(a.ball, b.ball))                                   # deterministic
        self.assertTrue(np.isfinite(a.ball).all())                                        # no gap left in the half
        self.assertTrue(np.array_equal(a.ball[seen], match.ball[seen]))                   # observations untouched
        self.assertTrue(report["filled_frames"] == int((~seen).sum()))
        self.assertTrue(np.array_equal(a.players, match.players))
        self.assertTrue(np.allclose(a.ball[590:], match.ball[589]))                       # open-ended gap: hold
        self.assertTrue(np.abs(a.ball[400:404] - match.ball[399]).max() < 1.5)            # short gap: a line

    def test_gaps(self):
        seen = np.array([1, 0, 0, 1, 1, 0], dtype=bool)
        self.assertTrue([(int(a), int(z)) for a, z in v31c_data.gaps(seen)] == [(1, 3), (5, 6)])

    def test_compromise_absorbs_only_short_sandwiched_possessions(self):
        fps = 25
        parts = [(0, 100), (1, 30), (0, 100), (1, 60), (0, 100), (2, 50), (1, 20), (2, 50), (1, 20), (0, 100)]
        labels = np.concatenate([np.full(n, k, dtype=np.int8) for k, n in parts])
        labels = np.r_[labels, np.int8(-1)]
        out = v31c_data.compromise(labels, [(1, 0, len(labels) - 2)], 2.0)
        expected = labels.copy()
        expected[100:130] = 0                                    # 1.2 s between two possessions of the other team
        self.assertTrue(np.array_equal(out, expected))           # 2.4 s kept; islands and edges next to D kept
        self.assertTrue(np.array_equal(v31c_data.compromise(labels, [(1, 0, len(labels) - 2)], 3.0)[230:290], np.zeros(60)))
        self.assertTrue(fps == 25 and out[-1] == -1)

    def test_vanish_alignment_diagnostic(self):
        labels = np.zeros(100, dtype=np.int8)
        labels[40:60] = 2
        seen = np.ones(100, dtype=bool)
        seen[41:60] = False
        self.assertTrue(v31c_data.vanish_alignment(seen, labels) == 1.0)
        seen[:] = True
        seen[10:20] = False
        self.assertTrue(v31c_data.vanish_alignment(seen, labels) == 0.0)
        codes = labels.copy()
        self.assertTrue(v31c_eval.dstart_at_vanish(codes, labels, seen) == (0.0, 1))

    def test_training_seeds_differ_from_evaluation_seeds(self):
        evaluation = {907 + g for g in (1, 2, 3)} | {7000 + g for g in (1, 2, 3)} | {5000 + g for g in (1, 2, 3)}
        for game in v31c_data.GAMES:
            for view in v31c_data.VIEWS:
                self.assertTrue(not {8000 + 100 * game + view, 9000 + 100 * game + view} & evaluation)

    def test_names_targets_and_bases(self):
        self.assertTrue(v31c_train.TARGETS["M"] == v31c_data.ABSORB_S == 2.0)
        names = {v31c_train.run_name(t, h, s) for t in v31c_train.TARGETS for h in (0, 1, 2, 3) for s in (0, 10)}
        self.assertTrue(len(names) == 32)
        self.assertTrue(v31c_train.metrica_games(2) == [1, 3] and v31c_train.metrica_games(0) == [1, 2, 3])
        self.assertTrue(v31c_train.mirrored(np.array([0, 1, 2, -1])).tolist() == [1, 0, 2, -1])
        self.assertTrue(all(np.isclose(sum(parts.values()), 1.0) for parts in v31c_eval.BASES.values()))
        self.assertTrue(v31c_eval.tag_of("M", 10) == "M_seed10" and v31c_eval.tag_of("A") == "Ab")
        self.assertTrue(str(v31c_eval.output_path("M", 1, "interp")).startswith("data/work/v31c/outputs/M/"))


if __name__ == "__main__":
    unittest.main()
