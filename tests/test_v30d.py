"""V30d: synthetic dead-ball sightings and the in-domain training rows (no challenge labels)."""
import unittest
import numpy as np
from possession import v30d_data, v30d_eval, v30d_train
from possession.sources import Match


def toy_match(n=400):
    t = np.arange(n, dtype=np.float32)
    ball = np.column_stack((10 + 0.05 * t, 30 + 0 * t)).astype(np.float32)
    dead = np.zeros(n, dtype=bool)
    dead[100:200] = True
    dead[300:340] = True
    ball[dead] = np.nan
    ball[150] = [50.0, 50.0]                                   # a real observation inside a stoppage
    players = np.stack([np.column_stack((20 + 0.01 * t, 20 + 0 * t)), np.column_stack((60 - 0.01 * t, 40 + 0 * t))], axis=1).astype(np.float32)
    return Match("toy", players, np.array([0, 1], dtype=np.int8), ball, (105.0, 68.0), [(1, 0, n - 1)]), dead


class V30dTest(unittest.TestCase):
    def test_sightings_fill_only_missing_dead_frames_and_are_deterministic(self):
        match, dead = toy_match()
        a, filled = v30d_data.sighted(match, dead, match.blocks, seed=7, p_fill=(1.0, 1.0), p_from_start=1.0)
        b, _ = v30d_data.sighted(match, dead, match.blocks, seed=7, p_fill=(1.0, 1.0), p_from_start=1.0)
        self.assertTrue(np.array_equal(a.ball, b.ball, equal_nan=True))
        self.assertTrue(np.array_equal(a.ball[~dead], match.ball[~dead], equal_nan=True))        # live play untouched
        self.assertTrue(np.array_equal(a.ball[150], match.ball[150]))                             # real observation kept
        self.assertTrue(np.isfinite(a.ball[dead]).all() and filled == dead.sum() - 1)             # every stoppage, from its first frame
        self.assertTrue(np.array_equal(a.players, match.players))

    def test_no_sightings_when_share_is_zero(self):
        match, dead = toy_match()
        a, filled = v30d_data.sighted(match, dead, match.blocks, seed=3, p_fill=(0.0, 0.0), p_from_start=0.4)
        self.assertTrue(filled == 0 and np.array_equal(a.ball, match.ball, equal_nan=True))

    def test_dead_runs_are_cut_at_the_halves(self):
        dead = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
        self.assertTrue(v30d_data.dead_runs(dead, [(1, 0, 3), (2, 4, 7)]) == [(0, 2), (3, 4), (4, 6), (7, 8)])

    def test_mirror_swaps_teams_and_keeps_dead_and_ungraded(self):
        self.assertTrue(v30d_train.mirrored(np.array([0, 1, 2, -1])).tolist() == [1, 0, 2, -1])

    def test_training_views_never_use_the_evaluation_seeds(self):
        for game in v30d_data.GAMES:
            for view in v30d_data.VIEWS:
                self.assertTrue(4000 + 100 * game + view != 907 + game and 3000 + 100 * game + view != 5000 + game)

    def test_hedged_target_and_bases(self):
        values = np.array([[0.2, 9.0, -0.1], [0.0, 9.0, 0.4]])                          # consistent, original, alternative
        self.assertTrue(np.allclose(v30d_eval.hedged(values, 1.0), [0.2, 0.0]))
        self.assertTrue(np.allclose(v30d_eval.hedged(values, 0.0), [-0.1, 0.4]))
        self.assertTrue(np.allclose(v30d_eval.hedged(values, 0.5), [0.05, 0.2]))
        self.assertTrue(all(np.isclose(sum(parts.values()), 1.0) for parts in v30d_eval.BASES.values()))

    def test_run_names_keep_held_out_folds_apart(self):
        names = {v30d_train.run_name(t, h, s) for t in "CA" for h in (0, 1, 2, 3) for s in (0, 10)}
        self.assertTrue(len(names) == 16)
        self.assertTrue(v30d_train.metrica_games(2) == [1, 3] and v30d_train.metrica_games(0) == [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
