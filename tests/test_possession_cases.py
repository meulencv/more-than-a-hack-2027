import unittest
import numpy as np
from possession.sequence_quality_gate import combine, missing_fraction
from possession.sources import Match
from possession.temporal_v3 import decode


def states(prob):
    return decode(np.log(np.clip(prob, 1e-4, 1)), [5, 5, 5], 4., 2.5, 2.5)


class PossessionCases(unittest.TestCase):
    def test_stoppage_splits_two_possessions_of_same_team(self):
        p = np.repeat([[.98, .01, .01], [.01, .01, .98], [.98, .01, .01]], 30, axis=0)
        np.testing.assert_array_equal(states(p), np.repeat([0, 2, 0], 30))

    def test_ball_missing_does_not_force_dead_ball(self):
        n = 100
        match = Match('synthetic', np.zeros((n, 2, 2)), np.array([0, 1]),
                      np.full((n, 2), np.nan), (105, 68), [(1, 0, n - 1)])
        hgb = np.tile([.1, .1, .8], (n, 1))
        temporal = np.tile([.98, .01, .01], (n, 1))
        p = combine(hgb, temporal, missing_fraction(match, np.arange(n)))
        np.testing.assert_array_equal(states(p), np.zeros(n))

    def test_restart_resets_previous_team(self):
        p = np.repeat([[.98, .01, .01], [.01, .01, .98], [.01, .98, .01]], 30, axis=0)
        np.testing.assert_array_equal(states(p), np.repeat([0, 2, 1], 30))


if __name__ == '__main__':
    unittest.main()
