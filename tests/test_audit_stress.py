import unittest
import numpy as np
from possession.sources import Match
from possession.audit_stress import blind_corruption


class BlindStressTests(unittest.TestCase):
    def test_observations_do_not_depend_on_labels(self):
        players = np.tile(np.array([[[10., 20.], [15., 25.], [80., 20.], [90., 25.]]]), (1000, 1, 1))
        ball = np.full((1000, 2), 25.)
        teams = np.array([0, 0, 1, 1])
        match = Match("synthetic", players, teams, ball, (105., 68.), [(1, 0, 999)],
                      {"labels": np.zeros(1000, dtype=int)})
        other = match.copy_with(meta={"labels": np.full(1000, 2)})
        a, b = blind_corruption(match), blind_corruption(other)
        np.testing.assert_equal(a.players, b.players)
        np.testing.assert_equal(a.ball, b.ball)
        self.assertFalse(np.array_equal(a.ball, match.ball))


if __name__ == "__main__":
    unittest.main()
