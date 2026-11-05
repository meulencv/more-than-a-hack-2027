import unittest
import numpy as np
import pandas as pd
from possession.external_training import weak_labels, feature_matrix, occlude


class ExternalTrainingTests(unittest.TestCase):
    def test_stoppage_requires_restart_in_same_period(self):
        events = pd.DataFrame([
            {"Type": "BALL OUT", "Period": 1, "Start Frame": 10, "End Frame": 20},
            {"Type": "SET PIECE", "Period": 2, "Start Frame": 100, "End Frame": 100},
        ])
        self.assertTrue((weak_labels(events, 200) == -1).all())

    def test_stoppage_and_pass_interiors(self):
        events = pd.DataFrame([
            {"Type": "BALL OUT", "Period": 1, "Start Frame": 10, "End Frame": 20},
            {"Type": "SET PIECE", "Period": 1, "Start Frame": 100, "End Frame": 100},
            {"Type": "PASS", "Period": 1, "Start Frame": 110, "End Frame": 150},
        ])
        y = weak_labels(events, 200)
        self.assertTrue((y[30:89] == 1).all())
        self.assertTrue((y[115:145] == 0).all())
        self.assertEqual(y[99], -1)

    def test_conflicting_live_observation_excludes_dead_gap(self):
        events = pd.DataFrame([
            {"Type": "BALL OUT", "Period": 1, "Start Frame": 10, "End Frame": 20},
            {"Type": "SET PIECE", "Period": 1, "Start Frame": 100, "End Frame": 100},
            {"Type": "PASS", "Period": 1, "Start Frame": 50, "End Frame": 80},
        ])
        self.assertFalse((weak_labels(events, 200) == 1).any())

    def test_features_have_no_infinities_and_dropout_is_repeatable(self):
        players = np.zeros((100, 4, 3), dtype=np.float32)
        ball = np.zeros((100, 3), dtype=np.float32)
        missing = occlude(ball, 2027)
        np.testing.assert_equal(missing, occlude(ball, 2027))
        x = feature_matrix(players, missing)
        self.assertEqual(x.shape, (100, 12))
        self.assertFalse(np.isinf(x).any())


if __name__ == "__main__":
    unittest.main()
