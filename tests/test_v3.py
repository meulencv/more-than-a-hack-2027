import unittest
import numpy as np
import pandas as pd
from possession.reference import event_reference, _one_touch
from possession.temporal_v3 import decode
from possession.features_v3 import attack_direction, frame_signals
from possession.sources import Match


class TestReference(unittest.TestCase):
    def test_one_touch_reversal(self):
        # A quick recovery followed within 5 frames by losing the ball is treated as a touch, not possession change
        events = pd.DataFrame([
            {"team": 0, "type": "SET PIECE", "subtype": "KICK OFF", "period": 1, "start": 0, "end": 0},
            {"team": 0, "type": "PASS", "subtype": "", "period": 1, "start": 0, "end": 20},
            {"team": 1, "type": "RECOVERY", "subtype": "INTERCEPTION", "period": 1, "start": 25, "end": 25},
            {"team": 1, "type": "BALL LOST", "subtype": "HEAD", "period": 1, "start": 27, "end": 27},
            {"team": 0, "type": "RECOVERY", "subtype": "", "period": 1, "start": 30, "end": 30},
            {"team": 0, "type": "PASS", "subtype": "", "period": 1, "start": 30, "end": 50},
        ])
        labels, blocks = event_reference(events, 100)
        self.assertEqual(len(blocks), 1)
        # Possession should stay with team 0 throughout frames 0..50
        self.assertTrue((labels[:50] == 0).all())


class TestTemporalV3(unittest.TestCase):
    def test_single_touch_ignored(self):
        # A single opposing touch is ignored
        log_emission = np.tile([1.0, -1.0, -2.0], (40, 1))
        log_emission[15] = [-1.0, 2.0, -2.0]
        states = decode(log_emission, min_steps=(4, 4, 4),
                        cost_turnover=3.0, cost_to_dead=2.0, cost_to_live=2.0)
        self.assertTrue((states == 0).all())

    def test_minimum_residence_guarantee(self):
        # Even with arbitrary switching impulses, every non-endpoint run is >= min_steps
        rng = np.random.default_rng(42)
        log_emission = rng.normal(0, 5, size=(100, 3))
        min_steps = (4, 4, 4)
        states = decode(log_emission, min_steps=min_steps,
                        cost_turnover=3.0, cost_to_dead=2.0, cost_to_live=2.0)
        # Check segment lengths
        boundaries = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
        lengths = [b - a for a, b in zip(boundaries[:-1], boundaries[1:])]
        # Any internal run (not the last which might be cut at the window boundary) must be >= 4
        for l in lengths[:-1]:
            self.assertGreaterEqual(l, 4)

    def test_turnover_with_sufficient_evidence(self):
        log_emission = np.tile([3.0, -3.0, -3.0], (40, 1))
        log_emission[20:] = [-3.0, 3.0, -3.0]  # Permanent turnover
        states = decode(log_emission, min_steps=(4, 4, 4),
                        cost_turnover=3.0, cost_to_dead=2.0, cost_to_live=2.0)
        self.assertTrue((states[:20] == 0).all())
        self.assertTrue((states[20:] == 1).all())


class TestFeaturesV3(unittest.TestCase):
    def test_attack_direction(self):
        # 10 frames, 2 players (home=0, away=1)
        # Home player is near x=10 (deep left), Away player is near x=90 (deep right)
        # So Home attacks +x (towards right, away goal), direction should be +1
        n = 10
        players = np.zeros((n, 2, 2))
        players[:, 0, 0] = 10.0  # home
        players[:, 1, 0] = 90.0  # away
        team = np.array([0, 1], dtype=np.int8)
        direction = attack_direction(players, team, 0, n - 1)
        self.assertEqual(direction, 1)


if __name__ == "__main__":
    unittest.main()
