import unittest
from types import SimpleNamespace

import numpy as np

from possession import v31_law as law

H, A, D = 0, 1, 2
PITCH = (100.0, 60.0)
BLOCKS = [(1, 100, 999), (2, 1500, 2499)]
KICKOFFS = (100, 1600)


def match(first_half_team, second_half_team, n=2600):
    """Two players per team; the kicking team has a player on the centre spot before each kick-off."""
    players = np.full((n, 4, 2), np.nan, dtype=np.float64)
    team = np.array([H, H, A, A], dtype=np.int8)
    players[:, 0] = (30.0, 30.0)
    players[:, 1] = (35.0, 20.0)
    players[:, 2] = (70.0, 30.0)
    players[:, 3] = (65.0, 40.0)
    for kickoff, kicker in zip(KICKOFFS, (first_half_team, second_half_team)):
        slot = 0 if kicker == H else 2
        players[kickoff - 60:kickoff + 5, slot] = (50.4, 30.2)
    return SimpleNamespace(players=players, team=team, pitch=PITCH)


def codes(second_half_first, n=2600):
    out = np.full(n, 3, dtype=np.int8)
    out[100:1000] = D
    out[100:300] = A                       # first half: A kicks off
    out[300:600] = H
    out[1500:2500] = D
    start = 1599                           # one frame before the published kick-off, as in the real decode
    for team, length in second_half_first:
        out[start:start + length] = team
        start += length
    return out


class KickoffLaw(unittest.TestCase):
    def test_short_wrong_team_possession_is_reassigned(self):
        before = codes([(A, 32), (H, 300)])
        after, report = law.kickoff_law(before, match(A, H), BLOCKS, KICKOFFS)
        self.assertEqual(report["reassigned_frames"], 32)
        self.assertEqual(report["dead_frames_before_kickoff"], 1)
        self.assertEqual(after[1599], D)
        self.assertTrue((after[1600:1931] == H).all())
        changed = np.flatnonzero(after != before)
        self.assertEqual((changed.min(), changed.max()), (1599, 1630))

    def test_right_team_is_left_alone(self):
        before = codes([(H, 332)])
        after, report = law.kickoff_law(before, match(A, H), BLOCKS, KICKOFFS)
        self.assertEqual(report["reassigned_frames"], 0)
        self.assertTrue((after[1600:] == before[1600:]).all())

    def test_long_wrong_team_possession_is_left_alone(self):
        before = codes([(A, 200), (H, 300)])
        after, report = law.kickoff_law(before, match(A, H), BLOCKS, KICKOFFS)
        self.assertEqual(report["reassigned_frames"], 0)
        self.assertTrue((after[1600:] == before[1600:]).all())

    def test_nothing_changes_when_tracking_contradicts_the_law(self):
        before = codes([(A, 32), (H, 300)])
        after, report = law.kickoff_law(before, match(A, A), BLOCKS, KICKOFFS)
        self.assertFalse(report["law_and_tracking_agree"])
        self.assertEqual(report["reassigned_frames"], 0)
        self.assertTrue((after[1600:] == before[1600:]).all())

    def test_wrong_team_not_followed_by_the_right_team_is_left_alone(self):
        before = codes([(A, 32)])          # then dead ball
        after, report = law.kickoff_law(before, match(A, H), BLOCKS, KICKOFFS)
        self.assertEqual(report["reassigned_frames"], 0)

    def test_input_is_not_modified(self):
        before = codes([(A, 32), (H, 300)])
        copy = before.copy()
        law.kickoff_law(before, match(A, H), BLOCKS, KICKOFFS)
        self.assertTrue((before == copy).all())


if __name__ == "__main__":
    unittest.main()
