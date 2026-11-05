import unittest
import pandas as pd
from possession.reference import _one_touch, event_reference
from possession.audit_reference import corrected_one_touch, alternative_reference


class ReferenceAuditTests(unittest.TestCase):
    def test_current_lost_event_must_not_match_itself(self):
        rows = pd.DataFrame([
            dict(team=1, type="BALL LOST", start=50, end=100),
            dict(team=0, type="PASS", start=150, end=180),
        ])
        rows = list(rows.itertuples())
        self.assertTrue(_one_touch(rows, 0))  # Demonstrates the frozen implementation's bug.
        self.assertFalse(corrected_one_touch(rows, 0))

    def test_actual_short_recovery_is_still_a_touch(self):
        rows = list(pd.DataFrame([
            dict(team=1, type="RECOVERY", start=50, end=50),
            dict(team=1, type="BALL LOST", start=52, end=52),
            dict(team=0, type="PASS", start=55, end=80),
        ]).itertuples())
        self.assertTrue(corrected_one_touch(rows, 0))

    def test_reference_contains_sustained_opposing_control(self):
        events = pd.DataFrame([
            dict(team=0, type="SET PIECE", subtype="KICK OFF", period=1, start=0, end=0),
            dict(team=0, type="PASS", subtype="", period=1, start=0, end=20),
            dict(team=1, type="BALL LOST", subtype="", period=1, start=50, end=100),
            dict(team=0, type="PASS", subtype="", period=1, start=150, end=180),
        ])
        original, _ = event_reference(events, 200)
        alternative, _ = alternative_reference(events, 200)
        self.assertTrue((original[50:150] == 0).all())
        self.assertTrue((alternative[50:150] == 1).all())


if __name__ == "__main__":
    unittest.main()
