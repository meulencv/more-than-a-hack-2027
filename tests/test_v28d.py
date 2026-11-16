import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from possession import v28d_relabel as relabel
from possession import v28d_semantic as semantic

HOME, AWAY, LOOSE = (10.0, 10.0), (50.0, 10.0), (30.0, 40.0)


def synthetic(plan):
    """plan: [(frames, label, where the ball is)] -> (labels, match with one player per team)."""
    labels, ball = [], []
    for frames, label, where in plan:
        labels += [label] * frames
        ball += [where] * frames
    n = len(labels)
    players = np.zeros((n, 2, 2), dtype=np.float32)
    players[:, 0] = HOME
    players[:, 1] = AWAY
    ball = np.asarray(ball, dtype=np.float32) + [0.5, 0.0]
    return np.asarray(labels, dtype=np.int8), SimpleNamespace(players=players, team=np.array([0, 1], dtype=np.int8), ball=ball)


class Relabel(unittest.TestCase):
    def test_uncontrolled_touch_between_two_possessions_is_absorbed(self):
        labels, match = synthetic([(100, 0, HOME), (5, 1, AWAY), (45, 1, LOOSE), (100, 0, HOME)])
        new, report = relabel.relabel(labels, match)
        self.assertTrue((new == 0).all())
        self.assertEqual(report["absorbed"], 1)
        self.assertEqual(report["frames_changed"], 50)

    def test_controlled_possession_is_kept(self):
        labels, match = synthetic([(100, 0, HOME), (40, 1, AWAY), (10, 1, LOOSE), (100, 0, HOME)])
        new, report = relabel.relabel(labels, match)
        self.assertTrue(np.array_equal(new, labels))
        self.assertEqual(report["absorbed"], 0)

    def test_contact_threshold_is_the_rule_constant(self):
        frames = int(relabel.RULE["min_contact_s"] * relabel.FPS)
        for contact, kept in ((frames - 1, False), (frames, True)):
            labels, match = synthetic([(100, 0, HOME), (contact, 1, AWAY), (20, 1, LOOSE), (100, 0, HOME)])
            new, _ = relabel.relabel(labels, match)
            self.assertEqual(bool((new == 1).any()), kept)

    def test_restart_and_stoppages_are_never_touched(self):
        # touch right after a stoppage takes the ball (restart); touch before a stoppage is kept too
        labels, match = synthetic([(50, 2, LOOSE), (5, 1, AWAY), (100, 0, HOME), (5, 1, AWAY), (50, 2, LOOSE), (80, 0, HOME)])
        new, report = relabel.relabel(labels, match)
        self.assertTrue(np.array_equal(new, labels))
        self.assertTrue(np.array_equal(new == 2, labels == 2))

    def test_chain_of_touches_keeps_the_holder_until_a_controlled_possession(self):
        labels, match = synthetic([(100, 0, HOME), (5, 1, AWAY), (5, 0, HOME), (60, 1, AWAY), (100, 0, HOME)])
        new, report = relabel.relabel(labels, match)
        self.assertTrue((new[:110] == 0).all())           # touch absorbed, the holder continues
        self.assertTrue((new[110:170] == 1).all())        # the controlled possession is kept
        self.assertEqual(report["absorbed"], 1)

    def test_unknown_frames_separate_live_stretches(self):
        labels, match = synthetic([(100, 0, HOME), (20, -1, LOOSE), (5, 1, AWAY), (100, 0, HOME)])
        new, _ = relabel.relabel(labels, match)
        self.assertTrue(np.array_equal(new, labels))      # first possession after a gap is a restart

    def test_statistics_of_relabelled_labels(self):
        labels, match = synthetic([(100, 0, HOME), (5, 1, AWAY), (45, 1, LOOSE), (100, 0, HOME), (50, 2, LOOSE)])
        self.assertEqual(relabel.stats(labels)["spells"], 3)
        self.assertEqual(relabel.stats(relabel.relabel(labels, match)[0])["spells"], 1)

    def test_written_labels_only_change_live_frames(self):
        for variant in relabel.VARIANTS:
            for game in relabel.GAMES:
                path = relabel.labels_dir(variant) / f"{game}.npz"
                if not path.exists():
                    self.skipTest("relabelled DFL games not written")
                saved = np.load(path)
                self.assertTrue(np.array_equal(saved["labels"] == 2, saved["native"] == 2))
                self.assertTrue(np.array_equal(saved["labels"] < 0, saved["native"] < 0))
                self.assertLess((saved["labels"] != saved["native"]).mean(), 0.12)


class Blend(unittest.TestCase):
    def test_equal_weight_blend(self):
        rng = np.random.default_rng(0)
        frames = np.arange(0, 500, 5)
        a = (rng.dirichlet([1, 1, 1], len(frames)), rng.random(len(frames)), frames)
        b = (rng.dirichlet([1, 1, 1], len(frames)), rng.random(len(frames)), frames)
        p, e, f = semantic.blend_arrays([a, b])
        self.assertTrue(np.allclose(p, (a[0] + b[0]) / 2))
        self.assertTrue(np.allclose(e, (a[1] + b[1]) / 2))
        self.assertTrue(np.allclose(p.sum(1), 1))
        self.assertTrue(np.array_equal(f, frames))

    def test_blend_rejects_other_timelines_and_weights(self):
        frames = np.arange(0, 50, 5)
        a = (np.full((10, 3), 1 / 3), np.zeros(10), frames)
        b = (np.full((10, 3), 1 / 3), np.zeros(10), frames + 5)
        with self.assertRaises(ValueError):
            semantic.blend_arrays([a, b])
        with self.assertRaises(ValueError):
            semantic.blend_arrays([a, a], weights=(0.7, 0.7))

    def test_adapter_without_camera_fusion_only_removes_possessions(self):
        if not semantic.verifier_path("consistent").exists():
            self.skipTest("verifier not fitted")
        from possession import v27_core as c
        from possession.v14_resolution import decode_fine
        rng = np.random.default_rng(3)
        frames = np.arange(0, 3000, 5)
        blocks = [(1, 0, 1499), (2, 1600, 2995)]

        def triple():
            logits = np.cumsum(rng.normal(0, 0.6, (len(frames), 3)), axis=0)
            logits -= logits.mean(1, keepdims=True)
            p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
            return p.astype("float32"), rng.random(len(frames)).astype("float32"), frames

        a, b = triple(), triple()
        after, before, report, _ = semantic.controlled_adapter(a, b, camera_fusion=False, blocks=blocks, n_frames=3000)
        p, e, _ = semantic.blend_arrays([a, b])
        match = SimpleNamespace(ball=np.empty((3000, 2)), blocks=blocks)
        self.assertTrue(np.array_equal(before, decode_fine(p, e, frames, match, "probability")))
        self.assertLessEqual(len(c.spells_of(c.segments(after, blocks))), len(c.spells_of(c.segments(before, blocks))))
        self.assertTrue(np.array_equal(after == 3, before == 3))
        again = semantic.controlled_adapter(a, b, camera_fusion=False, blocks=blocks, n_frames=3000)[0]
        self.assertTrue(np.array_equal(after, again))

    def test_selection_points_to_existing_checkpoints_with_frozen_recipe(self):
        path = Path("data/work/v28d/selection.json")
        if not path.exists():
            self.skipTest("retrained selection not written")
        new = json.loads(path.read_text())
        frozen = json.loads(Path("docs/FINAL_SELECTION.json").read_text())
        self.assertEqual([c["kind"] for c in new["components"]], [c["kind"] for c in frozen["components"]])
        self.assertEqual([c["weight"] for c in new["components"]], [c["weight"] for c in frozen["components"]])
        for spec in new["components"]:
            self.assertTrue(spec["path"].startswith("data/work/v28d/"))
            self.assertTrue(Path(spec["path"]).exists())


if __name__ == "__main__":
    unittest.main()
