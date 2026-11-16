import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from possession import v27_core as c
from possession import v28a_core as k28
from possession import v29_final as final
from possession import v29_lab as lab


class Always:
    """Stub verifier model: every candidate removal scores `value`."""
    columns = lab.V

    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


def synthetic(plan, fps=5):
    """plan: [(frames at 25 Hz, code)] -> one half of 5 Hz probabilities that decode to it."""
    codes = np.concatenate([np.full(n, k) for n, k in plan])
    frames = np.arange(0, len(codes), fps)
    probabilities = np.full((len(frames), 3), 0.01)
    probabilities[np.arange(len(frames)), codes[frames]] = 0.98
    return probabilities, np.zeros(len(frames)), frames, [(1, 0, len(codes) - 1)], len(codes)


class HedgedTarget(unittest.TestCase):
    def test_weighted_target_and_robust_minimum(self):
        values = np.zeros((3, len(c.REFERENCES)))
        values[:, c.REFERENCES.index("consistent")] = [0.010, -0.002, 0.004]
        values[:, c.REFERENCES.index("alternative")] = [-0.010, 0.006, 0.004]
        self.assertTrue(np.allclose(lab.hedge(values, "w1.00"), [0.010, -0.002, 0.004]))
        self.assertTrue(np.allclose(lab.hedge(values, "w0.70"), [0.004, 0.0004, 0.004]))
        self.assertTrue(np.allclose(lab.hedge(values, "w0.50"), [0.0, 0.002, 0.004]))
        self.assertTrue(np.allclose(lab.hedge(values, "robust"), [-0.010, -0.002, 0.004]))
        with self.assertRaises(ValueError):
            lab.hedge(values, "w0.30")

    def test_prior_and_a_priori_target_are_the_same_number(self):
        self.assertEqual(lab.TARGETS["w0.70"], lab.PRIOR)
        self.assertEqual(final.CANDIDATES["hedge"], dict(base="blend", family=final.FAMILY, target="w0.70"))
        self.assertEqual(final.CANDIDATES["bet"], dict(base="blend", family=final.FAMILY, target="w1.00"))
        self.assertEqual(final.CANDIDATES["safe"], dict(base="frozen", family="v27", target="robust"))

    def test_families_use_the_documented_columns(self):
        self.assertEqual(lab.FAMILIES["v27"]["columns"], list(range(len(c.FEATURES))))
        self.assertFalse(lab.needs_physics(lab.FAMILIES["v27_blend"]["columns"]))
        self.assertTrue(lab.needs_physics(lab.FAMILIES["v28a"]["columns"]))
        names = [k28.FEATURES[j] for j in lab.FAMILIES["v28a"]["columns"]]
        self.assertFalse({"seen_in", "pre_seen", "post_seen"} & set(names))      # no ball-availability feature


class Verifier(unittest.TestCase):
    PLAN = [(250, 0), (40, 1), (250, 0), (200, 2), (30, 1), (200, 2), (300, 1)]

    def test_features_without_signals_keep_the_v27_block_only(self):
        probabilities, edges, frames, blocks, n = synthetic(self.PLAN)
        codes, p, edge = final.decoded(probabilities, edges, frames, blocks, n)
        segs = c.segments(codes, blocks)
        cands = c.candidates(segs, lab.MAX_FRAMES)
        x = lab.features(segs, cands, p, edge)
        self.assertEqual(x.shape, (len(cands), len(k28.FEATURES)))
        self.assertTrue(np.array_equal(x[:, lab.V], c.candidate_features(segs, cands, p, edge)))
        self.assertTrue(np.isnan(x[:, len(c.FEATURES):]).all())

    def test_positive_scores_remove_short_possessions_and_negative_scores_nothing(self):
        probabilities, edges, frames, blocks, n = synthetic(self.PLAN)
        saved = dict(model=Always(1.0), max_frames=lab.MAX_FRAMES, base="test", family="stub", target="stub")
        after, before, report = final.verify(probabilities, edges, frames, blocks, n, saved)
        self.assertEqual(report["before"]["spells"], 5)
        self.assertEqual(report["after"]["spells"], 2)                    # H(absorbed) ... D ... A
        self.assertEqual(report["removals_by_kind"], dict(absorb=1, to_dead=1))
        self.assertTrue((after[:530] == 0).all() and (after[550:960] == 2).all() and (after[980:] == 1).all())
        saved["model"] = Always(-1.0)
        after, before, report = final.verify(probabilities, edges, frames, blocks, n, saved)
        self.assertTrue(np.array_equal(after, before))
        self.assertEqual(report["removals"], 0)

    def test_physical_verifier_refuses_to_run_without_signals(self):
        probabilities, edges, frames, blocks, n = synthetic(self.PLAN)
        model = Always(1.0)
        model.columns = lab.FAMILIES["v28a"]["columns"]
        with self.assertRaises(ValueError):
            final.verify(probabilities, edges, frames, blocks, n, dict(model=model, max_frames=lab.MAX_FRAMES))


class Files(unittest.TestCase):
    def test_existing_csv_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v29_test.csv"
            codes = np.array([0, 1, 2, 3], dtype=np.int8)
            first = final.write_csv(path, codes)
            self.assertEqual(final.write_csv(path, codes), first)          # identical bytes: accepted
            with self.assertRaises(RuntimeError):
                final.write_csv(path, codes[::-1])
            self.assertEqual(final.digest(path), first)

    def test_fitted_verifiers_reproduce_the_reviewed_ones(self):
        import joblib
        pairs = [("frozen", "w1.00", Path("data/work/v27/verifier_consistent.joblib")),
                 ("frozen", "robust", Path("data/work/v27/verifier_robust.joblib")),
                 ("blend", "w1.00", Path("data/work/v28d/verifier_blend_consistent.joblib"))]
        for base, target, old in pairs:
            new = lab.verifier_path(base, "v27", target)
            if not (old.exists() and new.exists() and (lab.ROOT / f"table_{base}.npz").exists()):
                self.skipTest("verifiers not fitted")
            table = lab.load_table(base)
            x = np.concatenate([table[f"x_{g}_{m}"] for g in lab.GAMES for m in c.MODES])
            self.assertTrue(np.array_equal(joblib.load(old)["model"].predict(x[:, lab.V]), joblib.load(new)["model"].predict(x)))

    def test_written_candidates(self):
        report = lab.ROOT / "run" / "run.json"
        if not report.exists():
            self.skipTest("pipeline not run")
        report = json.loads(report.read_text())
        self.assertTrue(report.get("frozen_reproduces_run_f2", True))
        self.assertFalse(report["challenge_labels_read"])
        codes = {}
        for name, row in report["candidates"].items():
            path = Path("evaluation/results") / f"v29_{name}.csv"
            if not path.exists():
                self.skipTest("CSV not written")
            self.assertEqual(final.digest(path), row["csv_sha256"])
            codes[name] = final.codes_of(path)
            self.assertEqual(len(codes[name]), row["rows"])
            self.assertLessEqual(row["after"]["spells"], row["before"]["spells"])
        if {"safe", "hedge_frozen"} <= set(codes) and Path("evaluation/results/v26b_video_ball.csv").exists():
            # frozen base: before the verifier the decode IS the adopted video-ball candidate
            self.assertEqual(report["candidates"]["safe"]["decoder_only_different_from"]["v26b_video_ball"], 0)
            self.assertTrue(np.array_equal(codes["safe"] == 3, final.codes_of("evaluation/results/v26b_video_ball.csv") == 3))


if __name__ == "__main__":
    unittest.main()
