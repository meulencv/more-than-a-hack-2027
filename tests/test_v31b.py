"""V31b: the a-priori compromise relabelling (no Metrica label, no challenge label)."""
import unittest
import numpy as np
from possession import v31b_eval, v31b_relabel


def sequence(*runs):
    return np.concatenate([np.full(int(round(seconds * 25)), code, dtype=np.int8) for code, seconds in runs])


class V31bTest(unittest.TestCase):
    def test_short_sandwiched_possession_is_absorbed(self):
        y = sequence((0, 5), (1, 1.5), (0, 5))
        out = v31b_relabel.compromise(y, 2.0)
        self.assertTrue((out == 0).all())

    def test_long_or_unsandwiched_possessions_are_kept(self):
        y = sequence((0, 5), (1, 2.5), (0, 5), (2, 3), (1, 1), (0, 4), (1, 1), (2, 2))
        out = v31b_relabel.compromise(y, 2.0)
        self.assertTrue(np.array_equal(out, y))                 # 2.5 s kept; D-X-T and T-X-D touches kept
        self.assertFalse(np.array_equal(v31b_relabel.compromise(y, 3.0), y))

    def test_stoppages_and_ungraded_frames_are_untouched(self):
        y = sequence((-1, 2), (0, 4), (1, 1), (0, 4), (2, 3), (1, 4), (0, 0.4), (1, 4), (-1, 1))
        out = v31b_relabel.compromise(y, 2.0)
        self.assertTrue(np.array_equal(out == 2, y == 2) and np.array_equal(out < 0, y < 0))
        self.assertEqual(v31b_relabel.stats(out)["spells"], 2)

    def test_chain_of_touches_collapses_to_the_long_holders(self):
        y = sequence((0, 6), (1, 1), (0, 1), (1, 6))
        out = v31b_relabel.compromise(y, 2.0)
        self.assertEqual(v31b_relabel.stats(out)["spells"], 2)   # two possessions remain, whichever touch went first

    def test_blend_weights_sum_to_one_and_frozen_is_half(self):
        for name, parts in v31b_eval.BASES.items():
            self.assertAlmostEqual(sum(parts.values()), 1.0)
            if name.endswith("_blend") and name != "v28d_blend":
                self.assertEqual(parts["frozen"], 0.5)

    def test_new_outputs_stay_inside_the_scope(self):
        self.assertTrue(str(v31b_eval.output_path("m2", 1, "clean")).startswith("data/work/v31b/"))
        self.assertTrue(str(v31b_eval.output_path("frozen", 1, "clean")).startswith("data/work/v16/"))


if __name__ == "__main__":
    unittest.main()
