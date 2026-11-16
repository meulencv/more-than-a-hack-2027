import unittest
from possession.v30_boundaries import shift


class ShiftTest(unittest.TestCase):
    def test_each_type_moves_by_its_constant(self):
        segs = [(0, 100, 0), (100, 200, 2), (200, 300, 1), (300, 400, 0)]
        self.assertEqual(shift(segs, to_dead=-12, from_dead=4, team=-4),
                         [(0, 88, 0), (88, 204, 2), (204, 296, 1), (296, 400, 0)])

    def test_zero_is_identity_and_frames_are_kept(self):
        segs = [(0, 3, 0), (3, 5, 1), (5, 40, 0)]
        self.assertEqual(shift(segs), segs)
        out = shift(segs, team=-10)
        self.assertEqual((out[0][0], out[-1][1]), (0, 40))
        self.assertTrue(all(z > a for a, z, _ in out))
        self.assertTrue(all(out[i][1] == out[i + 1][0] for i in range(len(out) - 1)))

    def test_half_edges_are_not_boundaries(self):
        segs = [(0, 50, 0), (60, 90, 1)]
        self.assertEqual(shift(segs, team=-4), segs)


if __name__ == "__main__":
    unittest.main()
