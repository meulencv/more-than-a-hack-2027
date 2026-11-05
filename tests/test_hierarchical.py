import unittest
import numpy as np
from possession.hierarchical import probabilities


class FakeBinary:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        p = np.asarray(x)[:, self.p]
        return np.column_stack((1 - p, p))


class HierarchyTests(unittest.TestCase):
    def test_normalized_probabilities_and_team_symmetry(self):
        model = {'dead': FakeBinary(0), 'team': FakeBinary(1)}
        x = np.array([[.2, .1], [.9, .7], [.0, .4]])
        mirror = np.array([[.2, .9], [.9, .3], [.0, .6]])
        p = probabilities(model, x, mirror)
        q = probabilities(model, mirror, x)
        np.testing.assert_allclose(p.sum(axis=1), 1)
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        np.testing.assert_allclose(p, q[:, [1, 0, 2]])
        np.testing.assert_allclose(p[:, 2], x[:, 0])


if __name__ == '__main__':
    unittest.main()
