import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import evaluation.possession_metrics as pm
from possession import mbr_decoder as mbr


def synthetic(n, rng, noise=1.0):
    """Alternating H/A/D runs with many short spells and noisy 5 Hz probabilities."""
    states, t = np.empty(n, dtype='int8'), 0
    label = 0
    while t < n:
        length = int(rng.choice([rng.integers(5, 40), rng.integers(40, 400)]))
        states[t:t + length] = label
        t += length
        label = int(rng.choice([x for x in range(3) if x != label], p=[.5, .5]))
    frames = np.arange(0, n, 5)
    logits = np.eye(3)[states[frames]] * 3 + rng.normal(0, noise, (len(frames), 3))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    edges = np.r_[0, (states[frames][1:] != states[frames][:-1]).astype(float)] * .8
    return states, frames, p, edges


class MBRTests(unittest.TestCase):
    def test_spell_f1_matches_official(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            n = int(rng.integers(200, 3000))
            gt = rng.choice(3, n, p=[.45, .45, .1]).astype('int8')
            gt = np.repeat(gt[: n // 20 + 1], 20)[:n]
            pred = gt.copy()
            flips = rng.integers(0, n, int(rng.integers(0, 40)))
            for f in flips:
                pred[f:f + int(rng.integers(1, 60))] = rng.integers(0, 3)
            official = pm.spell_scores(gt, pred, [(0, n)])['mean_f1']
            self.assertAlmostEqual(mbr.spell_f1(gt, pred), official, places=12)

    def test_viterbi_scale_one_is_frozen_decoder(self):
        # Load decode_fine's exact source without importing v14's heavy dependencies (torch).
        import ast
        source = Path('possession/v14_resolution.py').read_text()
        node = next(n for n in ast.parse(source).body if getattr(n, 'name', '') == 'decode_fine')
        scope = {'np': np}
        exec(compile(ast.Module([node], []), 'v14_resolution.py', 'exec'), scope)
        decode_fine = scope['decode_fine']
        rng = np.random.default_rng(2)
        truth, frames, p, edges = synthetic(4000, rng)
        match = SimpleNamespace(ball=np.empty((4000, 2)), blocks=[(1, 0, 1999), (2, 2000, 3995)])
        frozen = decode_fine(p, edges, frames, match, 'probability')
        ours = np.full(4000, 3, dtype='int8')
        for _, a, b in match.blocks:
            e, g = mbr.prepare_half(p, edges, frames, a, b)
            ours[a:b + 1] = mbr.viterbi(e, g, 1.0)
        np.testing.assert_array_equal(frozen, ours)

    def test_samples_follow_forward_marginal_at_last_frame(self):
        rng = np.random.default_rng(3)
        _, frames, p, edges = synthetic(1500, rng, noise=2.5)
        e, g = mbr.prepare_half(p, edges, frames, 0, 1499)
        la = mbr.forward(e, g, 1.0)
        paths = mbr.sample_paths(la, g, 1.0, 4000, rng)
        empirical = np.bincount(paths[:, -1], minlength=3) / 4000
        np.testing.assert_allclose(empirical, np.exp(la[-1]), atol=.03)

    def test_decode_preserves_layout_and_is_deterministic(self):
        rng = np.random.default_rng(4)
        _, frames, p, edges = synthetic(3000, rng)
        match = SimpleNamespace(ball=np.empty((3100, 2)), blocks=[(1, 0, 1499), (2, 1500, 2995)])
        a = mbr.decode_mbr(p, edges, frames, match, samples=12)
        b = mbr.decode_mbr(p, edges, frames, match, samples=12)
        np.testing.assert_array_equal(a, b)
        self.assertTrue((a[2996:] == 3).all())
        self.assertTrue(set(np.unique(a[:2996])) <= {0, 1, 2})


if __name__ == '__main__':
    unittest.main()
