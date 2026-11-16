import itertools
import unittest
from types import SimpleNamespace
import numpy as np
from possession import v27_core as c
from possession import v28b_core as b
from possession.v14_resolution import decode_fine


def synthetic(seed=0, n=1200):
    rng = np.random.default_rng(seed)
    frames = np.arange(0, n, 5)
    logits = np.cumsum(rng.normal(0, 0.6, (len(frames), 3)), axis=0)
    logits -= logits.mean(1, keepdims=True)
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return p.astype("float32"), rng.random(len(frames)).astype("float32"), frames, [(1, 0, 499), (2, 600, n - 5)]


def lattice_of(seed=0, n=1200):
    p, e, frames, blocks = synthetic(seed, n)
    return b.build_lattice(c.interpolate(p, e, frames, blocks), n), (p, e, frames, blocks)


def tiny_half(seed, atoms=6):
    """A half with a handful of atoms so that every segmentation can be enumerated."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(3, 30, atoms)
    bounds = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    p = rng.dirichlet(np.ones(3), int(bounds[-1]))
    live = p[:, 0] + p[:, 1]
    evidence = np.column_stack([np.log(live), np.log(p[:, 0] / live), np.log(p[:, 1] / live), np.log(p[:, 2])])
    return dict(first=0, length=int(bounds[-1]), bounds=bounds, evidence=evidence, edge=rng.random(atoms + 1),
                dead=np.zeros(atoms + 1), cache={})


def enumerate_segmentations(half, par, prior):
    """Every lattice segmentation with its score, by brute force."""
    emission, factor, cost, weights, use_prior = b.potentials(half, par, prior)
    bounds = half["bounds"]
    n = len(bounds) - 1
    out = []
    for cuts in range(2 ** (n - 1)):
        edges = [0] + [i for i in range(1, n) if cuts >> (i - 1) & 1] + [n]
        for states in itertools.product(range(3), repeat=len(edges) - 1):
            if any(x == y for x, y in zip(states[:-1], states[1:])):
                continue
            score = 0.0
            for j, k in enumerate(states):
                h, i = edges[j], edges[j + 1]
                previous = states[j - 1] if j else None
                score += emission[i, k] - emission[h, k]
                if previous is not None:
                    score -= cost[previous, k] * factor[h]
                if use_prior:
                    kind = 2 if k == 2 else (1 if previous is not None and previous < 2 else 0)
                    score += weights[kind] * prior[kind, min(bounds[i] - bounds[h], prior.shape[1] - 1)]
            out.append((score, [(int(bounds[edges[j]]), int(bounds[edges[j + 1]]), k) for j, k in enumerate(states)]))
    return out


RICH = dict(b.PARAMS, t_team=0.3, floor_team=1.2, floor_live=2.0, b_dead=0.3, c_ha=6.0, c_to_d=1.5, c_from_d=3.0,
            gamma=0.4, w_after_dead=0.7, w_after_team=1.3, w_dead=0.5)


def toy_prior(max_frames=400):
    return b.fit_prior({0: np.array([40., 90., 200.]), 1: np.array([20., 60., 150.]), 2: np.array([50., 120.])}, max_frames)


class V28bDecoder(unittest.TestCase):
    def test_default_lattice_decoder_is_decode_fine(self):
        for seed in range(3):
            lattice, (p, e, frames, blocks) = lattice_of(seed)
            match = SimpleNamespace(ball=np.empty((1200, 2)), blocks=blocks)
            self.assertTrue(np.array_equal(decode_fine(p, e, frames, match, "probability"), b.decode(lattice, 1200)))

    def test_semi_markov_viterbi_is_optimal(self):
        prior = toy_prior()
        for seed in range(4):
            half = tiny_half(seed)
            for par in (b.PARAMS, RICH):
                best = max(enumerate_segmentations(half, par, prior), key=lambda item: item[0])
                self.assertEqual(b.semi_viterbi(half, par, prior), best[1])

    def test_forward_is_the_log_partition_and_samples_follow_the_posterior(self):
        prior = toy_prior()
        half = tiny_half(5, atoms=5)
        everything = enumerate_segmentations(half, RICH, prior)
        beta = 0.7
        scores = np.array([s for s, _ in everything]) * beta
        log_z = scores.max() + np.log(np.exp(scores - scores.max()).sum())
        state = b.forward(half, RICH, prior, beta)
        alpha = state["alpha"][-1]
        self.assertAlmostEqual(alpha.max() + np.log(np.exp(alpha - alpha.max()).sum()), log_z, places=9)
        samples = b.sample_segmentations(half, state, 4000, np.random.default_rng(0))
        top = int(scores.argmax())
        share = np.mean([s == everything[top][1] for s in samples])
        self.assertAlmostEqual(share, float(np.exp(scores[top] - log_z)), delta=0.03)

    def test_locked_stoppages_are_kept(self):
        lattice, _ = lattice_of(3)
        base = b.decode(lattice, 1200)
        self.assertTrue(np.array_equal(b.decode(lattice, 1200, dict(b.PARAMS, lock_dead=1.0)), base))
        team = b.decode(lattice, 1200, dict(RICH, lock_dead=1.0, c_ha=20.0), toy_prior(600))
        self.assertTrue(np.array_equal(team == 2, base == 2))

    def test_evidence_floor_bounds_one_frame(self):
        lattice, _ = lattice_of(1)
        half = lattice[0]
        floored = b.cumulative(half, 0.5, 0.5)
        frames = np.diff(half["bounds"])[:, None]
        self.assertTrue((np.diff(floored, axis=0) >= -0.5 * frames - 1e-9).all())
        self.assertTrue(np.allclose(b.cumulative(half, b.NO_FLOOR, b.NO_FLOOR)[-1], half["evidence"].sum(0)))

    def test_prior_is_a_distribution(self):
        prior = toy_prior()
        self.assertTrue(np.allclose(np.exp(prior[:, 1:]).sum(1), 1.0))
        d = b.reference_durations(np.array([0] * 5 + [1] * 7 + [2] * 4 + [0] * 6 + [1] * 3 + [2] * 2), [(1, 0, 26)])
        self.assertEqual({k: v.tolist() for k, v in d.items()}, {0: [6.0], 1: [7.0, 3.0], 2: [4.0]})


class V28bConsensus(unittest.TestCase):
    def half(self):
        bounds = np.array([0, 100, 120, 300, 400, 420, 600], dtype=np.int64)
        return dict(first=0, length=600, bounds=bounds)

    def test_consensus_of_one_hypothesis_is_that_hypothesis(self):
        half = self.half()
        truth = [(0, 100, 0), (100, 300, 1), (300, 400, 2), (400, 600, 0)]
        reward, mean_spells = b.expected_matches(half, [truth])
        segs, info = b.best_consensus(half, reward, mean_spells)
        self.assertEqual(segs, truth)
        self.assertAlmostEqual(info["expected_f1"], 1.0, delta=0.03)       # the tie-break adds at most 2%

    def test_minority_island_is_dropped_and_majority_island_is_kept(self):
        half = self.half()
        plain = [(0, 300, 0), (300, 400, 2), (400, 600, 1)]
        island = [(0, 100, 0), (100, 120, 1), (120, 300, 0), (300, 400, 2), (400, 600, 1)]
        for votes, expected in ((1, plain), (4, island)):
            reward, mean_spells = b.expected_matches(half, [island] * votes + [plain] * (5 - votes))
            self.assertEqual(b.best_consensus(half, reward, mean_spells)[0], expected)

    def test_combine_rejects_boundaries_outside_the_lattice_and_reproduces_agreement(self):
        lattice, _ = lattice_of(2)
        base = b.decode(lattice, 1200)
        other = b.decode(lattice, 1200, dict(b.PARAMS, c_ha=16.0))
        self.assertTrue(np.array_equal(b.combine(lattice, 1200, [base, base, base])[0], base))
        merged = b.combine(lattice, 1200, [base, other, other])[0]
        self.assertTrue(np.array_equal(merged == 3, base == 3))
        shifted = base.copy()
        first = lattice[0]["first"]
        inner = [x for x in range(1, lattice[0]["length"]) if x not in set(lattice[0]["bounds"].tolist())][0]
        shifted[first:first + inner] = 0
        shifted[first + inner:first + inner + 1] = 1 if shifted[first + inner + 1] != 1 else 2
        with self.assertRaises(ValueError):
            b.combine(lattice, 1200, [shifted])

    def test_reward_counts_iou_thresholds(self):
        half = self.half()
        reward, mean_spells = b.expected_matches(half, [[(0, 300, 0), (300, 600, 2)]], tie=0.0)
        self.assertEqual(mean_spells, 1.0)
        self.assertAlmostEqual(reward[0, 0, 3], 1.0)            # the possession itself: all five thresholds
        self.assertAlmostEqual(reward[0, 0, 2], 2 / 5)          # 0..120 of 0..300: IoU 0.4 clears 0.3 and 0.4
        self.assertAlmostEqual(reward[0, 0, 4], 1.0)            # 0..400: IoU 0.75 clears every threshold
        self.assertAlmostEqual(reward[1].sum(), 0.0)


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


class V28bPipeline(unittest.TestCase):
    def model(self, configs, gain):
        return dict(family="duration", configs=configs, prior=toy_prior(600), verifier=Constant(gain), features=c.FEATURES,
                    max_frames=8 * c.FPS, threshold=0.0, lattice_configs=list(b.LATTICE_CONFIGS))

    def test_object_decoder(self):
        from possession import v28b_decoder as d
        p, e, frames, blocks = synthetic(6)
        match = SimpleNamespace(ball=np.empty((1200, 2)), blocks=blocks)
        baseline = decode_fine(p, e, frames, match, "probability")
        after, before, report = d.object_decoder(p, e, frames, blocks, 1200, verify=False, model=self.model([dict(b.PARAMS)], -1.0))
        self.assertTrue(np.array_equal(before, baseline))
        self.assertTrue(np.array_equal(after, baseline))          # one baseline decoder, no verifier: nothing changes
        self.assertEqual(report["graded_frames_changed"], 0)
        after, before, report = d.object_decoder(p, e, frames, blocks, 1200, model=self.model([dict(b.PARAMS), RICH, dict(RICH, c_ha=12.0)], -1.0))
        self.assertTrue(np.array_equal(after == 3, baseline == 3))
        self.assertEqual(report["verifier_removals"], 0)
        self.assertLessEqual(report["after"]["spells"], report["before"]["spells"])
        again = d.object_decoder(p, e, frames, blocks, 1200, model=self.model([dict(b.PARAMS), RICH, dict(RICH, c_ha=12.0)], -1.0))[0]
        self.assertTrue(np.array_equal(after, again))             # deterministic
        removed = d.object_decoder(p, e, frames, blocks, 1200, model=self.model([dict(b.PARAMS)], 1.0))[2]
        self.assertGreater(removed["verifier_removals"], 0)       # a verifier that always says yes removes short possessions


if __name__ == "__main__":
    unittest.main()
