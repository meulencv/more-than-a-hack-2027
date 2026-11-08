"""Sanity check of MBR decoding on synthetic posteriors (no match data involved).

Only shows that the machinery behaves; real evidence must come from
`python -m possession.v24_mbr` on the out-of-fold external predictions.
"""
from types import SimpleNamespace
import numpy as np
from possession import mbr_decoder as mbr


def make(n, rng, step=5):
    truth, t, label = np.empty(n, dtype='int8'), 0, 0
    while t < n:
        r = rng.random()
        length = rng.integers(5, 25) if r < .3 else rng.integers(25, 75) if r < .6 else rng.integers(75, 500)
        truth[t:t + length] = label
        t += int(length)
        label = int(rng.choice([1 - label, 2], p=[.75, .25])) if label < 2 else int(rng.integers(0, 2))
    frames = np.arange(0, n, step)
    hidden = np.zeros(len(frames), bool)  # occlusions: ~35 % of samples in long blocks
    k = 0
    while k < len(frames):
        span = int(rng.integers(5, 60))
        if rng.random() < .35:
            hidden[k:k + span] = True
        k += span
    signal = np.where(hidden, .6, 2.2)[:, None]
    noise = rng.normal(0, 1, (len(frames), 3))
    noise = .6 * noise + .4 * np.roll(noise, 1, 0)  # temporally correlated errors
    logits = np.eye(3)[truth[frames]] * signal + noise
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    change = np.r_[0, (truth[frames][1:] != truth[frames][:-1])].astype(float)
    edges = np.clip(np.convolve(change, [.3, .8, .3], 'same') + rng.normal(0, .1, len(frames)), 0, 1)
    edges[hidden] *= .3
    return truth, frames, p, edges


def main():
    configs = {'viterbi (congelado)': None,
               'MBR solo candidatos b=4': dict(beta=4.0, refine=False),
               'MBR + refinado b=2': dict(beta=2.0),
               'MBR + refinado b=4': dict(beta=4.0),
               'MBR + refinado b=8': dict(beta=8.0)}
    scores = {k: [] for k in configs}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n = 20000
        truth, frames, p, edges = make(n, rng)
        match = SimpleNamespace(ball=np.empty((n, 2)), blocks=[(1, 0, n - 1)])
        for name, cfg in configs.items():
            if cfg is None:
                e, g = mbr.prepare_half(p, edges, frames, 0, n - 1)
                codes = mbr.viterbi(e, g, 1.0)
            else:
                codes = mbr.decode_mbr(p, edges, frames, match, **cfg)
            scores[name].append(mbr.spell_f1(truth, codes))
    for name, v in scores.items():
        print(f'{name:28s} spellF1 {np.mean(v):.4f}  por semilla {np.round(v, 4).tolist()}')


if __name__ == '__main__':
    main()
