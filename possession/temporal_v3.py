"""v3 segment decoder: Viterbi over H/A/D with minimum durations and typed switch costs.

Each semantic state is a chain of age sub-states; only the last one can leave, so a state lasts
at least `min_steps` samples. Separate costs for a direct team change (H<->A, i.e. a turnover in
open play), live->dead and dead->live let the decoder treat a turnover as a rarer, more
expensive event than a stoppage. All parameters are selected on external games only.
"""
import numpy as np


def decode(log_emission, min_steps, cost_turnover, cost_to_dead, cost_to_live):
    e = np.asarray(log_emission, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != 3 or not np.isfinite(e).all():
        raise ValueError("Expected finite (time, 3) log emissions")
    if not len(e):
        return np.empty(0, dtype=np.int8)
    lengths = np.asarray(min_steps, dtype=int)
    if len(lengths) != 3 or (lengths < 1).any():
        raise ValueError("Three positive minimum durations required")
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    ends = starts + lengths - 1
    semantic = np.repeat(np.arange(3), lengths)
    size = len(semantic)
    trans = np.full((size, size), -np.inf)
    for s, t in zip(starts, ends):
        for age in range(s, t):
            trans[age, age + 1] = 0.0
        trans[t, t] = 0.0
    cost = np.array([[0, cost_turnover, cost_to_dead],
                     [cost_turnover, 0, cost_to_dead],
                     [cost_to_live, cost_to_live, 0]], dtype=float)
    for a in range(3):
        for b in range(3):
            if a != b:
                trans[ends[a], starts[b]] = -cost[a, b]
    n = len(e)
    score = np.full(size, -np.inf)
    score[starts] = e[0]
    back = np.zeros((n, size), dtype=np.int16)
    em = e[:, semantic]
    columns = np.arange(size)
    for t in range(1, n):
        cand = score[:, None] + trans
        best = np.argmax(cand, axis=0)
        back[t] = best
        score = cand[best, columns] + em[t]
        score -= score.max()
    states = np.empty(n, dtype=np.int8)
    cur = int(np.argmax(score))
    for t in range(n - 1, -1, -1):
        states[t] = semantic[cur]
        cur = int(back[t, cur])
    return states
