"""Viterbi decoder with mandatory minimum residence, separately for each half."""
import numpy as np


def decode(emission, minimum_steps=(3, 3, 4), switch_cost=3.0):
    """Return H/A/D integer states maximizing evidence minus transition costs.

    Each semantic state has a chain of age substates. Only the final substate may
    exit, so short opposing touches cannot create arbitrarily short possessions.
    Endpoints may end early because the observation window itself ends there.
    """
    emission = np.asarray(emission, dtype=float)
    if emission.ndim != 2 or emission.shape[1] != 3 or not np.isfinite(emission).all():
        raise ValueError("Expected finite (time, 3) emissions")
    if not len(emission):
        return np.empty(0, dtype=np.int8)
    lengths = np.asarray(minimum_steps, dtype=int)
    if len(lengths) != 3 or (lengths < 1).any():
        raise ValueError("Three positive minimum residence times required")
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    ends = starts + lengths - 1
    semantic = np.repeat(np.arange(3), lengths)
    size = len(semantic)
    transition = np.full((size, size), -np.inf)
    for start, end in zip(starts, ends):
        for age in range(start, end):
            transition[age, age + 1] = 0
        transition[end, end] = 0
    for old in range(3):
        for new in range(3):
            if old != new:
                transition[ends[old], starts[new]] = -switch_cost
    score = np.full(size, -np.inf)
    score[starts] = emission[0]
    back = np.zeros((len(emission), size), dtype=np.int16)
    for t in range(1, len(emission)):
        candidate = score[:, None] + transition
        back[t] = np.argmax(candidate, axis=0)
        score = candidate[back[t], np.arange(size)] + emission[t, semantic]
        score -= score.max()
    states = np.empty(len(emission), dtype=np.int8)
    current = int(np.argmax(score))
    for t in range(len(emission) - 1, -1, -1):
        states[t] = semantic[current]
        current = int(back[t, current])
    return states
