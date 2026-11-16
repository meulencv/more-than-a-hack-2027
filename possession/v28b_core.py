"""V28b building blocks: object-level decoding of the frozen-ensemble output.

The frame-level Viterbi of possession.v14_resolution.decode_fine is replaced by decoding over a
*segment lattice*: the boundaries of a few Viterbi decodes (permissive to conservative) cut each
half into atoms, and a segmentation is a sequence of H/A/D segments whose ends are atom ends.
On that lattice we can afford what the 25 Hz chain cannot:

* explicit-duration (semi-Markov) Viterbi, with a duration prior per state and context;
* exact sampling of segmentations from the tempered posterior (forward filter, backward sample);
* minimum-Bayes-risk decoding for the ranking metric: the segmentation that maximises the
  expected number of IoU-matched possessions minus a price per predicted possession.

With the default parameters and no duration prior the lattice Viterbi is exactly decode_fine,
because the baseline boundaries are lattice boundaries (tests/test_v28b.py).

Nothing here reads the challenge ground truth. Codes: 0 = H, 1 = A, 2 = D, 3 = not graded.
"""
import numpy as np
from . import v27_core as c

FPS = c.FPS
IOUS = np.array(c.IOUS)
# Decodes whose boundaries define the lattice: baseline, two permissive, two conservative.
LATTICE_CONFIGS = (
    c.DEFAULT,
    dict(c.DEFAULT, c_ha=1.5, c_to_d=1.0, c_from_d=1.0),
    dict(c.DEFAULT, c_ha=2.5, c_to_d=1.5, c_from_d=1.5, gamma=0.5),
    dict(c.DEFAULT, c_ha=8.0, c_to_d=4.0, c_from_d=4.0),
    dict(c.DEFAULT, c_ha=16.0, c_to_d=9.0, c_from_d=4.0, gamma=0.5),
)
# Baseline potentials (= decode_fine) and neutral values of everything V28b adds.
NO_FLOOR = float(-np.log(1e-5))
PARAMS = dict(t_live=0.2, t_team=0.2, t_dead=0.2, b_dead=0.0, floor_team=NO_FLOOR, floor_live=NO_FLOOR,
              c_ha=4.0, c_to_d=2.5, c_from_d=2.5, gamma=0.85, w_after_dead=0.0, w_after_team=0.0, w_dead=0.0,
              lock_dead=0.0)
LOCK = 1e3
# (state, previous state, duration-prior kind); kinds: 0 team spell after a stoppage or at the
# start of a half, 1 team spell after the other team, 2 stoppage.
PAIRS = ((0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 2, 0), (2, 0, 2), (2, 1, 2))
PAIR_STATE = np.array([p[0] for p in PAIRS])
PAIR_PREVIOUS = np.array([p[1] for p in PAIRS])
PAIR_KIND = np.array([p[2] for p in PAIRS])
KINDS = ("after_dead", "after_team", "dead")


# -- lattice ---------------------------------------------------------------------------------

def build_lattice(interpolated, n_frames, configs=LATTICE_CONFIGS, extra=None):
    """Per half: atom boundaries (frames relative to the half) and cumulative log-evidence.

    evidence columns: log(pH + pA), log(pH / (pH + pA)), log(pA / (pH + pA)), log(pD), so that
    t * (col0 + col1) is the baseline emission of H. `extra` (codes arrays) adds boundaries.
    `dead` counts the stoppage frames of the first decode (the baseline) up to each boundary; with
    par["lock_dead"] the decoder must keep exactly those stoppages and only re-decides the team.
    """
    codes = c.decode(interpolated, n_frames, list(configs))
    if extra is not None:
        codes = np.concatenate([codes, np.atleast_2d(extra).astype(np.int8)])
    halves = []
    for first, last, p, e in interpolated:
        window = codes[:, first:last + 1]
        change = np.flatnonzero((window[:, 1:] != window[:, :-1]).any(0)) + 1
        bounds = np.concatenate(([0], change, [window.shape[1]])).astype(np.int64)
        q = np.clip(p, 1e-5, 1)
        live = q[:, 0] + q[:, 1]
        evidence = np.column_stack([np.log(live), np.log(q[:, 0] / live), np.log(q[:, 1] / live), np.log(q[:, 2])])
        edge = e[np.minimum(bounds, len(e) - 1)].astype(np.float64)
        dead = np.concatenate(([0], np.cumsum(window[0] == 2)))[bounds].astype(np.float64)
        halves.append(dict(first=int(first), length=int(window.shape[1]), bounds=bounds, evidence=evidence,
                           edge=edge, dead=dead, cache={}))
    return halves


def cumulative(half, floor_team, floor_live):
    """Cumulative evidence at the lattice boundaries with per-frame log-evidence floored.

    A floor f keeps log(share) >= -f: one frame can then argue against a team by at most f, so
    a short confident burst (a touch the networks learned to call a possession) weighs less.
    f = 11.52 (= -log 1e-5) is the unfloored baseline.
    """
    key = (round(float(floor_team), 6), round(float(floor_live), 6))
    cache = half["cache"]
    if key not in cache:
        if len(cache) > 8:
            cache.clear()
        e = half["evidence"]
        floored = np.column_stack([np.maximum(e[:, 0], -floor_live), np.maximum(e[:, 1], -floor_team),
                                   np.maximum(e[:, 2], -floor_team), np.maximum(e[:, 3], -floor_live)])
        cache[key] = np.concatenate((np.zeros((1, 4)), np.cumsum(floored, 0)))[half["bounds"]]
    return cache[key]


def potentials(half, par, prior=None):
    """Cumulative emissions (N+1, 3), switch-cost multiplier per boundary, cost matrix [prev, new]."""
    cum = cumulative(half, par["floor_team"], par["floor_live"])
    seconds = half["bounds"] / FPS
    emission = np.column_stack([par["t_live"] * cum[:, 0] + par["t_team"] * cum[:, 1],
                                par["t_live"] * cum[:, 0] + par["t_team"] * cum[:, 2],
                                par["t_dead"] * cum[:, 3] + par["b_dead"] * seconds])
    if par.get("lock_dead"):
        emission[:, :2] -= LOCK * half["dead"][:, None]
        emission[:, 2] -= LOCK * (half["bounds"] - half["dead"])
    factor = 1 - par["gamma"] * half["edge"]
    factor[0] = 0.0                                   # the first segment of a half pays nothing
    cost = np.array([[0., par["c_ha"], par["c_to_d"]], [par["c_ha"], 0., par["c_to_d"]],
                     [par["c_from_d"], par["c_from_d"], 0.]])
    weights = np.array([par["w_after_dead"], par["w_after_team"], par["w_dead"]])
    use_prior = prior is not None and bool(np.any(weights != 0))
    return emission, factor, cost, weights, use_prior


def _base(value, emission_row, factor_h, cost):
    """Row h of the DP table A[pair] = V[h, prev] - cost[prev, state] * factor[h] - E[h, state]."""
    return value[PAIR_PREVIOUS] - cost[PAIR_PREVIOUS, PAIR_STATE] * factor_h - emission_row[PAIR_STATE]


def _duration(prior, weights, bounds, i):
    d = bounds[i] - bounds[:i]
    return (weights[:, None] * prior[:, np.minimum(d, prior.shape[1] - 1)])[PAIR_KIND]


def semi_viterbi(half, par, prior=None):
    """Best segmentation of one half: [(start_frame, stop_frame, state)] relative to the half.

    score = sum over segments of emission(state) - cost[prev, state] * (1 - gamma * edge[start])
            + w_kind * prior[kind, duration]          (prior: log pmf over frames, (3, Dmax + 1))
    """
    emission, factor, cost, weights, use_prior = potentials(half, par, prior)
    bounds = half["bounds"]
    n = len(bounds) - 1
    value = np.zeros((n + 1, 3))
    table = np.full((6, n + 1), -np.inf)
    table[:, 0] = 0.0
    table[[0, 2, 5], 0] = -np.inf                     # one start per state, never "after the other team"
    back_h = np.zeros((n + 1, 3), dtype=np.int64)
    back_j = np.zeros((n + 1, 3), dtype=np.int8)
    rows = np.arange(6)
    for i in range(1, n + 1):
        total = table[:, :i] + _duration(prior, weights, bounds, i) if use_prior else table[:, :i]
        h = total.argmax(1)
        best = total[rows, h]
        for k in range(3):
            a, b = 2 * k, 2 * k + 1
            pick = a if best[a] >= best[b] else b
            value[i, k] = best[pick] + emission[i, k]
            back_h[i, k] = h[pick]
            back_j[i, k] = PAIR_PREVIOUS[pick]
        if i < n:
            table[:, i] = _base(value[i], emission[i], factor[i], cost)
    k = int(value[n].argmax())
    i = n
    out = []
    while i > 0:
        h = int(back_h[i, k])
        out.append((int(bounds[h]), int(bounds[i]), k))
        k, i = int(back_j[i, k]), h
    return out[::-1]


def _logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def forward(half, par, prior=None, beta=1.0):
    """Forward log-sums of the tempered segment model (every potential multiplied by beta)."""
    emission, factor, cost, weights, use_prior = potentials(half, par, prior)
    emission, cost, weights = emission * beta, cost * beta, weights * beta
    bounds = half["bounds"]
    n = len(bounds) - 1
    alpha = np.zeros((n + 1, 3))
    table = np.full((6, n + 1), -np.inf)
    table[:, 0] = 0.0
    table[[0, 2, 5], 0] = -np.inf                     # one start per state, never "after the other team"
    for i in range(1, n + 1):
        total = table[:, :i] + _duration(prior, weights, bounds, i) if use_prior else table[:, :i]
        m = total.max(1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)             # a row of -inf stays -inf
        with np.errstate(divide="ignore"):
            part = m[:, 0] + np.log(np.exp(total - m).sum(1))
        for k in range(3):
            alpha[i, k] = np.logaddexp(part[2 * k], part[2 * k + 1]) + emission[i, k]
        if i < n:
            table[:, i] = _base(alpha[i], emission[i], factor[i], cost)
    return dict(alpha=alpha, table=table, emission=emission, weights=weights, use_prior=use_prior, prior=prior)


def sample_segmentations(half, state, n_samples, rng):
    """Exact draws from the tempered posterior over lattice segmentations (backward sampling)."""
    alpha, table, weights, prior, use_prior = state["alpha"], state["table"], state["weights"], state["prior"], state["use_prior"]
    bounds = half["bounds"]
    n = len(bounds) - 1
    final = np.exp(alpha[n] - _logsumexp(alpha[n]))
    out = []
    for _ in range(n_samples):
        k = int(rng.choice(3, p=final))
        i = n
        segs = []
        while i > 0:
            total = table[2 * k:2 * k + 2, :i]
            if use_prior:
                total = total + _duration(prior, weights, bounds, i)[2 * k:2 * k + 2]
            flat = total.ravel()
            w = np.exp(flat - flat.max())
            pick = int(np.searchsorted(np.cumsum(w), rng.random() * w.sum()))
            pick = min(pick, len(flat) - 1)
            pair, h = divmod(pick, i)
            segs.append((int(bounds[h]), int(bounds[i]), k))
            k, i = int(PAIR_PREVIOUS[2 * k + pair]), h
        out.append(segs[::-1])
    return out


# -- minimum Bayes risk for spellF1 ----------------------------------------------------------

def expected_matches(half, samples, tie=0.02):
    """reward[team, h, i]: expected spellF1 hits of predicting the possession bounds[h]..bounds[i].

    For every sampled possession and every lattice candidate of the same team, the number of
    IoU thresholds (0.3..0.7) the pair clears, averaged over samples and thresholds. The
    one-to-one constraint of the scorer is automatic from IoU 0.5 up (disjoint possessions) and
    ignored below it. `tie` adds a small multiple of the expected IoU itself, so that among
    candidates with the same hits the one with the best boundaries wins (the hit count alone is
    flat once every threshold is cleared and would let a possession leak into a stoppage).
    """
    bounds = half["bounds"]
    n = len(bounds) - 1
    reward = np.zeros((2, n + 1, n + 1), dtype=np.float64)
    count = {}
    for segs in samples:
        for a, z, k in segs:
            if k < 2:
                count[(a, z, k)] = count.get((a, z, k), 0) + 1
    slack = 1 / IOUS[0] - 1
    for (a, z, k), times in count.items():
        length = z - a
        h0 = int(np.searchsorted(bounds, a - slack * length, "left"))
        h1 = int(np.searchsorted(bounds, z, "left"))                 # starts strictly before z
        i0 = int(np.searchsorted(bounds, a, "right"))                # ends strictly after a
        i1 = int(np.searchsorted(bounds, z + slack * length, "right"))
        start = bounds[h0:h1][:, None].astype(np.float64)
        stop = bounds[i0:i1][None, :].astype(np.float64)
        inter = np.minimum(stop, z) - np.maximum(start, a)
        union = np.maximum(stop, z) - np.minimum(start, a)
        iou = np.where(stop > start, inter / union, 0.0)
        hits = (iou[:, :, None] >= IOUS[None, None, :] - 1e-12).sum(2)
        reward[k, h0:h1, i0:i1] += times * (hits + tie * len(IOUS) * iou * (hits > 0))
    reward /= len(samples) * len(IOUS)
    mean_spells = sum(count.values()) / len(samples)
    return reward, mean_spells


def consensus(half, reward, price):
    """Segmentation maximising sum over predicted possessions of (reward - price).

    Stoppages are free and fill whatever is not worth a possession. Returns (segments, value,
    total reward).
    """
    bounds = half["bounds"]
    n = len(bounds) - 1
    value = np.zeros((n + 1, 3))
    back_h = np.zeros((n + 1, 3), dtype=np.int64)
    back_j = np.zeros((n + 1, 3), dtype=np.int8)
    other = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    # previous-state maxima per boundary, filled as we go
    best_prev = np.full((3, n + 1), -np.inf)             # best_prev[k, h] = max_{j != k} value[h, j]
    arg_prev = np.zeros((3, n + 1), dtype=np.int8)
    best_prev[:, 0] = 0.0
    arg_prev[:, 0] = [2, 2, 0]
    for i in range(1, n + 1):
        for k in range(3):
            gain = (reward[k, :i, i] - price) if k < 2 else 0.0
            total = best_prev[k, :i] + gain
            h = int(total.argmax()) if k < 2 else int(i - 1 - total[::-1].argmax())   # shortest stoppage on ties
            value[i, k] = total[h]
            back_h[i, k] = h
            back_j[i, k] = arg_prev[k, h]
        for k in range(3):
            a, b = other[k]
            pick = a if value[i, a] >= value[i, b] else b
            best_prev[k, i] = value[i, pick]
            arg_prev[k, i] = pick
    k = int(value[n].argmax())
    total_value = float(value[n, k])
    i = n
    out = []
    while i > 0:
        h = int(back_h[i, k])
        out.append((int(bounds[h]), int(bounds[i]), k))
        k, i = int(back_j[i, k]), h
    out = out[::-1]
    index = {int(b): j for j, b in enumerate(bounds)}
    got = float(sum(reward[k, index[a], index[z]] for a, z, k in out if k < 2))
    return out, total_value, got


def best_consensus(half, reward, mean_spells, price=None, iterations=4):
    """Consensus for a fixed price per possession, or (price=None) for expected spellF1 itself.

    F = 2 E[matches] / (E[true possessions] + predicted possessions) is a ratio, so the maximiser
    of E[matches] - (F / 2) * predicted is found by a few Dinkelbach iterations on the price.
    """
    lam = 0.35 if price is None else price
    for _ in range(iterations if price is None else 1):
        segs, _, got = consensus(half, reward, lam)
        n_pred = sum(1 for _, _, k in segs if k < 2)
        if price is None:
            lam = got / max(1e-9, mean_spells + n_pred)
    return segs, dict(expected_f1=2 * got / max(1e-9, mean_spells + n_pred), sampled_spells=mean_spells,
                      predicted=n_pred, price=lam)


def relative_segments(codes, half):
    first = half["first"]
    return [(a - first, z - first, k) for a, z, k in c.segments(codes, [(0, first, first + half["length"] - 1)])]


# -- whole-match helpers ---------------------------------------------------------------------

def to_codes(per_half, lattice, n_frames):
    codes = np.full(n_frames, 3, dtype=np.int8)
    for half, segs in zip(lattice, per_half):
        for a, z, k in segs:
            codes[half["first"] + a:half["first"] + z] = k
    return codes


def decode(lattice, n_frames, par=PARAMS, prior=None):
    return to_codes([semi_viterbi(half, par, prior) for half in lattice], lattice, n_frames)


def decode_mbr(lattice, n_frames, par=PARAMS, prior=None, beta=1.0, n_samples=200, seed=0, prices=(None,)):
    """Posterior MBR: sample segmentations at inverse temperature beta, return the consensus per price."""
    rng = np.random.default_rng(seed)
    per_price = {price: [] for price in prices}
    infos = {price: [] for price in prices}
    for half in lattice:
        samples = sample_segmentations(half, forward(half, par, prior, beta), n_samples, rng)
        reward, mean_spells = expected_matches(half, samples)
        for price in prices:
            segs, info = best_consensus(half, reward, mean_spells, price)
            per_price[price].append(segs)
            infos[price].append(info)
    return {price: to_codes(per_price[price], lattice, n_frames) for price in prices}, infos


def combine(lattice, n_frames, decodes, price=None):
    """System combination: the decodes of several decoders are the hypotheses; return their consensus.

    Every boundary of every decode must be a lattice boundary (true for lattice decodes and for
    anything that only removes possessions from the baseline).
    """
    per_half, infos = [], []
    for half in lattice:
        samples = [relative_segments(codes, half) for codes in decodes]
        known = set(half["bounds"].tolist())
        if any(a not in known or z not in known for sample in samples for a, z, _ in sample):
            raise ValueError("A hypothesis has a boundary outside the lattice")
        reward, mean_spells = expected_matches(half, samples)
        segs, info = best_consensus(half, reward, mean_spells, price)
        per_half.append(segs)
        infos.append(info)
    return to_codes(per_half, lattice, n_frames), infos


# -- duration prior --------------------------------------------------------------------------

def reference_durations(reference, blocks):
    """Durations in frames by kind, from a reference labelling (0/1/2, -1 or 3 not graded)."""
    codes = np.where((reference >= 0) & (reference <= 2), reference, 3).astype(np.int8)
    out = {0: [], 1: [], 2: []}
    for _, first, last in blocks:
        segs = c.segments(codes, [(0, first, last)])
        for j, (a, z, k) in enumerate(segs):
            if k == 3 or j == 0 or j == len(segs) - 1:       # censored at the ends of a half
                continue
            if k == 2:
                out[2].append(z - a)
            else:
                out[1 if segs[j - 1][2] < 2 else 0].append(z - a)
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def fit_prior(durations, max_frames, bandwidth=0.35):
    """log pmf over the duration in frames, (3, max_frames + 1): Gaussian KDE on log-duration.

    Normalised over 1..max_frames per kind; index 0 is unused (set to the value at 1).
    """
    d = np.arange(1, max_frames + 1, dtype=np.float64)
    grid = np.linspace(0.0, np.log(max_frames), 600)
    out = np.zeros((3, max_frames + 1))
    for kind in range(3):
        x = np.log(np.clip(durations[kind], 1, None))
        density = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bandwidth) ** 2).mean(1) + 1e-12
        log_pmf = np.interp(np.log(d), grid, np.log(density)) - np.log(d)
        log_pmf -= _logsumexp(log_pmf)
        out[kind, 1:] = log_pmf
        out[kind, 0] = log_pmf[0]
    return out
