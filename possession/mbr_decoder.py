"""V24: minimum-Bayes-risk (MBR) decoding for the official spellF1.

The frozen decoder (`v14_resolution.decode_fine`) returns the single most probable
label path of a small CRF (Viterbi / MAP).  MAP maximises *sequence* likelihood, not
the ranking metric: spellF1 is a non-decomposable, object-level F1 over possessions
matched by temporal IoU.  Segmental MBR decoding (speech recognition: Goel & Byrne
2000; Kumar & Byrne 2002) instead picks the labelling with the highest *expected*
metric under the model posterior.

Here the posterior is the very same CRF that `decode_fine` maximises:

    score(s) = sum_t u * log p_t(s_t) - sum_t C[s_{t-1}, s_t] * (1 - 0.85 * edge_t)

turned into a distribution  P(s) ~ exp(beta * score(s)).  beta=1 is far too diffuse
(the 25 Hz emissions are interpolated, hence strongly correlated, and divided by 5):
its samples have ~10x more spells than the truth.  beta in [3, 8] gives samples with a
realistic number of spells; it is the one hyperparameter to select by cross-validation.  We

1. run forward filtering and draw M label paths by backward sampling (FFBS);
2. build candidates: Viterbi at several switch-cost scales (scale 1 == the frozen
   decoder, so MBR can always fall back to it);
3. pick the candidate with the highest mean spellF1 against the M samples, used as
   pseudo ground truth;
4. optionally refine it with local edits (delete short spells, insert short spells
   supported by many samples, move boundaries to the sample median), each accepted
   only if it raises the expected spellF1.

Nothing here reads labels: it only consumes model probabilities, so the same code
can decode the challenge match.  Pure numpy.
"""
import numpy as np

TRANSITION = np.array([[0., 4., 2.5], [4., 0., 2.5], [2.5, 2.5, 0.]])
UNARY = 1 / 5
EDGE = .85
THRESHOLDS = (0.7, 0.6, 0.5, 0.4, 0.3)  # descending: one greedy pass serves all five
DEFAULTS = dict(beta=4.0, samples=32, scales=(0.6, 0.8, 1.0, 1.25, 1.6), refine=True,
                passes=2, max_delete=50, max_insert=75, min_support=0.2,
                shift_window=25, seed=0)


# ----------------------------------------------------------------------------- inputs
def prepare_half(probabilities, boundaries, frames, first, last):
    """Same 25 Hz interpolation and energies as decode_fine(space='probability')."""
    selected = (frames >= first) & (frames <= last)
    f = frames[selected]
    if not len(f):
        raise ValueError('No emission in half')
    grid = np.arange(first, last + 1)
    source = probabilities[selected]
    interpolated = np.column_stack([np.interp(grid, f, source[:, j]) for j in range(3)])
    emission = np.log(np.clip(interpolated, 1e-5, 1)) * UNARY
    gate = 1 - EDGE * np.interp(grid, f, boundaries[selected])
    return emission, gate


def viterbi(emission, gate, scale=1.0):
    """MAP path; scale=1 reproduces decode_fine bit for bit."""
    n = len(emission)
    cost = scale * TRANSITION
    back = np.zeros((n, 3), dtype='int8')
    score = emission[0].copy()
    idx = np.arange(3)
    for t in range(1, n):
        previous = score[:, None] - cost * gate[t]
        back[t] = previous.argmax(0)
        score = previous[back[t], idx] + emission[t]
        score -= score.max()
    states = np.empty(n, dtype='int8')
    current = int(score.argmax())
    for t in range(n - 1, -1, -1):
        states[t] = current
        current = int(back[t, current])
    return states


def _lse0(m):
    top = m.max(0)
    return top + np.log(np.exp(m - top).sum(0))


def forward(emission, gate, beta=1.0):
    """Normalised log forward messages of P(s) ~ exp(beta * score(s))."""
    n = len(emission)
    out = np.empty((n, 3))
    cost = beta * TRANSITION
    em = beta * emission
    a = em[0] - _lse0(em[0][:, None])
    out[0] = a
    for t in range(1, n):
        a = _lse0(a[:, None] - cost * gate[t]) + em[t]
        a = a - _lse0(a[:, None])
        out[t] = a
    return out


def sample_paths(log_alpha, gate, beta, count, rng):
    """Backward sampling (FFBS) of `count` exact posterior paths, Gumbel-max."""
    n = len(log_alpha)
    cost = beta * TRANSITION
    paths = np.empty((count, n), dtype='int8')
    s = np.argmax(log_alpha[-1][None, :] + rng.gumbel(size=(count, 3)), 1)
    paths[:, -1] = s
    for t in range(n - 2, -1, -1):
        logits = log_alpha[t][None, :] - cost[:, s].T * gate[t + 1]
        s = np.argmax(logits + rng.gumbel(size=(count, 3)), 1)
        paths[:, t] = s
    return paths


# ----------------------------------------------------------------------------- metric
def spells(states):
    """(starts, stops, teams) of maximal H/A runs in a gap-free state array."""
    n = len(states)
    breaks = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [n]))
    labels = states[starts]
    keep = labels < 2
    return starts[keep], stops[keep], labels[keep]


def true_positives(gt, pred):
    """TP at each of THRESHOLDS, identical to the official greedy one-to-one match."""
    gs, ge, gk = gt
    ps, pe, pk = pred
    result = np.zeros(len(THRESHOLDS), dtype=int)
    if not len(gs) or not len(ps):
        return result
    lo = np.searchsorted(ge, ps, 'right')
    hi = np.searchsorted(gs, pe, 'left')
    count = np.maximum(hi - lo, 0)
    total = int(count.sum())
    if not total:
        return result
    j = np.repeat(np.arange(len(ps)), count)
    offsets = np.repeat(np.cumsum(count) - count, count)
    i = lo[j] + np.arange(total) - offsets
    same = gk[i] == pk[j]
    i, j = i[same], j[same]
    inter = np.minimum(ge[i], pe[j]) - np.maximum(gs[i], ps[j])
    union = np.maximum(ge[i], pe[j]) - np.minimum(gs[i], ps[j])
    iou = inter / union.astype(float)
    use = iou >= THRESHOLDS[-1]
    i, j, iou = i[use], j[use], iou[use]
    order = np.lexsort((j, i, -iou))
    used_g, used_p = set(), set()
    tp, k = 0, 0
    for index in order:
        value = iou[index]
        while k < len(THRESHOLDS) and value < THRESHOLDS[k]:
            result[k] = tp
            k += 1
        a, b = int(i[index]), int(j[index])
        if a in used_g or b in used_p:
            continue
        used_g.add(a)
        used_p.add(b)
        tp += 1
    result[k:] = tp
    return result


def spell_f1(gt_states, pred_states):
    gt, pred = spells(gt_states), spells(pred_states)
    tp = true_positives(gt, pred)
    denominator = len(gt[0]) + len(pred[0])
    return float(np.mean(2 * tp / denominator)) if denominator else 0.0


class Risk:
    """Expected spellF1 of a labelling against fixed posterior samples."""

    def __init__(self, paths):
        self.samples = [spells(p) for p in paths]
        self.sizes = np.array([len(s[0]) for s in self.samples])

    def __call__(self, states):
        pred = spells(states)
        n = len(pred[0])
        values = []
        for gt, size in zip(self.samples, self.sizes):
            tp = true_positives(gt, pred)
            values.append(np.mean(2 * tp / (size + n)) if size + n else 0.0)
        return float(np.mean(values))


# ----------------------------------------------------------------------------- refinement
def _runs(states):
    breaks = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(states)]))
    return starts, stops, states[starts]


def _deletions(states, max_length):
    starts, stops, labels = _runs(states)
    for k in range(len(starts)):
        a, b, label = starts[k], stops[k], labels[k]
        if label > 1 or b - a >= max_length:
            continue
        fills = set()
        if k > 0:
            fills.add(int(labels[k - 1]))
        if k + 1 < len(starts):
            fills.add(int(labels[k + 1]))
        for fill in fills:
            if fill != label:
                yield int(a), int(b), fill


def _insertions(states, paths, max_length, min_support):
    votes = {}
    for path in paths:
        s, e, k = spells(path)
        short = (e - s) <= max_length
        for a, b, team in zip(s[short], e[short], k[short]):
            if np.mean(states[a:b] == team) > .5:
                continue
            key = (int(a) // 4, int(b) // 4, int(team))
            votes.setdefault(key, []).append((int(a), int(b)))
    need = min_support * len(paths)
    for (_, _, team), spans in votes.items():
        if len(spans) >= need:
            spans = np.array(spans)
            yield int(np.median(spans[:, 0])), int(np.median(spans[:, 1])), team


def _median_shifts(states, paths, window):
    """Move every boundary to the median position of the same transition in the samples."""
    out = states.copy()
    t_cur = np.flatnonzero(states[1:] != states[:-1]) + 1
    if not len(t_cur):
        return out
    sample_t = []
    for path in paths:
        t = np.flatnonzero(path[1:] != path[:-1]) + 1
        sample_t.append((t, path[t - 1].astype(int) * 3 + path[t]))
    kinds = states[t_cur - 1].astype(int) * 3 + states[t_cur]
    target = t_cur.copy()
    for n, (t, kind) in enumerate(zip(t_cur, kinds)):
        found = []
        for times, types in sample_t:
            lo, hi = np.searchsorted(times, [t - window, t + window + 1])
            cand = times[lo:hi][types[lo:hi] == kind]
            if len(cand):
                found.append(cand[np.argmin(np.abs(cand - t))])
        if len(found) >= .5 * len(paths):
            target[n] = int(np.median(found))
    bounds = np.concatenate(([0], t_cur, [len(states)]))
    for n in range(len(t_cur)):
        new = int(np.clip(target[n], bounds[n] + 1, bounds[n + 2] - 1))
        old = t_cur[n]
        if new < old:
            out[new:old] = states[old]
        elif new > old:
            out[old:new] = states[old - 1]
    return out


def refine(states, paths, risk, cfg):
    current = states.copy()
    value = risk(current)
    for _ in range(cfg['passes']):
        shifted = _median_shifts(current, paths, cfg['shift_window'])
        shifted_value = risk(shifted)
        if shifted_value > value:
            current, value = shifted, shifted_value
        edits = list(_deletions(current, cfg['max_delete']))
        edits += list(_insertions(current, paths, cfg['max_insert'], cfg['min_support']))
        scored = []
        for a, b, label in edits:
            trial = current.copy()
            trial[a:b] = label
            gain = risk(trial) - value
            if gain > 1e-9:
                scored.append((gain, a, b, label))
        if not scored:
            break
        scored.sort(reverse=True)
        taken, trial = [], current.copy()
        for gain, a, b, label in scored:
            if any(a < y + 2 and x < b + 2 for x, y in taken):
                continue  # neighbouring edits interact; keep them for the next pass
            trial[a:b] = label
            taken.append((a, b))
        trial_value = risk(trial)
        if trial_value <= value:
            gain, a, b, label = scored[0]
            trial = current.copy()
            trial[a:b] = label
            trial_value = risk(trial)
        current, value = trial, trial_value
    return current, value


# ----------------------------------------------------------------------------- decoder
def decode_half(emission, gate, cfg, rng, report=None):
    log_alpha = forward(emission, gate, cfg['beta'])
    paths = sample_paths(log_alpha, gate, cfg['beta'], cfg['samples'], rng)
    risk = Risk(paths)
    candidates = [(s, viterbi(emission, gate, s)) for s in cfg['scales']]
    values = [risk(c) for _, c in candidates]
    best = int(np.argmax(values))
    states, value = candidates[best][1], values[best]
    if cfg['refine']:
        states, value = refine(states, paths, risk, cfg)
    if report is not None:
        report.append(dict(scale=float(candidates[best][0]), expected_map=float(values[list(cfg['scales']).index(1.0)])
                           if 1.0 in cfg['scales'] else None, expected_final=float(value),
                           sample_spells=float(risk.sizes.mean()), final_spells=int(len(spells(states)[0]))))
    return states


def decode_mbr(probabilities, boundaries, frames, match, report=None, **options):
    """Drop-in replacement for v14_resolution.decode_fine(..., 'probability')."""
    cfg = dict(DEFAULTS, **options)
    codes = np.full(len(match.ball), 3, dtype='int8')
    for index, (_, first, last) in enumerate(match.blocks):
        emission, gate = prepare_half(probabilities, boundaries, frames, first, last)
        rng = np.random.default_rng([cfg['seed'], index])
        codes[first:last + 1] = decode_half(emission, gate, cfg, rng, report)
    return codes
