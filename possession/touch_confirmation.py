"""Experimental temporal confirmation; observations only, no reference labels.

Dead-ball regions and half boundaries are immutable. A waiting margin confirms a
change retrospectively; it is never a new definition of possession duration.
"""
import math
import numpy as np

FPS = 25


def runs(labels):
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(cuts[:-1], cuts[1:]) if a < b]


def dense_evidence(probabilities, edges, frames, record):
    """Interpolate model outputs within each half; never invent ball visibility."""
    n = len(record['ball_seen'])
    p = np.zeros((n, 3)); e = np.zeros(n); control = np.zeros((n, 2))
    observed = np.zeros(n, dtype=bool)
    columns = {name: i for i, name in enumerate(record['columns'])}
    for _, first, last in record['blocks']:
        use = (frames >= first) & (frames <= last); f = frames[use]
        if not len(f): raise ValueError('Half without probabilities')
        grid = np.arange(first, last + 1)
        p[grid] = np.column_stack([np.interp(grid, f, probabilities[use, j]) for j in range(3)])
        e[grid] = np.interp(grid, f, edges[use])
        # Zero-order hold of valid physical measurements; gated by actual visibility.
        left = np.maximum(0, np.searchsorted(f, grid, side='right') - 1)
        for team, name in enumerate(['control_h', 'control_a']):
            source = record['direct'][use, columns[name]]
            control[grid, team] = np.nan_to_num(source[left], nan=0)
        observed[grid] = np.asarray(record['ball_seen'])[grid].astype(bool)
    return dict(p=p, edge=e, control=control, observed=observed)


def control_support(ev, team):
    c = ev['control']
    return ev['observed'] & (c[:, team] >= .5) & (c[:, team] - c[:, 1-team] >= .2)


def confirm(base, ev, blocks, wait_s, adaptive=False):
    """Confirm sustained opposing evidence and backdate to its first frame."""
    out = base.copy(); p = ev['p']; support = [control_support(ev, k) for k in [0, 1]]
    for _, first, last in blocks:
        for a, b, alive in runs((base[first:last+1] < 2).astype('int8')):
            if not alive: continue
            a += first; b += first; state = int(base[a]); start = -1
            for t in range(a, b):
                other = 1-state
                ratio = p[t, other] / max(1e-9, p[t, :2].sum())
                if ratio >= .6 and p[t, other] >= .35:
                    if start < 0: start = t
                    duration = wait_s
                    if adaptive:
                        duration = .12 if support[other][t] else wait_s if ev['observed'][t] else .8
                    ready = t-start+1 >= math.ceil(duration * FPS)
                    # No absolute duration floor: overwhelming model evidence can confirm immediately.
                    if ready or p[t, other] >= .95:
                        out[start:t+1] = other; state = other; start = -1
                else:
                    start = -1
                out[t] = state
    return out


def merge_interruptions(base, ev, blocks, maximum_s, weak_only):
    """One simultaneous pass over H-A-H interruptions. Never merge through D."""
    out = base.copy()
    for _, first, last in blocks:
        ss = runs(base[first:last+1])
        for i in range(1, len(ss)-1):
            a, b, middle = ss[i]; left = ss[i-1][2]; right = ss[i+1][2]
            if middle >= 2 or left >= 2 or left != right or middle == left: continue
            if b-a > round(maximum_s * FPS): continue
            a += first; b += first
            if weak_only:
                # Missing observations cannot be used as evidence that control never happened.
                if ev['observed'][a:b].mean() < .8: continue
                if ev['p'][a:b, middle].mean() >= .65: continue
                if ev['p'][a:b, middle].max() >= .9: continue
                if control_support(ev, middle)[a:b].any(): continue
            out[a:b] = left
    return out


def recover_interruptions(base, ev, blocks, maximum_s):
    """Recover short opposing pulses inside a baseline possession when supported."""
    out = base.copy(); p = ev['p']; edge = ev['edge']
    support = [control_support(ev, k) for k in [0, 1]]
    for _, first, last in blocks:
        for a, b, state in runs(base[first:last+1]):
            if state >= 2: continue
            a += first; b += first; other = 1-state
            ratio = p[a:b, other] / np.maximum(p[a:b, :2].sum(1), 1e-9)
            possible = (ratio >= .6) & (p[a:b, other] >= .4)
            for lo, hi, present in runs(possible.astype('int8')):
                if not present or hi-lo < 3 or hi-lo > round(maximum_s*FPS): continue
                lo += a; hi += a
                # Same-team anchors on both sides, within this original possession.
                if lo-a < 3 or b-hi < 3: continue
                physical = support[other][lo:hi].sum() >= 3
                confident = p[lo:hi, other].max() >= .9
                radius = 5
                limits = edge[max(a, lo-radius):min(b, lo+radius+1)].max() >= .3 and edge[max(a, hi-radius):min(b, hi+radius+1)].max() >= .3
                if confident or (physical and limits): out[lo:hi] = other
    return out


def apply_policy(base, ev, blocks, spec):
    kind = spec['kind']
    if kind == 'baseline': return base.copy()
    if kind == 'confirm':
        return confirm(base, ev, blocks, spec['seconds'], spec.get('adaptive', False))
    if kind == 'merge':
        return merge_interruptions(base, ev, blocks, spec['seconds'], spec['weak_only'])
    if kind == 'recover': return recover_interruptions(base, ev, blocks, spec['seconds'])
    raise ValueError('Unknown temporal policy')
