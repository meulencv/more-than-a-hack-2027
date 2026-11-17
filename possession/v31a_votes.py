"""V31a: disagreement between the four frozen networks as evidence (verifier and second stage).

Everything fitted so far on the frozen outputs sees only the ENSEMBLE MEAN. The four networks
(two tabular, two trajectory) are different readings of the same play; where they disagree the
mean hides it. This module gives the per-network tracks (possession.v31a_members) to

    votes     the verifier: 8 extra features of a candidate removal (how many networks prefer
              the possession over its replacement, the spread of their log-ratios, tabular vs
              trajectory), on top of the unchanged V27 feature set;
    members   the second stage: per-network P(D) / P(H) - P(A) at a few lags and their spread,
              on top of the possession.v30b_stack features.

Still network outputs only: no ball availability, no NaN pattern, no tracking feature.

    .venv/bin/python -m possession.v31a_votes verifier                   # frozen decoder + verifier with votes
    .venv/bin/python -m possession.v31a_votes stack --target M           # second stage with member features

Nested leave-one-game-out exactly as possession.v31a_lab; the label-blind ball conditions are
scored with the same fitted objects and never fitted on. No challenge label is read.
"""
import argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from . import v27_core as c
from . import v27_lab as lab27
from . import v28a_core as k28
from . import v30b_stack as s
from . import v31a_lab as L
from . import v31a_members as members

ROOT = L.ROOT
GAMES = L.GAMES
VOTES = ["votes", "ratio_min", "ratio_max", "ratio_std", "ratio_tabular", "ratio_trajectory", "own_min_network", "own_max_network"]
MEMBER_LAGS = (-5, -2, 0, 2, 5)


# -- data ------------------------------------------------------------------------------------

def load(game, mode):
    k = L.load(game, mode, "frozen")
    saved = np.load(members.path(game, mode))
    keep = np.isin(saved["frames"], k.frames)
    if not np.array_equal(saved["frames"][keep], k.frames):
        raise ValueError("Different timelines")
    k.members = saved["members"][:, keep].astype(np.float64)
    if np.abs(k.members.mean(0) - k.p0).max() > 1e-4:
        raise ValueError("Member mean is not the cached ensemble")
    even, odd = [], []
    for _, first, last in k.blocks:
        use = (k.frames >= first) & (k.frames <= last)
        e, o = member_half_features(k.members[:, use])
        even.append(e)
        odd.append(o)
    k.even_m, k.odd_m = np.concatenate(even), np.concatenate(odd)
    return k


def decoded(k, p):
    d = L.decoded(k, p)
    d.frames5, d.members = k.frames, k.members
    return d


# -- verifier with votes ---------------------------------------------------------------------

def vote_features(d, segs, cands):
    rows = np.zeros((len(cands), len(VOTES)))
    for row, (i, _, new) in enumerate(cands):
        a, z, k = segs[i]
        lo, hi = np.searchsorted(d.frames5, [a, z])
        if hi <= lo:
            lo = int(np.clip(np.searchsorted(d.frames5, (a + z) // 2), 0, len(d.frames5) - 1))
            hi = lo + 1
        p = np.clip(d.members[:, lo:hi], 1e-5, 1)
        ratio = (np.log(p[:, :, k]) - np.log(p[:, :, new])).mean(1)
        own = p[:, :, k].mean(1)
        rows[row] = [(ratio > 0).sum(), ratio.min(), ratio.max(), ratio.std(), ratio[:2].mean(), ratio[2:].mean(), own.min(), own.max()]
    return rows


def features(d, segs, cands):
    return np.hstack([c.candidate_features(segs, cands, d.p, d.edge), vote_features(d, segs, cands)])


def table_of(conds):
    table = {}
    for d in conds:
        cands = c.candidates(d.segs, L.MAX_FRAMES)
        table[(d.game, d.mode)] = (features(d, d.segs, cands), L.gains(d.segs, cands, d.gt))
    return table


def fit_verifier(table, games, target, seed=0, columns=None):
    keys = [(g, m) for g in games for m in c.MODES]
    x = np.concatenate([table[key][0] for key in keys])
    delta = np.concatenate([table[key][1] for key in keys])
    columns = list(range(x.shape[1])) if columns is None else columns
    return k28.Model(kind="gbm", columns=columns, parameters=dict(random_state=seed)).fit(x, L.target_of(delta, target))


def apply_verifier(model, d, threshold=0.0):
    def score(segs, cands):
        return model.predict(features(d, segs, cands))
    return c.apply_scored(d.segs, score, threshold, L.MAX_FRAMES, 3)


def verifier_study(seed=0, targets=("w0.50", "M", "w0.70", "w0.00")):
    conds = {(g, m): load(g, m) for g in GAMES for m in c.MODES + L.BLIND if members.path(g, m).exists()}
    dec = {key: decoded(k, k.p0) for key, k in conds.items()}
    table = table_of([d for (g, m), d in dec.items() if m in c.MODES])
    scores = L.Scores()
    for held in GAMES:
        train = [g for g in GAMES if g != held]
        for t in targets:
            with_votes = fit_verifier(table, train, t, seed)
            without = fit_verifier(table, train, t, seed, L.V)               # control: the same code path, V27 columns only
            for (g, m), d in dec.items():
                if g != held:
                    continue
                scores.record(f"votes:verifier:{t}", d, c.to_codes(apply_verifier(with_votes, d)[0], d.n))
                scores.record(f"votes-control:verifier:{t}", d, c.to_codes(apply_verifier(without, d)[0], d.n))
        print("FOLD", held, flush=True)
    scores.dump(ROOT / ("verifier_votes" + (f"_s{seed}" if seed else "") + ".json"), seed=seed)
    return scores


# -- second stage with member features -------------------------------------------------------

def member_half_features(m):
    """m (4, T, 3) of one half -> (even block, odd block); odd changes sign under a team swap."""
    dead, diff = m[:, :, 2], m[:, :, 0] - m[:, :, 1]
    even, odd = [], []
    for n in range(m.shape[0]):
        for lag in MEMBER_LAGS:
            even.append(s._lag(dead[n], lag))
            odd.append(s._lag(diff[n], lag))
        even.append(s._rolling(dead[n], 15))
        odd.append(s._rolling(diff[n], 15))
    for spread in (diff.std(0), dead.std(0)):
        even += [spread, s._rolling(spread, 5), s._rolling(spread, 15)]
    odd.append((diff > 0).sum(0) - m.shape[0] / 2)
    return np.column_stack(even), np.column_stack(odd)


def design(k, mirror=False):
    return np.hstack([k.even, k.even_m, -k.odd if mirror else k.odd, -k.odd_m if mirror else k.odd_m])


def fit_stacker(conds, target, seed=0):
    xs, ys, ws = [], [], []
    for k in conds:
        for mirror in (False, True):
            x = design(k, mirror)
            for r, w in L.STACK_TARGETS[target].items():
                y = k.labels[r]
                keep = y >= 0
                label = y[keep].astype(np.int64)
                xs.append(x[keep])
                ys.append(np.array([1, 0, 2])[label] if mirror else label)
                ws.append(np.full(int(keep.sum()), w))
    return HistGradientBoostingClassifier(**dict(s.MODEL, random_state=seed)).fit(np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(ws))


def emissions(model, k):
    return (model.predict_proba(design(k)) + model.predict_proba(design(k, True))[:, [1, 0, 2]]) / 2


def stack_study(target="M", seed=0, verifier_targets=("w0.50", "M")):
    conds = {(g, m): load(g, m) for g in GAMES for m in c.MODES + L.BLIND if members.path(g, m).exists()}
    single = {g: fit_stacker([conds[(g, m)] for m in c.MODES], target, seed) for g in GAMES}
    print("FITTED single-game stackers", target, flush=True)
    scores = L.Scores()
    emitted = {}
    tag0 = f"stack-members[{target}]"
    for held in GAMES:
        a, z = [g for g in GAMES if g != held]
        pair = fit_stacker([conds[(g, m)] for g in (a, z) for m in c.MODES], target, seed)
        inner = [conds[(g, m)] for g in (a, z) for m in c.MODES]
        inner_p = [emissions(single[z if k.game == a else a], k) for k in inner]
        for k, p in zip(inner, inner_p):
            emitted[f"inner_{held}_{k.game}_{k.mode}"] = p.astype(np.float32)
        train = [decoded(k, p) for k, p in zip(inner, inner_p)]
        plain = L.table_of(train)
        voted = table_of(train)
        models = {t: (L.fit_verifier(plain, [a, z], t, seed), fit_verifier(voted, [a, z], t, seed)) for t in verifier_targets}
        for (g, m), k in conds.items():
            if g != held:
                continue
            p = emissions(pair, k)
            emitted[f"test_{g}_{m}"] = p.astype(np.float32)
            test = decoded(k, p)
            scores.record(tag0 + ":decoder", test, test.codes)
            for t, (without, with_votes) in models.items():
                scores.record(tag0 + f":verifier:{t}", test, c.to_codes(L.apply_verifier(without, test)[0], test.n))
                scores.record(tag0 + f":votes-verifier:{t}", test, c.to_codes(apply_verifier(with_votes, test)[0], test.n))
        print("FOLD", held, flush=True)
    scores.dump(ROOT / f"stack_members_{target}_s{seed}.json", seed=seed, target=target)
    np.savez_compressed(ROOT / f"emissions_members_{target}_s{seed}.npz", **emitted)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["verifier", "stack"])
    parser.add_argument("--target", default="M")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.step == "verifier":
        verifier_study(args.seed)
    else:
        stack_study(args.target, args.seed)
