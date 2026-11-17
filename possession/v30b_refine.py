"""V30b boundary re-timing: move each predicted boundary to where the reference would put it.

A predicted boundary is a stoppage onset (team -> D), a restart (D -> team) or a change of team.
For every kind a regressor predicts (reference time - predicted time) from local evidence around
the predicted boundary, and the boundary is moved by that amount (never past a neighbour).

Protocol (leave one Metrica game out). The regressors are fitted on the boundaries of the
*frozen decoder* (no verifier, so nothing fitted on the held-out game enters) of the two other
games, three conditions each, against their consistent reference (boundaries paired one-to-one
within 3 s with equal labels on both sides). They are applied to the held-out game's recipe
output (decoder + equal-prior verifier, itself leave-one-game-out). Official metric, both
references. No challenge label is read.

    .venv/bin/python -m possession.v30b_refine logo        # all variants, per game / condition
    .venv/bin/python -m possession.v30b_refine noise       # reference timing against the tracking

Variants:
    const        one median shift per kind of boundary (three numbers)
    net          regressor on network-output features only (probability and edge shapes)
    net_phys     + ball kinematics and position relative to the lines (NaN where unseen)
    net_avail    + ball availability pattern (Metrica blanks the ball during stoppages: this
                 variant measures an artefact of the external games and must not be transferred)
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from . import v27_core as c
from . import v30b_budget as b

ROOT = b.ROOT
FPS = c.FPS
WINDOW = 3 * FPS
PAIR = 3 * FPS
OFFSETS = (-50, -25, -12, -5, 0, 5, 12, 25, 50)
MODEL = dict(loss="absolute_error", max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=20,
             l2_regularization=1.0, early_stopping=False, random_state=0)
VARIANTS = ("const", "net", "net_phys", "net_avail")


# -- features of one predicted boundary ------------------------------------------------------

def _at(x, t):
    return float(x[t]) if 0 <= t < len(x) else np.nan


def _first(mask, start):
    """Offset (frames, relative to the boundary) of the first True of a window, NaN if none."""
    hit = np.flatnonzero(mask)
    return float(hit[0] + start) if len(hit) else np.nan


def _last(mask, start):
    hit = np.flatnonzero(mask)
    return float(hit[-1] + start) if len(hit) else np.nan


def after_signal(p, before, after):
    """Probability that the label after the boundary already holds, among the two labels involved
    (onset: P(D); restart: 1 - P(D); change of team: share of the new team)."""
    if after == 2:
        return p[:, 2]
    if before == 2:
        return 1.0 - p[:, 2]
    return p[:, after] / np.clip(p[:, before] + p[:, after], 1e-6, None)


NET = ([f"s{o:+d}" for o in OFFSETS] + ["cross30", "cross50", "cross70", "last_below50", "slope_at", "slope_max",
                                         "edge_at", "edge_max", "edge_mass", "edge_here", "previous_s", "next_s", "dead_here"])
PHYS = (["speed-25", "speed-12", "speed+0", "speed+12", "speed+25", "line-25", "line-12", "line+0", "line+12", "line_min",
         "out_first", "near_line_first", "kink_before", "kink_after", "d_before-12", "d_before+0", "d_before+12",
         "d_after-12", "d_after+0", "d_after+12", "closer_first"])
AVAIL = ["seen-25", "seen-12", "seen+0", "seen+12", "seen+25", "seen_share_before", "seen_share_after", "last_seen", "first_seen"]
FEATURES = NET + PHYS + AVAIL
COLUMNS = dict(net=list(range(len(NET))), net_phys=list(range(len(NET) + len(PHYS))), net_avail=list(range(len(FEATURES))))


def boundary_features(segs, j, p, edge, s):
    """Features of the boundary between segs[j] and segs[j + 1] (same half)."""
    (a0, f, before), (_, z1, after) = segs[j], segs[j + 1]
    n = len(edge)
    lo, hi = max(0, f - WINDOW), min(n, f + WINDOW + 1)
    signal = np.nan_to_num(after_signal(p[lo:hi], before, after), nan=0.5)
    start = lo - f
    slope = np.zeros_like(signal)
    slope[2:-2] = (signal[4:] - signal[:-4]) / 4.0
    e = np.nan_to_num(edge[lo:hi])
    row = [_at(signal, f + o - lo) for o in OFFSETS]
    row += [_first(signal >= 0.3, start), _first(signal >= 0.5, start), _first(signal >= 0.7, start), _last(signal < 0.5, start),
            float(np.argmax(slope) + start), float(slope.max())]
    near = slice(max(0, f - 50 - lo), f + 51 - lo)
    weights = e[near]
    row += [float(np.argmax(weights) + near.start + start), float(weights.max()),
            float((weights * (np.arange(len(weights)) + near.start + start)).sum() / max(weights.sum(), 1e-6)), _at(e, f - lo),
            min(30.0, (f - a0) / FPS), min(30.0, (z1 - f) / FPS), float(before == 2 or after == 2)]
    # ball kinematics and lines
    line = np.fmin(s["touchline"], s["goalline"])
    row += [_at(s["speed"], f + o) for o in (-25, -12, 0, 12, 25)]
    row += [_at(line, f + o) for o in (-25, -12, 0, 12)]
    window = line[lo:hi]
    with np.errstate(invalid="ignore"):
        row += [float(np.nanmin(window[:f + 26 - lo])) if np.isfinite(window[:f + 26 - lo]).any() else np.nan,
                _first(window < 0, start), _first(window < 1.0, start)]
    kinks = np.flatnonzero(s["event"][lo:hi] == 1) + start
    row += [float(kinks[kinks <= 0][-1]) if (kinks <= 0).any() else np.nan, float(kinks[kinks > 0][0]) if (kinks > 0).any() else np.nan]
    if before < 2 and after < 2:
        d0, d1 = s["distance"][:, before], s["distance"][:, after]
    elif after == 2:                                    # onset: the team that had the ball / the other one
        d0, d1 = s["distance"][:, before], s["distance"][:, 1 - before]
    else:
        d0, d1 = s["distance"][:, 1 - after], s["distance"][:, after]
    row += [_at(d0, f + o) for o in (-12, 0, 12)] + [_at(d1, f + o) for o in (-12, 0, 12)]
    with np.errstate(invalid="ignore"):
        row += [_first(d1[lo:hi] < d0[lo:hi], start)]
    seen = s["seen"].astype(float)
    row += [_at(seen, f + o) for o in (-25, -12, 0, 12, 25)]
    row += [float(seen[max(0, f - 25):f].mean()) if f > 0 else np.nan, float(seen[f:f + 25].mean()) if f < n else np.nan,
            _last(seen[lo:min(n, f + 51)] == 1, start), _first(seen[max(0, f - 50):hi] == 1, max(0, f - 50) - f)]
    return row


def half_segments(segs, blocks):
    return [[s for s in segs if first <= s[0] <= last] for _, first, last in blocks]


def rows_of(k, segs, reference=None):
    """Per kind: (features, target offset or NaN, (half, j)) of every boundary of `segs`."""
    out = {kind: ([], [], []) for kind in b.KINDS}
    for h, half in enumerate(half_segments(segs, k.blocks)):
        bounds = b.boundaries(half)
        index = [j for j in range(len(half) - 1) if half[j][1] == half[j + 1][0]]
        target = {}
        if reference is not None:
            rb = b.boundaries(half_segments(reference, k.blocks)[h])
            target = {jj: rb[i][0] - bounds[jj][0] for i, jj in b.pair_boundaries(rb, bounds, PAIR)}
        for position, j in enumerate(index):
            kind = bounds[position][1]
            out[kind][0].append(boundary_features(half, j, k.p, k.edge, k.signals))
            out[kind][1].append(target.get(position, np.nan))
            out[kind][2].append((h, j))
    return {kind: (np.array(x, dtype=np.float64).reshape(len(x), len(FEATURES)), np.array(y, dtype=np.float64), where)
            for kind, (x, y, where) in out.items()}


# -- fit and apply ---------------------------------------------------------------------------

def fit(conds, variant, reference="consistent"):
    """{kind: model or constant} from the frozen-decoder boundaries of `conds`."""
    data = {kind: ([], []) for kind in b.KINDS}
    for k in conds:
        rows = k.train_rows[reference]
        for kind in b.KINDS:
            x, y, _ = rows[kind]
            keep = np.isfinite(y)
            data[kind][0].append(x[keep])
            data[kind][1].append(y[keep])
    models = {}
    for kind in b.KINDS:
        x, y = np.concatenate(data[kind][0]), np.concatenate(data[kind][1])
        if variant == "const":
            models[kind] = float(np.median(y))
        else:
            models[kind] = HistGradientBoostingRegressor(**MODEL).fit(x[:, COLUMNS[variant]], y)
    return models


def predict(models, variant, kind, x):
    if not len(x):
        return np.zeros(0)
    if variant == "const":
        return np.full(len(x), models[kind])
    return models[kind].predict(x[:, COLUMNS[variant]])


def apply(k, segs, models, variant, kinds=b.KINDS, limit=WINDOW, minimum=3):
    """Re-timed copy of `segs`. A boundary never comes closer than `minimum` frames to a neighbour."""
    rows = rows_of(k, segs)
    halves = [[list(s) for s in half] for half in half_segments(segs, k.blocks)]
    shift = {}
    for kind in kinds:
        x, _, where = rows[kind]
        for (h, j), d in zip(where, predict(models, variant, kind, x)):
            shift[(h, j)] = int(np.clip(round(float(d)), -limit, limit))
    for (h, j) in sorted(shift):
        half = halves[h]
        lower, upper = half[j][0] + minimum, half[j + 1][1] - minimum
        if lower > upper:
            continue
        new = int(min(max(half[j][1] + shift[(h, j)], lower), upper))
        half[j][1] = new
        half[j + 1][0] = new
    return [tuple(s) for half in halves for s in half]


def prepare(conds):
    for k in conds:
        k.train_rows = {"consistent": rows_of(k, k.base_segs, k.ref_segs["consistent"])}
    return conds


def logo():
    ROOT.mkdir(parents=True, exist_ok=True)
    conds = prepare(b.recipe())
    report = dict(rows=[], shifts={})
    scores = {}
    for held in b.GAMES:
        train = [k for k in conds if k.game != held]
        for variant in VARIANTS:
            models = fit(train, variant)
            if variant == "const":
                report["shifts"][str(held)] = {kind: models[kind] / FPS for kind in b.KINDS}
            for k in [k for k in conds if k.game == held]:
                for kinds, tag in ((b.KINDS, variant), (("onset", "restart"), variant + ":dead"), (("change",), variant + ":change")):
                    codes = c.to_codes(apply(k, k.pred_segs, models, variant, kinds), k.n)
                    for r in b.REFS:
                        scores[(tag, r, k.game, k.mode)] = b.score(k, codes, r)
                # accuracy of the re-timing itself on the held-out recipe boundaries
                if variant != "const":
                    rows = rows_of(k, k.pred_segs, k.ref_segs["consistent"])
                    for kind in b.KINDS:
                        x, y, _ = rows[kind]
                        keep = np.isfinite(y)
                        d = predict(models, variant, kind, x[keep])
                        report["rows"].append(dict(variant=variant, game=k.game, mode=k.mode, kind=kind, n=int(keep.sum()),
                                                   before=float(np.median(np.abs(y[keep])) / FPS), after=float(np.median(np.abs(y[keep] - d)) / FPS),
                                                   within05_before=float((np.abs(y[keep]) <= 12.5).mean()),
                                                   within05_after=float((np.abs(y[keep] - d) <= 12.5).mean())))
        print("FOLD", held, "done", flush=True)
    for k in conds:
        for r in b.REFS:
            scores[("recipe", r, k.game, k.mode)] = b.score(k, k.pred, r)
    tags = ["recipe"] + [v + suffix for v in VARIANTS for suffix in ("", ":dead", ":change")]
    report["scores"] = {f"{tag}|{r}|{g}|{m}": v for (tag, r, g, m), v in scores.items()}
    print("\nConstant shift per kind (s, reference minus predicted), fitted without the held-out game:", json.dumps(report["shifts"]))
    for r in b.REFS:
        print(f"\nOfficial spellF1 change versus the recipe, {r} reference (held-out game)")
        print(f"{'':<20}" + "".join(f"{n:<9}" for n in ("g1", "g2", "g3", "mean", "clean", "moder", "stress", "worst", "wins/9")))
        for tag in tags:
            d = np.array([[scores[(tag, r, g, m)] - scores[("recipe", r, g, m)] for m in c.MODES] for g in b.GAMES])
            head = f"{scores[('recipe', r, 1, 'clean')] and tag:<20}"
            if tag == "recipe":
                d = np.array([[scores[(tag, r, g, m)] for m in c.MODES] for g in b.GAMES])
            print(head + "".join(f"{v:<+9.4f}" for v in list(d.mean(1)) + [d.mean()] + list(d.mean(0)) + [d.min()]) + f"{int((d > 0).sum())}")
    print("\nRe-timing accuracy on held-out recipe boundaries (paired within 3 s): median |error| s and share within 0.5 s")
    for variant in VARIANTS[1:]:
        for kind in b.KINDS:
            for m in c.MODES:
                use = [x for x in report["rows"] if x["variant"] == variant and x["kind"] == kind and x["mode"] == m]
                print(f"{variant:<10}{kind:<8}{m:<9} |e| {np.mean([x['before'] for x in use]):.2f} -> {np.mean([x['after'] for x in use]):.2f}   "
                      f"<=0.5s {np.mean([x['within05_before'] for x in use]):.2f} -> {np.mean([x['within05_after'] for x in use]):.2f}")
    (ROOT / "refine.json").write_text(json.dumps(report, indent=1))


# -- how exact is the reference timing? ------------------------------------------------------

def noise():
    """Reference stoppage timing against the clean tracking of the same game.

    (1) Metrica blanks the ball while play is stopped: start/end of the blank against the
        reference onset/restart (same annotation, so this only shows they are one source).
    (2) Ball-out stoppages: when did the tracked ball physically cross the line, relative to the
        reference onset? That is an independent physical clock.
    (3) Restarts: when does the ball start to move (first frame >= 3 m/s after the restart)?
    """
    from .sources import external_match
    report = {}
    for game in b.GAMES:
        match = external_match(game)
        length, width = match.pitch
        meta = np.load(c.ROOT / f"metrica_{game}.npz")
        reference = meta["consistent"]
        blocks = [tuple(int(v) for v in x) for x in meta["blocks"]]
        ball = match.ball.astype(np.float64)
        seen = np.isfinite(ball).all(1)
        margin = np.minimum(np.minimum(ball[:, 0], length - ball[:, 0]), np.minimum(ball[:, 1], width - ball[:, 1]))
        edges = np.diff(np.r_[0, (~seen).astype(np.int8), 0])
        blank = [(a, z) for a, z in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if z - a >= FPS]
        starts, ends = np.array([a for a, _ in blank]), np.array([z for _, z in blank])
        segs = c.segments(np.where(reference >= 0, reference, 3), blocks)
        onset = [f for f, kind, _, _ in b.boundaries(segs) if kind == "onset"]
        restart = [f for f, kind, _, _ in b.boundaries(segs) if kind == "restart"]
        blank_on = np.array([starts[np.abs(starts - f).argmin()] - f for f in onset]) / FPS
        blank_re = np.array([ends[np.abs(ends - f).argmin()] - f for f in restart]) / FPS
        crossing, inside = [], 0
        for f in onset:
            t = f - 1
            while t > f - 2 * FPS and not seen[t]:
                t -= 1
            if not seen[t]:
                continue
            window = np.arange(max(0, f - 3 * FPS), t + 1)
            outside = np.flatnonzero(margin[window] < 0)
            runs = np.split(outside, np.flatnonzero(np.diff(outside) > 1) + 1) if len(outside) else []
            if runs and runs[-1][-1] == len(window) - 1:              # the ball is outside when the stoppage starts
                crossing.append((f - window[runs[-1][0]]) / FPS)
            elif margin[t] < 3.0:
                inside += 1
        live = (reference >= 0) & (reference < 2)
        missed = [(a, z) for a, z in blank if z - a >= 3 * FPS and live[a:z].mean() > 0.9]
        crossing = np.array(crossing)
        report[game] = dict(
            onsets=len(onset), blank_start_within_02=float((np.abs(blank_on) <= 0.2).mean()), blank_end_within_02=float((np.abs(blank_re) <= 0.2).mean()),
            ball_out_onsets=int(len(crossing)), reference_after_crossing_median=float(np.median(crossing)),
            reference_after_crossing_iqr=[float(v) for v in np.percentile(crossing, [25, 75])],
            reference_after_crossing_p90=float(np.percentile(crossing, 90)), near_line_but_inside=inside,
            blanks_3s_inside_reference_live=len(missed), seconds=float(sum(z - a for a, z in missed) / FPS),
            share_of_reference_live=float(sum(z - a for a, z in missed) / live.sum()))
        print("NOISE", game, json.dumps(report[game]), flush=True)
    (ROOT / "reference_noise.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    steps = dict(logo=logo, noise=noise)
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=list(steps))
    steps[parser.parse_args().step]()
