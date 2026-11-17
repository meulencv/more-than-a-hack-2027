"""V30b second stage: the ensemble's emissions re-read against the provider's own definition.

The error budget (possession.v30b_budget) puts most of the loss in the team layer and in the
stoppage layer, not in boundary timing. The frozen networks learned DFL's native labels; the
decoder then reads their probabilities with fixed costs. The simplest fix that touches both
layers: a small gradient-boosting classifier that maps a +-5 s window of the ensemble's own
output (three probabilities and the boundary score, 5 Hz) to H/A/D as the open Metrica games
label it, with the two readings of the definition weighted 0.5/0.5. Its output replaces the
emissions; the decoder and the equal-prior verifier are unchanged.

Only network outputs enter (no tracking signal), so the stage can be applied to the challenge
match as it is; the open games' ball-availability artefact can enter only through the networks.

Strictly nested leave-one-game-out. For held-out game g with training games A, B:
    stacker(A, B) -> g              the scored emissions
    stacker(A) -> B, stacker(B) -> A   out-of-fold emissions on which the decoder configuration
                                       (optional) and the verifier are fitted
so no label of g reaches anything applied to g.

    .venv/bin/python -m possession.v30b_stack logo            # data/work/v30b/stack.json
    .venv/bin/python -m possession.v30b_stack logo --dead     # only P(D) is re-read (team split kept)

No challenge label is read.
"""
import argparse
import json
from types import SimpleNamespace
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from . import v27_core as c
from . import v27_lab as lab27
from . import v28a_core as k28
from . import v28a_lab as lab28
from . import v29_lab as v29
from . import v30b_budget as b

ROOT = b.ROOT
GAMES = b.GAMES
LAGS = (-25, -15, -10, -7, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 10, 15, 25)      # 5 Hz steps: up to +-5 s
EDGE_LAGS = (-5, -3, -2, -1, 0, 1, 2, 3, 5)
MEANS = (5, 15, 50)                                                                    # centred windows: 1 s, 3 s, 10 s
MODEL = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=24, min_samples_leaf=80, l2_regularization=1.0,
             early_stopping=False, random_state=0)
GRID = [dict(c.DEFAULT, temp=t, c_ha=a, c_to_d=d, c_from_d=d) for t in (0.1, 0.2, 0.35) for a in (3.0, 4.0, 6.0) for d in (2.5, 4.0)]


# -- features on the 5 Hz grid ---------------------------------------------------------------

def _lag(x, k):
    out = np.full(len(x), np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def _rolling(x, width, side=0):
    """Mean over a window of `width` samples: centred (0), past (-1, excluding t) or future (+1)."""
    total = np.concatenate(([0.0], np.cumsum(x)))
    index = np.arange(len(x))
    if side == 0:
        lo, hi = index - width // 2, index + width // 2 + 1
    elif side < 0:
        lo, hi = index - width, index
    else:
        lo, hi = index + 1, index + 1 + width
    lo, hi = np.clip(lo, 0, len(x)), np.clip(hi, 0, len(x))
    return (total[hi] - total[lo]) / np.maximum(hi - lo, 1)


def half_features(p, edge):
    """Features of one half. Columns: [dead-like block | team-difference block]; the second block
    changes sign when home and away are swapped."""
    dead, diff = p[:, 2], p[:, 0] - p[:, 1]
    even = [_lag(dead, k) for k in LAGS] + [_lag(edge, k) for k in EDGE_LAGS]
    odd = [_lag(diff, k) for k in LAGS]
    for width in MEANS:
        even.append(_rolling(dead, width))
        odd.append(_rolling(diff, width))
    for side in (-1, 1):
        even.append(_rolling(dead, 10, side))
        odd.append(_rolling(diff, 10, side))
        even.append(_rolling(edge, 5, side))
    even.append(_rolling(np.abs(diff), 15))
    return np.column_stack(even), np.column_stack(odd)


def load(game, mode):
    meta = np.load(c.ROOT / f"metrica_{game}.npz")
    saved = np.load(f"data/work/v16/game{game}_{mode}_final.npz")
    blocks = [tuple(int(v) for v in x) for x in meta["blocks"]]
    frames, p, edge = saved["frames"], saved["probabilities"].astype(np.float64), saved["edges"].astype(np.float64)
    even, odd, keep = [], [], np.zeros(len(frames), dtype=bool)
    for _, first, last in blocks:
        use = (frames >= first) & (frames <= last)
        e, o = half_features(p[use], edge[use])
        even.append(e)
        odd.append(o)
        keep |= use
    k = SimpleNamespace(game=game, mode=mode, blocks=blocks, n=len(meta["original"]), references={r: meta[r] for r in c.REFERENCES},
                        frames=frames[keep], p0=p[keep], edge0=edge[keep], even=np.concatenate(even), odd=np.concatenate(odd))
    k.labels = {r: meta[r][k.frames] for r in b.REFS}
    k.signals = k28.cached_signals(game, mode)
    k.gt = {r: c.spells_of(c.segments(k.references[r], k.blocks)) for r in c.REFERENCES}
    return k


def design(k, mirror=False):
    return np.hstack([k.even, -k.odd if mirror else k.odd])


# -- the stacker -----------------------------------------------------------------------------

def fit(conds):
    """Classifier of H/A/D. Both readings weigh 0.5; home and away are exchangeable."""
    xs, ys, ws = [], [], []
    for k in conds:
        for mirror in (False, True):
            x = design(k, mirror)
            for r in b.REFS:
                y = k.labels[r]
                keep = y >= 0
                label = y[keep].astype(np.int64)
                if mirror:
                    label = np.array([1, 0, 2])[label]
                xs.append(x[keep])
                ys.append(label)
                ws.append(np.full(int(keep.sum()), 0.5))
    return HistGradientBoostingClassifier(**MODEL).fit(np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(ws))


def emissions(model, k, dead_only=False):
    direct = model.predict_proba(design(k))
    mirrored = model.predict_proba(design(k, True))[:, [1, 0, 2]]
    p = (direct + mirrored) / 2
    if dead_only:                                          # keep the networks' team split, re-read only P(D)
        team = k.p0[:, :2] / np.clip(k.p0[:, :2].sum(1, keepdims=True), 1e-6, None)
        p = np.column_stack([team * (1 - p[:, 2:3]), p[:, 2]])
    return p


def decoded(k, p, config=c.DEFAULT):
    """A condition object in the layout possession.v29_lab expects, for emissions p."""
    out = SimpleNamespace(game=k.game, mode=k.mode, blocks=k.blocks, n=k.n, references=k.references, gt=k.gt, signals=k.signals)
    out.interpolated = c.interpolate(p, k.edge0, k.frames, k.blocks)
    out.codes = c.decode(out.interpolated, k.n, [config])[0]
    out.segs = c.segments(out.codes, k.blocks)
    out.p, out.edge = c.dense(out.interpolated, k.n)
    return out


def verifier_table(conds):
    table = {}
    for k in conds:
        cands = c.candidates(k.segs, v29.MAX_FRAMES)
        tag = f"{k.game}_{k.mode}"
        table["x_" + tag] = v29.features(k.segs, cands, k.p, k.edge, None)
        table["d_" + tag] = lab28.exact_deltas(k.segs, cands, k.gt)
        table["u_" + tag] = np.zeros((len(cands), len(c.REFERENCES)))
    return table


def equal_prior(k, segs):
    spells = c.spells_of(segs)
    return 0.5 * c.spell_f1(k.gt["consistent"], spells) + 0.5 * c.spell_f1(k.gt["alternative"], spells)


def select_config(conds, emitted):
    """Decoder configuration with the best equal-prior spellF1 on the given (training) conditions."""
    total = np.zeros(len(GRID))
    for k, p in zip(conds, emitted):
        interpolated = c.interpolate(p, k.edge0, k.frames, k.blocks)
        codes = c.decode(interpolated, k.n, GRID)
        total += [equal_prior(k, c.segments(codes[i], k.blocks)) for i in range(len(GRID))]
    return GRID[int(total.argmax())]


def logo(dead_only=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    conds = {(g, m): load(g, m) for g in GAMES for m in c.MODES}
    single = {g: fit([conds[(g, m)] for m in c.MODES]) for g in GAMES}
    print("FITTED single-game stackers", flush=True)
    scores, report = {}, dict(configs={}, dead_only=dead_only)

    def record(tag, k, codes):
        for r in b.REFS:
            scores[(tag, r, k.game, k.mode)] = c.official(k.references[r], codes)["spell_f1"]

    for held in GAMES:
        a, z = [g for g in GAMES if g != held]
        pair = fit([conds[(g, m)] for g in (a, z) for m in c.MODES])
        # out-of-fold emissions of the two training games (each from the other one's stacker)
        inner = [conds[(g, m)] for g in (a, z) for m in c.MODES]
        inner_p = [emissions(single[z if k.game == a else a], k, dead_only) for k in inner]
        config = select_config(inner, inner_p)
        report["configs"][str(held)] = {name: config[name] for name in ("temp", "c_ha", "c_to_d")}
        for label, cfg in (("stack", c.DEFAULT), ("stack_sel", config)):
            train = [decoded(k, p, cfg) for k, p in zip(inner, inner_p)]
            model = v29.fit(verifier_table(train), [a, z], "v27", "w0.50")
            for m in c.MODES:
                k = conds[(held, m)]
                test = decoded(k, emissions(pair, k, dead_only), cfg)
                record(label + ":decoder", k, test.codes)
                new, _ = v29.apply(model, test.segs, test.p, test.edge, None)
                record(label + ":verified", k, c.to_codes(new, k.n))
        # the current recipe on the same fold, for the paired comparison
        frozen_table = v29.load_table("frozen")
        model = v29.fit(frozen_table, [a, z], "v27", "w0.50")
        for m in c.MODES:
            k = conds[(held, m)]
            base = decoded(k, k.p0)
            record("frozen:decoder", k, base.codes)
            new, _ = v29.apply(model, base.segs, base.p, base.edge, None)
            record("frozen:verified", k, c.to_codes(new, k.n))
        print("FOLD", held, "config", report["configs"][str(held)], flush=True)
    tags = ["frozen:decoder", "frozen:verified", "stack:decoder", "stack:verified", "stack_sel:decoder", "stack_sel:verified"]
    for r in b.REFS:
        print(f"\nOfficial spellF1, {r} reference, held-out game (nested leave-one-game-out)" + ("  [P(D) only]" if dead_only else ""))
        print(f"{'':<20}" + "".join(f"{n:<9}" for n in ("g1", "g2", "g3", "mean", "clean", "moder", "stress")) + "vs recipe: mean   worst   wins/9")
        for tag in tags:
            v = np.array([[scores[(tag, r, g, m)] for m in c.MODES] for g in GAMES])
            d = v - np.array([[scores[("frozen:verified", r, g, m)] for m in c.MODES] for g in GAMES])
            print(f"{tag:<20}" + "".join(f"{x:<9.4f}" for x in list(v.mean(1)) + [v.mean()] + list(v.mean(0)))
                  + f"           {d.mean():+.4f} {d.min():+.4f} {int((d > 0).sum())}")
    report["scores"] = {f"{tag}|{r}|{g}|{m}": v for (tag, r, g, m), v in scores.items()}
    (ROOT / ("stack_dead.json" if dead_only else "stack.json")).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["logo"])
    parser.add_argument("--dead", action="store_true")
    arguments = parser.parse_args()
    logo(arguments.dead)
