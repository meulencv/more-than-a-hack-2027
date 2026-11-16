"""V30: systematic boundary-timing correction after the spell verifier.

Finding (three open Metrica games, frozen ensemble): the decoded boundaries are biased against
the event-derived references, in the same direction in every game and sensor condition:

    possession -> dead ball   starts late   (median +18 frames)
    possession -> possession  changes late  (median +4..5 frames)
    dead ball  -> possession  starts early  (median -4 frames)

The bias does not depend on the reading of one-touch interceptions (both references share
those boundaries). `shift` moves every boundary of a type by a constant number of frames.
Nothing here reads the challenge ground truth; constants are chosen leave-one-game-out.

    .venv/bin/python -m possession.v30_boundaries offsets   # the bias, per boundary type
    .venv/bin/python -m possession.v30_boundaries logo      # leave-one-game-out table

Inputs: data/work/v27/metrica_{g}.npz (`v27_lab prepare`) and
data/work/v16/game{g}_{mode}_final.npz (`v16_transfer`).

Caveat: next to a stoppage the Metrica open games have no ball track, and on the challenge
match P(D) is fused with camera evidence that does not exist here. The T->T constant rests on
the networks and the decoder alone and is the safer one to transfer; `TT_ONLY` keeps just it.
"""
import argparse
import itertools
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from . import v27_core as c

GAMES = (1, 2, 3)
READINGS = ("consistent", "alternative")
MAX_FRAMES = 8 * c.FPS
GBM = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=25,
           l2_regularization=1.0, early_stopping=False, random_state=0)      # = v27_lab.MODEL
GRID = list(itertools.product((0, -4, -8, -12, -16), (0, 2, 4), (0, -2, -4, -6)))
SHIFT = dict(to_dead=-12, from_dead=4, team=-4)          # majority of the leave-one-game-out choices
TT_ONLY = dict(to_dead=0, from_dead=0, team=-4)


def shift(segs, to_dead=0, from_dead=0, team=0, min_frames=1):
    """Move boundaries by a constant per type (negative = earlier). segs: [(start, stop, code)]
    half-open, as v27_core.segments. A boundary never crosses its neighbours and no segment
    gets shorter than min_frames; boundaries at the edge of a half are not touched."""
    out = [list(s) for s in segs]
    for i in range(1, len(out)):
        if out[i - 1][1] != out[i][0]:
            continue
        a, b = out[i - 1][2], out[i][2]
        move = to_dead if (a < 2 and b == 2) else from_dead if (a == 2 and b < 2) else team if (a < 2 and b < 2) else 0
        if not move:
            continue
        t = max(out[i - 1][0] + min_frames, min(out[i][1] - min_frames, out[i][0] + move))
        if out[i - 1][0] < t < out[i][1]:
            out[i - 1][1] = t
            out[i][0] = t
    return [tuple(s) for s in out]


# -- study -----------------------------------------------------------------------------------

def conditions():
    out = []
    for k in c.all_conditions():
        k.segs = c.segments(c.decode(k.interpolated, k.n)[0], k.blocks)
        k.p, k.edge = c.dense(k.interpolated, k.n)
        k.gt = {r: c.spells_of(c.segments(np.where(k.references[r] >= 0, k.references[r], 3), k.blocks)) for r in READINGS}
        out.append(k)
    return out


def expected(k, segs):
    return float(np.mean([c.spell_f1(k.gt[r], c.spells_of(segs)) for r in READINGS]))


def verifier(conds, held):
    """The V29 `final` recipe (V27 features, target 0.5/0.5), fitted without game `held`."""
    x, y = [], []
    for k in conds:
        if k.game == held:
            continue
        cands = c.candidates(k.segs, MAX_FRAMES)
        x.append(c.candidate_features(k.segs, cands, k.p, k.edge))
        base = {r: c.spell_f1(k.gt[r], c.spells_of(k.segs)) for r in READINGS}
        y.append([np.mean([c.spell_f1(k.gt[r], c.spells_of(c.relabel(k.segs, {i: new}))) - base[r] for r in READINGS])
                  for i, _, new in cands])
    return HistGradientBoostingRegressor(**GBM).fit(np.concatenate(x), np.concatenate(y))


def offsets():
    """Signed offset (frames) of each decoded boundary to the reference boundary of the same
    transition, for possessions matched at IoU >= 0.3."""
    found = {}
    for k in conditions():
        gt, pred = k.gt["consistent"], c.spells_of(k.segs)
        kind_p = _kinds(k.segs)
        kind_g = _kinds(c.segments(np.where(k.references["consistent"] >= 0, k.references["consistent"], 3), k.blocks))
        used_g, used_p = set(), set()
        for iou, i, j in c.overlaps(gt, pred):
            if iou < 0.3 or i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            for side in (0, 1):
                a, b = kind_g[(gt[i][side], side)], kind_p[(pred[j][side], side)]
                if a == b and a is not None:
                    found.setdefault(a, []).append(pred[j][side] - gt[i][side])
    for name, v in sorted(found.items()):
        print(f"{name:<10} n={len(v):<5} quartiles {np.percentile(v, [25, 50, 75]).round(0)}")


def _kinds(segs):
    out = {}
    for i, (a, z, k) in enumerate(segs):
        if k > 1:
            continue
        before = segs[i - 1][2] if i and segs[i - 1][1] == a else None
        after = segs[i + 1][2] if i + 1 < len(segs) and segs[i + 1][0] == z else None
        out[(a, 0)] = None if before is None else "from_dead" if before == 2 else "team"
        out[(z, 1)] = None if after is None else "to_dead" if after == 2 else "team"
    return out


def logo():
    conds = conditions()
    after = {}
    for held in GAMES:
        model = verifier(conds, held)
        for k in conds:
            if k.game == held:
                after[(k.game, k.mode)] = c.apply_scored(
                    k.segs, lambda segs, cands, k=k: model.predict(c.candidate_features(segs, cands, k.p, k.edge)), 0.0, MAX_FRAMES, 3)[0]
    score = {(k.game, k.mode, s): expected(k, shift(after[(k.game, k.mode)], *s)) for k in conds for s in GRID}
    chosen = {h: max(GRID, key=lambda s: sum(v for (g, _, t), v in score.items() if g != h and t == s)) for h in GAMES}
    rows = {"frozen decoder": lambda k: k.segs, "verifier 0.5 (V29 final)": lambda k: after[(k.game, k.mode)],
            "+ shift, constants chosen without the game": lambda k: shift(after[(k.game, k.mode)], *chosen[k.game]),
            "+ shift, SHIFT": lambda k: shift(after[(k.game, k.mode)], **SHIFT),
            "+ shift, TT_ONLY": lambda k: shift(after[(k.game, k.mode)], **TT_ONLY)}
    report = dict(chosen={str(h): chosen[h] for h in GAMES}, rows=[])
    print("leave-one-game-out choices (to_dead, from_dead, team):", chosen)
    for name, fn in rows.items():
        row = dict(variant=name)
        for r in READINGS:
            row[r] = [round(float(np.mean([c.spell_f1(k.gt[r], c.spells_of(fn(k))) for k in conds if k.game == g])), 4) for g in GAMES]
        row["mean"] = round(float(np.mean([row[r] for r in READINGS])), 4)
        report["rows"].append(row)
        print(f"{name:<44}", {r: row[r] for r in READINGS}, "mean", row["mean"], flush=True)
    (c.ROOT / "v30_boundaries.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    steps = dict(offsets=offsets, logo=logo)
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=list(steps))
    steps[parser.parse_args().step]()
