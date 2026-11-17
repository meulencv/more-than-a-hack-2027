"""V30e: re-timing of predicted stoppage boundaries (onset earlier / restart later).

Finding on the three open Metrica games (possession.v30e_external): the frozen decoder starts
its stoppages a median ~1 s AFTER the reference (ball out / foul event) and ends them ~0.15 s
before the reference restart, in every game and under every degradation.

Two re-timers, both applied AFTER the spell verifier and both unable to create or delete a
possession (a boundary moves inside the neighbouring possession, which keeps MIN_KEEP frames):

    constant(k, r)      every stoppage starts k frames earlier and ends r frames later
    hysteresis(theta)   a stoppage starts at the first frame from which P(D) stays >= theta
                        until the decoded onset, looking back at most LOOK frames (the decoder
                        needs P(D) to dominate; the evidence usually starts rising earlier)

    .venv/bin/python -m possession.v30e_retime external      # leave-one-game-out on Metrica

Writes data/work/v30e/retime_external.json. No challenge data is read by `external`.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import v27_core as c

ROOT = Path("data/work/v30e")
MIN_KEEP = 12                   # a possession next to a re-timed stoppage keeps at least 0.5 s
LOOK = 75                       # hysteresis looks back at most 3 s
SHIFTS = (0, 5, 10, 15, 20, 25, 30, 40)
RESTARTS = (-5, 0, 5, 10)
THETAS = (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05)


def neighbours(segs, i):
    a, z, _ = segs[i]
    previous = segs[i - 1] if i > 0 and segs[i - 1][1] == a else None
    following = segs[i + 1] if i + 1 < len(segs) and segs[i + 1][0] == z else None
    return previous, following


def move(codes, segs, onsets=None, restarts=None):
    """Apply {segment index of a D run: new onset frame} / {index: new restart frame}.

    Restarts are placed first; a new restart leaves the next possession MIN_KEEP frames. A new
    onset must then lie in what is left of the previous possession and leave it MIN_KEEP frames
    (or its whole length when it is shorter than that), so no possession is ever removed.
    Returns new codes and the list of (index, kind, old, new) actually applied.
    """
    out = codes.copy()
    applied = []
    start = {}                                              # possession segment index -> its start after the restart moves
    for i, (a, z, k) in enumerate(segs):
        if k != 2 or not restarts or i not in restarts:
            continue
        _, following = neighbours(segs, i)
        if following is None or following[2] > 1:
            continue
        target = int(max(min(restarts[i], following[1] - MIN_KEEP), a + 1))
        if target > z:
            out[z:target] = 2
        elif target < z:
            out[target:z] = following[2]
        if target != z:
            applied.append((i, "restart", z, target))
            start[i + 1] = target
    for i, (a, z, k) in enumerate(segs):
        if k != 2 or not onsets or i not in onsets:
            continue
        previous, _ = neighbours(segs, i)
        if previous is None or previous[2] > 1:
            continue
        floor = min(start.get(i - 1, previous[0]) + MIN_KEEP, a)
        target = int(min(max(onsets[i], floor), z - 1))
        if target < a:
            out[target:a] = 2
        elif target > a:
            out[a:target] = previous[2]
        if target != a:
            applied.append((i, "onset", a, target))
    return out, applied


def constant(codes, blocks, k, r=0):
    segs = c.segments(codes, blocks)
    dead = [i for i, s in enumerate(segs) if s[2] == 2]
    return move(codes, segs, {i: segs[i][0] - k for i in dead} if k else None, {i: segs[i][1] + r for i in dead} if r else None)


def hysteresis_onsets(codes, blocks, p_dead, theta, look=LOOK):
    segs = c.segments(codes, blocks)
    onsets = {}
    for i, (a, z, k) in enumerate(segs):
        if k != 2:
            continue
        previous, _ = neighbours(segs, i)
        if previous is None or previous[2] > 1:
            continue
        first = max(previous[0], a - look)
        below = np.flatnonzero(p_dead[first:a] < theta)
        onsets[i] = first + (below[-1] + 1 if len(below) else 0)
    return segs, onsets


def hysteresis(codes, blocks, p_dead, theta, r=0):
    segs, onsets = hysteresis_onsets(codes, blocks, p_dead, theta)
    dead = [i for i, s in enumerate(segs) if s[2] == 2]
    return move(codes, segs, onsets, {i: segs[i][1] + r for i in dead} if r else None)


# -- external study --------------------------------------------------------------------------

def verified_conditions():
    """The nine Metrica conditions with the submission recipe applied leave-one-game-out:
    frozen decoder -> w=0.5 spell verifier fitted on the two other games."""
    from . import v29_lab as lab
    conds = lab.conditions("frozen")
    table = lab.load_table("frozen")
    models = {held: lab.fit(table, [g for g in lab.GAMES if g != held], "v27", "w0.50") for held in lab.GAMES}
    for k in conds:
        new, _ = lab.apply(models[k.game], k.segs, k.p, k.edge, None)
        k.verified = c.to_codes(new, k.n)
        k.model = models[k.game]
    return conds


def external(args):
    conds = verified_conditions()
    names = ("consistent", "alternative")
    variants = {"decoder": lambda k: k.codes, "verified": lambda k: k.verified}
    for s in SHIFTS:
        for r in RESTARTS:
            variants[f"const_k{s}_r{r}"] = lambda k, s=s, r=r: constant(k.verified, k.blocks, s, r)[0]
    for theta in THETAS:
        for r in (0, 5):
            variants[f"hyst_{theta}_r{r}"] = lambda k, theta=theta, r=r: hysteresis(k.verified, k.blocks, k.p[:, 2], theta, r)[0]
    scores = {}
    for name, function in variants.items():
        for k in conds:
            codes = function(k)
            for reference in names:
                scores[(name, k.game, k.mode, reference)] = c.spell_f1(k.gt[reference], c.spells_of(c.segments(codes, k.blocks)))
    table = {}
    for name in variants:
        table[name] = {reference: [float(np.mean([scores[(name, g, m, reference)] for m in c.MODES])) for g in (1, 2, 3)] for reference in names}
        table[name]["worst_condition_delta"] = {reference: float(min(scores[(name, g, m, reference)] - scores[("verified", g, m, reference)]
                                                                      for g in (1, 2, 3) for m in c.MODES)) for reference in names}
    for name, row in table.items():
        print(f"RETIME {name:<16}", " ".join(f"{reference[:4]} " + " ".join(f"{v:.3f}" for v in row[reference]) + f" mean {np.mean(row[reference]):.4f}"
                                              for reference in names), "worst", {r[:4]: round(v, 3) for r, v in row["worst_condition_delta"].items()}, flush=True)
    # leave-one-game-out choice of the re-timer: equal weight on both references (as the verifier)
    logo = []
    for family, keys in (("constant", [n for n in variants if n.startswith("const")]), ("hysteresis", [n for n in variants if n.startswith("hyst")]),
                         ("constant_onset_only", [n for n in variants if n.startswith("const") and n.endswith("_r0")]),
                         ("hysteresis_onset_only", [n for n in variants if n.startswith("hyst") and n.endswith("_r0")])):
        for held in (1, 2, 3):
            train = [g for g in (1, 2, 3) if g != held]
            value = lambda n: np.mean([0.5 * table[n]["consistent"][g - 1] + 0.5 * table[n]["alternative"][g - 1] for g in train])
            best = max(keys, key=value)
            row = dict(family=family, held_game=held, chosen=best,
                       **{reference: dict(verified=round(table["verified"][reference][held - 1], 4), retimed=round(table[best][reference][held - 1], 4),
                                          delta=round(table[best][reference][held - 1] - table["verified"][reference][held - 1], 4)) for reference in names})
            logo.append(row)
            print("RETIME LOGO", json.dumps(row), flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "retime_external.json").write_text(json.dumps(dict(table=table, leave_one_game_out=logo, min_keep=MIN_KEEP, look=LOOK,
                                                               challenge_data_read=False), indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["external"])
    args = parser.parse_args()
    dict(external=external)[args.step](args)
