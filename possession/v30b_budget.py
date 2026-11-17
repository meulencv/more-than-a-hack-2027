"""V30b error budget: where the remaining spellF1 of the current recipe is lost.

The recipe is the submission's: frozen-ensemble outputs on the three open Metrica games
(data/work/v16) -> 25 Hz Viterbi -> V27-feature verifier fitted to the equal-prior target
(`w0.50`), leave-one-game-out. Every number below is the official metric
(evaluation.possession_metrics) against an event-derived reference; no challenge label is read.

    .venv/bin/python -m possession.v30b_budget oracles     # oracle substitutions (A.1-A.5)
    .venv/bin/python -m possession.v30b_budget misses      # unmatched reference possessions by cause
    .venv/bin/python -m possession.v30b_budget latency     # onset / restart / team-change timing
    .venv/bin/python -m possession.v30b_budget dfl         # the same timing on the DFL out-of-fold outputs
    .venv/bin/python -m possession.v30b_budget ceilings    # what no model / no post-processing can remove
    .venv/bin/python -m possession.v30b_budget semantic    # team-layer oracles that do not need to know the reading
    .venv/bin/python -m possession.v30b_budget all

Outputs: data/work/v30b/{oracles,misses,latency,dfl,ceilings}.json.

Oracle substitutions (reference information enters only where stated):
    ref_dead       reference stoppages; the team on live frames is the model's (two-state
                   Viterbi with the decoder's change-of-team cost inside every live stretch)
    ref_team       predicted stoppages; reference team on the frames both call live (a predicted
                   live frame the reference calls dead takes the nearest reference team)
    ref_bounds     reference possession boundaries, predicted team by majority per reference
                   possession = team-identification ceiling
    ref_label      predicted segmentation, reference team by majority per predicted possession
    snap_{1,2}s    every predicted boundary moved onto the nearest unused reference boundary of
                   the same kind (onset / restart / change of team) within 1 s / 2 s
    snap_{kind}    the same for one kind of boundary only (2 s)
    removal        the best removal of short predicted possessions (v27_lab.oracle, <= 8 s)
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import v27_core as c
from . import v27_lab as lab27
from . import v29_lab as v29

ROOT = Path("data/work/v30b")
GAMES = (1, 2, 3)
REFS = ("consistent", "alternative")
FPS = c.FPS
KINDS = ("onset", "restart", "change")
BINS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 1e9))
BIN_NAMES = ("<1s", "1-2s", "2-4s", "4-8s", "8-16s", ">16s")


# -- the recipe ------------------------------------------------------------------------------

def recipe(base="frozen", family="v27", target="w0.50"):
    """Nine conditions with the held-out prediction of the current recipe in k.pred_segs."""
    conds = v29.conditions(base)
    table = v29.load_table(base)
    models = {held: v29.fit(table, [g for g in GAMES if g != held], family, target) for held in GAMES}
    for k in conds:
        k.base_segs = k.segs
        k.model = models[k.game]
        k.pred_segs, k.applied = v29.apply(k.model, k.segs, k.p, k.edge, k.signals)
        k.pred = c.to_codes(k.pred_segs, k.n)
        k.ref_segs = {r: c.segments(np.where(k.references[r] >= 0, k.references[r], 3), k.blocks) for r in REFS}
    return conds


def score(k, codes, r):
    return c.official(k.references[r], codes)["spell_f1"]


# -- oracle substitutions --------------------------------------------------------------------

def nearest_fill(values, known):
    """values with the entries where ~known replaced by the nearest known entry (ties: earlier)."""
    index = np.flatnonzero(known)
    if not len(index):
        return values.copy()
    position = np.searchsorted(index, np.arange(len(values)))
    left = index[np.clip(position - 1, 0, len(index) - 1)]
    right = index[np.clip(position, 0, len(index) - 1)]
    pick = np.where(np.abs(np.arange(len(values)) - left) <= np.abs(right - np.arange(len(values))), left, right)
    return values[pick]


def ref_dead(k, r, verifier=None):
    """Reference stoppages + the model's team inside every reference live stretch."""
    reference = k.references[r]
    codes = np.full(k.n, 3, dtype=np.int8)
    for first, last, p, e in k.interpolated:
        dead = reference[first:last + 1] == 2
        codes[first:last + 1] = c.team_viterbi(p, e, dead, [(c.DEFAULT["c_ha"], c.DEFAULT["gamma"])])[0]
    return codes


def ref_team(k, r):
    """Predicted stoppages + reference team where both are live."""
    reference = k.references[r]
    codes = k.pred.copy()
    for _, first, last in k.blocks:
        window = slice(first, last + 1)
        ref = reference[window]
        team = nearest_fill(ref, ref < 2)
        live = codes[window] < 2
        codes[window] = np.where(live, team, 2)
    return codes


def majority_team(codes_inside, p_inside):
    counts = [(codes_inside == 0).sum(), (codes_inside == 1).sum()]
    if counts[0] == counts[1]:
        return int(np.nansum(p_inside[:, 1]) > np.nansum(p_inside[:, 0]))
    return int(counts[1] > counts[0])


def ref_bounds(k, r):
    """Reference segmentation, the team of each reference possession is the predicted majority."""
    codes = np.where(k.references[r] >= 0, k.references[r], 3).astype(np.int8)
    for a, z, code in k.ref_segs[r]:
        if code < 2:
            codes[a:z] = majority_team(k.pred[a:z], k.p[a:z])
    return codes


def ref_label(k, r):
    """Predicted segmentation, the team of each predicted possession is the reference majority."""
    reference = k.references[r]
    codes = k.pred.copy()
    for a, z, code in k.pred_segs:
        if code < 2:
            inside = reference[a:z]
            counts = [(inside == 0).sum(), (inside == 1).sum()]
            if counts[0] != counts[1]:
                codes[a:z] = int(counts[1] > counts[0])
    return codes


def kind_of(before, after):
    if after == 2:
        return "onset"
    if before == 2:
        return "restart"
    return "change"


def boundaries(segs):
    """[(frame, kind, code before, code after)] of the label changes inside a half."""
    out = []
    for (a, z, k0), (a1, z1, k1) in zip(segs[:-1], segs[1:]):
        if z == a1 and k0 < 3 and k1 < 3:
            out.append((a1, kind_of(k0, k1), k0, k1))
    return out


def pair_boundaries(reference, predicted, tolerance, exact=True):
    """Greedy one-to-one pairs (reference index, predicted index) by |distance| <= tolerance.

    exact: the two labels must be equal on both sides; otherwise only the kind must be.
    """
    pairs = []
    for i, (fr, kr, br, ar) in enumerate(reference):
        for j, (fp, kp, bp, ap) in enumerate(predicted):
            if abs(fp - fr) > tolerance or kr != kp:
                continue
            if exact and (br != bp or ar != ap):
                continue
            pairs.append((abs(fp - fr), i, j))
    pairs.sort()
    used_r, used_p, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        out.append((i, j))
    return out


def snap(k, r, tolerance, kinds=KINDS):
    """Move predicted boundaries onto matched reference boundaries; never past a neighbour."""
    out = []
    for _, first, last in k.blocks:
        ref = [s for s in k.ref_segs[r] if first <= s[0] <= last]
        pred = [list(s) for s in k.pred_segs if first <= s[0] <= last]
        rb, pb = boundaries(ref), boundaries([tuple(s) for s in pred])
        target = {j: rb[i][0] for i, j in pair_boundaries(rb, pb, tolerance) if rb[i][1] in kinds}
        for j in sorted(target):                               # boundary j sits between pred[j] and pred[j + 1]
            lower = pred[j][0] + 1
            upper = pred[j + 1][1] - 1
            new = int(min(max(target[j], lower), upper))
            pred[j][1] = new
            pred[j + 1][0] = new
        out += [tuple(s) for s in pred]
    return c.to_codes(out, k.n)


def removal_oracle(k, r, segs=None):
    return lab27.oracle(k.pred_segs if segs is None else segs, k.gt[r], lab27.MAX_FRAMES)


def oracle_scores(k, r):
    out = dict(decoder=score(k, k.codes, r), recipe=score(k, k.pred, r))
    out["ref_dead"] = score(k, ref_dead(k, r), r)
    out["ref_team"] = score(k, ref_team(k, r), r)
    out["ref_bounds"] = score(k, ref_bounds(k, r), r)
    out["ref_label"] = score(k, ref_label(k, r), r)
    for seconds in (0.5, 1, 2):
        out[f"snap_{seconds}s"] = score(k, snap(k, r, int(seconds * FPS)), r)
    for kind in KINDS:
        out[f"snap_{kind}_2s"] = score(k, snap(k, r, 2 * FPS, (kind,)), r)
    out["snap_dead_2s"] = score(k, snap(k, r, 2 * FPS, ("onset", "restart")), r)
    out["removal"] = removal_oracle(k, r)
    snapped = c.segments(snap(k, r, 2 * FPS), k.blocks)
    out["snap_2s+removal"] = lab27.oracle(snapped, k.gt[r], lab27.MAX_FRAMES)
    # reference stoppages, then the removal oracle on the team layer = what is left is team identification
    dead = c.segments(ref_dead(k, r), k.blocks)
    out["ref_dead+removal"] = lab27.oracle(dead, k.gt[r], lab27.MAX_FRAMES)
    # predicted stoppages moved onto the reference when within 2 s, reference team elsewhere
    return out


def table(rows, names, title):
    print("\n" + title)
    print(f"{'':<18}" + "".join(f"{n:<8}" for n in ("g1", "g2", "g3", "mean", "clean", "moder", "stress")))
    for name in names:
        values = {(row["game"], row["mode"]): row[name] for row in rows}
        game = [np.mean([values[(g, m)] for m in c.MODES]) for g in GAMES]
        mode = [np.mean([values[(g, m)] for g in GAMES]) for m in c.MODES]
        print(f"{name:<18}" + "".join(f"{v:<8.3f}" for v in game + [np.mean(game)] + mode))


def oracles():
    ROOT.mkdir(parents=True, exist_ok=True)
    conds = recipe()
    report = {}
    for r in REFS:
        rows = []
        for k in conds:
            row = dict(game=k.game, mode=k.mode, **oracle_scores(k, r))
            rows.append(row)
            print("ORACLE", r, json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()}), flush=True)
        report[r] = rows
        table(rows, [n for n in rows[0] if n not in ("game", "mode")], f"Official spellF1 against the {r} reference (held-out recipe and oracles)")
    (ROOT / "oracles.json").write_text(json.dumps(report, indent=1))


# -- unmatched reference possessions by cause ------------------------------------------------

CAUSES = ("lost_in_stoppage", "wrong_team", "split", "merged", "shift_stoppage_side", "shift_change_side", "other")


def duration_bin(frames):
    seconds = frames / FPS
    return next(i for i, (lo, hi) in enumerate(BINS) if lo <= seconds < hi)


def classify(gt, pred, reference, predicted, matched_pred):
    """Cause of one unmatched reference possession gt = (a, z, team).

    reference / predicted: label arrays (reference -1 where not graded, predicted 3).
    """
    a, z, team = gt
    length = z - a
    inside = predicted[a:z]
    same, dead, other = (inside == team).mean(), (inside == 2).mean(), (inside == 1 - team).mean()
    if same < 0.5:
        return "lost_in_stoppage" if dead >= other else "wrong_team"
    own = [(min(z, pz) - max(a, pa), pa, pz) for pa, pz, pk in pred if pk == team and pz > a and pa < z]
    overlap, pa, pz = max(own)
    under, over = length - overlap, (pz - pa) - overlap
    if under >= over:                                   # the best predicted possession is too short
        rest = np.ones(length, dtype=bool)
        rest[max(a, pa) - a:min(z, pz) - a] = False
        missing = inside[rest]
        if (missing == team).mean() >= 0.5:
            return "split"
        return "shift_stoppage_side" if (missing == 2).sum() >= (missing == 1 - team).sum() else "shift_change_side"
    outside = np.concatenate([reference[pa:min(a, pz)], reference[max(z, pa):pz]])          # reference under the excess
    if (outside == team).sum() >= max(FPS, 0.3 * over):
        return "merged"
    return "shift_stoppage_side" if (outside == 2).sum() >= (outside == 1 - team).sum() else "shift_change_side"


def misses():
    ROOT.mkdir(parents=True, exist_ok=True)
    conds = recipe()
    report = {}
    for r in REFS:
        counts = np.zeros((len(conds), len(CAUSES), len(BINS)))
        totals = np.zeros((len(conds), len(BINS)))
        spurious = np.zeros((len(conds), 4, len(BINS)))          # unmatched predicted: in ref stoppage / wrong team / fragment / other
        n_pred = np.zeros((len(conds), len(BINS)))
        for row, k in enumerate(conds):
            gt, pred = k.gt[r], c.spells_of(k.pred_segs)
            g, p = c.matching(gt, pred)[0.5]
            reference = k.references[r]
            for i, spell in enumerate(gt):
                b = duration_bin(spell[1] - spell[0])
                totals[row, b] += 1
                if not g[i]:
                    counts[row, CAUSES.index(classify(spell, pred, reference, k.pred, p)), b] += 1
            for j, (a, z, team) in enumerate(pred):
                b = duration_bin(z - a)
                n_pred[row, b] += 1
                if not p[j]:
                    inside = reference[a:z]
                    same, dead, other = (inside == team).mean(), (inside == 2).mean(), (inside == 1 - team).mean()
                    kind = 2 if same >= 0.5 else (0 if dead >= other else 1)
                    spurious[row, kind, b] += 1
        per_game = counts.reshape(3, 3, len(CAUSES), len(BINS)).mean(1)          # mean over conditions
        total_game = totals.reshape(3, 3, len(BINS)).mean(1)
        report[r] = dict(causes=CAUSES, bins=BIN_NAMES, unmatched=per_game.tolist(), reference=total_game.tolist(),
                         unmatched_by_mode=counts.reshape(3, 3, len(CAUSES), len(BINS)).mean(0).tolist(),
                         predicted=n_pred.reshape(3, 3, len(BINS)).mean(1).tolist(),
                         unmatched_predicted=spurious.reshape(3, 3, 4, len(BINS)).mean(1).tolist())
        mean = per_game.mean(0)
        total = total_game.mean(0)
        print(f"\nUnmatched reference possessions at IoU 0.5, {r} reference (mean count per game and condition)")
        print(f"{'cause':<22}" + "".join(f"{b:<8}" for b in BIN_NAMES) + f"{'all':<8}{'share':<8}{'g1':<7}{'g2':<7}{'g3':<7}{'clean':<7}{'moder':<7}{'stress':<7}")
        by_mode = counts.reshape(3, 3, len(CAUSES), len(BINS)).mean(0).sum(2)
        for j, cause in enumerate(CAUSES):
            print(f"{cause:<22}" + "".join(f"{v:<8.1f}" for v in mean[j]) + f"{mean[j].sum():<8.1f}{mean[j].sum() / mean.sum():<8.2f}"
                  + "".join(f"{v:<7.1f}" for v in per_game[:, j].sum(1)) + "".join(f"{v:<7.1f}" for v in by_mode[:, j]))
        print(f"{'unmatched':<22}" + "".join(f"{v:<8.1f}" for v in mean.sum(0)) + f"{mean.sum():<8.1f}")
        print(f"{'reference':<22}" + "".join(f"{v:<8.1f}" for v in total) + f"{total.sum():<8.1f}")
        print(f"{'recall@0.5':<22}" + "".join(f"{1 - u / max(t, 1e-9):<8.2f}" for u, t in zip(mean.sum(0), total)) + f"{1 - mean.sum() / total.sum():<8.2f}")
        s = spurious.reshape(3, 3, 4, len(BINS)).mean((0, 1))
        npred = n_pred.reshape(3, 3, len(BINS)).mean((0, 1))
        print("Unmatched PREDICTED possessions at IoU 0.5")
        for j, name in enumerate(("inside_ref_stoppage", "wrong_team", "same_team_bad_iou")):
            print(f"{name:<22}" + "".join(f"{v:<8.1f}" for v in s[j]) + f"{s[j].sum():<8.1f}")
        print(f"{'predicted':<22}" + "".join(f"{v:<8.1f}" for v in npred) + f"{npred.sum():<8.1f}")
        print(f"{'precision@0.5':<22}" + "".join(f"{1 - u / max(t, 1e-9):<8.2f}" for u, t in zip(s.sum(0), npred)) + f"{1 - s.sum() / npred.sum():<8.2f}")
    (ROOT / "misses.json").write_text(json.dumps(report, indent=1))


# -- timing of the three kinds of boundary ---------------------------------------------------

def latencies(ref_segs, pred_segs, blocks, window=3 * FPS):
    """{kind: (signed pred - ref in seconds for paired boundaries, reference count, predicted count)}."""
    out = {kind: [[], 0, 0] for kind in KINDS}
    for _, first, last in blocks:
        rb = boundaries([s for s in ref_segs if first <= s[0] <= last])
        pb = boundaries([s for s in pred_segs if first <= s[0] <= last])
        for kind in KINDS:
            out[kind][1] += sum(b[1] == kind for b in rb)
            out[kind][2] += sum(b[1] == kind for b in pb)
        for i, j in pair_boundaries(rb, pb, window, exact=False):
            out[rb[i][1]][0].append((pb[j][0] - rb[i][0]) / FPS)
    return out


def summary(values, n_ref, n_pred):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return dict(n_ref=n_ref, n_pred=n_pred, paired=0)
    q1, median, q3 = np.percentile(v, [25, 50, 75])
    return dict(n_ref=int(n_ref), n_pred=int(n_pred), paired=int(len(v)), found=round(len(v) / max(n_ref, 1), 3),
                median=round(float(median), 2), q1=round(float(q1), 2), q3=round(float(q3), 2), mean=round(float(v.mean()), 2),
                abs_median=round(float(np.median(np.abs(v))), 2), within_05=round(float((np.abs(v) <= 0.5).mean()), 3),
                within_1=round(float((np.abs(v) <= 1.0).mean()), 3), within_2=round(float((np.abs(v) <= 2.0).mean()), 3))


def show_latency(title, rows):
    print("\n" + title)
    print(f"{'':<28}{'ref':<6}{'pred':<6}{'found':<7}{'median':<8}{'IQR':<16}{'|d|med':<8}{'<=0.5s':<8}{'<=1s':<7}{'<=2s':<7}")
    for name, s in rows:
        if not s.get("paired"):
            print(f"{name:<28}{s['n_ref']:<6}{s['n_pred']:<6}-")
            continue
        print(f"{name:<28}{s['n_ref']:<6}{s['n_pred']:<6}{s['found']:<7.2f}{s['median']:<+8.2f}{'[%+.2f, %+.2f]' % (s['q1'], s['q3']):<16}"
              f"{s['abs_median']:<8.2f}{s['within_05']:<8.2f}{s['within_1']:<7.2f}{s['within_2']:<7.2f}")


def latency():
    ROOT.mkdir(parents=True, exist_ok=True)
    conds = recipe()
    report = {}
    for r in REFS:
        rows = []
        for kind in KINDS:
            for label, keep in [("all", lambda k: True)] + [(f"game {g}", lambda k, g=g: k.game == g) for g in GAMES] \
                    + [(m, lambda k, m=m: k.mode == m) for m in c.MODES]:
                values, n_ref, n_pred = [], 0, 0
                for k in conds:
                    if keep(k):
                        v, a, b = latencies(k.ref_segs[r], k.pred_segs, k.blocks)[kind]
                        values += v
                        n_ref += a
                        n_pred += b
                rows.append((f"{kind} {label}", summary(values, n_ref, n_pred)))
        show_latency(f"Predicted minus reference time (s), paired within 3 s, {r} reference, current recipe", rows)
        report[r] = {name: s for name, s in rows}
    (ROOT / "latency.json").write_text(json.dumps(report, indent=1))


# -- DFL native cross-validation (in-domain) -------------------------------------------------

def dfl():
    """The same oracles and timing on the out-of-fold DFL outputs (native possession labels)."""
    from . import v13_cv
    from .v6_experiment import as_match
    ROOT.mkdir(parents=True, exist_ok=True)
    rows, timing = [], {kind: [[], 0, 0] for kind in KINDS}
    by_mode = {m: {kind: [[], 0, 0] for kind in KINDS} for m in c.MODES}
    for index, held in enumerate(v13_cv.settings()["test_folds"]):
        for game in held:
            for mode in c.MODES:
                path = v13_cv.ROOT / f"fold{index}" / f"{game}_{mode}.npz"
                if not path.exists():
                    print("DFL missing", path, flush=True)
                    continue
                r = v13_cv.record(game, mode)
                saved = np.load(path)
                labels = r["full_labels"]
                blocks = [tuple(int(v) for v in b) for b in as_match(r).blocks]
                interpolated = c.interpolate(saved["combined"], saved["edges"], saved["frames"], blocks)
                codes = c.decode(interpolated, len(labels))[0]
                reference = np.where(labels >= 0, labels, 3).astype(np.int8)
                graded = c.graded_blocks(labels)
                ref_segs, pred_segs = c.segments(reference, graded), c.segments(np.where(reference < 3, codes, 3), graded)
                k = type("K", (), {})()
                k.n, k.blocks, k.references, k.pred, k.pred_segs, k.ref_segs = len(labels), graded, {"native": labels}, np.where(reference < 3, codes, 3).astype(np.int8), pred_segs, {"native": ref_segs}
                k.gt = {"native": c.spells_of(ref_segs)}
                row = dict(game=game, mode=mode, decoder=c.official(labels, codes)["spell_f1"],
                           snap_1s=score(k, snap(k, "native", FPS), "native"), snap_2s=score(k, snap(k, "native", 2 * FPS), "native"),
                           removal=lab27.oracle(pred_segs, k.gt["native"], lab27.MAX_FRAMES))
                for kind in KINDS:
                    row[f"snap_{kind}_2s"] = score(k, snap(k, "native", 2 * FPS, (kind,)), "native")
                rows.append(row)
                found = latencies(ref_segs, pred_segs, graded)
                for kind in KINDS:
                    for store in (timing, by_mode[mode]):
                        store[kind][0] += found[kind][0]
                        store[kind][1] += found[kind][1]
                        store[kind][2] += found[kind][2]
                print("DFL", json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()}), flush=True)
    names = [n for n in rows[0] if n not in ("game", "mode")]
    print("\nDFL native out-of-fold (frozen-architecture CV, decoder only): official spellF1")
    for name in names:
        print(f"{name:<18}" + "".join(f"{m}={np.mean([row[name] for row in rows if row['mode'] == m]):.3f}  " for m in c.MODES)
              + f"mean={np.mean([row[name] for row in rows]):.3f}")
    shown = [(f"{kind} all", summary(*timing[kind])) for kind in KINDS]
    shown += [(f"{kind} {m}", summary(*by_mode[m][kind])) for kind in KINDS for m in c.MODES]
    show_latency("Predicted minus native-label time (s), paired within 3 s, DFL out-of-fold", shown)
    (ROOT / "dfl.json").write_text(json.dumps(dict(rows=rows, timing={name: s for name, s in shown}), indent=1))


# -- ceilings: reference noise, semantics, resolution ----------------------------------------

def blank_runs(seen, minimum):
    edges = np.diff(np.r_[0, (~seen).astype(np.int8), 0])
    return [(a, z) for a, z in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if z - a >= minimum]


def repaired(reference, seen, minimum=3 * FPS):
    """Reference with the stoppages its converter missed: Metrica blanks the ball while play is
    stopped, so a blank of >= 3 s inside reference-live play is a stoppage (mostly the wait before
    a goal kick that the events tag as a lost ball, which the converter does not read as a restart)."""
    out = reference.copy()
    live = (reference >= 0) & (reference < 2)
    for a, z in blank_runs(seen, minimum):
        out[a:z][live[a:z]] = 2
    return out


def jittered(segs, sigma, rng, n):
    """Reference segments with every boundary moved by N(0, sigma) frames (order preserved)."""
    out = [list(s) for s in segs]
    for j in range(len(out) - 1):
        if out[j][1] != out[j + 1][0]:
            continue
        lower, upper = out[j][0] + 1, out[j + 1][1] - 1
        if lower > upper:
            continue
        new = int(min(max(out[j][1] + int(round(rng.normal(0, sigma))), lower), upper))
        out[j][1] = new
        out[j + 1][0] = new
    return c.to_codes([tuple(s) for s in out], n)


def insertion(k, r, causes=("wrong_team", "merged")):
    """Prediction with the missed reference possessions of the given causes painted in."""
    gt, pred = k.gt[r], c.spells_of(k.pred_segs)
    g, p = c.matching(gt, pred)[0.5]
    codes = k.pred.copy()
    for i, spell in enumerate(gt):
        if not g[i] and spell[1] - spell[0] <= lab27.MAX_FRAMES and classify(spell, pred, k.references[r], k.pred, p) in causes:
            codes[spell[0]:spell[1]] = spell[2]
    return codes


def ceilings():
    from .sources import external_match
    ROOT.mkdir(parents=True, exist_ok=True)
    conds = recipe()
    seen = {g: np.isfinite(external_match(g).ball).all(1) for g in GAMES}
    report = dict(per_iou={}, repaired={}, jitter={}, semantic={}, resolution={}, team_layer={})
    # 1. F1 at each IoU threshold: existence (0.3) against timing (the drop to 0.7)
    for r in REFS:
        rows = []
        for k in conds:
            m = c.matching(k.gt[r], c.spells_of(k.pred_segs))
            total = len(k.gt[r]) + len(c.spells_of(k.pred_segs))
            rows.append([2.0 * m[t][0].sum() / total for t in c.IOUS])
        rows = np.array(rows).reshape(3, 3, len(c.IOUS))
        report["per_iou"][r] = dict(ious=list(c.IOUS), per_game=rows.mean(1).tolist(), mean=rows.mean((0, 1)).tolist())
        print(f"F1 by IoU threshold {c.IOUS}, {r}: mean", rows.mean((0, 1)).round(3), "per game", rows.mean(1).round(3).tolist(), flush=True)
    # 2. reference stoppages the converter missed
    for r in REFS:
        rows = []
        for k in conds:
            fixed = repaired(k.references[r], seen[k.game])
            gt = c.spells_of(c.segments(np.where(fixed >= 0, fixed, 3), k.blocks))
            rows.append([score(k, k.codes, r), c.official(fixed, k.codes)["spell_f1"], score(k, k.pred, r), c.official(fixed, k.pred)["spell_f1"],
                         removal_oracle(k, r), lab27.oracle(k.pred_segs, gt, lab27.MAX_FRAMES),
                         float(((fixed == 2) & (k.references[r] != 2)).sum() / FPS), len(k.gt[r]) - len(gt)])
        rows = np.array(rows).reshape(3, 3, 8)
        report["repaired"][r] = dict(columns=["decoder", "decoder_repaired", "recipe", "recipe_repaired", "removal", "removal_repaired", "seconds_repaired", "spells_removed"],
                                     per_game=rows.mean(1).tolist(), per_mode=rows.mean(0).tolist())
        print(f"Repaired reference ({r}): decoder {rows[..., 0].mean():.3f} -> {rows[..., 1].mean():.3f}, recipe {rows[..., 2].mean():.3f} -> {rows[..., 3].mean():.3f}, "
              f"removal oracle {rows[..., 4].mean():.3f} -> {rows[..., 5].mean():.3f}; per game recipe", rows[..., 2].mean(1).round(3), "->", rows[..., 3].mean(1).round(3),
              "seconds", rows[:, 0, 6], flush=True)
    # 3. timing noise: the reference against itself with jittered boundaries
    for r in REFS:
        for sigma_s in (0.1, 0.2, 0.3, 0.5, 1.0):
            values = []
            for g in GAMES:
                k = conds[3 * (g - 1)]
                for seed in range(3):
                    codes = jittered(k.ref_segs[r], sigma_s * FPS, np.random.default_rng(seed), k.n)
                    values.append(score(k, codes, r))
            values = np.array(values).reshape(3, 3).mean(1)
            report["jitter"][f"{r}|{sigma_s}"] = values.tolist()
            print(f"Reference against itself with N(0, {sigma_s} s) boundary jitter, {r}: per game", values.round(3), "mean", values.mean().round(3), flush=True)
    # 4. semantics: one reading scored against the other; the resolution ceiling
    for g in GAMES:
        k = conds[3 * (g - 1)]
        cross = score(k, np.where(k.references["consistent"] >= 0, k.references["consistent"], 3).astype(np.int8), "alternative")
        report["semantic"][str(g)] = cross
        print(f"Game {g}: the consistent reference scored against the alternative one {cross:.3f} (a perfect model of the wrong reading); "
              f"perfect model at the equal prior {(1 + cross) / 2:.3f}", flush=True)
        for r in REFS:
            for limit in (1, 2, 3):
                segs = lab27.duration_rule(k.ref_segs[r], absorb_s=limit, island_s=limit, edge_s=limit)
                value = score(k, c.to_codes(segs, k.n), r)
                report["resolution"][f"{g}|{r}|{limit}"] = value
                print(f"   {r}: perfect except possessions shorter than {limit} s are not resolved: {value:.3f}", flush=True)
    # 5. the team layer: removal, insertion, both; reference stoppages followed by the same verifier
    for r in REFS:
        rows = []
        for k in conds:
            dead_segs = c.segments(ref_dead(k, r), k.blocks)
            verified, _ = v29.apply(k.model, dead_segs, k.p, k.edge, k.signals)
            inserted = insertion(k, r)
            rows.append([score(k, k.pred, r), removal_oracle(k, r), score(k, inserted, r),
                         lab27.oracle(c.segments(inserted, k.blocks), k.gt[r], lab27.MAX_FRAMES),
                         score(k, ref_dead(k, r), r), score(k, c.to_codes(verified, k.n), r), score(k, ref_team(k, r), r)])
        rows = np.array(rows).reshape(3, 3, 7)
        names = ["recipe", "removal", "insertion", "insertion+removal", "ref_dead", "ref_dead+verifier", "ref_team"]
        report["team_layer"][r] = dict(columns=names, per_game=rows.mean(1).tolist(), per_mode=rows.mean(0).tolist())
        print(f"Team layer ({r}):", {n: round(float(v), 3) for n, v in zip(names, rows.mean((0, 1)))}, flush=True)
        for g in GAMES:
            print("   game", g, {n: round(float(v), 3) for n, v in zip(names, rows[g - 1].mean(0))}, flush=True)
    (ROOT / "ceilings.json").write_text(json.dumps(report, indent=1))


# -- which part of the team layer is semantics? ----------------------------------------------

def robust_removal(k, passes=6):
    """Removal oracle that only takes removals that do not lose under either reading and gain
    under at least one: what better evidence could deliver without knowing the reading."""
    segs = k.pred_segs
    gts = [k.gt[r] for r in REFS]
    current = [c.spell_f1(gt, c.spells_of(segs)) for gt in gts]
    for _ in range(passes):
        gains = []
        for i, _, new in c.candidates(segs, lab27.MAX_FRAMES):
            spells = c.spells_of(c.relabel(segs, {i: new}))
            delta = [c.spell_f1(gt, spells) - f for gt, f in zip(gts, current)]
            if min(delta) > -1e-12 and max(delta) > 1e-12:
                gains.append((sum(delta), i, new))
        changes = {}
        for _, i, new in sorted(gains, reverse=True):
            if not any(i + d in changes for d in (-2, -1, 0, 1, 2)):
                changes[i] = new
        if not changes:
            break
        segs = c.relabel(segs, changes)
        current = [c.spell_f1(gt, c.spells_of(segs)) for gt in gts]
    return segs


def common_insertion(k):
    """Insert the missed possessions that exist in BOTH readings (same team, IoU >= 0.7)."""
    pred = c.spells_of(k.pred_segs)
    other = k.gt["alternative"]
    g, p = c.matching(k.gt["consistent"], pred)[0.5]
    codes = k.pred.copy()
    for i, spell in enumerate(k.gt["consistent"]):
        if g[i] or spell[1] - spell[0] > lab27.MAX_FRAMES:
            continue
        if classify(spell, pred, k.references["consistent"], k.pred, p) not in ("wrong_team", "merged"):
            continue
        shared = any(o[2] == spell[2] and (min(o[1], spell[1]) - max(o[0], spell[0])) / float(max(o[1], spell[1]) - min(o[0], spell[0])) >= 0.7
                     for o in other if o[1] > spell[0] and o[0] < spell[1])
        if shared:
            codes[spell[0]:spell[1]] = spell[2]
    return codes


def semantic():
    ROOT.mkdir(parents=True, exist_ok=True)
    conds = recipe()
    names = ["recipe", "robust_removal", "common_insertion", "robust_both", "robust_both+snap_2s", "reading_known"]
    rows = {r: [] for r in REFS}
    for k in conds:
        removed = c.to_codes(robust_removal(k), k.n)
        inserted = common_insertion(k)
        both = SimpleCondition(k, c.segments(inserted, k.blocks))
        robust_both = c.to_codes(robust_removal(both), k.n)
        for r in REFS:
            known = lab27.oracle(c.segments(insertion(k, r), k.blocks), k.gt[r], lab27.MAX_FRAMES)
            timed = snap(SimpleCondition(k, c.segments(robust_both, k.blocks)), r, 2 * FPS)
            rows[r].append([score(k, k.pred, r), score(k, removed, r), score(k, inserted, r), score(k, robust_both, r), score(k, timed, r), known])
        print("SEMANTIC", k.game, k.mode, {r: [round(v, 3) for v in rows[r][-1]] for r in REFS}, flush=True)
    report = {}
    for r in REFS:
        a = np.array(rows[r]).reshape(3, 3, len(names))
        report[r] = dict(columns=names, per_game=a.mean(1).tolist(), per_mode=a.mean(0).tolist())
        print(f"\n{r}: " + "  ".join(f"{n} {v:.3f}" for n, v in zip(names, a.mean((0, 1)))))
        for g in GAMES:
            print(f"   game {g}: " + "  ".join(f"{n} {v:.3f}" for n, v in zip(names, a[g - 1].mean(0))))
    (ROOT / "semantic.json").write_text(json.dumps(report, indent=1))


class SimpleCondition:
    """A copy of a condition with another predicted segmentation (for the oracles above)."""

    def __init__(self, k, segs):
        self.pred_segs, self.gt, self.n, self.blocks, self.ref_segs = segs, k.gt, k.n, k.blocks, k.ref_segs


if __name__ == "__main__":
    steps = dict(oracles=oracles, misses=misses, latency=latency, dfl=dfl, ceilings=ceilings, semantic=semantic)
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=list(steps) + ["all"])
    step = parser.parse_args().step
    for name in (steps if step == "all" else [step]):
        steps[name]()
