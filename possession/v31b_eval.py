"""V31b step 3: external test on the three open Metrica games (never used to train the networks).

Conditions per game:
    clean / moderate / stress        cached inputs of possession.v28d_data (the standard yardstick)
    interp / interpclean / blind3    the reviewer's label-blind ball conditions
                                     (data/work/review_v30d/build_conditions.py; cached records)
    sight                            synthetic dead-ball sightings (possession.v30d_data)

References: consistent, alternative (the two readings, 50/50) and, as a diagnostic, the compromise
labelling M itself (alternative with T-X-T possessions shorter than 2 s absorbed).

A base is a weighted probability blend of network ensembles. Read-outs per base:
    decoder          unchanged 25 Hz Viterbi (possession.v27_core.decode)
    verifier_wX      possession.v29_lab procedure (V27 features, gain regressor, boundary 0),
                     target X * gain_consistent + (1 - X) * gain_alternative, fitted on the
                     base's own outputs on the STANDARD conditions of the two other games and
                     applied to every condition of the held-out game.
    verifier_M       the same regressor with ONE target: the gain under the compromise labelling M.

    .venv/bin/python -m possession.v31b_eval predict --tag m2 [--device cpu]     # 21 records
    .venv/bin/python -m possession.v31b_eval predict --tag v28d_native --modes interp interpclean blind3
    .venv/bin/python -m possession.v31b_eval score                               # data/work/v31b/score.json
    .venv/bin/python -m possession.v31b_eval score --bases m2 --out parts/m2.json   (parallel) then
    .venv/bin/python -m possession.v31b_eval merge --files parts/m2.json ... --out score.json
    .venv/bin/python -m possession.v31b_eval bar                                 # adoption-bar table

No challenge ground truth is read.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
from . import v27_core as c
from . import v29_lab as lab
from . import v30d_eval as e30
from . import v30d_data
from .v28d_data import metrica_record

ROOT = Path("data/work/v31b")
GAMES = (1, 2, 3)
STANDARD = c.MODES
BLIND = ("interp", "interpclean", "blind3")
MODES = STANDARD + BLIND + ("sight",)
REFS = ("consistent", "alternative")
TARGET_ABSORB_S = 2.0                                            # the compromise labelling "M" (diagnostic third reference)
SCORED = REFS + ("M",)
WEIGHTS = (0.5, 1.0, 0.0)
REVIEW = Path("data/work/review_v30d")
# cached outputs of earlier rounds (read-only): tag -> {mode group: path pattern}
EXISTING = {
    "frozen": dict(standard="data/work/v16/game{g}_{m}_final.npz", blind=str(REVIEW / "outputs/frozen/game{g}_{m}.npz"),
                   sight="data/work/v30d/outputs/frozen/game{g}_{m}.npz"),
    "v28d": dict(standard="data/work/v28d/outputs/retrained/game{g}_{m}.npz", blind=str(REVIEW / "outputs/v28d/game{g}_{m}.npz"),
                 sight="data/work/v30d/outputs/v28d/game{g}_{m}.npz"),
    "v28d_native": dict(standard="data/work/v28d/outputs/retrained_native/game{g}_{m}.npz", sight="data/work/v30d/outputs/v28d_native/game{g}_{m}.npz"),
}
SELECTIONS = {"v28d_native": "data/work/v28d/selection_native.json", "frozen": "docs/FINAL_SELECTION.json", "v28d": "data/work/v28d/selection.json"}
BASES = {
    "frozen": {"frozen": 1.0},                                   # the submitted networks (DFL native labels)
    "v28d": {"v28d": 1.0},                                       # control: DFL labels relabelled with rule "C"
    "v28d_blend": {"frozen": 0.5, "v28d": 0.5},                  # control: V28d / V29 "blend" base
    "native_blend": {"frozen": 0.5, "v28d_native": 0.5},         # control: same second stage WITHOUT the new target
}
for _v in ("m2", "m3"):
    BASES.update({
        _v: {_v: 1.0}, f"{_v}_seed10": {f"{_v}_seed10": 1.0}, f"{_v}x2": {_v: 0.5, f"{_v}_seed10": 0.5},
        f"{_v}_blend": {"frozen": 0.5, _v: 0.5}, f"{_v}_seed10_blend": {"frozen": 0.5, f"{_v}_seed10": 0.5},
        f"{_v}x2_blend": {"frozen": 0.5, _v: 0.25, f"{_v}_seed10": 0.25}})
BAR = ("frozen", "verifier_w0.50")                               # the submission recipe


def group_of(mode):
    return "standard" if mode in STANDARD else "blind" if mode in BLIND else "sight"


def output_path(tag, game, mode):
    pattern = EXISTING.get(tag, {}).get(group_of(mode))
    if pattern:
        return Path(pattern.format(g=game, m=mode))
    return ROOT / "outputs" / tag / f"game{game}_{mode}.npz"


def input_record(game, mode):
    if mode in STANDARD:
        return metrica_record(game, mode)
    if mode == "sight":
        return v30d_data.record(game, None)
    return joblib.load(REVIEW / "records" / f"game{game}_{mode}.joblib")


def selection_of(tag):
    if tag in SELECTIONS:
        return Path(SELECTIONS[tag])
    from . import v31b_train
    variant, _, seed = tag.partition("_seed")
    return v31b_train.selection_path(variant, int(seed or 0))


def predict(tag, device="cpu", modes=MODES, games=GAMES, threads=4):
    import torch
    from .competitive_bundle import CompetitiveBundle
    torch.set_num_threads(threads)
    bundle = None
    for game in games:
        for mode in modes:
            path = output_path(tag, game, mode)
            if path.exists():
                continue
            if ROOT not in path.parents:
                raise RuntimeError(f"{path} is outside {ROOT}")
            bundle = bundle or CompetitiveBundle(json.loads(selection_of(tag).read_text()), device)
            path.parent.mkdir(parents=True, exist_ok=True)
            r = input_record(game, mode)
            p, e = bundle.probabilities(r)
            temporary = path.with_suffix(".part.npz")
            np.savez_compressed(temporary, probabilities=p, edges=e, frames=r["frames"])
            temporary.replace(path)
            print("V31B PREDICT", tag, game, mode, p.shape, flush=True)


# -- conditions and scoring ----------------------------------------------------------------------

def compromise_reference(meta):
    """M: the alternative reference with T-X-T possessions shorter than 2 s absorbed (a-priori constant)."""
    from .v31b_relabel import compromise
    return compromise(meta["alternative"], TARGET_ABSORB_S)


def conditions(base, modes=MODES):
    parts = BASES[base]
    out = []
    for game in GAMES:
        meta = np.load(c.ROOT / f"metrica_{game}.npz")
        blocks = [tuple(int(v) for v in b) for b in meta["blocks"]]
        n = len(meta["original"])
        target = compromise_reference(meta)
        for mode in modes:
            saved = [(np.load(output_path(tag, game, mode)), w) for tag, w in parts.items()]
            frames = saved[0][0]["frames"]
            if any(not np.array_equal(frames, s["frames"]) for s, _ in saved):
                raise ValueError("Different timelines")
            p = sum(w * s["probabilities"].astype(np.float64) for s, w in saved)
            e = sum(w * s["edges"].astype(np.float64) for s, w in saved)
            k = SimpleNamespace(game=game, mode=mode, blocks=blocks, n=n, references={r: meta[r] for r in c.REFERENCES},
                                interpolated=c.interpolate(p, e, frames, blocks), scored=dict({r: meta[r] for r in REFS}, M=target))
            k.codes = c.decode(k.interpolated, k.n)[0]
            k.segs = c.segments(k.codes, k.blocks)
            k.p, k.edge = c.dense(k.interpolated, k.n)
            k.gt = {r: c.spells_of(c.segments(k.references[r], k.blocks)) for r in c.REFERENCES}
            out.append(k)
    return out


def target_models(conds):
    """Verifier aimed at M: the same V27 gain regressor, target = exact spellF1 gain of a removal
    under the compromise labelling M (one target instead of the 0.5/0.5 mixture of the two readings).
    Leave-one-game-out on the standard conditions, like the mixture verifiers."""
    from . import v28a_core as k28
    x, d = {}, {}
    for k in conds:
        if k.mode not in STANDARD:
            continue
        cands = c.candidates(k.segs, lab.MAX_FRAMES)
        gt = c.spells_of(c.segments(np.where(k.scored["M"] >= 0, k.scored["M"], 3).astype(np.int8), k.blocks))
        base = c.spell_f1(gt, c.spells_of(k.segs))
        x[(k.game, k.mode)] = lab.features(k.segs, cands, k.p, k.edge, None)
        d[(k.game, k.mode)] = np.array([c.spell_f1(gt, c.spells_of(c.relabel(k.segs, {i: new}))) - base for i, _, new in cands])
    models = {}
    for held in GAMES:
        keys = [(g, m) for g in GAMES if g != held for m in STANDARD]
        models[held] = k28.Model(**lab.FAMILIES["v27"]).fit(np.concatenate([x[key] for key in keys]), np.concatenate([d[key] for key in keys]))
    return models


def per_game(values, modes):
    return [round(float(np.mean([values[(g, m)] for m in modes])), 4) for g in GAMES]


def score_base(base, modes=MODES):
    conds = conditions(base, modes)
    table = e30.table_of([k for k in conds if k.mode in STANDARD])       # verifier sees standard conditions only
    readouts = {"decoder": {(k.game, k.mode): k.codes for k in conds}}
    removed = {}
    for w in WEIGHTS + ("M",):
        if w == "M":
            models, name = target_models(conds), "verifier_M"
        else:
            models = {held: e30.fit_verifier(table, [g for g in GAMES if g != held], w) for held in GAMES}
            name = f"verifier_w{w:.2f}"
        readouts[name], removed[name] = {}, {}
        for k in conds:
            new, applied = lab.apply(models[k.game], k.segs, k.p, k.edge, None)
            readouts[name][(k.game, k.mode)] = c.to_codes(new, k.n)
            removed[name][(k.game, k.mode)] = len(applied)
    blocks = {(k.game, k.mode): k.blocks for k in conds}
    refs = {(k.game, k.mode): k.scored for k in conds}
    report = {}
    groups = {"standard": STANDARD, "blind": tuple(m for m in BLIND if m in modes), "sight": tuple(m for m in ("sight",) if m in modes)}
    for name, codes in readouts.items():
        row = {}
        for r in SCORED:
            f1 = {key: c.official(refs[key][r], v)["spell_f1"] for key, v in codes.items()}
            row[r] = dict(per_condition={f"{g}_{m}": round(f1[(g, m)], 4) for g in GAMES for m in modes})
            for group, members in groups.items():
                if members:
                    row[r][group] = per_game(f1, members)
            row[r]["mean"] = round(float(np.mean(row[r]["standard"])), 4)
        stats = {key: c.duration_stats(v, blocks[key]) for key, v in codes.items()}
        for s in ("per_live_minute", "share_lt_3s", "dead_share"):
            for group, members in groups.items():
                if members:
                    row[f"{s}_{group}"] = per_game({key: v[s] for key, v in stats.items()}, members)
        if name in removed:
            row["removals_standard"] = per_game(removed[name], STANDARD)
        report[name] = row
    return report


def score(bases, out="score.json"):
    path = ROOT / out
    report = json.loads(path.read_text()) if path.exists() else {}
    for base in bases:
        report[base] = score_base(base)
        print("SCORED", base, {r: report[base]["decoder"][r]["standard"] for r in REFS}, flush=True)
        path.write_text(json.dumps(report, indent=1))
    show(report, bases)
    return report


def merge(files, out="score.json"):
    """Scores computed by parallel processes (one file each) -> one report."""
    report = {}
    for name in files:
        report.update(json.loads((ROOT / name).read_text()))
    (ROOT / out).write_text(json.dumps(report, indent=1))
    print("MERGED", list(report), flush=True)
    return report


def show(report, bases=None):
    for r in SCORED:
        print(f"\n== reference: {r} == held-out spellF1 per game g1 g2 g3 (mean): standard (clean/moderate/stress) | label-blind ball (interp/interpclean/blind3) | sight")
        for base in bases or report:
            for name, row in report[base].items():
                v = row[r]
                cell = lambda x: " ".join(f"{y:.3f}" for y in x) + f" ({np.mean(x):.3f})"
                print(f"{base:<17}{name:<15}" + " | ".join(cell(v[g]) for g in ("standard", "blind", "sight") if g in v)
                      + (f"   /min {row['per_live_minute_standard']} <3s {row['share_lt_3s_standard']}" if r == REFS[0] else ""))


def snap(bases, modes=("moderate", "interp", "interpclean", "blind3", "sight")):
    """D-start diagnostic of the reviewer (data/work/review_v30d/metrica_snap.py): how often a predicted
    stoppage starts within 2 frames of a ball-vanish event, and how well reference stoppages are found."""
    def near(a, b):
        if len(a) == 0 or len(b) == 0:
            return np.full(len(a), 10 ** 6)
        i = np.searchsorted(b, a)
        return np.minimum(np.abs(b[np.clip(i - 1, 0, len(b) - 1)] - a), np.abs(b[np.clip(i, 0, len(b) - 1)] - a))
    rows = []
    for base in bases:
        for k in conditions(base, modes):
            seen = input_record(k.game, k.mode)["ball_seen"]
            vanish = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
            y, ref = k.codes, k.references["consistent"]
            g = ref >= 0
            starts = np.flatnonzero((y[1:] == 2) & (y[:-1] != 2) & g[1:] & g[:-1]) + 1
            ref_starts = np.flatnonzero((ref[1:] == 2) & (ref[:-1] != 2) & g[1:] & g[:-1]) + 1
            row = dict(base=base, game=k.game, mode=k.mode, d_starts=int(len(starts)), reference_d_starts=int(len(ref_starts)),
                       pred_start_within2_of_vanish=round(float(np.mean(near(starts, vanish) <= 2)), 3),
                       ref_start_found_within25=round(float(np.mean(near(ref_starts, starts) <= 25)), 3),
                       d_iou=round(float(((y == 2) & (ref == 2) & g).sum() / (((y == 2) | (ref == 2)) & g).sum()), 3),
                       seen_given_reference_dead=round(float(seen[ref == 2].mean()), 3))
            rows.append(row)
            print("SNAP", json.dumps(row), flush=True)
    (ROOT / "snap.json").write_text(json.dumps(rows, indent=1))
    for base in bases:
        for mode in modes:
            sel = [r for r in rows if r["base"] == base and r["mode"] == mode]
            print(f"{base:<14}{mode:<12}" + " ".join(f"{key} {np.mean([r[key] for r in sel]):.3f}" for key in
                  ("pred_start_within2_of_vanish", "ref_start_found_within25", "d_iou", "seen_given_reference_dead")))


def bar(out="score.json", readout="verifier_w0.50", bases=None):
    """Every base's read-out against the SUBMISSION RECIPE (frozen + verifier w0.5), per game."""
    report = json.loads((ROOT / out).read_text())
    reference = report[BAR[0]][BAR[1]]
    rows = []
    for base in bases or [b for b in report if b != BAR[0]]:
        for name in (readout, "decoder") if readout != "decoder" else ("decoder",):
            row = dict(base=base, readout=name)
            ok = True
            for r in REFS:
                v = report[base][name][r]
                d = {g: (np.array(v[g]) - np.array(reference[r][g])).round(4) for g in ("standard", "blind", "sight") if g in v}
                blind_conditions = np.array([[v["per_condition"][f"{g}_{m}"] - reference[r]["per_condition"][f"{g}_{m}"] for g in GAMES] for m in BLIND])
                row[r] = dict(per_game=v["standard"], mean=v["mean"], gain_per_game=d["standard"].tolist(), gain_mean=round(float(d["standard"].mean()), 4),
                              blind_gain_per_game=d["blind"].tolist(), blind_gain_mean=round(float(d["blind"].mean()), 4),
                              blind_gain_per_condition={m: blind_conditions[i].round(4).tolist() for i, m in enumerate(BLIND)},
                              sight_gain_per_game=d["sight"].tolist())
                ok &= bool((d["standard"] > 0).all() and d["standard"].mean() >= 0.015 and (d["blind"] > 0).all())
            row["equal_prior_mean"] = round(float(np.mean([row[r]["mean"] for r in REFS])), 4)
            row["equal_prior_gain"] = round(float(np.mean([row[r]["gain_mean"] for r in REFS])), 4)
            row["meets_bar_this_run"] = ok
            rows.append(row)
    (ROOT / f"bar_{readout}.json").write_text(json.dumps(dict(bar=dict(base=BAR[0], readout=BAR[1], **{r: reference[r] for r in REFS}), rows=rows), indent=1))
    print("BAR (submission recipe) " + " | ".join(f"{r} {reference[r]['standard']} ({reference[r]['mean']:.4f})" for r in REFS))
    for row in rows:
        fmt = lambda x: " ".join(f"{y:+.3f}" for y in x)
        print(f"{row['base']:<17}{row['readout']:<15}" + " | ".join(
            f"{r[:4]} {row[r]['per_game']} gain {fmt(row[r]['gain_per_game'])} ({row[r]['gain_mean']:+.4f}) blind {fmt(row[r]['blind_gain_per_game'])} ({row[r]['blind_gain_mean']:+.4f})"
            for r in REFS) + f" | E {row['equal_prior_gain']:+.4f} | {'MEETS' if row['meets_bar_this_run'] else 'no'}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["predict", "score", "show", "bar", "merge", "snap"])
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--tag")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--games", nargs="+", type=int, default=list(GAMES))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bases", nargs="+", default=list(BASES))
    parser.add_argument("--readout", default="verifier_w0.50")
    parser.add_argument("--out", default="score.json")
    args = parser.parse_args()
    if args.step == "predict":
        predict(args.tag, args.device, tuple(args.modes), tuple(args.games), args.threads)
    elif args.step == "score":
        score(args.bases, args.out)
    elif args.step == "snap":
        snap(args.bases)
    elif args.step == "merge":
        merge(args.files, args.out)
    elif args.step == "show":
        show(json.loads((ROOT / args.out).read_text()))
    else:
        bar(args.out, args.readout)
