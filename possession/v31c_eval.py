"""V31c evaluation: networks trained on the compromise target M with a label-blind ball,
leave-one-Metrica-game-out, official metric, both readings, standard AND label-blind conditions.

Conditions of the held-out game g (networks of possession.v31c_train --held g, never trained on g):
    standard     clean / moderate / stress   (cached inputs of possession.v28d_data, seed 907 + g;
                                              the Metrica export artefact "ball missing <=> dead" is present)
    label-blind  interp / interpclean / blind3   (the reviewer's records, data/work/review_v30d/records,
                                              built by data/work/review_v30d/build_conditions.py)
    sight        the V30d transfer condition (possession.v30d_data: synthetic sightings on 75% of the
                 stoppages + moderate): P(seen | dead) / P(seen | live) = 0.34-0.40 / 0.62-0.67, the
                 ratio of the challenge match (0.52 / ~0.93); placement uses labels, evaluation only

Read-outs per base (a base = weighted blend of network ensembles' probabilities), all of
possession.v30d_eval (unchanged code): 25 Hz decoder, + verifier wX (possession.v29_lab procedure)
fitted on the base's own out-of-fold outputs on the STANDARD conditions of the two other games
and applied to every condition of the held-out one.

Bases: frozen (the submitted networks), M / M_seed10 (one seed), M2 (two seeds, eight networks),
M+frozen, M2+frozen (50/50 with the frozen ensemble), C2 / A2 (V30d references), Ab* (control:
alternative target on the same label-blind views).

Diagnostics per condition: share of predicted D starts within 2 frames of a ball-vanish event,
spellF1 against M itself, possessions per live minute, D share.

    .venv/bin/python -m possession.v31c_eval predict --target M --held 1 [--seed-offset 10] [--device cpu]
    .venv/bin/python -m possession.v31c_eval score --bases frozen M M_seed10 M2 M+frozen M2+frozen C2 A2
    .venv/bin/python -m possession.v31c_eval score --bases M2 --out score_part_M2.json     # one base per process (parallel)
    .venv/bin/python -m possession.v31c_eval merge --parts score_controls.json score_part_M2.json ...
    .venv/bin/python -m possession.v31c_eval show

No challenge ground truth is read.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from . import v27_core as c
from . import v29_lab as lab
from . import v30d_eval as e
from . import v31c_data, v31c_train
from .v28d_data import metrica_record

ROOT = Path("data/work/v31c")
REVIEW = Path("data/work/review_v30d")
GAMES = (1, 2, 3)
STANDARD = c.MODES
BLIND = ("interp", "interpclean", "blind3")
SIGHT = ("sight",)
MODES = STANDARD + BLIND + SIGHT
GROUPS = (("standard", STANDARD), ("blind", BLIND), ("sight", SIGHT))
REFS = ("consistent", "alternative")
WEIGHTS = (0.5, 1.0, 0.0)
SUBMISSION = ("frozen", "verifier_w0.50")
V30D_TAGS = ("C", "A", "C_seed10", "A_seed10")
BASES = {
    "frozen": {"frozen": 1.0},
    "M": {"M": 1.0}, "M_seed10": {"M_seed10": 1.0},
    "M2": {"M": 0.5, "M_seed10": 0.5},
    "M+frozen": {"M": 0.5, "frozen": 0.5},
    "M_seed10+frozen": {"M_seed10": 0.5, "frozen": 0.5},
    "M2+frozen": {"M": 0.25, "M_seed10": 0.25, "frozen": 0.5},
    "Ab": {"Ab": 1.0}, "Ab_seed10": {"Ab_seed10": 1.0}, "Ab2": {"Ab": 0.5, "Ab_seed10": 0.5},
    "Ab+frozen": {"Ab": 0.5, "frozen": 0.5},
    "Ab2+frozen": {"Ab": 0.25, "Ab_seed10": 0.25, "frozen": 0.5}, "Ab_seed10+frozen": {"Ab_seed10": 0.5, "frozen": 0.5},
    "M+Ab": {"M": 0.5, "Ab": 0.5}, "M+Ab+frozen": {"M": 0.25, "Ab": 0.25, "frozen": 0.5},
    "M15": {"M15": 1.0}, "M30": {"M30": 1.0},
    "C2": {"C": 0.5, "C_seed10": 0.5}, "A2": {"A": 0.5, "A_seed10": 0.5},
    "A2+frozen": {"A": 0.25, "A_seed10": 0.25, "frozen": 0.5},
}
TARGET_OF_TAG = {"M": "M", "Ab": "A", "M15": "M15", "M30": "M30"}


def tag_of(target, seed_offset=0):
    name = {v: k for k, v in TARGET_OF_TAG.items()}[target]
    return name + (f"_seed{seed_offset}" if seed_offset else "")


def output_path(tag, game, mode):
    if tag == "frozen" and mode in STANDARD:
        return Path(f"data/work/v16/game{game}_{mode}_final.npz")
    if tag == "frozen" or tag in V30D_TAGS:
        return (REVIEW / "outputs" if mode in BLIND else Path("data/work/v30d/outputs")) / tag / f"game{game}_{mode}.npz"
    return ROOT / "outputs" / tag / f"game{game}_{mode}.npz"


def input_record(game, mode):
    if mode in STANDARD:
        return metrica_record(game, mode)
    if mode in SIGHT:
        from .v30d_data import record
        return record(game, None)
    return joblib.load(REVIEW / "records" / f"game{game}_{mode}.joblib")


def ball_seen(game, mode):
    path = ROOT / "seen" / f"game{game}_{mode}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(input_record(game, mode)["ball_seen"], dtype=bool))
    return np.load(path)


def predict(target, held, device="cpu", seed_offset=0):
    import torch
    from .competitive_bundle import CompetitiveBundle
    torch.set_num_threads(4)
    tag = tag_of(target, seed_offset)
    selection = json.loads(v31c_train.selection_path(target, held, seed_offset).read_text())
    if selection["held_out_metrica_game"] != held:
        raise ValueError("Selection of another fold")
    bundle = None
    for mode in MODES:
        path = output_path(tag, held, mode)
        if path.exists():
            continue
        bundle = bundle or CompetitiveBundle(selection, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        r = input_record(held, mode)
        p, edges = bundle.probabilities(r)
        np.savez_compressed(path, probabilities=p, edges=edges, frames=r["frames"])
        print("V31C PREDICT", tag, held, mode, p.shape, flush=True)


# -- scoring -------------------------------------------------------------------------------------

def patched():
    """possession.v30d_eval with this module's conditions, paths and bases (this process only)."""
    e.MODES = MODES
    e.output_path = output_path
    e.BASES = dict(BASES)
    return e


def near(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), 10 ** 6)
    i = np.searchsorted(b, a)
    return np.minimum(np.abs(b[np.clip(i - 1, 0, len(b) - 1)] - a), np.abs(b[np.clip(i, 0, len(b) - 1)] - a))


def dstart_at_vanish(codes, reference, seen, k=2):
    """(share of predicted D starts within k frames of a ball-vanish event, number of predicted D starts)."""
    g = reference >= 0
    starts = np.flatnonzero((codes[1:] == 2) & (codes[:-1] != 2) & g[1:] & g[:-1]) + 1
    vanish = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
    return (float((near(starts, vanish) <= k).mean()) if len(starts) else 0.0), int(len(starts))


def groups(values):
    """Per-game means over the standard and over the label-blind conditions."""
    out = {}
    for name, modes in GROUPS:
        out[name] = [round(float(np.mean([values[(g, m)] for m in modes])), 4) for g in GAMES]
        out[name + "_mean"] = round(float(np.mean(out[name])), 4)
    out["per_condition"] = {f"{g}_{m}": round(float(values[(g, m)]), 4) for g in GAMES for m in MODES}
    return out


def score_base(base):
    ev = patched()
    conds = ev.conditions(base)
    table = ev.table_of([k for k in conds if k.mode in STANDARD])      # the verifier never sees the label-blind conditions
    readouts = {"decoder": {(k.game, k.mode): k.codes for k in conds}}
    removed = {}
    for w in WEIGHTS:
        models = {held: ev.fit_verifier(table, [g for g in GAMES if g != held], w) for held in GAMES}
        name = f"verifier_w{w:.2f}"
        readouts[name], removed[name] = {}, {}
        for k in conds:
            new, applied = lab.apply(models[k.game], k.segs, k.p, k.edge, None)
            readouts[name][(k.game, k.mode)] = c.to_codes(new, k.n)
            removed[name][(k.game, k.mode)] = len(applied)
    refs = {(k.game, k.mode): k.references for k in conds}
    blocks = {(k.game, k.mode): k.blocks for k in conds}
    target = {g: v31c_data.metrica_target(g) for g in GAMES}
    report = {}
    for name, codes in readouts.items():
        row = {r: groups({key: c.official(refs[key][r], v)["spell_f1"] for key, v in codes.items()}) for r in REFS}
        row["M"] = groups({key: c.official(target[key[0]], v)["spell_f1"] for key, v in codes.items()})
        stats = {key: c.duration_stats(v, blocks[key]) for key, v in codes.items()}
        for s in ("per_live_minute", "share_lt_3s", "dead_share"):
            row[s] = groups({key: v[s] for key, v in stats.items()})
        snap = {key: dstart_at_vanish(v, refs[key]["consistent"], ball_seen(*key)) for key, v in codes.items()}
        row["dstart_within2_of_vanish"] = {m: round(float(np.mean([snap[(g, m)][0] for g in GAMES])), 3) for m in MODES}
        row["d_starts"] = {m: round(float(np.mean([snap[(g, m)][1] for g in GAMES])), 1) for m in MODES}
        if name in removed:
            row["removals"] = groups(removed[name])
        report[name] = row
    return report


def verdict(row, bar):
    """Adoption bar versus the submission recipe (frozen + verifier w0.5)."""
    out = {}
    ok = True
    for r in REFS:
        d = np.array(row[r]["standard"]) - np.array(bar[r]["standard"])
        db = np.array(row[r]["blind"]) - np.array(bar[r]["blind"])
        ds = np.array(row[r]["sight"]) - np.array(bar[r]["sight"])
        out[r] = dict(standard_delta=d.round(4).tolist(), standard_mean_delta=round(float(d.mean()), 4),
                      blind_delta=db.round(4).tolist(), blind_mean_delta=round(float(db.mean()), 4),
                      sight_delta=ds.round(4).tolist(), sight_mean_delta=round(float(ds.mean()), 4))
        ok &= bool((d > 0).all() and d.mean() >= 0.015 and db.mean() > 0)
    out["equal_prior_standard_delta"] = round(float(np.mean([out[r]["standard_mean_delta"] for r in REFS])), 4)
    out["equal_prior_blind_delta"] = round(float(np.mean([out[r]["blind_mean_delta"] for r in REFS])), 4)
    out["equal_prior_sight_delta"] = round(float(np.mean([out[r]["sight_mean_delta"] for r in REFS])), 4)
    out["meets_bar_this_seed_set"] = ok
    return out


def show(report):
    bar = report["frozen"]["verifier_w0.50"]
    for r in REFS + ("M",):
        print(f"\n== reference: {r} == held-out spellF1 per game g1 g2 g3 (mean) | STANDARD clean/moderate/stress || LABEL-BLIND interp/interpclean/blind3 || SIGHT"
              + (" || delta vs submission recipe: standard per game (mean) | blind (mean) | sight (mean)" if r != "M" else ""))
        for base, rows in report.items():
            for name, row in rows.items():
                v = row[r]
                line = f"{base:<16}{name:<15}" + " ".join(f"{x:.4f}" for x in v["standard"]) + f" ({v['standard_mean']:.4f}) || " \
                       + " ".join(f"{x:.4f}" for x in v["blind"]) + f" ({v['blind_mean']:.4f}) || " \
                       + " ".join(f"{x:.4f}" for x in v["sight"]) + f" ({v['sight_mean']:.4f})"
                if r != "M":
                    d = np.array(v["standard"]) - np.array(bar[r]["standard"])
                    db = np.array(v["blind"]) - np.array(bar[r]["blind"])
                    ds = np.array(v["sight"]) - np.array(bar[r]["sight"])
                    line += " || " + " ".join(f"{x:+.3f}" for x in d) + f" ({d.mean():+.4f}) | " + " ".join(f"{x:+.3f}" for x in db) + f" ({db.mean():+.4f})" \
                            + " | " + " ".join(f"{x:+.3f}" for x in ds) + f" ({ds.mean():+.4f})"
                print(line)
    print("\n== equal-prior mean (average of the two references), standard | blind | sight; /min, D share (standard, per game); predicted D starts within 2 frames of a ball vanish per condition; verdict")
    for base, rows in report.items():
        for name, row in rows.items():
            eq = [np.mean([row[r][g + "_mean"] for r in REFS]) for g, _ in GROUPS]
            v = verdict(row, bar)
            print(f"{base:<16}{name:<15}{eq[0]:.4f} | {eq[1]:.4f} | {eq[2]:.4f}   /min {row['per_live_minute']['standard']} D {row['dead_share']['standard']}"
                  f"   Dstart@vanish {row['dstart_within2_of_vanish']}   BAR {'MEETS' if v['meets_bar_this_seed_set'] else 'no'}"
                  f" (eq-prior delta {v['equal_prior_standard_delta']:+.4f} | blind {v['equal_prior_blind_delta']:+.4f} | sight {v['equal_prior_sight_delta']:+.4f})")


def score(bases, out="score.json"):
    path = ROOT / out
    report = json.loads(path.read_text()) if path.exists() else {}
    for base in bases:
        report[base] = score_base(base)
        print("SCORED", base, {r: report[base]["decoder"][r]["standard"] for r in REFS}, flush=True)
    bar = report.get("frozen", {}).get("verifier_w0.50")
    if bar:
        for rows in report.values():
            for row in rows.values():
                row["versus_submission_recipe"] = verdict(row, bar)
    path.write_text(json.dumps(report, indent=1))
    if bar:
        show(report)
    return report


def snap(bases, out="snap.json"):
    """Where the decoder's stoppages sit (reviewer's metrica_snap quantities), per group of conditions:
    frame IoU of D, share of reference D starts / ends with a predicted one within 1 s, team accuracy
    on reference-live frames, predicted D starts within 2 frames of a ball vanish."""
    ev = patched()
    report = {}
    for base in bases:
        rows = {}
        for k in ev.conditions(base):
            seen = ball_seen(k.game, k.mode)
            y, ref = k.codes, k.references["consistent"]
            g = ref >= 0
            edge = lambda v, into: np.flatnonzero(((v[1:] == 2) & (v[:-1] != 2) if into else (v[1:] != 2) & (v[:-1] == 2)) & g[1:] & g[:-1]) + 1
            live = g & (ref < 2)
            rows[(k.game, k.mode)] = dict(
                d_iou=float(((y == 2) & (ref == 2) & g).sum() / (((y == 2) | (ref == 2)) & g).sum()),
                ref_dstart_found_1s=float((near(edge(ref, True), edge(y, True)) <= 25).mean()),
                ref_dend_found_1s=float((near(edge(ref, False), edge(y, False)) <= 25).mean()),
                team_accuracy_live=float((y[live] == ref[live]).mean()),
                dstart_within2_of_vanish=dstart_at_vanish(y, ref, seen)[0])
        report[base] = {name: {q: round(float(np.mean([rows[(g_, m)][q] for g_ in GAMES for m in modes])), 4)
                               for q in next(iter(rows.values()))} for name, modes in GROUPS}
        print(f"SNAP {base:<12}" + " || ".join(f"{name}: " + " ".join(f"{q}={v:.3f}" for q, v in row.items()) for name, row in report[base].items()), flush=True)
    (ROOT / out).write_text(json.dumps(report, indent=1))


def merge(parts, out="score.json"):
    """Join score files written by parallel `score --bases X --out score_part_X.json` runs and add the verdicts."""
    report = {}
    for name in parts:
        report.update(json.loads((ROOT / name).read_text()))
    bar = report["frozen"]["verifier_w0.50"]
    for rows in report.values():
        for row in rows.values():
            row["versus_submission_recipe"] = verdict(row, bar)
    (ROOT / out).write_text(json.dumps(report, indent=1))
    show(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["predict", "score", "show", "merge", "snap"])
    parser.add_argument("--parts", nargs="+", default=[])
    parser.add_argument("--target", choices=list(v31c_train.TARGETS))
    parser.add_argument("--held", type=int, choices=[1, 2, 3])
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--bases", nargs="+", default=["frozen", "M", "M_seed10", "M2", "M+frozen", "M2+frozen", "C2", "A2"])
    parser.add_argument("--out", default="score.json")
    args = parser.parse_args()
    if args.step == "predict":
        predict(args.target, args.held, args.device, args.seed_offset)
    elif args.step == "snap":
        snap(args.bases, args.out if args.out != "score.json" else "snap.json")
    elif args.step == "merge":
        merge(args.parts, args.out)
    elif args.step == "show":
        show(json.loads((ROOT / args.out).read_text()))
    else:
        score(args.bases, args.out)
