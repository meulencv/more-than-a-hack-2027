"""V30d evaluation: in-domain networks, leave-one-Metrica-game-out, official metric.

For held-out game g the networks of possession.v30d_train --held g (never trained on g) are
run on g's three evaluation conditions (the byte-identical cached inputs of possession.v28d_data:
clean / moderate / stress, seed 907 + g) and on the extra "sight" condition of
possession.v30d_data (synthetic dead-ball sightings; a transfer check for the data artefact
"ball seen => live" of the open Metrica games, which the challenge match does not share).

Read-outs per base (a base = a weighted blend of network ensembles' probabilities):
    decoder          unchanged 25 Hz Viterbi (possession.v27_core.decode)
    + verifier wX    possession.v29_lab procedure (V27 feature set, gain regressor, boundary 0),
                     target = X * gain_consistent + (1 - X) * gain_alternative, fitted on the
                     base's own outputs on the two other games and applied to the held-out one
                     (those outputs are themselves out-of-fold: every game is predicted by the
                     networks that did not train on it).

    .venv/bin/python -m possession.v30d_eval predict --target C --held 1 [--device cpu]
    .venv/bin/python -m possession.v30d_eval baselines            # frozen / V28d on the sight records
    .venv/bin/python -m possession.v30d_eval score                # data/work/v30d/score.json + table
    .venv/bin/python -m possession.v30d_eval score --bases C A CA C25A75 --suffix _seed10 --out score_seed10.json
    .venv/bin/python -m possession.v30d_eval score --bases C2 A2 CA2 C25A75_2 --out score_two_seeds.json
    .venv/bin/python -m possession.v30d_eval summary              # data/work/v30d/summary.json

No challenge ground truth is read.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import v27_core as c
from . import v28a_core as k28
from . import v28a_lab as lab28
from . import v29_lab as lab
from . import v30d_data, v30d_train
from .v28d_data import metrica_record

ROOT = Path("data/work/v30d")
GAMES = (1, 2, 3)
MODES = c.MODES + ("sight",)
REFS = ("consistent", "alternative")
WEIGHTS = (1.0, 0.5, 0.0)                      # P(consistent) of the verifier target
EXISTING = {"frozen": "data/work/v16/game{g}_{m}_final.npz", "v28d": "data/work/v28d/outputs/retrained/game{g}_{m}.npz",
            "v28d_native": "data/work/v28d/outputs/retrained_native/game{g}_{m}.npz"}
BASELINE_SELECTIONS = {"frozen": "docs/FINAL_SELECTION.json", "v28d": "data/work/v28d/selection.json",
                       "v28d_native": "data/work/v28d/selection_native.json"}
BASES = {
    "frozen": {"frozen": 1.0},                           # DFL native labels (the submitted networks)
    "v28d": {"v28d": 1.0},                               # DFL controlled-possession labels
    "v28d_native": {"v28d_native": 1.0},                 # DFL native labels, retrained (noise control)
    "v28d_blend": {"frozen": 0.5, "v28d": 0.5},          # the V28d / V29 "blend" base
    "C": {"C": 1.0},
    "A": {"A": 1.0},
    "CA": {"C": 0.5, "A": 0.5},
    "C25A75": {"C": 0.25, "A": 0.75},
    # two-seed ensembles (frozen seeds and seeds + 10): eight networks per target
    "C2": {"C": 0.5, "C_seed10": 0.5},
    "A2": {"A": 0.5, "A_seed10": 0.5},
    "CA2": {"C": 0.25, "C_seed10": 0.25, "A": 0.25, "A_seed10": 0.25},
    "C25A75_2": {"C": 0.125, "C_seed10": 0.125, "A": 0.375, "A_seed10": 0.375},
    "C+v28d": {"C": 0.5, "v28d": 0.5},
    "A+frozen": {"A": 0.5, "frozen": 0.5},
    "all4": {"C": 0.25, "A": 0.25, "frozen": 0.25, "v28d": 0.25},
}


def output_path(tag, game, mode):
    if tag in EXISTING and mode != "sight":
        return Path(EXISTING[tag].format(g=game, m=mode))
    return ROOT / "outputs" / tag / f"game{game}_{mode}.npz"


def input_record(game, mode):
    return v30d_data.record(game, None) if mode == "sight" else metrica_record(game, mode)


def run_bundle(selection, device, pairs, tag):
    import torch
    from .competitive_bundle import CompetitiveBundle
    torch.set_num_threads(4)
    bundle = CompetitiveBundle(json.loads(Path(selection).read_text()), device)
    for game, mode in pairs:
        path = output_path(tag, game, mode)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        r = input_record(game, mode)
        p, e = bundle.probabilities(r)
        np.savez_compressed(path, probabilities=p, edges=e, frames=r["frames"])
        print("PREDICT", tag, game, mode, p.shape, flush=True)


def predict(target, held, device="cpu", seed_offset=0, repeat=v30d_train.METRICA_REPEAT):
    tag = target + (f"_seed{seed_offset}" if seed_offset else "") + (f"_r{repeat}" if repeat != v30d_train.METRICA_REPEAT else "")
    run_bundle(v30d_train.selection_path(target, held, seed_offset, repeat), device, [(held, m) for m in MODES], tag)


def baselines(device="cpu"):
    for tag, selection in BASELINE_SELECTIONS.items():
        run_bundle(selection, device, [(g, "sight") for g in GAMES], tag)


# -- conditions and the verifier -----------------------------------------------------------------

def conditions(base, suffix=""):
    """Every (game, mode) of one base: blended probabilities, decode, references."""
    parts = {(tag + suffix if tag in ("C", "A") else tag): w for tag, w in BASES[base].items()}
    out = []
    for game in GAMES:
        meta = np.load(c.ROOT / f"metrica_{game}.npz")
        blocks = [tuple(int(v) for v in b) for b in meta["blocks"]]
        n = len(meta["original"])
        for mode in MODES:
            saved = [(np.load(output_path(tag, game, mode)), w) for tag, w in parts.items()]
            frames = saved[0][0]["frames"]
            if any(not np.array_equal(frames, s["frames"]) for s, _ in saved):
                raise ValueError("Different timelines")
            p = sum(w * s["probabilities"].astype(np.float64) for s, w in saved)
            e = sum(w * s["edges"].astype(np.float64) for s, w in saved)
            k = SimpleNamespace(game=game, mode=mode, blocks=blocks, n=n, references={r: meta[r] for r in c.REFERENCES},
                                interpolated=c.interpolate(p, e, frames, blocks))
            k.codes = c.decode(k.interpolated, k.n)[0]
            k.segs = c.segments(k.codes, k.blocks)
            k.p, k.edge = c.dense(k.interpolated, k.n)
            k.gt = {r: c.spells_of(c.segments(k.references[r], k.blocks)) for r in c.REFERENCES}
            out.append(k)
    return out


def table_of(conds):
    """Candidate removals of the three standard conditions: features, exact gains, purity gaps."""
    table = {}
    for k in conds:
        if k.mode == "sight":
            continue
        cands = c.candidates(k.segs, lab.MAX_FRAMES)
        table[(k.game, k.mode)] = (lab.features(k.segs, cands, k.p, k.edge, None), lab28.exact_deltas(k.segs, cands, k.gt),
                                   lab28.purity_gaps(k.segs, cands, k.references))
    return table


def hedged(values, w):
    return w * values[:, c.REFERENCES.index("consistent")] + (1.0 - w) * values[:, c.REFERENCES.index("alternative")]


def fit_verifier(table, games, w):
    keys = [(g, m) for g in games for m in c.MODES]
    x = np.concatenate([table[key][0] for key in keys])
    d = np.concatenate([table[key][1] for key in keys])
    u = np.concatenate([table[key][2] for key in keys])
    return k28.Model(**lab.FAMILIES["v27"]).fit(x, hedged(d, w), hedged(u, w))


def per_game(values, modes=c.MODES):
    return [round(float(np.mean([values[(g, m)] for m in modes])), 4) for g in GAMES]


def score_base(base, suffix=""):
    conds = conditions(base, suffix)
    table = table_of(conds)
    readouts = {"decoder": {(k.game, k.mode): k.codes for k in conds}}
    removed = {}
    for w in WEIGHTS:
        models = {held: fit_verifier(table, [g for g in GAMES if g != held], w) for held in GAMES}
        name = f"verifier_w{w:.2f}"
        readouts[name], removed[name] = {}, {}
        for k in conds:
            new, applied = lab.apply(models[k.game], k.segs, k.p, k.edge, None)
            readouts[name][(k.game, k.mode)] = c.to_codes(new, k.n)
            removed[name][(k.game, k.mode)] = len(applied)
    report = {}
    blocks = {(k.game, k.mode): k.blocks for k in conds}
    refs = {(k.game, k.mode): k.references for k in conds}
    for name, codes in readouts.items():
        row = {}
        for r in REFS:
            official = {key: c.official(refs[key][r], v) for key, v in codes.items()}
            row[r] = dict(per_game=per_game({key: v["spell_f1"] for key, v in official.items()}),
                          per_condition={f"{g}_{m}": round(official[(g, m)]["spell_f1"], 4) for g in GAMES for m in MODES},
                          sight_per_game=per_game({key: v["spell_f1"] for key, v in official.items()}, ("sight",)),
                          macro_f1_per_game=per_game({key: v["macro_f1"] for key, v in official.items()}))
            row[r]["mean"] = round(float(np.mean(row[r]["per_game"])), 4)
            row[r]["sight_mean"] = round(float(np.mean(row[r]["sight_per_game"])), 4)
        stats = {key: c.duration_stats(v, blocks[key]) for key, v in codes.items()}
        for s in ("per_live_minute", "share_lt_3s", "dead_share"):
            row[s] = per_game({key: v[s] for key, v in stats.items()})
            row[s + "_sight"] = per_game({key: v[s] for key, v in stats.items()}, ("sight",))
        if name in removed:
            row["removals"] = per_game(removed[name])
        report[name] = row
    return report


def show(report):
    for r in REFS:
        print(f"\n== reference: {r} == held-out spellF1 per Metrica game (mean of clean/moderate/stress) | mean | sight condition per game | mean")
        for base, rows in report.items():
            for name, row in rows.items():
                v = row[r]
                print(f"{base:<12}{name:<16}" + " ".join(f"{x:.4f}" for x in v["per_game"]) + f" | {v['mean']:.4f} | "
                      + " ".join(f"{x:.4f}" for x in v["sight_per_game"]) + f" | {v['sight_mean']:.4f}"
                      + (f"   /min {row['per_live_minute']} D {row['dead_share']}" if r == REFS[0] else ""))


def score(bases, suffix="", out="score.json"):
    report = {}
    for base in bases:
        report[base] = score_base(base, suffix)
        print("SCORED", base, {r: report[base]["decoder"][r]["per_game"] for r in REFS}, flush=True)
    (ROOT / out).write_text(json.dumps(report, indent=1))
    show(report)
    return report


BARS = {"consistent": [("frozen", "decoder"), ("frozen", "verifier_w0.50"), ("v28d_blend", "verifier_w1.00")],
        "alternative": [("frozen", "decoder"), ("frozen", "verifier_w0.50"), ("frozen", "verifier_w0.00")]}
RECIPES = [("C", "decoder"), ("C", "verifier_w1.00"), ("C2", "verifier_w1.00"), ("A", "decoder"), ("A", "verifier_w0.00"), ("A2", "verifier_w0.00"),
           ("CA", "verifier_w0.50"), ("CA2", "verifier_w0.50"), ("C25A75_2", "verifier_w0.50"), ("A2", "verifier_w0.50")]


def summary(files, out="summary.json"):
    """Per-game held-out spellF1 of the recipes next to the bars they must beat on every game."""
    report = {}
    for name in files:
        for base, rows in json.loads((ROOT / name).read_text()).items():
            report[(base if name != "score_seed10.json" else base + "_seed10")] = rows
    table = []
    recipes = RECIPES + [(b + "_seed10", n) for b, n in RECIPES if b in ("C", "A", "CA") and b + "_seed10" in report]
    for base, name in recipes:
        if base not in report:
            continue
        row = dict(base=base, readout=name)
        for r in REFS:
            v = report[base][name][r]
            row[r] = dict(per_game=v["per_game"], mean=v["mean"], sight_per_game=v["sight_per_game"], sight_mean=v["sight_mean"])
            for bar_base, bar_name in BARS[r]:
                b = report[bar_base][bar_name][r]
                d = np.array(v["per_game"]) - np.array(b["per_game"])
                ds = np.array(v["sight_per_game"]) - np.array(b["sight_per_game"])
                dc = np.array([v["per_condition"][key] - b["per_condition"][key] for key in v["per_condition"] if not key.endswith("sight")])
                row[r][f"vs {bar_base} {bar_name}"] = dict(per_game=d.round(4).tolist(), wins_every_game=bool((d > 0).all()),
                                                          conditions_won=int((dc > 0).sum()), sight_per_game=ds.round(4).tolist())
        row["per_live_minute"], row["share_lt_3s"], row["dead_share"] = (report[base][name][k] for k in ("per_live_minute", "share_lt_3s", "dead_share"))
        table.append(row)
    (ROOT / out).write_text(json.dumps(dict(bars={r: {f"{b} {n}": report[b][n][r]["per_game"] for b, n in BARS[r]} for r in REFS}, rows=table), indent=1))
    for r in REFS:
        print(f"\n== {r}: bars " + "; ".join(f"{b} {n} {report[b][n][r]['per_game']}" for b, n in BARS[r]))
        for row in table:
            v = row[r]
            print(f"{row['base']:<12}{row['readout']:<15}{v['per_game']} mean {v['mean']:.4f} sight {v['sight_mean']:.4f} | "
                  + " | ".join(f"{k[3:]}: {x['per_game']} {'ALL' if x['wins_every_game'] else 'no'} {x['conditions_won']}/9" for k, x in v.items() if k.startswith("vs ")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["predict", "baselines", "score", "show", "summary"])
    parser.add_argument("--files", nargs="+", default=["score.json", "score_seed10.json", "score_two_seeds.json"])
    parser.add_argument("--target", choices=["C", "A"])
    parser.add_argument("--held", type=int, choices=[1, 2, 3])
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=v30d_train.METRICA_REPEAT)
    parser.add_argument("--bases", nargs="+", default=list(BASES))
    parser.add_argument("--suffix", default="", help="tag suffix of the C / A outputs (e.g. _seed10)")
    parser.add_argument("--out", default="score.json")
    args = parser.parse_args()
    if args.step == "predict":
        predict(args.target, args.held, args.device, args.seed_offset, args.repeat)
    elif args.step == "baselines":
        baselines(args.device)
    elif args.step == "summary":
        summary(args.files)
    elif args.step == "show":
        show(json.loads((ROOT / args.out).read_text()))
    else:
        score(args.bases, args.suffix, args.out)
