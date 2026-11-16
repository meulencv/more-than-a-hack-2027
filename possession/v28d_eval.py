"""V28d step 3: external test of a (re)trained ensemble on the three open Metrica games.

The Metrica games are never used to train the networks. Inputs are the cached records of
possession.v28d_data (same construction as possession.v16_transfer). Two read-outs per ensemble:

* the unchanged decoder (25 Hz Viterbi, costs 4 / 2.5), official metric, three references;
* the V27 spell verifier REFIT leave-one-game-out on this ensemble's own outputs (identical
  features, model and candidate definition as possession.v27_lab; fit on two games, applied to
  the third; the whole threshold curve is reported, nothing is chosen on the scored game).

    .venv/bin/python -m possession.v28d_eval predict --tag retrained --selection data/work/v28d/selection.json
    .venv/bin/python -m possession.v28d_eval predict --tag frozen_check --selection docs/FINAL_SELECTION.json
    .venv/bin/python -m possession.v28d_eval blend --tag blend --sources frozen retrained
    .venv/bin/python -m possession.v28d_eval score --tags frozen retrained blend
    .venv/bin/python -m possession.v28d_eval summary --tags retrained blend     # table against the adoption bar

Tag "frozen" always means the cached outputs data/work/v16/game{g}_{mode}_final.npz.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import v27_core as c
from . import v27_lab as lab
from .v28d_data import metrica_record

ROOT = Path("data/work/v28d")
GAMES = (1, 2, 3)


def output_path(tag, game, mode):
    if tag == "frozen":
        return Path(f"data/work/v16/game{game}_{mode}_final.npz")
    return ROOT / "outputs" / tag / f"game{game}_{mode}.npz"


def predict(tag, selection, device="cpu"):
    import torch
    from .competitive_bundle import CompetitiveBundle
    if tag == "frozen":
        raise ValueError("'frozen' is reserved for the cached v16 outputs")
    torch.set_num_threads(6)
    bundle = CompetitiveBundle(json.loads(Path(selection).read_text()), device)
    for game in GAMES:
        for mode in c.MODES:
            path = output_path(tag, game, mode)
            path.parent.mkdir(parents=True, exist_ok=True)
            r = metrica_record(game, mode)
            p, e = bundle.probabilities(r)
            np.savez_compressed(path, probabilities=p, edges=e, frames=r["frames"])
            print("PREDICT", tag, game, mode, p.shape, flush=True)


def blend(tag, sources, weights=None):
    weights = weights or [1 / len(sources)] * len(sources)
    for game in GAMES:
        for mode in c.MODES:
            saved = [np.load(output_path(s, game, mode)) for s in sources]
            if any(not np.array_equal(saved[0]["frames"], s["frames"]) for s in saved):
                raise ValueError("Different timelines")
            path = output_path(tag, game, mode)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, probabilities=sum(w * s["probabilities"] for w, s in zip(weights, saved)),
                                edges=sum(w * s["edges"] for w, s in zip(weights, saved)), frames=saved[0]["frames"])
    (ROOT / "outputs" / tag / "blend.json").write_text(json.dumps(dict(sources=sources, weights=weights)))


def conditions(tag):
    out = []
    for game in GAMES:
        meta = np.load(c.ROOT / f"metrica_{game}.npz")
        blocks = [tuple(int(v) for v in b) for b in meta["blocks"]]
        n = len(meta["original"])
        for mode in c.MODES:
            saved = np.load(output_path(tag, game, mode))
            k = SimpleNamespace(game=game, mode=mode, blocks=blocks, n=n, references={r: meta[r] for r in c.REFERENCES},
                                interpolated=c.interpolate(saved["probabilities"], saved["edges"], saved["frames"], blocks))
            k.codes = c.decode(k.interpolated, k.n)[0]
            k.segs = c.segments(k.codes, k.blocks)
            k.p, k.edge = c.dense(k.interpolated, k.n)
            k.gt = {r: c.spells_of(c.segments(k.references[r], k.blocks)) for r in c.REFERENCES}
            out.append(k)
    return out


def candidate_table(conds):
    """possession.v27_lab.build_table for arbitrary ensemble outputs (kept in memory)."""
    table = {}
    for k in conds:
        cands = c.candidates(k.segs, lab.MAX_FRAMES)
        x = c.candidate_features(k.segs, cands, k.p, k.edge)
        delta = np.zeros((len(cands), len(c.REFERENCES)))
        for j, reference in enumerate(c.REFERENCES):
            base = c.spell_f1(k.gt[reference], c.spells_of(k.segs))
            for row, (i, _, new) in enumerate(cands):
                delta[row, j] = c.spell_f1(k.gt[reference], c.spells_of(c.relabel(k.segs, {i: new}))) - base
        table[f"x_{k.game}_{k.mode}"] = x
        table[f"d_{k.game}_{k.mode}"] = delta
    return table


def grid(values):
    """{(game, mode): value} -> per-game means (list of 3) and overall mean."""
    per_game = [float(np.mean([values[(g, m)] for m in c.MODES])) for g in GAMES]
    return dict(per_game=[round(v, 4) for v in per_game], mean=round(float(np.mean(per_game)), 4),
                per_condition=[[round(float(values[(g, m)]), 4) for m in c.MODES] for g in GAMES])


def score(tag):
    conds = conditions(tag)
    report = dict(tag=tag, decoder={}, stats={}, verifier={})
    for reference in c.REFERENCES:
        official = {(k.game, k.mode): c.official(k.references[reference], k.codes) for k in conds}
        report["decoder"][reference] = dict(spell_f1=grid({key: v["spell_f1"] for key, v in official.items()}),
                                            macro_f1=grid({key: v["macro_f1"] for key, v in official.items()}))
    stats = {(k.game, k.mode): c.duration_stats(k.codes, k.blocks) for k in conds}
    for name in ("spells", "per_live_minute", "mean_s", "share_lt_1s", "share_lt_3s", "dead_share"):
        report["stats"][name] = grid({key: v[name] for key, v in stats.items()})
    table = candidate_table(conds)
    for target in ("consistent", "robust"):
        models = {held: lab.fit(*lab.training(table, [g for g in GAMES if g != held], target)) for held in GAMES}
        for threshold in lab.THRESHOLDS:
            after = {}
            for k in conds:
                new, applied = c.apply_scored(k.segs, lab.scorer(models[k.game], k.p, k.edge), threshold, lab.MAX_FRAMES)
                codes = c.to_codes(new, k.n)
                after[(k.game, k.mode)] = dict(spells=len(c.spells_of(new)), **{
                    r: c.official(k.references[r], codes) for r in c.REFERENCES})
            row = dict(spells=grid({key: v["spells"] for key, v in after.items()}))
            for r in c.REFERENCES:
                row[r] = dict(spell_f1=grid({key: v[r]["spell_f1"] for key, v in after.items()}),
                              macro_f1=grid({key: v[r]["macro_f1"] for key, v in after.items()}))
            report["verifier"][f"{target}@{threshold}"] = row
    (ROOT / f"score_{tag}.json").write_text(json.dumps(report, indent=1))
    return report


def show(reports):
    print("\nHELD-OUT spellF1, per Metrica game (mean of clean/moderate/stress) and mean of games")
    for reference in c.REFERENCES:
        print(f"-- reference: {reference}")
        for r in reports:
            d = r["decoder"][reference]["spell_f1"]
            print(f"   {r['tag']:<14} decoder                {d['per_game']}  mean {d['mean']:.4f}")
            for name in ("consistent@0.0", "consistent@0.0005", "robust@0.0"):
                v = r["verifier"][name][reference]["spell_f1"]
                print(f"   {r['tag']:<14} + verifier {name:<17} {v['per_game']}  mean {v['mean']:.4f}")
    print("-- decoder spell statistics (per game)")
    for r in reports:
        print(f"   {r['tag']:<14}", {k: v["per_game"] for k, v in r["stats"].items() if k in ("per_live_minute", "mean_s", "share_lt_3s", "dead_share")})


# -- experiment: verifier that also sees each ensemble separately ------------------------------

SOURCE_COLUMNS = [i for i, name in enumerate(c.FEATURES) if name not in (
    "duration_s", "previous_s", "next_s", "is_absorb", "is_to_dead", "is_to_team", "next_to_dead", "boundaries_10s")]


def dual_features(segs, cands, k):
    """V27 features on the blended output + the probability features of each source ensemble."""
    parts = [c.candidate_features(segs, cands, k.p, k.edge)]
    parts += [c.candidate_features(segs, cands, p, e)[:, SOURCE_COLUMNS] for p, e in k.sources]
    return np.hstack(parts)


def score_dual(tag, sources):
    """Same leave-one-game-out protocol as `score`, verifier features = blended + per-source."""
    conds = conditions(tag)
    for k in conds:
        k.sources = []
        for source in sources:
            saved = np.load(output_path(source, k.game, k.mode))
            k.sources.append(c.dense(c.interpolate(saved["probabilities"], saved["edges"], saved["frames"], k.blocks), k.n))
    base = candidate_table(conds)
    x = {(k.game, k.mode): dual_features(k.segs, c.candidates(k.segs, lab.MAX_FRAMES), k) for k in conds}
    index = {r: i for i, r in enumerate(c.REFERENCES)}
    report = dict(tag=tag + "_dual", sources=list(sources), verifier={})
    for target in ("consistent", "robust"):
        models = {}
        for held in GAMES:
            train = [(g, m) for g in GAMES if g != held for m in c.MODES]
            d = np.concatenate([base[f"d_{g}_{m}"] for g, m in train])
            y = d[:, index["consistent"]] if target == "consistent" else np.minimum(d[:, index["consistent"]], d[:, index["alternative"]])
            models[held] = lab.fit(np.concatenate([x[key] for key in train]), y)
        for threshold in lab.THRESHOLDS:
            after = {}
            for k in conds:
                scorer = lambda segs, cands, k=k: models[k.game].predict(dual_features(segs, cands, k))
                new, _ = c.apply_scored(k.segs, scorer, threshold, lab.MAX_FRAMES)
                codes = c.to_codes(new, k.n)
                after[(k.game, k.mode)] = dict(spells=len(c.spells_of(new)), **{r: c.official(k.references[r], codes) for r in c.REFERENCES})
            row = dict(spells=grid({key: v["spells"] for key, v in after.items()}))
            for r in c.REFERENCES:
                row[r] = dict(spell_f1=grid({key: v[r]["spell_f1"] for key, v in after.items()}),
                              macro_f1=grid({key: v[r]["macro_f1"] for key, v in after.items()}))
            report["verifier"][f"{target}@{threshold}"] = row
            print("DUAL", tag, f"{target}@{threshold}", {r: row[r]["spell_f1"]["per_game"] for r in c.REFERENCES}, flush=True)
    plain = json.loads((ROOT / f"score_{tag}.json").read_text())
    report["decoder"], report["stats"] = plain["decoder"], plain["stats"]
    (ROOT / f"score_{tag}_dual.json").write_text(json.dumps(report, indent=1))
    return report


def summary(tags, bar="frozen"):
    """Per-game held-out spellF1 next to the adoption bar (frozen ensemble + V27 consistent verifier)."""
    reports = {t: json.loads((ROOT / f"score_{t}.json").read_text()) for t in [bar] + [t for t in tags if t != bar]}
    rows = []
    for tag, r in reports.items():
        for name in ["decoder"] + list(r["verifier"]):
            row = dict(tag=tag, readout=name)
            for reference in c.REFERENCES:
                v = (r["decoder"][reference] if name == "decoder" else r["verifier"][name][reference])["spell_f1"]
                row[reference] = v
                if name != "decoder":
                    b = np.array(reports[bar]["verifier"][name][reference]["spell_f1"]["per_condition"])
                    d = np.array(v["per_condition"]) - b
                    row[reference + "_vs_bar"] = dict(per_game=d.mean(1).round(4).tolist(), mean=round(float(d.mean()), 4),
                                                      conditions_won=int((d > 0).sum()), worst_condition=round(float(d.min()), 4))
            row["macro_f1_consistent"] = (r["decoder"]["consistent"] if name == "decoder" else r["verifier"][name]["consistent"])["macro_f1"]["per_game"]
            rows.append(row)
    best_bar = np.max([reports[bar]["verifier"][n]["consistent"]["spell_f1"]["per_game"] for n in reports[bar]["verifier"]
                       if n.startswith("consistent@")], axis=0)
    (ROOT / "summary.json").write_text(json.dumps(dict(bar=bar, bar_best_threshold_per_game=best_bar.round(4).tolist(), rows=rows), indent=1))
    print("BAR (frozen + V27 consistent verifier), best threshold per game:", best_bar.round(4).tolist())
    for row in rows:
        if row["readout"] in ("decoder", "consistent@0.0", "consistent@0.0005", "robust@0.0"):
            extra = ""
            if row["readout"] != "decoder":
                extra = "  vs bar " + " | ".join(f"{r[:4]} {row[r + '_vs_bar']['per_game']} won {row[r + '_vs_bar']['conditions_won']}/9" for r in c.REFERENCES)
            print(f"{row['tag']:<18}{row['readout']:<19}" + " ".join(f"{r[:4]} {row[r]['per_game']} {row[r]['mean']:.4f}" for r in c.REFERENCES) + extra)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["predict", "blend", "score", "summary", "dual"])
    parser.add_argument("--tag")
    parser.add_argument("--tags", nargs="+")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    if args.step == "predict":
        predict(args.tag, args.selection, args.device)
    elif args.step == "blend":
        blend(args.tag, args.sources)
    elif args.step == "summary":
        summary(args.tags)
    elif args.step == "dual":
        score_dual(args.tag, args.sources)
    else:
        show([score(tag) for tag in args.tags])
