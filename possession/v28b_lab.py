"""V28b experiments: object-level decoders, leave-one-game-out on the three open Metrica games.

    .venv/bin/python -m possession.v28b_lab lattice                      # segment lattices of the 9 conditions (cache)
    for f in cost calibrated duration team; do
        .venv/bin/python -m possession.v28b_lab search --family $f       # fixed-seed random search, every fold
    done
    .venv/bin/python -m possession.v28b_lab logo      # select on two games, score the third: decoders, consensus,
                                                      # posterior MBR, verifier after the decoder, V27 for reference
    .venv/bin/python -m possession.v28b_lab extras    # consensus size, macroF1, paired block bootstrap against V27
    .venv/bin/python -m possession.v28b_lab fit       # final decoders + verifier on the three games

Protocol. Fold g holds game g out: the duration prior is fitted on the consistent references of
the two other games, every parameter is selected on those two games (3 conditions each), the
verifier is fitted on them, and the held-out game is only scored. Fold 0 uses the three games and
is only used for the final model. No challenge ground truth is read anywhere.

Inputs: data/work/v27/metrica_{1,2,3}.npz (references, `python -m possession.v27_lab prepare`),
data/work/v16/game{g}_{mode}_final.npz (frozen-ensemble outputs) and, for the V27 reference rows,
data/work/v27/candidate_table.npz. Outputs: data/work/v28b/.
"""
import argparse
import json
import pickle
import time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from . import v27_core as c
from . import v28b_core as b

ROOT = Path("data/work/v28b")
GAMES = (1, 2, 3)
WORKERS = 8
STATE = {}


# -- data ------------------------------------------------------------------------------------

def build_lattices():
    ROOT.mkdir(parents=True, exist_ok=True)
    out = []
    for k in c.all_conditions():
        k.lattice = b.build_lattice(k.interpolated, k.n)
        for half in k.lattice:
            half["cache"] = {}
        out.append(k)
        print("LATTICE", k.game, k.mode, [len(h["bounds"]) - 1 for h in k.lattice], flush=True)
    with open(ROOT / "lattices.pkl", "wb") as handle:
        pickle.dump(out, handle)
    return out


def conditions():
    if "conditions" not in STATE:
        path = ROOT / "lattices.pkl"
        if path.exists():
            with open(path, "rb") as handle:
                loaded = pickle.load(handle)
        else:
            loaded = build_lattices()
        for k in loaded:
            k.gt = {r: c.spells_of(c.segments(k.references[r], k.blocks)) for r in c.REFERENCES}
            k.base = b.decode(k.lattice, k.n)
        STATE["conditions"] = loaded
    return STATE["conditions"]


def prior_for(games, bandwidth=0.35):
    """Duration prior from the consistent references of `games` (identical across conditions)."""
    key = (tuple(games), bandwidth)
    if key not in STATE:
        merged = {0: [], 1: [], 2: []}
        longest = 0
        for k in conditions():
            longest = max(longest, max(h["length"] for h in k.lattice))
            if k.game in games and k.mode == "clean":
                d = b.reference_durations(k.references["consistent"], k.blocks)
                for kind in range(3):
                    merged[kind].append(d[kind])
        STATE[key] = b.fit_prior({kind: np.concatenate(v) for kind, v in merged.items()}, longest, bandwidth)
    return STATE[key]


def fold_prior(fold):
    """fold 0 = all three games (final model); fold g = game g held out."""
    return prior_for(tuple(g for g in GAMES if g != fold))


def spells(codes, k):
    return c.spells_of(c.segments(codes, k.blocks))


def scores(codes, k):
    pred = spells(codes, k)
    return [c.spell_f1(k.gt[r], pred) for r in c.REFERENCES] + [len(pred)]


# -- parameter search ------------------------------------------------------------------------

FAMILIES = ("cost", "calibrated", "duration", "team")
FREE = dict(cost=("c_ha", "c_to_d", "c_from_d", "gamma"),
            calibrated=("c_ha", "c_to_d", "c_from_d", "gamma", "t_team", "floor_team", "floor_live", "b_dead"),
            duration=("c_ha", "c_to_d", "c_from_d", "gamma", "t_team", "floor_team", "floor_live", "b_dead",
                      "w_after_dead", "w_after_team", "w_dead"),
            team=("c_ha", "gamma", "t_team", "floor_team", "w_after_dead", "w_after_team"))


def random_configs(n, seed, family="duration"):
    """Random segment-decoder parameters around the baseline. t_live = t_dead = 0.2 fixes the scale.

    cost        switch costs and boundary weight only (the V27 three-state sweep, on the lattice)
    calibrated  + emission calibration: team temperature, per-frame evidence floors, stoppage bias
    duration    + explicit duration priors (semi-Markov), weights per kind
    team        stoppages of the baseline locked; team temperature, floor, switch cost, durations

    Every family consumes the same random stream (all parameters are drawn, the family keeps its
    own), so the families are nested views of one fixed-seed sample.
    """
    rng = np.random.default_rng(seed)

    def log_uniform(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    out = []
    for _ in range(n):
        drawn = dict(t_team=log_uniform(0.02, 0.4), floor_team=log_uniform(0.5, b.NO_FLOOR),
                     floor_live=log_uniform(1.0, b.NO_FLOOR), b_dead=float(rng.uniform(-1.0, 1.0)),
                     c_ha=log_uniform(2, 40), c_to_d=log_uniform(1, 20), c_from_d=log_uniform(1, 20),
                     gamma=float(rng.uniform(0, 1)))
        for name in ("w_after_dead", "w_after_team", "w_dead"):
            drawn[name] = 0.0 if rng.random() < 0.25 else log_uniform(0.2, 8)
        par = dict(b.PARAMS, **{name: drawn[name] for name in FREE[family]})
        if family == "team":
            par["lock_dead"] = 1.0
        out.append(par)
    return out


def uses_prior(par):
    return any(par[name] != 0 for name in ("w_after_dead", "w_after_team", "w_dead"))


def _evaluate(job):
    index, par = job
    out = np.zeros((4, 9, 4))                           # fold (0 = all games), condition, 3 references + spells
    for fold in range(4):
        if fold and not uses_prior(par):
            out[fold] = out[0]
            continue
        prior = fold_prior(fold)
        for j, k in enumerate(conditions()):
            out[fold, j] = scores(b.decode(k.lattice, k.n, par, prior), k)
    return index, out


def evaluate_configs(configs, workers=WORKERS):
    conditions()
    for fold in range(4):
        fold_prior(fold)
    result = np.zeros((len(configs), 4, 9, 4))
    with get_context("fork").Pool(workers) as pool:
        for done, (index, out) in enumerate(pool.imap_unordered(_evaluate, list(enumerate(configs)), chunksize=4)):
            result[index] = out
            if (done + 1) % 200 == 0:
                print("EVALUATED", done + 1, len(configs), flush=True)
    return result


SEARCH_SIZE = dict(cost=800, calibrated=1500, duration=3000, team=1500)


def search(family="duration", n=None, seed=28):
    """Random search of one decoder family, scored on every condition for every fold's prior."""
    start = time.time()
    configs = [dict(b.PARAMS, lock_dead=float(family == "team"))] + random_configs(n or SEARCH_SIZE[family], seed, family)
    result = evaluate_configs(configs)
    names = sorted(b.PARAMS)
    np.savez_compressed(ROOT / f"search_{family}.npz", result=result, names=np.array(names),
                        values=np.array([[p[x] for x in names] for p in configs]))
    print("SEARCH", family, len(configs), "configs", round(time.time() - start, 1), "s", flush=True)


def load_search(family="duration"):
    saved = np.load(ROOT / f"search_{family}.npz")
    names = [str(x) for x in saved["names"]]
    configs = [dict(b.PARAMS, **dict(zip(names, (float(v) for v in row)))) for row in saved["values"]]
    return configs, saved["result"]


def train_columns(held):
    return [j for j, k in enumerate(conditions()) if k.game != held]


def held_columns(held):
    return [j for j, k in enumerate(conditions()) if k.game == held]


# -- V27 verifier (reference method) and verifier on top of another decoder --------------------

def segs_of(codes, k):
    return c.segments(codes, k.blocks)


def candidate_table(codes, k, references=("consistent",)):
    """Candidate removals on a decode with their exact spellF1 change (as v27_lab.build_table)."""
    from . import v27_lab as lab
    segs = segs_of(codes, k)
    p, edge = dense(k)
    cands = c.candidates(segs, lab.MAX_FRAMES)
    x = c.candidate_features(segs, cands, p, edge)
    delta = np.zeros((len(cands), len(references)))
    for j, reference in enumerate(references):
        base = c.spell_f1(k.gt[reference], c.spells_of(segs))
        for row, (i, _, new) in enumerate(cands):
            delta[row, j] = c.spell_f1(k.gt[reference], c.spells_of(c.relabel(segs, {i: new}))) - base
    return x, delta


def dense(k):
    if not hasattr(k, "dense"):
        k.dense = c.dense(k.interpolated, k.n)
    return k.dense


def fit_verifier(tables):
    """tables: [(x, delta)] of the training conditions; target = gain under the consistent reference."""
    from . import v27_lab as lab
    x = np.concatenate([t[0] for t in tables])
    y = np.concatenate([t[1][:, 0] for t in tables])
    return lab.fit(x, y)


def apply_verifier(model, codes, k, threshold=0.0):
    from . import v27_lab as lab
    p, edge = dense(k)
    new, applied = c.apply_scored(segs_of(codes, k), lab.scorer(model, p, edge), threshold, lab.MAX_FRAMES)
    return c.to_codes(new, k.n), len(applied)


def verifier_logo(decoded, threshold=0.0):
    """decoded[held] = [codes per condition] produced with parameters selected without game `held`.

    For each held-out game the verifier is fitted on the candidate tables of the two other games
    (same decoder) and applied to the held-out game. Returns [codes per condition] out of fold.
    """
    out = [None] * 9
    for held in GAMES:
        tables = [candidate_table(decoded[held][j], conditions()[j]) for j in train_columns(held)]
        model = fit_verifier(tables)
        for j in held_columns(held):
            out[j] = apply_verifier(model, decoded[held][j], conditions()[j], threshold)[0]
    return out


def v27_reference(threshold):
    """The V27 mode='consistent' verifier, leave-one-game-out, exactly as v27_lab.verifier does it."""
    from . import v27_lab as lab
    table = np.load(c.ROOT / "candidate_table.npz")
    out = [None] * 9
    for held in GAMES:
        model = lab.fit(*lab.training(table, [g for g in GAMES if g != held], "consistent"))
        for j in held_columns(held):
            out[j] = apply_verifier(model, conditions()[j].base, conditions()[j], threshold)[0]
    return out


def score_table(per_condition):
    """(9 conditions, 3 references + spells) for a list of codes."""
    return np.array([scores(codes, k) for codes, k in zip(per_condition, conditions())])


def per_game(table):
    """Mean over the three conditions of each game: (3 games, columns)."""
    return table.reshape(3, 3, -1).mean(1)


def describe(name, table, base, references=None):
    """One printed row + dict: per-game held-out deltas against `base` per reference."""
    row = dict(method=name, spells=table[:, 3].astype(int).tolist())
    for j, reference in enumerate(c.REFERENCES):
        delta = table[:, j] - base[:, j]
        row[reference] = dict(per_game_score=per_game(table)[:, j].round(4).tolist(),
                              per_game_delta=per_game(delta[:, None])[:, 0].round(4).tolist(),
                              mean_delta=round(float(delta.mean()), 4), worst_condition=round(float(delta.min()), 4))
    print("ROW", json.dumps(row), flush=True)
    return row


# -- leave-one-game-out comparison -------------------------------------------------------------

TOP = 10
BETAS = (0.5, 1.0, 2.0)
PRICES = (None, 0.3, 0.4)


def ranked(result, held):
    """Configs ordered by mean consistent spellF1 on the two games that are not `held` (fold `held`).

    held = 0: all nine conditions with the prior of the three games (the final model).
    """
    columns = train_columns(held) if held else list(range(9))
    return np.argsort(-result[:, held][:, columns, 0].mean(1), kind="stable")


def decode_fold(configs, order, held, top):
    """decoded[i][j]: codes of condition j with the i-th best config of fold `held`."""
    prior = fold_prior(held)
    return [[b.decode(k.lattice, k.n, configs[i], prior) for k in conditions()] for i in order[:top]]


def out_of_fold(by_held):
    """by_held[held][j] -> the list over conditions that only keeps each game's held-out codes."""
    return [by_held[k.game][j] for j, k in enumerate(conditions())]


def logo(top=TOP, n_samples=150):
    ks = conditions()
    base = score_table([k.base for k in ks])
    rows = [describe("frozen ensemble (decode_fine)", base, base)]
    reference = {thr: v27_reference(thr) for thr in (0.0, 0.0005)}
    v27 = {thr: score_table(codes) for thr, codes in reference.items()}
    for thr in v27:
        rows.append(describe(f"V27 consistent verifier, threshold {thr}", v27[thr], base))
    tables = {}
    kept = {}
    for family in FAMILIES:
        if not (ROOT / f"search_{family}.npz").exists():
            continue
        configs, result = load_search(family)
        decoded, single, combined = {}, {}, {}
        for held in GAMES:
            order = ranked(result, held)
            decoded[held] = decode_fold(configs, order, held, top)
            single[held] = decoded[held][0]
            combined[held] = [b.combine(k.lattice, k.n, [d[j] for d in decoded[held]])[0] for j, k in enumerate(ks)]
            print("SELECTED", family, "held", held, json.dumps({x: round(v, 3) for x, v in configs[order[0]].items()}), flush=True)
        kept[family] = (single, combined, decoded)
        tables[f"{family}: best config"] = score_table(out_of_fold(single))
        tables[f"{family}: consensus of the {top} best configs"] = score_table(out_of_fold(combined))
        for name in list(tables)[-2:]:
            rows.append(describe(name, tables[name], base))
    # decoder first, V27-style verifier after (re-fitted on the candidate removals of that decoder)
    for family in ("duration", "team"):
        if family not in kept:
            continue
        single, combined, decoded = kept[family]
        for label, source in (("best config", single), (f"consensus of the {top} best", combined)):
            for thr in (0.0, 0.0005):
                name = f"{family}: {label} + verifier, threshold {thr}"
                tables[name] = score_table(verifier_logo(source, thr))
                rows.append(describe(name, tables[name], base))
        # system combination with the V27 verifier outputs as two more hypotheses
        mixed = {held: [b.combine(k.lattice, k.n, [d[j] for d in decoded[held]] + [reference[0.0][j], reference[0.0005][j]])[0]
                        for j, k in enumerate(ks)] for held in GAMES}
        name = f"{family}: consensus of the {top} best configs and the two V27 verifier outputs"
        tables[name] = score_table(out_of_fold(mixed))
        rows.append(describe(name, tables[name], base))
    # posterior MBR on the best semi-Markov model: (beta, price) selected on the two training games
    if "duration" in kept:
        configs, result = load_search("duration")
        chosen = {}
        mbr = {}
        for held in GAMES:
            par, prior = configs[ranked(result, held)[0]], fold_prior(held)
            grid = {}
            for beta in BETAS:
                per_price = [b.decode_mbr(k.lattice, k.n, par, prior, beta, n_samples, seed=0, prices=PRICES)[0] for k in ks]
                for price in PRICES:
                    grid[(beta, price)] = [codes[price] for codes in per_price]
                print("MBR", held, beta, flush=True)
            train = train_columns(held)
            value = {key: float(np.mean([scores(codes[j], ks[j])[0] for j in train])) for key, codes in grid.items()}
            chosen[held] = max(value, key=value.get)
            mbr[held] = grid[chosen[held]]
            print("MBR SELECTED", held, chosen[held], {str(k): round(v, 4) for k, v in value.items()}, flush=True)
        name = "duration: posterior MBR (forward-filter backward-sample, beta and price selected in fold)"
        tables[name] = score_table(out_of_fold(mbr))
        rows.append(describe(name, tables[name], base))
        rows[-1]["selected"] = {str(g): [chosen[g][0], chosen[g][1]] for g in GAMES}
    bar = np.maximum(per_game(v27[0.0])[:, 0], per_game(v27[0.0005])[:, 0])
    verdict = []
    for name, table in tables.items():
        score = per_game(table)[:, 0]
        verdict.append(dict(method=name, per_game_consistent=score.round(4).tolist(),
                            margin_over_v27_threshold_0=(score - per_game(v27[0.0])[:, 0]).round(4).tolist(),
                            margin_over_v27_threshold_0_0005=(score - per_game(v27[0.0005])[:, 0]).round(4).tolist(),
                            beats_both_on_every_game=bool((score > bar).all())))
        print("VERDICT", json.dumps(verdict[-1]), flush=True)
    (ROOT / "logo.json").write_text(json.dumps(dict(rows=rows, verdict=verdict, top=top, challenge_labels_read=False), indent=1))


# -- the candidate pipeline, its sensitivity and its uncertainty ------------------------------

FAMILY = "duration"
THRESHOLD = 0.0


def candidate_logo(family=FAMILY, top=TOP, threshold=THRESHOLD, verify=True):
    """Out-of-fold codes of: consensus of the `top` best decoders of a family (+ re-fitted verifier)."""
    ks = conditions()
    configs, result = load_search(family)
    by_held = {}
    for held in GAMES:
        decoded = decode_fold(configs, ranked(result, held), held, top)
        by_held[held] = [b.combine(k.lattice, k.n, [d[j] for d in decoded])[0] for j, k in enumerate(ks)]
    return verifier_logo(by_held, threshold) if verify else out_of_fold(by_held)


def block_counts(codes, k, reference, block):
    """Per time block (half, minute): matched pairs per IoU threshold, true and predicted possessions."""
    gt, pred = k.gt[reference], spells(codes, k)
    match = c.matching(gt, pred)
    out = {}

    def key(frame):
        for half, first, last in k.blocks:
            if first <= frame <= last:
                return (half, (frame - first) // block)
        raise ValueError("Possession outside the graded halves")

    for i, (a, _, _) in enumerate(gt):
        row = out.setdefault(key(a), np.zeros(len(c.IOUS) + 2))
        row[:len(c.IOUS)] += [float(match[t][0][i]) for t in c.IOUS]
        row[-2] += 1
    for a, _, _ in pred:
        out.setdefault(key(a), np.zeros(len(c.IOUS) + 2))[-1] += 1
    return out


def paired_bootstrap(first, second, reference="consistent", draws=2000, seed=0, block=60 * c.FPS):
    """spellF1(first) - spellF1(second) under a paired bootstrap of one-minute blocks, per game.

    The three conditions of a game share the resampled minutes (they are the same football), and
    the score of a draw is the mean over conditions of the spellF1 of the resampled blocks.
    """
    ks = conditions()
    rng = np.random.default_rng(seed)
    out = {}
    pooled = np.zeros(draws)
    for game in GAMES:
        index = [j for j, k in enumerate(ks) if k.game == game]
        counts = [[block_counts(codes[j], ks[j], reference, block) for j in index] for codes in (first, second)]
        keys = sorted(set().union(*[set(d) for side in counts for d in side]))
        zero = np.zeros(len(c.IOUS) + 2)
        table = np.array([[[d.get(key, zero) for key in keys] for d in side] for side in counts])   # side, condition, block, 7
        pick = rng.integers(0, len(keys), (draws, len(keys)))
        total = table[:, :, pick].sum(3)                                                           # side, condition, draw, 7
        f1 = (2 * total[..., :len(c.IOUS)] / np.maximum(1, total[..., -2:].sum(-1))[..., None]).mean(-1).mean(1)   # side, draw
        diff = f1[0] - f1[1]
        pooled += diff / len(GAMES)
        out[str(game)] = dict(mean=round(float(diff.mean()), 4), low5=round(float(np.percentile(diff, 5)), 4),
                              high95=round(float(np.percentile(diff, 95)), 4), p_positive=round(float((diff > 0).mean()), 3))
    out["mean_of_games"] = dict(mean=round(float(pooled.mean()), 4), low5=round(float(np.percentile(pooled, 5)), 4),
                                high95=round(float(np.percentile(pooled, 95)), 4), p_positive=round(float((pooled > 0).mean()), 3))
    return out


def macro_table(per_condition, reference="consistent"):
    return np.array([c.official(k.references[reference], codes)["macro_f1"] for codes, k in zip(per_condition, conditions())])


def extras():
    """Sensitivity to the size of the consensus, macroF1, and paired bootstrap against V27."""
    ks = conditions()
    base_codes = [k.base for k in ks]
    base = score_table(base_codes)
    report = dict(sensitivity=[], bootstrap={}, macro_f1={})
    for family in ("duration", "team"):
        for top in (1, 5, 10, 20, 30):
            for verify in (False, True):
                name = f"{family}: consensus of {top}" + (" + verifier" if verify else "")
                report["sensitivity"].append(describe(name, score_table(candidate_logo(family, top, THRESHOLD, verify)), base))
    candidate = candidate_logo()
    references = {f"V27 consistent, threshold {thr}": v27_reference(thr) for thr in (0.0, 0.0005)}
    references["frozen ensemble"] = base_codes
    for name, codes in references.items():
        report["bootstrap"][name] = {r: paired_bootstrap(candidate, codes, r) for r in c.REFERENCES}
        print("BOOTSTRAP candidate minus", name, json.dumps(report["bootstrap"][name]), flush=True)
    for name, codes in dict(candidate=candidate, **references).items():
        report["macro_f1"][name] = {r: per_game(macro_table(codes, r)[:, None])[:, 0].round(4).tolist() for r in c.REFERENCES}
        print("MACRO", name, json.dumps(report["macro_f1"][name]), flush=True)
    (ROOT / "extras.json").write_text(json.dumps(report, indent=1))


# -- final model -----------------------------------------------------------------------------

def fit_final(top=TOP):
    """Decoders, duration prior and verifier fitted on the three open Metrica games (no hold-out).

    One file per family: `duration` re-decides stoppages and teams, `team` keeps the stoppages of
    the baseline decode and only re-decides the team.
    """
    import joblib
    from . import v27_lab as lab
    ks = conditions()
    for family in ("duration", "team"):
        configs, result = load_search(family)
        order = ranked(result, 0)[:top]
        decoded = decode_fold(configs, order, 0, top)
        combined = [b.combine(k.lattice, k.n, [d[j] for d in decoded])[0] for j, k in enumerate(ks)]
        verifier = fit_verifier([candidate_table(combined[j], ks[j]) for j in range(9)])
        after = [apply_verifier(verifier, combined[j], ks[j], THRESHOLD)[0] for j in range(9)]
        in_sample = {name: per_game(score_table(codes))[:, 0].round(4).tolist()
                     for name, codes in (("baseline", [k.base for k in ks]), ("consensus", combined), ("consensus_verifier", after))}
        joblib.dump(dict(family=family, configs=[configs[i] for i in order], prior=fold_prior(0), verifier=verifier,
                         features=c.FEATURES, max_frames=lab.MAX_FRAMES, threshold=THRESHOLD, verifier_parameters=lab.MODEL,
                         lattice_configs=list(b.LATTICE_CONFIGS), in_sample_consistent_spell_f1=in_sample,
                         fitted_on="Metrica open sample games 1-3, consistent reference", challenge_labels_read=False),
                    ROOT / f"decoder_{family}.joblib")
        print("FITTED", family, json.dumps(in_sample), flush=True)
        for par in (configs[i] for i in order):
            print("  CONFIG", json.dumps({x: round(v, 3) for x, v in par.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["lattice", "search", "logo", "extras", "fit"])
    parser.add_argument("--family", choices=FAMILIES, default="duration")
    parser.add_argument("--n", type=int)
    args = parser.parse_args()
    if args.step == "lattice":
        build_lattices()
    elif args.step == "search":
        search(args.family, args.n)
    elif args.step == "logo":
        logo()
    elif args.step == "extras":
        extras()
    else:
        fit_final()
