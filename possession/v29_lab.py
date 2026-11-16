"""V29 hedge study: spell verifiers fitted to the EXPECTED spellF1 gain under semantic uncertainty.

The open question (see data/work/review_v27_reference): does a one-touch interception change
possession in the challenge ground truth? "consistent" says no (~4.2 possessions per live
minute), "alternative" says yes (~6). V27/V28a fit the verifier to one reading (the bet) or to
the minimum of both (robust). Here the target of a candidate removal is

    w * gain_consistent + (1 - w) * gain_alternative          (w = P(consistent))

and, for the V28a recipe, the same blend of the two purity gaps. Decision boundary 0, never
tuned. w = 0.7 was the study-time choice (the reviewer's probability, later found to depend on the public window; the submission uses w = 0.5); the table below exists to
check that the trade-off is sane, not to pick w.

NOTE (after the final audit): the SUBMISSION uses the equal prior, target `w0.50` on the frozen
base with the V27 feature set (scripts/reproduce_submission.py, variant `final`). The 0.7 above
was informed by the public window of the challenge and was replaced for that reason; it stays as
the variant `prior70`. PRIOR below is still 0.7 because the `expected` field of hedge.json was
computed with it. See docs/TECHNICAL_REPORT.md, sections 3.3 and 6.

    .venv/bin/python -m possession.v29_lab mix       # mixture references q=0.5 (reviewer's rv_refs)
    .venv/bin/python -m possession.v29_lab table     # candidate removals of both bases, 9 conditions
    .venv/bin/python -m possession.v29_lab hedge     # leave-one-game-out table (official metric)
    .venv/bin/python -m possession.v29_lab fit       # final verifiers on the three games

Bases (cached network outputs on the three open Metrica games x clean/moderate/stress):
    frozen   data/work/v16/game{g}_{mode}_final.npz              (frozen four-network ensemble)
    blend    data/work/v28d/outputs/blend/game{g}_{mode}.npz     (0.5 frozen + 0.5 retrained, V28d)
Families:
    v27        V27 feature set (23 network-output features), GBM of the exact gain
    v27_blend  V27 feature set, gain regressor + purity-gap regressor (V28a's target, no physics)
    v28a       V28a adopted variant `physical_live_blend` (physics for T-X-T candidates only)

No challenge ground truth is read. Set OMP_NUM_THREADS=1 on a shared machine.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from . import v27_core as c
from . import v27_lab as v27
from . import v28a_core as k28
from . import v28a_lab as lab28

ROOT = Path("data/work/v29")
GAMES = (1, 2, 3)
BASES = ("frozen", "blend")
MAX_FRAMES = v27.MAX_FRAMES
N27 = len(c.FEATURES)
V = list(range(N27))
PHYSICAL = list(range(N27, N27 + len(k28.PHYSICAL)))
P_NO_SEEN = [j for j in PHYSICAL if k28.FEATURES[j] not in ("seen_in", "pre_seen", "post_seen")]
FAMILIES = {
    "v27": dict(kind="gbm", columns=V),
    "v27_blend": dict(kind="blend", columns=V),
    "v28a": dict(kind="blend", columns=V + P_NO_SEEN, live_only=True),      # = v28a_lab.ADOPTED
}
TARGETS = {"w1.00": 1.0, "w0.85": 0.85, "w0.70": 0.7, "w0.50": 0.5, "robust": None}
PRIOR = 0.7                                   # P(consistent), fixed a priori
MIX_Q, MIX_SEEDS = 0.5, (0, 1, 2)
REVIEW = Path("data/work/review_v27_reference")


# -- data ------------------------------------------------------------------------------------

def output_path(base, game, mode):
    if base == "frozen":
        return Path(f"data/work/v16/game{game}_{mode}_final.npz")
    if base == "blend":
        return Path(f"data/work/v28d/outputs/blend/game{game}_{mode}.npz")
    raise ValueError("Unknown base")


def conditions(base):
    """The nine Metrica conditions of one base, with decode, references and tracking signals."""
    from .v28d_eval import conditions as load
    out = load("frozen" if base == "frozen" else "blend") if base in BASES else None
    if out is None:
        raise ValueError("Unknown base")
    for k in out:
        k.signals = k28.cached_signals(k.game, k.mode)
    return out


def needs_physics(columns):
    return any(j >= N27 for j in columns)


def features(segs, cands, p, edge, signals=None):
    """Rows in the V28a column layout. The physical block is filled only when signals are
    given; the structure/shape blocks (rejected in V28a, used by no family here) stay NaN."""
    x = np.full((len(cands), len(k28.FEATURES)), np.nan)
    x[:, :N27] = c.candidate_features(segs, cands, p, edge)
    if signals is not None and len(cands):
        x[:, PHYSICAL] = np.array([k28.physical_features(segs[i][0], segs[i][1], segs[i][2], signals) for i, _, _ in cands],
                                  dtype=np.float64)
    return x


def build_table():
    """Per base and condition: features, exact spellF1 change and purity gap of every candidate
    removal on the base's own decode, for the three references."""
    ROOT.mkdir(parents=True, exist_ok=True)
    for base in BASES:
        out = {}
        for k in conditions(base):
            cands = c.candidates(k.segs, MAX_FRAMES)
            tag = f"{k.game}_{k.mode}"
            out["x_" + tag] = features(k.segs, cands, k.p, k.edge, k.signals)
            out["d_" + tag] = d = lab28.exact_deltas(k.segs, cands, k.gt)
            out["u_" + tag] = lab28.purity_gaps(k.segs, cands, k.references)
            out["a_" + tag] = np.array([c.ACTIONS.index(action) for _, action, _ in cands])
            print("TABLE", base, k.game, k.mode, len(cands), "positive", (d > 0).mean(0).round(3), flush=True)
        np.savez_compressed(ROOT / f"table_{base}.npz", **out)
    old = np.load(k28.ROOT / "candidate_table.npz")              # the frozen base must reproduce V28a's table
    new = np.load(ROOT / "table_frozen.npz")
    columns = V + PHYSICAL
    same = all(np.array_equal(old[f"x_{g}_{m}_0"][:, columns], new[f"x_{g}_{m}"][:, columns], equal_nan=True)
               and np.array_equal(old[f"d_{g}_{m}_0"], new[f"d_{g}_{m}"]) and np.array_equal(old[f"u_{g}_{m}_0"], new[f"u_{g}_{m}"])
               for g in GAMES for m in c.MODES)
    print("TABLE frozen base identical to data/work/v28a/candidate_table.npz (state 0):", same, flush=True)
    if not same:
        raise RuntimeError("The frozen-base table does not reproduce V28a")


def load_table(base):
    path = ROOT / f"table_{base}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m possession.v29_lab table`")
    return np.load(path)


# -- hedged targets and models ---------------------------------------------------------------

def hedge(values, target):
    """values (n, references) -> the training target. w * consistent + (1 - w) * alternative,
    or their minimum for 'robust'."""
    consistent = values[:, c.REFERENCES.index("consistent")]
    alternative = values[:, c.REFERENCES.index("alternative")]
    if target not in TARGETS:
        raise ValueError("Unknown target")
    w = TARGETS[target]
    if w is None:
        return np.minimum(consistent, alternative)
    return w * consistent + (1.0 - w) * alternative


def training(table, games, target):
    keys = [f"{g}_{m}" for g in games for m in c.MODES]
    return (np.concatenate([table["x_" + key] for key in keys]),
            hedge(np.concatenate([table["d_" + key] for key in keys]), target),
            hedge(np.concatenate([table["u_" + key] for key in keys]), target))


def fit(table, games, family, target):
    return k28.Model(**FAMILIES[family]).fit(*training(table, games, target))


def scorer(model, p, edge, signals):
    use = signals if needs_physics(model.columns) else None

    def score(segs, cands):
        return model.predict(features(segs, cands, p, edge, use))
    return score


def apply(model, segs, p, edge, signals, threshold=0.0, passes=3):
    return c.apply_scored(segs, scorer(model, p, edge, signals), threshold, MAX_FRAMES, passes)


# -- references ------------------------------------------------------------------------------

def mix_references():
    """Mixture references of the reviewer (rv_refs): a share q of the one-touch interceptions
    counts as possession. q=0 must reproduce 'consistent'."""
    sys.path.insert(0, str(REVIEW.resolve()))
    from rv_common import load_events
    from rv_refs import reference, make_touch
    out = {}
    for game in GAMES:
        events, n = load_events(game)
        events = events[["team", "type", "subtype", "period", "start", "end"]]
        events = events[events.type != "CARRY"].reset_index(drop=True)
        meta = np.load(c.ROOT / f"metrica_{game}.npz")
        if not np.array_equal(reference(events, n, make_touch("mix", q=0.0))[0], meta["consistent"]):
            raise RuntimeError("rv_refs at q=0 does not reproduce the consistent reference")
        for seed in MIX_SEEDS:
            labels = reference(events, n, make_touch("mix", q=MIX_Q, seed=seed))[0]
            if not np.array_equal(labels >= 0, meta["consistent"] >= 0):
                raise RuntimeError("Mixture reference changed the graded frames")
            out[f"g{game}_s{seed}"] = labels
        k = c.duration_stats(np.where(labels >= 0, labels, 3), [tuple(int(v) for v in b) for b in meta["blocks"]])
        print("MIX", game, "q", MIX_Q, {name: round(k[name], 3) for name in ("spells", "per_live_minute", "share_lt_3s")}, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "mix_refs.npz", **out)


def scoring_references(game):
    """{name: [label arrays]} -- a mixture is the mean over its selection seeds."""
    meta = np.load(c.ROOT / f"metrica_{game}.npz")
    refs = {r: [meta[r]] for r in c.REFERENCES}
    path = ROOT / "mix_refs.npz"
    if path.exists():
        saved = np.load(path)
        refs["mix_q0.5"] = [saved[f"g{game}_s{seed}"] for seed in MIX_SEEDS]
    return refs


def score_codes(codes, refs):
    out = {}
    for name, arrays in refs.items():
        scores = [c.official(a, codes) for a in arrays]
        out[name] = dict(spell_f1=float(np.mean([s["spell_f1"] for s in scores])),
                         macro_f1=float(np.mean([s["macro_f1"] for s in scores])))
    return out


# -- the study -------------------------------------------------------------------------------

def per_game(values):
    """{(game, mode): v} -> [mean over modes per game]."""
    return [float(np.mean([values[(g, m)] for m in c.MODES])) for g in GAMES]


def hedge_study():
    refs = {g: scoring_references(g) for g in GAMES}
    names = list(refs[1])
    conds = {base: conditions(base) for base in BASES}
    floor = {(k.game, k.mode): score_codes(k.codes, refs[k.game]) for k in conds["frozen"]}     # the frozen decoder
    report = dict(prior_consistent=PRIOR, references=names, mix=dict(q=MIX_Q, seeds=list(MIX_SEEDS)) if "mix_q0.5" in names else None,
                  frozen_decoder={r: per_game({key: v[r]["spell_f1"] for key, v in floor.items()}) for r in names}, rows=[])
    for g in GAMES:
        blocks = conds["frozen"][3 * (g - 1)].blocks
        report.setdefault("reference_stats", {})[str(g)] = {
            r: {s: round(float(np.mean([c.duration_stats(np.where(a >= 0, a, 3), blocks)[s] for a in arrays])), 4)
                for s in ("spells", "per_live_minute", "share_lt_3s", "mean_s")} for r, arrays in refs[g].items()}

    def row_of(base, family, target, after):
        row = dict(base=base, family=family, target=target)
        for r in names:
            row[r] = [round(v, 4) for v in per_game({key: after[key]["score"][r]["spell_f1"] - floor[key][r]["spell_f1"] for key in after})]
            row[r + "_worst_condition"] = round(min(after[key]["score"][r]["spell_f1"] - floor[key][r]["spell_f1"] for key in after), 4)
        row["macro_f1_consistent"] = [round(v, 4) for v in per_game({key: after[key]["score"]["consistent"]["macro_f1"] - floor[key]["consistent"]["macro_f1"] for key in after})]
        row["expected"] = [round(PRIOR * a + (1 - PRIOR) * b, 4) for a, b in zip(row["consistent"], row["alternative"])]
        for s in ("spells", "per_live_minute", "share_lt_3s"):
            row[s] = [round(v, 3) for v in per_game({key: after[key]["stats"][s] for key in after})]
        row["removals"] = [round(v, 1) for v in per_game({key: after[key]["removals"] for key in after})]
        report["rows"].append(row)
        print("HEDGE", json.dumps(row), flush=True)

    for base in BASES:
        table = load_table(base)
        row_of(base, "decoder", "-", {(k.game, k.mode): dict(score=score_codes(k.codes, refs[k.game]),
                                                             stats=c.duration_stats(k.codes, k.blocks), removals=0) for k in conds[base]})
        for family in FAMILIES:
            for target in TARGETS:
                models = {held: fit(table, [g for g in GAMES if g != held], family, target) for held in GAMES}
                after = {}
                for k in conds[base]:
                    new, applied = apply(models[k.game], k.segs, k.p, k.edge, k.signals)
                    codes = c.to_codes(new, k.n)
                    after[(k.game, k.mode)] = dict(score=score_codes(codes, refs[k.game]), stats=c.duration_stats(codes, k.blocks),
                                                   removals=len(applied))
                row_of(base, family, target, after)
    (ROOT / "hedge.json").write_text(json.dumps(report, indent=1))
    show(report)


def show(report=None):
    report = report or json.loads((ROOT / "hedge.json").read_text())
    names = report["references"]
    print("\nHeld-out spellF1 change versus the FROZEN decoder, per Metrica game (mean of clean/moderate/stress)")
    print("frozen decoder:", {r: [round(v, 4) for v in report["frozen_decoder"][r]] for r in names})
    head = f"{'base':<7}{'family':<10}{'target':<7}" + "".join(f"{r[:11]:<25}" for r in names) + f"{'E[0.7/0.3]':<25}{'mean E':<8}{'spells/live min':<21}{'<3s':<6}"
    print(head)
    for row in report["rows"]:
        cells = "".join("{:<25}".format(" ".join(f"{v:+.3f}" for v in row[r])) for r in names)
        print(f"{row['base']:<7}{row['family']:<10}{row['target']:<7}{cells}" + "{:<25}".format(" ".join(f"{v:+.3f}" for v in row["expected"]))
              + f"{np.mean(row['expected']):+.4f} " + "{:<21}".format(" ".join(f"{v:.2f}" for v in row["per_live_minute"]))
              + f"{np.mean(row['share_lt_3s']):.3f}")


def verifier_path(base, family, target):
    return ROOT / "verifiers" / f"{base}_{family}_{target}.joblib"


def fit_final(bases=BASES, families=tuple(FAMILIES), targets=tuple(TARGETS)):
    """Final verifiers: all three games, on each base's own Metrica outputs."""
    import joblib
    for base in bases:
        table = load_table(base)
        for family in families:
            for target in targets:
                model = fit(table, list(GAMES), family, target)
                path = verifier_path(base, family, target)
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(dict(model=model, features=k28.FEATURES, max_frames=MAX_FRAMES, base=base, family=family, target=target,
                                 weight_consistent=TARGETS[target], parameters=k28.GBM), path)
                print("FITTED", path, flush=True)


if __name__ == "__main__":
    steps = dict(mix=mix_references, table=build_table, hedge=hedge_study, show=show, fit=fit_final)
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=list(steps))
    steps[parser.parse_args().step]()
