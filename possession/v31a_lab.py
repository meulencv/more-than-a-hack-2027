"""V31a: learners aimed at the compromise target M, on the FROZEN networks' outputs only.

M = the alternative event reading with every possession shorter than ABSORB_S (2 s, fixed a
priori) that is sandwiched between two possessions of the other team absorbed into them
(possession.v27_lab.duration_rule(alternative segments, absorb_s=2.0, island_s=0, edge_s=0)).
Used as a labelling it scores >= 0.85 under BOTH readings of the definition, so a learner that
predicts M gains under both at once instead of sliding along the semantic axis.

Nothing here retrains a network and no feature is built from ball availability, NaN patterns or
tracking: the only inputs are the ensemble's probabilities and boundary score (5 Hz) and what
the decoder makes of them.

    .venv/bin/python -m possession.v31a_lab verifier                 # B: frozen decoder + verifier on M-gain
    .venv/bin/python -m possession.v31a_lab bag                      # bagged verifier (variance control)
    .venv/bin/python -m possession.v31a_lab stack --target M         # A: second stage aimed at M
    .venv/bin/python -m possession.v31a_lab stack --target mixed     # control: the V30b target (0.5/0.5)
    .venv/bin/python -m possession.v31a_lab blend --target M         # 0.5 frozen + 0.5 second stage (after `stack`)
    .venv/bin/python -m possession.v31a_lab verifier --base tta      # C + B (needs possession.v31a_tta)
    .venv/bin/python -m possession.v31a_lab insert                   # insertion candidates: oracle + learner
    .venv/bin/python -m possession.v31a_lab show                     # tables of every JSON written so far

Strictly nested leave-one-game-out as in possession.v30b_stack: for held-out game g with
training games A, B: stacker(A, B) -> g; stacker(A) -> B and stacker(B) -> A give the
out-of-fold emissions on which the verifier is fitted; verifiers are fitted on the standard
conditions (clean / moderate / stress) of A and B only. The reviewer's label-blind ball
conditions (interp / interpclean / blind3, data/work/review_v30d) are scored with those same
fitted objects and are never fitted on.

Outputs under data/work/v31a/. No challenge label is read.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from . import v27_core as c
from . import v27_lab as lab27
from . import v28a_core as k28
from . import v30b_stack as s

ROOT = Path("data/work/v31a")
GAMES = (1, 2, 3)
REFS = ("consistent", "alternative")
BLIND = ("interp", "interpclean", "blind3")
ABSORB_S = 2.0                                           # fixed a priori; 1.5 and 3.0 are sensitivity only
M_NAMES = {"M": 2.0, "M15": 1.5, "M30": 3.0}
RNAMES = REFS + tuple(M_NAMES)                           # columns of every gain table
MAX_FRAMES = lab27.MAX_FRAMES
V = list(range(len(c.FEATURES)))
# verifier targets: weights over RNAMES
TARGETS = {
    "w0.50": dict(consistent=0.5, alternative=0.5),      # the submission's target
    "w0.70": dict(consistent=0.7, alternative=0.3),
    "w0.30": dict(consistent=0.3, alternative=0.7),
    "w0.85": dict(consistent=0.85, alternative=0.15),
    "w1.00": dict(consistent=1.0),
    "w0.00": dict(alternative=1.0),
    "M": dict(M=1.0),
    "M15": dict(M15=1.0),
    "M30": dict(M30=1.0),
    "M+w0.50": dict(M=0.5, consistent=0.25, alternative=0.25),
}
STACK_TARGETS = {
    "M": dict(M=1.0), "M15": dict(M15=1.0), "M30": dict(M30=1.0),
    "mixed": dict(consistent=0.5, alternative=0.5),      # = possession.v30b_stack
    "alternative": dict(alternative=1.0), "consistent": dict(consistent=1.0),
}


# -- references and conditions ---------------------------------------------------------------

def m_reference(alternative, blocks, absorb_s=ABSORB_S):
    """The compromise labelling (frame codes, -1 where not graded)."""
    segs = c.segments(np.where(alternative >= 0, alternative, 3).astype(np.int8), blocks)
    codes = c.to_codes(lab27.duration_rule(segs, absorb_s=absorb_s, island_s=0, edge_s=0), len(alternative)).astype(np.int8)
    codes[alternative < 0] = -1
    codes[codes == 3] = -1
    return codes


def output_path(base, game, mode):
    if mode in BLIND:
        folder = dict(frozen="data/work/review_v30d/outputs/frozen", tta=str(ROOT / "tta" / "avg_blind"))[base]
        return Path(folder) / f"game{game}_{mode}.npz"
    if base == "frozen":
        return Path(f"data/work/v16/game{game}_{mode}_final.npz")
    if base == "tta":
        return ROOT / "tta" / "avg" / f"game{game}_{mode}.npz"
    raise ValueError("Unknown base")


def load(game, mode, base="frozen"):
    """One game under one observation regime: 5 Hz network outputs, second-stage features, references."""
    meta = np.load(c.ROOT / f"metrica_{game}.npz")
    saved = np.load(output_path(base, game, mode))
    blocks = [tuple(int(v) for v in x) for x in meta["blocks"]]
    frames, p, edge = saved["frames"], saved["probabilities"].astype(np.float64), saved["edges"].astype(np.float64)
    even, odd, keep = [], [], np.zeros(len(frames), dtype=bool)
    for _, first, last in blocks:
        use = (frames >= first) & (frames <= last)
        e, o = s.half_features(p[use], edge[use])
        even.append(e)
        odd.append(o)
        keep |= use
    references = {r: meta[r] for r in REFS}
    for name, limit in M_NAMES.items():
        references[name] = m_reference(meta["alternative"], blocks, limit)
    k = SimpleNamespace(game=game, mode=mode, base=base, blocks=blocks, n=len(meta["consistent"]), references=references,
                        frames=frames[keep], p0=p[keep], edge0=edge[keep], even=np.concatenate(even), odd=np.concatenate(odd))
    k.labels = {r: references[r][k.frames] for r in RNAMES}
    k.gt = {r: c.spells_of(c.segments(np.where(references[r] >= 0, references[r], 3).astype(np.int8), blocks)) for r in RNAMES}
    return k


def decoded(k, p, config=c.DEFAULT):
    out = SimpleNamespace(game=k.game, mode=k.mode, blocks=k.blocks, n=k.n, references=k.references, gt=k.gt)
    out.interpolated = c.interpolate(p, k.edge0, k.frames, k.blocks)
    out.codes = c.decode(out.interpolated, k.n, [config])[0]
    out.segs = c.segments(out.codes, k.blocks)
    out.p, out.edge = c.dense(out.interpolated, k.n)
    return out


# -- verifier --------------------------------------------------------------------------------

def gains(segs, cands, gt):
    """Exact spellF1 change of every candidate removal under every reference of RNAMES."""
    delta = np.zeros((len(cands), len(RNAMES)))
    spells = c.spells_of(segs)
    for j, r in enumerate(RNAMES):
        base = c.spell_f1(gt[r], spells)
        for row, (i, _, new) in enumerate(cands):
            delta[row, j] = c.spell_f1(gt[r], c.spells_of(c.relabel(segs, {i: new}))) - base
    return delta


def table_of(conds):
    table = {}
    for k in conds:
        cands = c.candidates(k.segs, MAX_FRAMES)
        table[(k.game, k.mode)] = (c.candidate_features(k.segs, cands, k.p, k.edge), gains(k.segs, cands, k.gt))
    return table


def target_of(d, target):
    return sum(w * d[:, RNAMES.index(r)] for r, w in TARGETS[target].items())


def fit_verifier(table, games, target, seed=0):
    keys = [(g, m) for g in games for m in c.MODES]
    x = np.concatenate([table[key][0] for key in keys])
    d = np.concatenate([table[key][1] for key in keys])
    return k28.Model(kind="gbm", columns=V, parameters=dict(random_state=seed)).fit(x, target_of(d, target))


def apply_verifier(model, k, threshold=0.0):
    def score(segs, cands):
        return model.predict(c.candidate_features(segs, cands, k.p, k.edge))
    return c.apply_scored(k.segs, score, threshold, MAX_FRAMES, 3)


# -- scoring ---------------------------------------------------------------------------------

class Scores:
    def __init__(self):
        self.values = {}

    def record(self, tag, k, codes):
        stats = c.duration_stats(codes, k.blocks)
        row = dict(per_live_minute=stats["per_live_minute"], share_lt_3s=stats["share_lt_3s"], spells=stats["spells"])
        for r in RNAMES:
            row[r] = c.official(k.references[r], codes)["spell_f1"]
        self.values[f"{tag}|{k.game}|{k.mode}"] = row

    def dump(self, path, **extra):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(scores=self.values, **extra), indent=1))


def per_game(values, tag, field, modes):
    return np.array([np.mean([values[f"{tag}|{g}|{m}"][field] for m in modes]) for g in GAMES])


def show_table(values, reference_tag, title, tags=None):
    """Per-game held-out table against the submission recipe (reference_tag in the same dict)."""
    tags = tags or sorted({key.split("|")[0] for key in values}, key=lambda t: list(values).index(next(k for k in values if k.startswith(t + "|"))))
    print("\n" + title)
    for modes, name in ((c.MODES, "standard conditions (clean/moderate/stress)"), (BLIND, "label-blind ball conditions (interp/interpclean/blind3)")):
        if not any(f"{tags[0]}|1|{m}" in values for m in modes):
            continue
        print(f"-- {name}")
        print(f"{'':<34}{'consistent g1 g2 g3 (mean)':<34}{'alternative g1 g2 g3 (mean)':<34}{'equal':<8}{'d cons':<22}{'d alt':<22}{'d eq':<8}{'M':<7}{'/min':<6}{'<3s':<5}")
        ref = {r: per_game(values, reference_tag, r, modes) for r in REFS} if f"{reference_tag}|1|{modes[0]}" in values else None
        for tag in tags:
            if f"{tag}|1|{modes[0]}" not in values:
                continue
            v = {r: per_game(values, tag, r, modes) for r in RNAMES + ("per_live_minute", "share_lt_3s")}
            line = f"{tag:<34}"
            for r in REFS:
                line += "{:<34}".format(" ".join(f"{x:.3f}" for x in v[r]) + f" ({v[r].mean():.3f})")
            line += f"{(v['consistent'].mean() + v['alternative'].mean()) / 2:<8.3f}"
            if ref is not None:
                for r in REFS:
                    line += "{:<22}".format(" ".join(f"{x:+.3f}" for x in v[r] - ref[r]))
                line += f"{((v['consistent'] - ref['consistent']).mean() + (v['alternative'] - ref['alternative']).mean()) / 2:<+8.3f}"
            line += f"{v['M'].mean():<7.3f}{v['per_live_minute'].mean():<6.2f}{v['share_lt_3s'].mean():<5.2f}"
            print(line)


def bar(values, tag, reference_tag):
    """The adoption bar on one row: every game under both references, mean >= 0.015 each, sign on blind."""
    out = {}
    for modes, name in ((c.MODES, "standard"), (BLIND, "blind")):
        if f"{tag}|1|{modes[0]}" not in values:
            continue
        d = {r: per_game(values, tag, r, modes) - per_game(values, reference_tag, r, modes) for r in REFS}
        out[name] = dict(every_game=bool(all((d[r] > 0).all() for r in REFS)), mean={r: round(float(d[r].mean()), 4) for r in REFS},
                         per_game={r: [round(float(x), 4) for x in d[r]] for r in REFS})
    if "standard" in out:
        out["meets_bar_single_seed"] = bool(out["standard"]["every_game"] and all(v >= 0.015 for v in out["standard"]["mean"].values())
                                            and ("blind" not in out or all(v > 0 for v in out["blind"]["mean"].values())))
    return out


# -- B: verifier-only ------------------------------------------------------------------------

def base_conditions(base, modes):
    out = {}
    for g in GAMES:
        for m in modes:
            if output_path(base, g, m).exists():
                k = load(g, m, base)
                out[(g, m)] = (k, decoded(k, k.p0))
    return out


def verifier_study(base="frozen", seed=0, thresholds=(0.0,)):
    conds = base_conditions(base, c.MODES + BLIND)
    table = table_of([d for (g, m), (k, d) in conds.items() if m in c.MODES])
    scores = Scores()
    removed = {}
    prefix = "" if base == "frozen" else base + ":"
    for held in GAMES:
        train = [g for g in GAMES if g != held]
        models = {t: fit_verifier(table, train, t, seed) for t in TARGETS}
        for (g, m), (k, d) in conds.items():
            if g != held:
                continue
            scores.record(prefix + "decoder", d, d.codes)
            for limit in (1.5, 2.0, 3.0):                # a-priori duration rule on the prediction (no fit)
                scores.record(prefix + f"decoder+absorb<{limit}s", d, c.to_codes(lab27.duration_rule(d.segs, absorb_s=limit), d.n))
            for t, model in models.items():
                for threshold in thresholds:
                    new, applied = apply_verifier(model, d, threshold)
                    tag = prefix + f"verifier:{t}" + (f"@{threshold}" if threshold else "")
                    scores.record(tag, d, c.to_codes(new, d.n))
                    removed[f"{tag}|{g}|{m}"] = len(applied)
                    if t == "w0.50" and not threshold:
                        scores.record(prefix + "verifier:w0.50+absorb<2.0s", d, c.to_codes(lab27.duration_rule(new, absorb_s=2.0), d.n))
        print("FOLD", held, flush=True)
    name = f"verifier_{base}" + (f"_s{seed}" if seed else "")
    scores.dump(ROOT / f"{name}.json", removed=removed, base=base, seed=seed)
    show_table(scores.values, prefix + "verifier:w0.50", f"B. verifier-only on the {base} decoder (seed {seed}); reference row = {prefix}verifier:w0.50")
    return scores


class Bag:
    """Mean of verifiers fitted on bootstrap resamples of the candidate rows (variance reduction
    of the removal decision; the decision boundary stays 0)."""

    def __init__(self, models):
        self.models = models

    def predict(self, x):
        return np.mean([m.predict(x) for m in self.models], axis=0)


def fit_bag(table, games, target, n=25, seed=0):
    keys = [(g, m) for g in games for m in c.MODES]
    x = np.concatenate([table[key][0] for key in keys])
    y = target_of(np.concatenate([table[key][1] for key in keys]), target)
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(n):
        rows = rng.integers(0, len(x), len(x))
        models.append(k28.Model(kind="gbm", columns=V).fit(x[rows], y[rows]))
    return Bag(models)


def bag_study(base="frozen", seeds=(0, 1), n=25, targets=("w0.50", "M")):
    conds = base_conditions(base, c.MODES + BLIND)
    table = table_of([d for (g, m), (k, d) in conds.items() if m in c.MODES])
    scores = Scores()
    for seed in seeds:
        for held in GAMES:
            models = {t: fit_bag(table, [g for g in GAMES if g != held], t, n, seed) for t in targets}
            for (g, m), (k, d) in conds.items():
                if g == held:
                    for t, model in models.items():
                        scores.record(f"verifier:{t}:bag{n}/seed{seed}", d, c.to_codes(apply_verifier(model, d)[0], d.n))
        print("BAG seed", seed, flush=True)
    scores.dump(ROOT / f"verifier_bag_{base}.json", base=base, n=n)
    return scores


# -- A: second stage aimed at M --------------------------------------------------------------

def fit_stacker(conds, target, seed=0):
    xs, ys, ws = [], [], []
    for k in conds:
        for mirror in (False, True):
            x = s.design(k, mirror)
            for r, w in STACK_TARGETS[target].items():
                y = k.labels[r]
                keep = y >= 0
                label = y[keep].astype(np.int64)
                if mirror:
                    label = np.array([1, 0, 2])[label]
                xs.append(x[keep])
                ys.append(label)
                ws.append(np.full(int(keep.sum()), w))
    model = dict(s.MODEL, random_state=seed)
    return HistGradientBoostingClassifier(**model).fit(np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(ws))


def stack_study(target="M", base="frozen", seed=0, verifier_targets=("w0.50", "M", "M+w0.50", "w0.70", "w1.00"), save=True):
    import joblib
    conds = {(g, m): load(g, m, base) for g in GAMES for m in c.MODES + BLIND if output_path(base, g, m).exists()}
    single = {g: fit_stacker([conds[(g, m)] for m in c.MODES], target, seed) for g in GAMES}
    print("FITTED single-game stackers", target, flush=True)
    scores = Scores()
    removed, emitted = {}, {}
    tag0 = f"stack[{target}]" + ("" if base == "frozen" else f"<{base}>")
    for held in GAMES:
        a, z = [g for g in GAMES if g != held]
        pair = fit_stacker([conds[(g, m)] for g in (a, z) for m in c.MODES], target, seed)
        if save:
            (ROOT / "stackers").mkdir(parents=True, exist_ok=True)
            joblib.dump(pair, ROOT / "stackers" / f"{target}_{base}_s{seed}_held{held}.joblib")
        inner = [conds[(g, m)] for g in (a, z) for m in c.MODES]
        inner_p = [s.emissions(single[z if k.game == a else a], k) for k in inner]
        for k, p in zip(inner, inner_p):                 # out-of-fold emissions of game k.game from the other training game's stacker
            emitted[f"inner_{held}_{k.game}_{k.mode}"] = p.astype(np.float32)
        train = [decoded(k, p) for k, p in zip(inner, inner_p)]
        table = table_of(train)
        models = {t: fit_verifier(table, [a, z], t, seed) for t in verifier_targets}
        for (g, m), k in conds.items():
            if g != held:
                continue
            p = s.emissions(pair, k)
            emitted[f"test_{g}_{m}"] = p.astype(np.float32)
            test = decoded(k, p)
            scores.record(tag0 + ":decoder", test, test.codes)
            scores.record(tag0 + ":decoder+absorb<2.0s", test, c.to_codes(lab27.duration_rule(test.segs, absorb_s=2.0), test.n))
            for t, model in models.items():
                new, applied = apply_verifier(model, test)
                scores.record(tag0 + f":verifier:{t}", test, c.to_codes(new, test.n))
                removed[f"{tag0}:verifier:{t}|{g}|{m}"] = len(applied)
        print("FOLD", held, flush=True)
    scores.dump(ROOT / f"stack_{target}_{base}_s{seed}.json", removed=removed, base=base, seed=seed, target=target)
    np.savez_compressed(ROOT / f"emissions_{target}_{base}_s{seed}.npz", **emitted)
    return scores


def mixed(k, p, weight, part="all"):
    """weight * frozen + (1 - weight) * second stage. part: 'all' (the three probabilities),
    'dead' (only P(D) is mixed, the frozen team split is kept) or 'team' (only the team split
    is mixed, the frozen P(D) is kept) -- the last two attribute the effect to a layer."""
    p = p.astype(np.float64)
    if part == "all":
        return weight * k.p0 + (1 - weight) * p
    share0 = k.p0[:, :2] / np.clip(k.p0[:, :2].sum(1, keepdims=True), 1e-6, None)
    share1 = p[:, :2] / np.clip(p[:, :2].sum(1, keepdims=True), 1e-6, None)
    if part == "dead":
        dead = weight * k.p0[:, 2] + (1 - weight) * p[:, 2]
        return np.column_stack([share0 * (1 - dead[:, None]), dead])
    if part == "team":
        share = weight * share0 + (1 - weight) * share1
        return np.column_stack([share * (1 - k.p0[:, 2:3]), k.p0[:, 2]])
    raise ValueError("Unknown part")


def blend_study(target="M", base="frozen", seed=0, weight=0.5, verifier_targets=("w0.50", "M", "w0.30", "w0.00", "w0.70"),
                parts=("all", "dead", "team")):
    """Emissions = weight * frozen + (1 - weight) * second stage (weight fixed a priori at 0.5),
    verifier refitted nested on the blended out-of-fold emissions. Needs the emissions cache of
    `stack --target ... --seed ...`."""
    if "+" in target:                                    # "alternative:0+M:1+M15:0": mean of several second stages (target:seed)
        specs = [item.split(":") for item in target.split("+")]
        files = [np.load(ROOT / f"emissions_{name}_{base}_s{int(sd)}.npz") for name, sd in specs]
        emitted = {key: np.mean([f[key] for f in files], axis=0) for key in files[0].files}
        target = "+".join(name for name, _ in specs)
    else:
        emitted = np.load(ROOT / f"emissions_{target}_{base}_s{seed}.npz")
    conds = {(g, m): load(g, m, base) for g in GAMES for m in c.MODES + BLIND if output_path(base, g, m).exists()}
    scores = Scores()
    for part in parts:
        tag0 = f"blend{weight:g}{'' if part == 'all' else '-' + part}[{target}]" + ("" if base == "frozen" else f"<{base}>")
        targets = verifier_targets if part == "all" else ("w0.50",)
        for held in GAMES:
            a, z = [g for g in GAMES if g != held]
            train = [decoded(conds[(g, m)], mixed(conds[(g, m)], emitted[f"inner_{held}_{g}_{m}"], weight, part)) for g in (a, z) for m in c.MODES]
            table = table_of(train)
            models = {t: fit_verifier(table, [a, z], t, seed) for t in targets}
            for (g, m), k in conds.items():
                if g != held:
                    continue
                test = decoded(k, mixed(k, emitted[f"test_{g}_{m}"], weight, part))
                scores.record(tag0 + ":decoder", test, test.codes)
                for t, model in models.items():
                    scores.record(tag0 + f":verifier:{t}", test, c.to_codes(apply_verifier(model, test)[0], test.n))
            print("FOLD", part, held, flush=True)
    scores.dump(ROOT / f"stack_blend{weight:g}_{target}_{base}_s{seed}.json", base=base, seed=seed, target=target, weight=weight)
    return scores


def dstart_diagnostic(target="M", base="frozen", seed=1):
    """Reviewer's D-start diagnostic (data/work/review_v30d/metrica_snap.py) for the second stage:
    share of predicted stoppage starts within 2 frames of the ball vanishing, frozen decoder vs
    second stage, standard vs label-blind conditions. Ball availability is read HERE ONLY, to
    measure the artefact; it is not a feature of anything."""
    import joblib
    from .v28d_data import metrica_record
    emitted = np.load(ROOT / f"emissions_{target}_{base}_s{seed}.npz")

    def near(a, b):
        if not len(a) or not len(b):
            return np.full(len(a), 10 ** 6)
        i = np.searchsorted(b, a)
        return np.minimum(np.abs(b[np.clip(i - 1, 0, len(b) - 1)] - a), np.abs(b[np.clip(i, 0, len(b) - 1)] - a))

    rows = {}
    for g in GAMES:
        for m in c.MODES + BLIND:
            if not output_path(base, g, m).exists():
                continue
            k = load(g, m, base)
            r = joblib.load(f"data/work/review_v30d/records/game{g}_{m}.joblib") if m in BLIND else metrica_record(g, m)
            seen = r["ball_seen"]
            vanish = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
            reference = k.references["consistent"]
            graded = reference >= 0
            ref_starts = np.flatnonzero((reference[1:] == 2) & (reference[:-1] != 2) & graded[1:] & graded[:-1]) + 1
            row = dict(seen_given_dead=float(seen[reference == 2].mean()), reference=float((near(ref_starts, vanish) <= 2).mean()))
            for name, p in (("frozen", k.p0), ("stack", emitted[f"test_{g}_{m}"])):
                y = decoded(k, p).codes
                starts = np.flatnonzero((y[1:] == 2) & (y[:-1] != 2) & graded[1:] & graded[:-1]) + 1
                row[name] = float((near(starts, vanish) <= 2).mean())
                row[name + "_ref_start_found_1s"] = float((near(ref_starts, starts) <= 25).mean())
            rows[f"{g}_{m}"] = row
            print("DSTART", g, m, json.dumps({key: round(v, 3) for key, v in row.items()}), flush=True)
    (ROOT / f"dstart_{target}_{base}_s{seed}.json").write_text(json.dumps(rows, indent=1))
    for modes, name in ((c.MODES, "standard"), (BLIND, "label-blind")):
        keys = [f"{g}_{m}" for g in GAMES for m in modes if f"{g}_{m}" in rows]
        if keys:
            print(name, {field: round(float(np.mean([rows[key][field] for key in keys])), 3) for field in rows[keys[0]]})


# -- reporting -------------------------------------------------------------------------------

def collect():
    values = {}
    for path in sorted(ROOT.glob("verifier_*.json")) + sorted(ROOT.glob("stack_*.json")) + sorted(ROOT.glob("insert_*.json")):
        saved = json.loads(path.read_text())
        suffix = f" s{saved['seed']}" if saved.get("seed") else ""
        for key, row in saved["scores"].items():
            tag, g, m = key.split("|")
            values[f"{tag}{suffix}|{g}|{m}"] = row
    return values


def show():
    values = collect()
    show_table(values, "verifier:w0.50", "V31a held-out official spellF1 per Metrica game; d = versus the SUBMISSION RECIPE (frozen decoder + verifier w0.50)")
    report = {tag: bar(values, tag, "verifier:w0.50") for tag in sorted({key.split("|")[0] for key in values})}
    (ROOT / "bar.json").write_text(json.dumps(report, indent=1))
    print("\nRows meeting the single-seed adoption bar:", [t for t, r in report.items() if r.get("meets_bar_single_seed")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["verifier", "bag", "stack", "blend", "dstart", "show"])
    parser.add_argument("--target", default="M")
    parser.add_argument("--base", default="frozen")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.step == "verifier":
        verifier_study(args.base, args.seed)
    elif args.step == "stack":
        stack_study(args.target, args.base, args.seed)
        show_table(collect(), "verifier:w0.50", "after stack " + args.target) if (ROOT / "verifier_frozen.json").exists() else None
    elif args.step == "bag":
        bag_study(args.base)
    elif args.step == "dstart":
        dstart_diagnostic(args.target, args.base, args.seed)
    elif args.step == "blend":
        blend_study(args.target, args.base, args.seed)
    else:
        show()
