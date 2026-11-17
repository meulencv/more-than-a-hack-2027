"""V31a insertion candidates: short possessions the decoder did not open, read from the emissions.

The verifier of the submission can only REMOVE predicted possessions. The compromise target M
keeps every sandwiched possession of >= 2 s, and the insertion oracle is large under the
alternative reading, so this module asks whether a possession the frozen decoder merged into a
longer one of the other team can be recovered from the same network outputs:

    candidates   runs of the other team in MORE LIBERAL decodes of the same emissions (lower
                 change-of-team cost / higher unary weight) that lie strictly inside one predicted
                 possession (T-X-T after insertion). Nothing but probabilities and edge scores.
    features     possession.v27_core.candidate_features of the inserted possession (the V27
                 feature set, as if it had been decoded) + the liberal level that found it +
                 its position inside the host possession.
    target       exact official-metric gain of the insertion under M (or the equal prior).
    decision     predicted gain > 0, never tuned.

Two uses:
    liberal      liberal decode -> verifier refitted on that decode (removal only, more candidates)
    insert       submission recipe (frozen decode + verifier w0.50) -> learned insertions

    .venv/bin/python -m possession.v31a_insert oracle      # how much insertions could give (data/work/v31a/insert_oracle.json)
    .venv/bin/python -m possession.v31a_insert liberal     # data/work/v31a/insert_liberal.json
    .venv/bin/python -m possession.v31a_insert insert      # data/work/v31a/insert_learned.json

Nested leave-one-game-out: for held-out g the recipe output of training game A comes from the
verifier fitted on B alone (and vice versa), the insertion model is fitted on A and B, and is
applied to g on the output of the verifier fitted on A and B. No challenge label is read.
"""
import argparse
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from . import v27_core as c
from . import v27_lab as lab27
from . import v28a_core as k28
from . import v31a_lab as L

ROOT = L.ROOT
GAMES = L.GAMES
FPS = c.FPS
LIBERAL = [dict(c.DEFAULT, c_ha=3.0), dict(c.DEFAULT, c_ha=2.0), dict(c.DEFAULT, c_ha=2.0, temp=0.35), dict(c.DEFAULT, c_ha=1.0, temp=0.35)]
MIN_FRAMES = 12                                          # 0.5 s
MARGIN = 5                                               # frames from the host's ends
EXTRA = ("level", "from_start_s", "to_end_s", "host_s")
GBM = dict(k28.GBM)


def liberal_codes(k):
    return c.decode(k.interpolated, k.n, LIBERAL)


def insertions(segs, liberal):
    """[(host index, start, stop, team, level)] non-overlapping; the least liberal decode wins."""
    out, taken = [], np.zeros(liberal.shape[1], dtype=bool)
    hosts = [(i, a, z, t) for i, (a, z, t) in enumerate(segs) if t < 2]
    for level in range(len(liberal)):
        codes = liberal[level]
        for i, a, z, t in hosts:
            window = codes[a:z]
            other = (window == 1 - t).astype(np.int8)
            edges = np.flatnonzero(np.diff(np.concatenate(([0], other, [0]))))
            for ra, rz in zip(edges[::2] + a, edges[1::2] + a):
                if rz - ra < MIN_FRAMES or ra - a < MARGIN or z - rz < MARGIN or taken[ra:rz].any():
                    continue
                taken[max(a, ra - MARGIN):min(z, rz + MARGIN)] = True
                out.append((i, int(ra), int(rz), 1 - t, level))
    return sorted(out, key=lambda item: item[1])


def inserted(segs, items):
    """Segmentation with the given insertions applied (items of one host must not overlap)."""
    by_host = {}
    for item in items:
        by_host.setdefault(item[0], []).append(item)
    out = []
    for i, (a, z, t) in enumerate(segs):
        position = a
        for _, ra, rz, team, _ in sorted(by_host.get(i, []), key=lambda item: item[1]):
            out += [(position, ra, t), (ra, rz, team)]
            position = rz
        out.append((position, z, t))
    return out


def features(segs, items, p, edge):
    rows = np.zeros((len(items), len(c.FEATURES) + len(EXTRA)))
    for row, item in enumerate(items):
        i, ra, rz, team, level = item
        a, z, _ = segs[i]
        new = inserted(segs, [item])
        index = next(j for j, seg in enumerate(new) if seg[0] == ra and seg[1] == rz)
        rows[row, :len(c.FEATURES)] = c.candidate_features(new, [(index, "absorb", 1 - team)], p, edge)[0]
        rows[row, len(c.FEATURES):] = [level, (ra - a) / FPS, (z - rz) / FPS, (z - a) / FPS]
    return rows


def gains(segs, items, gt):
    delta = np.zeros((len(items), len(L.RNAMES)))
    spells = c.spells_of(segs)
    for j, r in enumerate(L.RNAMES):
        base = c.spell_f1(gt[r], spells)
        for row, item in enumerate(items):
            delta[row, j] = c.spell_f1(gt[r], c.spells_of(inserted(segs, [item]))) - base
    return delta


def greedy_oracle(segs, items, gt, passes=3):
    """Insert every candidate that raises spellF1 under gt, best first, re-evaluated each pass."""
    chosen = []
    remaining = list(items)
    for _ in range(passes):
        current = inserted(segs, chosen)
        base = c.spell_f1(gt, c.spells_of(current))
        scored = [(c.spell_f1(gt, c.spells_of(inserted(segs, chosen + [item]))) - base, item) for item in remaining]
        good = [item for g, item in sorted(scored, key=lambda x: -x[0]) if g > 1e-12]
        if not good:
            break
        chosen += good
        remaining = [item for item in remaining if item not in good]
    return inserted(segs, chosen), len(chosen)


# -- recipe outputs (nested) -----------------------------------------------------------------

def recipe_outputs(base="frozen", target="w0.50", modes=c.MODES + L.BLIND):
    """{(game, mode): (decoded condition, held-out recipe segments, training-fold recipe segments {other game: segs})}"""
    conds = L.base_conditions(base, modes)
    table = L.table_of([d for (g, m), (k, d) in conds.items() if m in c.MODES])
    single = {g: L.fit_verifier(table, [g], target) for g in GAMES}
    pair = {held: L.fit_verifier(table, [g for g in GAMES if g != held], target) for held in GAMES}
    out = {}
    for (g, m), (k, d) in conds.items():
        held = L.apply_verifier(pair[g], d)[0]
        nested = {o: L.apply_verifier(single[o], d)[0] for o in GAMES if o != g} if m in c.MODES else {}
        out[(g, m)] = (d, held, nested)
    return out


def oracle(base="frozen"):
    outputs = recipe_outputs(base, modes=c.MODES)
    scores = L.Scores()
    counts = {}
    for (g, m), (d, segs, _) in outputs.items():
        items = insertions(segs, liberal_codes(d))
        delta = gains(segs, items, d.gt)
        duration = np.array([(rz - ra) / FPS for _, ra, rz, _, _ in items])
        counts[f"{g}_{m}"] = dict(n=len(items), ge2s=int((duration >= 2).sum()),
                                  positive={r: int((delta[:, j] > 0).sum()) for j, r in enumerate(L.RNAMES)},
                                  negative={r: int((delta[:, j] < 0).sum()) for j, r in enumerate(L.RNAMES)},
                                  positive_both=int(((delta[:, 0] > 0) & (delta[:, 1] > 0)).sum()),
                                  positive_M_ge2s=int(((delta[:, 2] > 0) & (duration >= 2)).sum()))
        scores.record("verifier:w0.50", d, c.to_codes(segs, d.n))
        scores.record("insert:all", d, c.to_codes(inserted(segs, items), d.n))
        scores.record("insert:all>=2s", d, c.to_codes(inserted(segs, [it for it, s_ in zip(items, duration) if s_ >= 2]), d.n))
        for r in L.RNAMES[:3]:
            new, n = greedy_oracle(segs, items, d.gt[r])
            scores.record(f"insert:oracle[{r}]", d, c.to_codes(new, d.n))
        print("ORACLE", g, m, json.dumps(counts[f"{g}_{m}"]), flush=True)
    scores.dump(ROOT / "oracle_insert.json", counts=counts)
    L.show_table(scores.values, "verifier:w0.50", "Insertion candidates on the submission recipe: oracles (reference information used; NOT held-out results)")


# -- learned insertion -----------------------------------------------------------------------

def fit_inserter(x, y, seed=0):
    return HistGradientBoostingRegressor(**dict(GBM, random_state=seed)).fit(x, y)


def apply_inserter(model, d, segs, threshold=0.0):
    items = insertions(segs, liberal_codes(d))
    if not items:
        return segs, 0
    score = model.predict(features(segs, items, d.p, d.edge))
    chosen = [item for item, v in zip(items, score) if v > threshold]
    return inserted(segs, chosen), len(chosen)


def learned(base="frozen", targets=("M", "w0.50", "M+w0.50"), seed=0):
    outputs = recipe_outputs(base)
    train = {}
    for (g, m), (d, _, nested) in outputs.items():
        if m not in c.MODES:
            continue
        liberal = liberal_codes(d)
        for other, segs in nested.items():                 # game g's recipe output from the verifier of `other` alone
            items = insertions(segs, liberal)
            train[(g, m, other)] = (features(segs, items, d.p, d.edge), gains(segs, items, d.gt))
        print("TABLE", g, m, flush=True)
    scores = L.Scores()
    counts = {}
    prefix = "" if base == "frozen" else base + ":"
    for held in GAMES:
        a, z = [g for g in GAMES if g != held]
        keys = [(a, m, z) for m in c.MODES] + [(z, m, a) for m in c.MODES]
        x = np.concatenate([train[key][0] for key in keys])
        delta = np.concatenate([train[key][1] for key in keys])
        models = {t: fit_inserter(x, L.target_of(delta, t), seed) for t in targets}
        for (g, m), (d, segs, _) in outputs.items():
            if g != held:
                continue
            scores.record(prefix + "verifier:w0.50", d, c.to_codes(segs, d.n))
            for t, model in models.items():
                new, n = apply_inserter(model, d, segs)
                scores.record(prefix + f"recipe+insert[{t}]", d, c.to_codes(new, d.n))
                counts[f"{t}|{g}|{m}"] = n
        print("FOLD", held, flush=True)
    scores.dump(ROOT / (f"insert_learned_{base}" + (f"_s{seed}" if seed else "") + ".json"), counts=counts, base=base, seed=seed)
    L.show_table(scores.values, prefix + "verifier:w0.50", "Learned insertions on the submission recipe (held-out, nested)")


def liberal(base="frozen", targets=("w0.50", "M"), seed=0):
    """Liberal decode + verifier refitted on that decode (the verifier sees the extra possessions as candidates)."""
    scores = L.Scores()
    for level, config in enumerate(LIBERAL):
        conds = {}
        for g in GAMES:
            for m in c.MODES + L.BLIND:
                if L.output_path(base, g, m).exists():
                    k = L.load(g, m, base)
                    conds[(g, m)] = L.decoded(k, k.p0, config)
        table = L.table_of([d for (g, m), d in conds.items() if m in c.MODES])
        name = f"liberal[c_ha={config['c_ha']},temp={config['temp']}]"
        for held in GAMES:
            models = {t: L.fit_verifier(table, [g for g in GAMES if g != held], t, seed) for t in targets}
            for (g, m), d in conds.items():
                if g != held:
                    continue
                scores.record(name + ":decoder", d, d.codes)
                for t, model in models.items():
                    scores.record(name + f":verifier:{t}", d, c.to_codes(L.apply_verifier(model, d)[0], d.n))
        print("LEVEL", level, flush=True)
    scores.dump(ROOT / f"insert_liberal_{base}.json", base=base, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["oracle", "learned", "liberal"])
    parser.add_argument("--base", default="frozen")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dict(oracle=oracle, learned=lambda b: learned(b, seed=args.seed), liberal=lambda b: liberal(b, seed=args.seed))[args.step](args.base)
