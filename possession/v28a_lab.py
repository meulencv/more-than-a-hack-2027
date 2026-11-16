"""V28a experiments: a physically informed spell verifier, leave-one-game-out on Metrica's games.

    .venv/bin/python -m possession.v28a_lab signals   # per-frame tracking signals, 9 conditions
    .venv/bin/python -m possession.v28a_lab headroom  # oracle ceilings: removals, D removals, splits
    .venv/bin/python -m possession.v28a_lab table     # candidate removals + exact metric deltas
    .venv/bin/python -m possession.v28a_lab logo      # model comparison, leave-one-game-out (+ nested)
    .venv/bin/python -m possession.v28a_lab importance  # held-out discrimination + permutation importance
    .venv/bin/python -m possession.v28a_lab fit       # final models on the three games
    .venv/bin/python -m possession.v28a_lab shift     # label-free covariate check against the challenge match

Set OMP_NUM_THREADS=1 when the machine is shared (the models are tiny; threads only contend).

No challenge ground truth is read. Inputs: the V27 references (data/work/v27/metrica_{g}.npz),
the cached frozen-ensemble outputs (data/work/v16) and the open Metrica tracking, re-degraded
with the seeds of possession.v16_transfer so the tracking is what the ensemble saw.
"""
import argparse
import json
import numpy as np
from . import v27_core as c
from . import v27_lab as v27
from . import v28a_core as k28

MAX_FRAMES = v27.MAX_FRAMES
GAMES = (1, 2, 3)
STATES = 3                                  # baseline decode + two oracle passes (training only)
N27 = len(c.FEATURES)
BLOCKS = {}
_first = 0
for _name, _names in (("v27", c.FEATURES), ("physical", k28.PHYSICAL), ("structure", k28.STRUCTURE), ("shape", k28.SHAPE)):
    BLOCKS[_name] = list(range(_first, _first + len(_names)))
    _first += len(_names)

# -- data ------------------------------------------------------------------------------------

def write_signals():
    from .sources import challenge_match
    from .degrade import challenge_profile
    k28.ROOT.mkdir(parents=True, exist_ok=True)
    profile = challenge_profile(challenge_match())
    for game in GAMES:
        for mode in c.MODES:
            s = k28.metrica_signals(game, mode, profile)
            np.savez_compressed(k28.ROOT / f"signals_{game}_{mode}.npz", **s)
            print("SIGNALS", game, mode, "ball tracked", round(float(s["seen"].mean()), 3), flush=True)


def prepared():
    out = v27.prepared()
    for k in out:
        k.signals = k28.cached_signals(k.game, k.mode)
    return out


def exact_deltas(segs, cands, gts):
    delta = np.zeros((len(cands), len(c.REFERENCES)))
    for j, reference in enumerate(c.REFERENCES):
        gt = gts[reference]
        base = c.spell_f1(gt, c.spells_of(segs))
        for row, (i, _, new) in enumerate(cands):
            delta[row, j] = c.spell_f1(gt, c.spells_of(c.relabel(segs, {i: new}))) - base
    return delta


def purity_gaps(segs, cands, references):
    """Share of the spell's frames the reference gives to the new code minus to the spell's team.

    A smoother view of the same question than the exact metric change: > 0 means the reference
    agrees more with the removal than with the predicted possession."""
    gap = np.zeros((len(cands), len(c.REFERENCES)))
    for j, reference in enumerate(c.REFERENCES):
        labels = references[reference]
        for row, (i, _, new) in enumerate(cands):
            a, z, k = segs[i]
            gap[row, j] = (labels[a:z] == new).mean() - (labels[a:z] == k).mean()
    return gap


def oracle_pass(segs, gt, max_frames=MAX_FRAMES):
    """One greedy pass of exact-gain removals (never two neighbours). Uses the reference."""
    score = c.spell_f1(gt, c.spells_of(segs))
    gains = []
    for i, _, new in c.candidates(segs, max_frames):
        f = c.spell_f1(gt, c.spells_of(c.relabel(segs, {i: new})))
        if f > score + 1e-12:
            gains.append((f - score, i, new))
    changes = {}
    for _, i, new in sorted(gains, reverse=True):
        if not any(i + d in changes for d in (-2, -1, 0, 1, 2)):
            changes[i] = new
    return c.relabel(segs, changes), len(changes)


def build_table():
    """Candidates with features, exact spellF1 deltas and purity gaps, in the baseline state and
    after one and two oracle passes (to test training on later-pass states too)."""
    out = {}
    for k in prepared():
        segs = k.segs
        for state in range(STATES):
            cands = c.candidates(segs, MAX_FRAMES)
            tag = f"{k.game}_{k.mode}_{state}"
            out["x_" + tag] = k28.candidate_features(segs, cands, k.p, k.edge, k.signals)
            out["d_" + tag] = d = exact_deltas(segs, cands, k.gt)
            out["u_" + tag] = purity_gaps(segs, cands, k.references)
            out["a_" + tag] = np.array([c.ACTIONS.index(action) for _, action, _ in cands])
            print("TABLE", k.game, k.mode, state, len(cands), "positive", (d > 0).mean(0).round(3), flush=True)
            segs, _ = oracle_pass(segs, k.gt["consistent"])
    np.savez_compressed(k28.ROOT / "candidate_table.npz", **out)


def training(table, games, target="consistent", states=(0,), modes=c.MODES):
    """x, exact gain, purity gap of the candidates of `games` under the `target` reference."""
    keys = [f"{g}_{m}_{s}" for g in games for m in modes for s in states]
    j = c.REFERENCES.index(target)
    return (np.concatenate([table["x_" + key] for key in keys]),
            np.concatenate([table["d_" + key][:, j] for key in keys]),
            np.concatenate([table["u_" + key][:, j] for key in keys]))


# -- models (defined in v28a_core so a saved verifier unpickles from any entry point) --------

Model, MONOTONE, GBM = k28.Model, k28.MONOTONE, k28.GBM


V, P, S, H = (BLOCKS[name] for name in ("v27", "physical", "structure", "shape"))
P_NO_SEEN = [j for j in P if k28.FEATURES[j] not in ("seen_in", "pre_seen", "post_seen")]
VARIANTS = {
    "v27_gbm": dict(kind="gbm", columns=V),                 # = possession.v27_lab verifier, mode consistent
    "v27_blend": dict(kind="blend", columns=V),             # the new target alone, without new features
    "v27_shape_gbm": dict(kind="gbm", columns=V + H),
    "v27_structure_gbm": dict(kind="gbm", columns=V + S),
    "physical_gbm": dict(kind="gbm", columns=V + P),
    "physical_blend": dict(kind="blend", columns=V + P),
    "physical_noseen_blend": dict(kind="blend", columns=V + P_NO_SEEN),          # no ball-availability features
    "physical_live_blend": dict(kind="blend", columns=V + P_NO_SEEN, live_only=True),   # physics only for T-X-T
    "physical_live_gbm": dict(kind="gbm", columns=V + P_NO_SEEN, live_only=True),
    "all_gbm": dict(kind="gbm"),
    "all_gbm_mono": dict(kind="gbm_mono"),
    "all_gbc": dict(kind="gbc"),
    "all_logit": dict(kind="logit"),
    "all_ridge": dict(kind="ridge"),
    "all_blend": dict(kind="blend"),
    "all_blend_mono": dict(kind="blend_mono"),
}
SELECTABLE = [name for name in VARIANTS if name != "v27_gbm"]


def scorer(model, k):
    def score(segs, cands):
        return model.predict(k28.candidate_features(segs, cands, k.p, k.edge, k.signals))
    return score


def evaluate(conditions, model, threshold=0.0, passes=3):
    rows = []
    for k in conditions:
        new, applied = c.apply_scored(k.segs, scorer(model, k), threshold, MAX_FRAMES, passes)
        row = dict(game=k.game, mode=k.mode, applied=len(applied), n_after=len(c.spells_of(new)))
        for reference in c.REFERENCES:
            row[reference] = c.spell_f1(k.gt[reference], c.spells_of(new)) - c.spell_f1(k.gt[reference], c.spells_of(k.segs))
        rows.append(row)
    return rows


def summarise(rows):
    out = dict(spells_after=[r["n_after"] for r in rows])
    for reference in c.REFERENCES:
        delta = np.array([r[reference] for r in rows]).reshape(3, 3)
        out[reference] = dict(mean=round(float(delta.mean()), 4), per_game=delta.mean(1).round(4).tolist(),
                              per_mode=delta.mean(0).round(4).tolist(), worst_condition=round(float(delta.min()), 4),
                              per_condition=delta.round(4).tolist())
    return out


def fitted(table, games, variant, states=(0,), modes=c.MODES):
    return Model(**VARIANTS[variant]).fit(*training(table, games, "consistent", states, modes))


def logo(table, conditions, variant, states=(0,), modes=c.MODES, threshold=0.0, passes=3):
    rows = []
    for held in GAMES:
        model = fitted(table, [g for g in GAMES if g != held], variant, states, modes)
        rows += evaluate([k for k in conditions if k.game == held], model, threshold, passes)
    return summarise(rows)


def compare():
    """Leave-one-game-out for every variant, then a nested choice: for each held-out game the
    variant is chosen on the two other games only (fit on one, score the other, both ways)."""
    table = np.load(k28.ROOT / "candidate_table.npz")
    conditions = prepared()
    report = dict(baseline={r: np.array([c.spell_f1(k.gt[r], c.spells_of(k.segs)) for k in conditions]).reshape(3, 3).round(4).tolist()
                            for r in c.REFERENCES}, variants={}, extras={})
    for name in VARIANTS:
        report["variants"][name] = logo(table, conditions, name)
        print("LOGO", name, json.dumps(report["variants"][name]), flush=True)
    extras = dict(v27_gbm_threshold_0005=("v27_gbm", dict(threshold=0.0005)),     # = evaluation/results/v27_semantic_bet.csv
                  adopted_passes_6=(ADOPTED, dict(passes=6)),
                  adopted_oracle_states=(ADOPTED, dict(states=(0, 1, 2))),
                  adopted_degraded_only=(ADOPTED, dict(modes=("moderate", "stress"))))
    for name, (variant, arguments) in extras.items():
        report["extras"][name] = logo(table, conditions, variant, **arguments)
        print("EXTRA", name, json.dumps(report["extras"][name]), flush=True)
    inner, rows = {}, []
    for held in GAMES:
        first, second = [g for g in GAMES if g != held]
        score = {}
        for name in SELECTABLE:
            deltas = []
            for fit_game, score_game in ((first, second), (second, first)):
                model = fitted(table, [fit_game], name)
                deltas += [r["consistent"] for r in evaluate([k for k in conditions if k.game == score_game], model)]
            score[name] = float(np.mean(deltas))
        chosen = max(score, key=score.get)
        inner[held] = dict(chosen=chosen, inner_scores={n: round(v, 4) for n, v in score.items()})
        rows += evaluate([k for k in conditions if k.game == held], fitted(table, [first, second], chosen))
        print("NESTED", held, json.dumps(inner[held]), flush=True)
    report["nested"] = dict(choice=inner, **summarise(rows))
    print("NESTED RESULT", json.dumps(report["nested"]), flush=True)
    (k28.ROOT / "logo.json").write_text(json.dumps(report, indent=1))


def importance(variant=None, repeats=5):
    """Held-out discrimination (AUC of "the removal helps") and permutation importance."""
    from sklearn.metrics import roc_auc_score
    variant = variant or adopted_variant()
    table = np.load(k28.ROOT / "candidate_table.npz")
    seen = k28.FEATURES.index("seen_in")
    held_out = {}
    for name in ("v27_gbm", variant):
        scores, gains, modes, tracked, actions = [], [], [], [], []
        for held in GAMES:
            model = fitted(table, [g for g in GAMES if g != held], name)
            for mode in c.MODES:
                x = table[f"x_{held}_{mode}_0"]
                scores.append(model.predict(x))
                gains.append(table[f"d_{held}_{mode}_0"][:, 0])
                modes += [mode] * len(x)
                tracked.append(x[:, seen])
                actions.append(table[f"a_{held}_{mode}_0"])
        scores, gains, modes, tracked, actions = (np.concatenate(scores), np.concatenate(gains), np.array(modes),
                                                  np.concatenate(tracked), np.concatenate(actions))
        row = dict(overall=round(float(roc_auc_score(gains > 0, scores)), 3))
        for mode in c.MODES:
            m = modes == mode
            row[mode] = dict(all=round(float(roc_auc_score(gains[m] > 0, scores[m])), 3),
                             ball_tracked_over_80pct=round(float(roc_auc_score(gains[m & (tracked > 0.8)] > 0, scores[m & (tracked > 0.8)])), 3),
                             ball_tracked_under_20pct=round(float(roc_auc_score(gains[m & (tracked < 0.2)] > 0, scores[m & (tracked < 0.2)])), 3),
                             n_over_80=int((m & (tracked > 0.8)).sum()), n_under_20=int((m & (tracked < 0.2)).sum()),
                             **{c.ACTIONS[a]: round(float(roc_auc_score(gains[m & (actions == a)] > 0, scores[m & (actions == a)])), 3)
                                for a in range(len(c.ACTIONS))})
        held_out[name] = row
        print("AUC", name, json.dumps(row), flush=True)
    rng = np.random.default_rng(0)
    drop = np.zeros((len(GAMES), len(k28.FEATURES)))
    for g, held in enumerate(GAMES):
        model = fitted(table, [x for x in GAMES if x != held], variant)
        x = np.concatenate([table[f"x_{held}_{m}_0"] for m in c.MODES])
        y = np.concatenate([table[f"d_{held}_{m}_0"][:, 0] for m in c.MODES]) > 0
        base = roc_auc_score(y, model.predict(x))
        for j in model.columns:
            for _ in range(repeats):
                shuffled = x.copy()
                shuffled[:, j] = rng.permutation(shuffled[:, j])
                drop[g, j] += (base - roc_auc_score(y, model.predict(shuffled))) / repeats
    order = np.argsort(-drop.mean(0))
    block_of = {j: name for name, columns in BLOCKS.items() for j in columns}
    ranking = [dict(feature=k28.FEATURES[j], block=block_of[j], auc_drop=round(float(drop[:, j].mean()), 4),
                    per_game=drop[:, j].round(4).tolist()) for j in order]
    blocks = {name: round(float(drop[:, columns].sum(1).mean()), 4) for name, columns in BLOCKS.items()}
    for row in ranking[:25]:
        print("IMPORTANCE", json.dumps(row), flush=True)
    print("IMPORTANCE BY BLOCK (sum of AUC drops)", json.dumps(blocks), flush=True)
    (k28.ROOT / "importance.json").write_text(json.dumps(dict(variant=variant, held_out_auc=held_out, by_block=blocks, ranking=ranking), indent=1))


def headroom():
    """Exact-gain oracles of the other action kinds on the baseline decode (consistent reference):
    removing a short predicted stoppage between possessions, and ADDING a possession where a
    cheaper team-only decode sees an opposite-team episode inside a predicted possession."""
    def f1(gt, codes, blocks):
        return c.spell_f1(gt, c.spells_of(c.segments(codes, blocks)))

    def greedy(codes, candidates_of, gt, blocks, passes):
        score, applied = f1(gt, codes, blocks), 0
        for _ in range(passes):
            gains = []
            for a, z, new in candidates_of(codes):
                trial = codes.copy()
                trial[a:z] = new
                gain = f1(gt, trial, blocks) - score
                if gain > 1e-12:
                    gains.append((gain, a, z, new))
            used = []
            for _, a, z, new in sorted(gains, reverse=True):
                if any(not (z + 10 * c.FPS <= ua or a >= uz + 10 * c.FPS) for ua, uz in used):
                    continue
                codes = codes.copy()
                codes[a:z] = new
                used.append((a, z))
                applied += 1
            if not used:
                break
            score = f1(gt, codes, blocks)
        return score, applied

    report = []
    for k in v27.prepared():
        base = c.to_codes(k.segs, k.n)
        gt = k.gt["consistent"]
        start = f1(gt, base, k.blocks)

        def dead_islands(codes):
            segs, out = c.segments(codes, k.blocks), set()
            for i, (a, z, code) in enumerate(segs):
                if code != 2 or z - a > MAX_FRAMES:
                    continue
                for j in (i - 1, i + 1):
                    if 0 <= j < len(segs) and segs[j][2] < 2 and (segs[j][1] == a or segs[j][0] == z):
                        out.add((a, z, segs[j][2]))
            return sorted(out)

        cheap = c.redecode_teams(k.interpolated, base, [(2.0, 0.85), (1.0, 0.85)])
        row = dict(game=k.game, mode=k.mode, baseline=round(start, 4),
                   remove_possession_le_8s=round(v27.oracle(k.segs, gt, MAX_FRAMES) - start, 4))
        score, applied = greedy(base, dead_islands, gt, k.blocks, 4)
        row.update(remove_stoppage_le_8s=round(score - start, 4), stoppages_removed=applied, stoppage_candidates=len(dead_islands(base)))
        for name, alternative in (("split_cost_2", cheap[0]), ("split_cost_1", cheap[1])):
            def splits(codes, alternative=alternative):
                marked = np.where((alternative != base) & (base < 2), alternative + 10, 5).astype(np.int8)
                return [(a, z, code - 10) for a, z, code in c.segments(marked, k.blocks) if code >= 10]
            score, applied = greedy(base, splits, gt, k.blocks, 1)
            row.update({name: round(score - start, 4), name + "_applied": applied, name + "_candidates": len(splits(base))})
        report.append(row)
        print("HEADROOM", json.dumps(row), flush=True)
    (k28.ROOT / "headroom.json").write_text(json.dumps(report, indent=1))


# Adopted for the challenge match. `physical_blend` has the best held-out mean, but next to a
# stoppage its physical block meets a different data source on the challenge (see `shift`): the
# open Metrica games never track the ball while play is stopped, the challenge tracker often
# does. `physical_live_blend` uses the physics only for T-X-T candidates, where both sources
# look alike, and still beats the V27 verifier on every held-out game.
ADOPTED = "physical_live_blend"
FULL = "physical_blend"


def adopted_variant():
    return ADOPTED


def fit_final():
    import joblib
    table = np.load(k28.ROOT / "candidate_table.npz")
    for variant, name in ((ADOPTED, "verifier.joblib"), (FULL, "verifier_full.joblib")):
        model = fitted(table, list(GAMES), variant)
        joblib.dump(dict(model=model, features=k28.FEATURES, max_frames=MAX_FRAMES, variant=variant, target="consistent",
                         parameters=GBM), k28.ROOT / name)
        print("FITTED", variant, name, flush=True)


def shift():
    """Label-free covariate check: the candidates of the challenge match next to those of the
    Metrica conditions, separately for T-X-T (absorb) and for candidates next to a stoppage."""
    from .v28a_verifier import challenge_candidates
    x = challenge_candidates()
    table = np.load(k28.ROOT / "candidate_table.npz")
    reference = {m: np.concatenate([table[f"x_{g}_{m}_0"] for g in GAMES]) for m in c.MODES}
    absorb = k28.FEATURES.index("is_absorb")
    names = ["duration_s", "own_mean", "seen_in", "near_own_share", "d_own_med", "d_own_min", "own_close_s", "own_carry_share",
             "free_share", "speed_mean", "speed_max", "straight", "events", "line_dev_m", "speed_before", "speed_after",
             "pre_near_opp", "post_near_opp"]
    report = dict(candidates=dict(challenge=len(x), **{m: len(reference[m]) // 3 for m in c.MODES}), groups={})
    for group, flag in (("between_two_possessions", 1.0), ("next_to_a_stoppage", 0.0)):
        rows = {}
        for name in names:
            j = k28.FEATURES.index(name)
            row = {}
            for source, values in [("challenge", x)] + list(reference.items()):
                v = values[values[:, absorb] == flag][:, j]
                row[source] = dict(median=round(float(np.nanmedian(v)), 2) if np.isfinite(v).any() else None,
                                   finite=round(float(np.isfinite(v).mean()), 2))
            rows[name] = row
            print("SHIFT", group, name, json.dumps(row), flush=True)
        report["groups"][group] = rows
    (k28.ROOT / "shift.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    steps = dict(signals=write_signals, headroom=headroom, table=build_table, logo=compare, importance=importance, fit=fit_final, shift=shift)
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=list(steps))
    steps[parser.parse_args().step]()
