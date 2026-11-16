"""V27 experiments on the three open Metrica games (leave-one-game-out). No challenge labels.

    .venv/bin/python -m possession.v27_lab prepare    # three event-derived references per game
    .venv/bin/python -m possession.v27_lab measure    # spell-duration distributions + baseline scores
    .venv/bin/python -m possession.v27_lab decoder    # switch-cost sweeps (3-state and team-only)
    .venv/bin/python -m possession.v27_lab rules      # a-priori duration rules + oracle headroom
    .venv/bin/python -m possession.v27_lab table      # candidate removals with exact metric deltas
    .venv/bin/python -m possession.v27_lab verifier   # learned spell verifier, leave-one-game-out
    .venv/bin/python -m possession.v27_lab fit        # final verifiers on the three games

Inputs: data/external/metrica_sample and the frozen-ensemble outputs on those games written by
`python -m possession.v16_transfer` (data/work/v16/game{1,2,3}_{clean,moderate,stress}_final.npz).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from . import v27_core as c

MAX_FRAMES = 8 * c.FPS
MODEL = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=25,
             l2_regularization=1.0, early_stopping=False, random_state=0)
THRESHOLDS = (0.0, 0.0005, 0.001, 0.002)


def build_table():
    """Every candidate removal on the baseline decode, with its exact spellF1 change per reference."""
    out = {}
    for k in c.all_conditions():
        segs = c.segments(c.decode(k.interpolated, k.n)[0], k.blocks)
        p, edge = c.dense(k.interpolated, k.n)
        cands = c.candidates(segs, MAX_FRAMES)
        x = c.candidate_features(segs, cands, p, edge)
        delta = np.zeros((len(cands), len(c.REFERENCES)))
        for j, reference in enumerate(c.REFERENCES):
            gt = c.spells_of(c.segments(k.references[reference], k.blocks))
            base = c.spell_f1(gt, c.spells_of(segs))
            for row, (i, _, new) in enumerate(cands):
                delta[row, j] = c.spell_f1(gt, c.spells_of(c.relabel(segs, {i: new}))) - base
        out[f"x_{k.game}_{k.mode}"] = x
        out[f"d_{k.game}_{k.mode}"] = delta
        print("TABLE", k.game, k.mode, len(cands), (delta > 0).mean(0).round(3), flush=True)
    np.savez_compressed(c.ROOT / "candidate_table.npz", **out)


def training(table, games, target):
    x = np.concatenate([table[f"x_{g}_{m}"] for g in games for m in c.MODES])
    d = np.concatenate([table[f"d_{g}_{m}"] for g in games for m in c.MODES])
    j = {r: i for i, r in enumerate(c.REFERENCES)}
    if target == "robust":
        y = np.minimum(d[:, j["consistent"]], d[:, j["alternative"]])
    else:
        y = d[:, j[target]]
    return x, y


def fit(x, y, columns=None):
    model = HistGradientBoostingRegressor(**MODEL).fit(x if columns is None else x[:, columns], y)
    return model


def scorer(model, p, edge, columns=None):
    def score(segs, cands):
        x = c.candidate_features(segs, cands, p, edge)
        return model.predict(x if columns is None else x[:, columns])
    return score


def prepared():
    out = []
    for k in c.all_conditions():
        k.segs = c.segments(c.decode(k.interpolated, k.n)[0], k.blocks)
        k.p, k.edge = c.dense(k.interpolated, k.n)
        k.gt = {r: c.spells_of(c.segments(k.references[r], k.blocks)) for r in c.REFERENCES}
        out.append(k)
    return out


def evaluate(conditions, model, threshold, columns=None):
    rows = []
    for k in conditions:
        new, applied = c.apply_scored(k.segs, scorer(model, k.p, k.edge, columns), threshold, MAX_FRAMES)
        row = dict(game=k.game, mode=k.mode, applied=len(applied), n_before=len(c.spells_of(k.segs)), n_after=len(c.spells_of(new)))
        for reference in c.REFERENCES:
            row[reference + "_before"] = c.spell_f1(k.gt[reference], c.spells_of(k.segs))
            row[reference + "_after"] = c.spell_f1(k.gt[reference], c.spells_of(new))
            row["macro_" + reference] = (c.official(k.references[reference], c.to_codes(new, k.n))["macro_f1"]
                                         - c.official(k.references[reference], c.to_codes(k.segs, k.n))["macro_f1"])
        rows.append(row)
    return rows


def verifier(columns=None, tag="all"):
    """Leave-one-game-out: fit the gain regressor on two games, apply it to the third.

    The whole threshold curve is reported instead of a nested choice; a threshold is the
    predicted spellF1 gain a removal must reach.
    """
    table = np.load(c.ROOT / "candidate_table.npz")
    conditions = prepared()
    report = []
    for target in ("consistent", "robust"):
        models = {held: fit(*training(table, [g for g in (1, 2, 3) if g != held], target), columns) for held in (1, 2, 3)}
        for threshold in THRESHOLDS:
            rows = []
            for held in (1, 2, 3):
                rows += evaluate([k for k in conditions if k.game == held], models[held], threshold, columns)
            summary = dict(target=target, threshold=threshold, spells_after=[r["n_after"] for r in rows],
                           macro_f1_change={r: round(float(np.mean([x["macro_" + r] for x in rows])), 5) for r in c.REFERENCES})
            for reference in c.REFERENCES:
                delta = np.array([r[reference + "_after"] - r[reference + "_before"] for r in rows]).reshape(3, 3)
                summary[reference] = dict(mean=round(float(delta.mean()), 4), per_game=delta.mean(1).round(4).tolist(),
                                          worst_condition=round(float(delta.min()), 4))
            report.append(summary)
            print("VERIFIER", tag, json.dumps(summary), flush=True)
    (c.ROOT / f"verifier_logo_{tag}.json").write_text(json.dumps(report, indent=1))


def prepare():
    """original / alternative (repo references) + 'consistent' (same touch rule in all games).

    Games 1-2 have no CARRY events, so a recovery whose next action is a lost ball struck within
    0.2 s is a touch without control and does not change possession. Game 3 (EPTS/JSON) inserts
    a CARRY after every recovery, which silently switches that rule off (10 recoveries flagged
    instead of ~60). Dropping CARRY rows makes game 3 follow the rule games 1-2 already follow;
    for games 1-2 the result is identical to 'original'.
    """
    from .sources import external_match
    from .reference import event_reference
    from .audit_reference import alternative_reference
    c.ROOT.mkdir(parents=True, exist_ok=True)
    for game in (1, 2, 3):
        source = external_match(game)
        events = source.meta["events"]
        alternative, blocks = alternative_reference(events, len(source.ball))
        consistent, blocks2 = event_reference(events[events.type != "CARRY"].reset_index(drop=True), len(source.ball))
        if blocks != source.blocks or blocks2 != source.blocks:
            raise ValueError("Reference variant changed the graded periods")
        if game < 3 and not np.array_equal(consistent, source.meta["labels"]):
            raise ValueError("Games 1-2 must be unchanged")
        np.savez_compressed(c.ROOT / f"metrica_{game}.npz", original=source.meta["labels"], alternative=alternative,
                            consistent=consistent, blocks=np.array(source.blocks))
        print("PREPARED", game, {r: len(c.spells_of(c.segments(x, source.blocks)))
                                 for r, x in [("original", source.meta["labels"]), ("alternative", alternative), ("consistent", consistent)]}, flush=True)


def measure():
    """Task 1-2: spell statistics of references and of the frozen ensemble, and its official scores."""
    rows = []
    for k in c.all_conditions():
        codes = c.decode(k.interpolated, k.n)[0]
        if k.mode == "clean":
            for reference in c.REFERENCES:
                rows.append(dict(what=f"metrica {k.game} reference {reference}", **c.duration_stats(k.references[reference], k.blocks)))
        row = dict(what=f"metrica {k.game} ensemble {k.mode}", **c.duration_stats(codes, k.blocks))
        for reference in c.REFERENCES:
            score = c.official(k.references[reference], codes)
            row["spell_f1_" + reference] = score["spell_f1"]
            row["macro_f1_" + reference] = score["macro_f1"]
        rows.append(row)
    from . import v13_cv
    for game in v13_cv.settings()["games"]:           # DFL native labels the networks learned
        r = v13_cv.record(game, "clean")
        rows.append(dict(what=f"DFL native {game}", **c.duration_stats(np.where(r["full_labels"] >= 0, r["full_labels"], 3), c.graded_blocks(r["full_labels"]))))
    import pandas as pd
    from games import GAME, half_video_blocks
    for name in ("final_candidate", "v24_camera_fusion", "v27_semantic"):   # predictions only, no ground truth
        path = Path("evaluation/results") / f"{name}.csv"
        if path.exists():
            codes = pd.read_csv(path)["label"].map(dict(H=0, A=1, D=2, X=3)).to_numpy()
            rows.append(dict(what=f"challenge prediction {name}", **c.duration_stats(codes, list(half_video_blocks(GAME)))))
    (c.ROOT / "measure.json").write_text(json.dumps(rows, indent=1))
    for row in rows:
        print("MEASURE", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


def logo_table(scores, base, names, n, tag):
    """scores[reference]: (configs, games, modes). Select on two games ('consistent'), score the third."""
    out = []
    for held in range(3):
        train = [g for g in range(3) if g != held]
        best = int((scores["consistent"][:, train] - base["consistent"][train]).mean((1, 2)).argmax())
        row = dict(family=tag, held_game=held + 1, config=names[best], spells=n[best, held].tolist(),
                   **{r: (scores[r][best, held] - base[r][held]).round(4).tolist() for r in c.REFERENCES})
        out.append(row)
        print("LOGO", json.dumps(row), flush=True)
    return out


def decoder():
    """(a) switch costs. Full 3-state sweep, and a team-only re-decode that keeps every stoppage."""
    full = [dict(c.DEFAULT, c_ha=a, c_to_d=b, c_from_d=d, gamma=g) for a in (2, 3, 4, 6, 8, 12, 16, 24)
            for b in (1.5, 2.5, 4, 6, 9) for d in (1.5, 2.5, 4, 6, 9) for g in (0.5, 0.85, 1.0)]
    team = [(a, g) for a in (3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32) for g in (0., 0.25, 0.5, 0.7, 0.85)]
    shape = (3, 3)
    base = {r: np.zeros(shape) for r in c.REFERENCES}
    s_full = {r: np.zeros((len(full),) + shape) for r in c.REFERENCES}
    s_team = {r: np.zeros((len(team),) + shape) for r in c.REFERENCES}
    n_full = np.zeros((len(full),) + shape)
    n_team = np.zeros((len(team),) + shape)
    for k in c.all_conditions():
        g, m = k.game - 1, c.MODES.index(k.mode)
        default = c.decode(k.interpolated, k.n)[0]
        a = c.decode(k.interpolated, k.n, full)
        b = c.redecode_teams(k.interpolated, default, team)
        for r in c.REFERENCES:
            gt = c.spells_of(c.segments(k.references[r], k.blocks))
            base[r][g, m] = c.spell_f1(gt, c.spells_of(c.segments(default, k.blocks)))
            for codes, store, count in ((a, s_full, n_full), (b, s_team, n_team)):
                for i in range(len(codes)):
                    spells = c.spells_of(c.segments(codes[i], k.blocks))
                    store[r][i, g, m] = c.spell_f1(gt, spells)
                    count[i, g, m] = len(spells)
        print("DECODED", k.game, k.mode, flush=True)
    report = logo_table(s_full, base, [{x: f[x] for x in ("c_ha", "c_to_d", "c_from_d", "gamma")} for f in full], n_full, "three_state")
    report += logo_table(s_team, base, [dict(cost=a, gamma=g) for a, g in team], n_team, "team_only")
    curve = [dict(cost=a, gamma=g, spells=n_team[i].mean(1).round(0).tolist(),
                  **{r: dict(mean=round(float((s_team[r][i] - base[r]).mean()), 4), per_game=(s_team[r][i] - base[r]).mean(1).round(4).tolist(),
                             worst_condition=round(float((s_team[r][i] - base[r]).min()), 4)) for r in c.REFERENCES})
             for i, (a, g) in enumerate(team)]
    for row in curve:
        print("TEAM", json.dumps(row), flush=True)
    (c.ROOT / "decoder.json").write_text(json.dumps(dict(logo=report, team_curve=curve,
                                                         baseline={r: base[r].round(4).tolist() for r in base}), indent=1))


def duration_rule(segs, absorb_s=0.0, island_s=0.0, edge_s=0.0):
    """(c) constants fixed a priori: remove T-X-T / D-X-D / T-X-D possessions shorter than a limit."""
    for _ in range(4):
        changes = {}
        for i, action, new in c.candidates(segs, 10 ** 9):
            d = (segs[i][1] - segs[i][0]) / c.FPS
            if i - 1 in changes or i - 2 in changes or i in changes:
                continue
            island = action == "to_dead" and segs[i - 1][2] == 2 and segs[i + 1][2] == 2
            if (action == "absorb" and d < absorb_s) or (island and d < island_s) or (action in ("to_previous", "to_next") and d < edge_s):
                changes[i] = new
        if not changes:
            break
        segs = c.relabel(segs, changes)
    return segs


def oracle(segs, gt, max_frames):
    score = c.spell_f1(gt, c.spells_of(segs))
    for _ in range(6):
        gains = []
        for i, _, new in c.candidates(segs, max_frames):
            f = c.spell_f1(gt, c.spells_of(c.relabel(segs, {i: new})))
            if f > score + 1e-12:
                gains.append((f - score, i, new))
        changes = {}
        for _, i, new in sorted(gains, reverse=True):
            if not any(i + d in changes for d in (-2, -1, 0, 1, 2)):
                changes[i] = new
        if not changes:
            break
        segs = c.relabel(segs, changes)
        score = c.spell_f1(gt, c.spells_of(segs))
    return score


def rules():
    variants = dict(absorb_1s=dict(absorb_s=1), absorb_2s=dict(absorb_s=2), absorb_3s=dict(absorb_s=3), absorb_4s=dict(absorb_s=4),
                    island_2s=dict(island_s=2), island_3s=dict(island_s=3), edge_2s=dict(edge_s=2))
    delta = {name: {r: np.zeros((3, 3)) for r in c.REFERENCES} for name in list(variants) + ["oracle_8s"]}
    for k in prepared():
        g, m = k.game - 1, c.MODES.index(k.mode)
        for r in c.REFERENCES:
            base = c.spell_f1(k.gt[r], c.spells_of(k.segs))
            for name, arguments in variants.items():
                delta[name][r][g, m] = c.spell_f1(k.gt[r], c.spells_of(duration_rule(k.segs, **arguments))) - base
            delta["oracle_8s"][r][g, m] = oracle(k.segs, k.gt[r], MAX_FRAMES) - base
    report = {name: {r: dict(mean=round(float(d[r].mean()), 4), per_game=d[r].mean(1).round(4).tolist(),
                             worst_condition=round(float(d[r].min()), 4)) for r in c.REFERENCES} for name, d in delta.items()}
    for name, row in report.items():
        print("RULE", name, json.dumps(row), flush=True)
    (c.ROOT / "rules.json").write_text(json.dumps(report, indent=1))


def fit_final():
    import joblib
    table = np.load(c.ROOT / "candidate_table.npz")
    for target in ("robust", "consistent"):
        model = fit(*training(table, [1, 2, 3], target))
        joblib.dump(dict(model=model, features=c.FEATURES, max_frames=MAX_FRAMES, target=target, parameters=MODEL),
                    c.ROOT / f"verifier_{target}.joblib")
        print("FITTED", target, flush=True)


if __name__ == "__main__":
    steps = dict(prepare=prepare, measure=measure, decoder=decoder, rules=rules, table=build_table, verifier=verifier, fit=fit_final)
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=list(steps))
    steps[parser.parse_args().step]()
