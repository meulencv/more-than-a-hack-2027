"""V28d step 1: DFL native per-frame possession -> "controlled possession" (Metrica semantics).

DFL/Sportec BallPossession flips at every touch of the ball by the other team (7-11 possessions
per live minute, about half of them shorter than 3 s). Metrica's definition (CHALLENGE.md) only
changes possession when the other team gains CONTROLLED possession: a simple touch, a deflection
or an aerial duel without control does not count.

Rule (algorithmic, tracking only, no hand labels, no event data):

  Walk through every live stretch (maximal run without a stoppage) in time order and keep a
  "current holder". The first native possession after a stoppage/kick-off always takes the ball
  (a restart is a controlled action). A later native possession X of the other team that is
  followed by the holder again (T-X-T) takes the ball only if it is CONTROLLED:

      contact_s(X) >= MIN_CONTACT_S

  where contact_s is the time, inside X, during which the ball is within RADIUS_M (2-D) of a
  player of X's team (the ball is at somebody's feet, not just struck once and gone). Otherwise
  X is an uncontrolled touch and its frames stay with the current holder (T-X-T becomes T).
  Stoppage frames (D) and unlabelled frames are never modified, so the live/dead structure is
  exactly DFL's. Chains are handled by the state machine: in T-x-t-X the touch x is absorbed,
  t is then a continuation of T, and X is tested on its own.

Calibration (`calibrate`, DFL games only): constants are scored by the relative squared distance
of the six-game mean spell statistics to the Metrica controlled-possession statistics (4.20
possessions per live minute, mean 14.3 s, 6.0 % under 1 s, 20.4 % under 3 s). Two optional
clauses were rejected by that criterion: "a second player of X's team touches the ball later"
(a completed pass counts as control; leaves >= 5.4 possessions per live minute because duels put
several players next to the ball) and "absorb a touch that is followed by a stoppage" (drives
the share of possessions under 1 s to 0.5-4 %, against 5-8 %). For the remaining one-clause rule
the criterion is flat (0.19-0.23) for RADIUS 1.5-2 m and MIN_CONTACT 1.2-2.5 s, so it cannot pin
the constants on its own. `diagnostic` breaks the tie with a cross-provider check: applied to the
touch-level ("alternative") event reference of an open Metrica game, the rule should reproduce
that game's controlled ("consistent") reference. Among the settings inside the flat region the
winner is RADIUS 1.5 m / MIN_CONTACT 1.2 s whichever Metrica game is left out (spellF1 between
the converted and the controlled reference 0.94 / 0.93 / 0.92, against 0.79 without conversion
and 0.87-0.90 for the best pure-duration rule). With these constants the DFL games keep 4.97
possessions per live minute (they are more fragmented than the Metrica games; the rule is not
pushed further because that starts to absorb genuinely controlled possessions).

    .venv/bin/python -m possession.v28d_relabel calibrate   # grid + table -> data/work/v28d/relabel_calibration.json
    .venv/bin/python -m possession.v28d_relabel diagnostic  # Metrica alternative -> consistent conversion check
    .venv/bin/python -m possession.v28d_relabel write       # data/work/v28d/labels/{game}.npz
    .venv/bin/python -m possession.v28d_relabel write --variant c08   # sensitivity variants (c08, c20)
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import v27_core as c

ROOT = Path("data/work/v28d")
GAMES = ("J03WMX", "J03WN1", "J03WOH", "J03WPY", "J03WOY", "J03WQQ")
FPS = 25
# Selected by `calibrate` (see relabel_calibration.json); frozen here for reproducibility.
RULE = dict(radius_m=1.5, min_contact_s=1.2, pass_gap_s=None, absorb_before_dead=False)
FLAT_LOSS = 0.23
# Sensitivity variants of the same one-clause rule (trained separately, compared leave-one-game-out).
VARIANTS = {"main": RULE,
            "c08": dict(RULE, min_contact_s=0.8),                 # lighter: best Metrica conversion, DFL criterion 0.32
            "c20": dict(RULE, radius_m=2.0, min_contact_s=2.0)}   # heavier: minimum of the DFL criterion (0.19)


def labels_dir(variant="main"):
    return ROOT / ("labels" if variant == "main" else f"labels_{variant}")
TARGET = dict(per_live_minute=4.20, mean_s=14.29, share_lt_1s=0.060, share_lt_3s=0.204)   # mean of the 3 Metrica games


def team_distances(match):
    """Per team: distance of the ball to its nearest player and that player's slot, per frame."""
    d = np.linalg.norm(match.players.astype(np.float64) - match.ball[:, None].astype(np.float64), axis=2)
    d = np.where(np.isfinite(d), d, 1e3)
    out = []
    for k in (0, 1):
        slots = np.flatnonzero(match.team == k)
        dk = d[:, slots]
        out.append((dk.min(1), slots[dk.argmin(1)]))
    return out


def controlled(a, z, k, distances, rule):
    near, who = distances[k]
    contact = near[a:z] < rule["radius_m"]
    if contact.sum() >= rule["min_contact_s"] * FPS:
        return True
    if rule["pass_gap_s"] is not None and contact.any():
        index = np.flatnonzero(contact)
        players = who[a:z][index]
        first = players[0]
        other = index[players != first]
        if len(other) and other[-1] - index[0] >= rule["pass_gap_s"] * FPS:
            return True
    return False


def relabel(labels, match, rule=RULE, distances=None):
    """labels: native per-frame codes (0 H, 1 A, 2 D, -1 unknown). Returns (new labels, report)."""
    labels = np.asarray(labels)
    distances = distances or team_distances(match)
    codes = np.where(labels >= 0, labels, 3).astype(np.int8)
    blocks = c.graded_blocks(labels)
    segs = c.segments(codes, blocks)
    out = labels.copy()
    absorbed = []
    holder = None
    previous_stop = None
    for i, (a, z, k) in enumerate(segs):
        contiguous = previous_stop == a
        previous_stop = z
        if k > 1:
            holder = None
            continue
        if holder is None or not contiguous:
            holder = k
            continue
        if k == holder:
            continue
        following = segs[i + 1][2] if i + 1 < len(segs) and segs[i + 1][0] == z else None
        if following != holder and not (rule["absorb_before_dead"] and following == 2):
            holder = k            # last possession of a half, or touch before a stoppage kept
            continue
        if controlled(a, z, k, distances, rule):
            holder = k
        else:
            out[a:z] = holder
            absorbed.append((z - a) / FPS)
    report = dict(absorbed=len(absorbed), absorbed_seconds=float(np.sum(absorbed)),
                  absorbed_median_s=float(np.median(absorbed)) if absorbed else 0.0,
                  frames_changed=int((out != labels).sum()),
                  share_of_live_frames_changed=float((out != labels).sum() / max(1, (labels >= 0).sum() - (labels == 2).sum())))
    return out, report


def stats(labels):
    return c.duration_stats(np.where(labels >= 0, labels, 3).astype(np.int8), c.graded_blocks(labels))


def loss(row):
    return float(sum(((row[k] - v) / v) ** 2 for k, v in TARGET.items()))


def calibrate():
    from .sportec_native import external_native_match
    matches = {g: external_native_match(g) for g in GAMES}
    native = {g: stats(m.meta["labels"]) for g, m in matches.items()}
    keys = ("spells", "per_live_minute", "mean_s", "median_s", "share_lt_1s", "share_lt_3s", "dead_share")
    grid = []
    for radius in (1.0, 1.5, 2.0):
        distances = {g: team_distances(m) for g, m in matches.items()}
        for contact in (0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0):
            for gap in (None, 0.4, 0.8):
                for before_dead in (False, True):
                    rule = dict(radius_m=radius, min_contact_s=contact, pass_gap_s=gap, absorb_before_dead=before_dead)
                    per_game = {g: stats(relabel(m.meta["labels"], m, rule, distances[g])[0]) for g, m in matches.items()}
                    mean = {k: float(np.mean([per_game[g][k] for g in GAMES])) for k in keys}
                    grid.append(dict(rule=rule, mean=mean, loss=loss(mean),
                                     per_game_rate=[round(per_game[g]["per_live_minute"], 3) for g in GAMES]))
                    print("GRID", json.dumps(rule), {k: round(v, 3) for k, v in mean.items()}, "loss", round(grid[-1]["loss"], 4), flush=True)
    grid.sort(key=lambda r: r["loss"])
    report = dict(target=TARGET, native={g: native[g] for g in GAMES},
                  native_mean={k: float(np.mean([native[g][k] for g in GAMES])) for k in keys}, grid=grid, selected=RULE)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "relabel_calibration.json").write_text(json.dumps(report, indent=1))
    for row in grid[:12]:
        print("BEST", json.dumps(row["rule"]), {k: round(v, 3) for k, v in row["mean"].items()}, "loss", round(row["loss"], 4), row["per_game_rate"])


def diagnostic():
    """Cross-provider check of the one-clause rule (tie-break inside the flat DFL region).

    Converts the touch-level Metrica reference with the rule and compares it with the controlled
    reference of the same game (official spell matching). The tie-break is repeated leaving each
    Metrica game out, so no constant depends on the game it is later scored on.
    """
    from .sources import external_match
    calibration = json.loads((ROOT / "relabel_calibration.json").read_text())
    flat = [r["rule"] for r in calibration["grid"] if r["loss"] <= FLAT_LOSS and r["rule"]["pass_gap_s"] is None
            and not r["rule"]["absorb_before_dead"]]
    scores = np.zeros((len(flat), 3))
    before = np.zeros(3)
    for j, game in enumerate((1, 2, 3)):
        match = external_match(game)
        distances = team_distances(match)
        saved = np.load(c.ROOT / f"metrica_{game}.npz")
        blocks = [tuple(int(v) for v in b) for b in saved["blocks"]]
        spells = lambda y: c.spells_of(c.segments(np.where(y >= 0, y, 3).astype(np.int8), blocks))
        gt = spells(saved["consistent"])
        before[j] = c.spell_f1(gt, spells(saved["alternative"]))
        for i, rule in enumerate(flat):
            scores[i, j] = c.spell_f1(gt, spells(relabel(saved["alternative"], match, rule, distances)[0]))
    rows = [dict(rule=rule, conversion_spell_f1=scores[i].round(4).tolist()) for i, rule in enumerate(flat)]
    chosen = [flat[int(np.delete(scores, held, axis=1).sum(1).argmax())] for held in range(3)]
    report = dict(without_conversion=before.round(4).tolist(), flat_region=rows, chosen_leaving_out_game_1_2_3=chosen,
                  selected=RULE, unanimous=all(k == RULE for k in chosen))
    (ROOT / "relabel_diagnostic.json").write_text(json.dumps(report, indent=1))
    for row in rows:
        print("DIAGNOSTIC", json.dumps(row), flush=True)
    print("WITHOUT", before.round(4).tolist(), "CHOSEN", json.dumps(chosen), "UNANIMOUS", report["unanimous"], flush=True)


def write(variant="main"):
    from .sportec_native import external_native_match
    rule = VARIANTS[variant]
    labels_dir(variant).mkdir(parents=True, exist_ok=True)
    rows = {}
    for game in GAMES:
        match = external_native_match(game)
        native = match.meta["labels"]
        new, report = relabel(native, match, rule)
        if not np.array_equal(new == 2, native == 2) or not np.array_equal(new < 0, native < 0):
            raise ValueError("The relabelling must not touch stoppages or unknown frames")
        np.savez_compressed(labels_dir(variant) / f"{game}.npz", labels=new, native=native)
        rows[game] = dict(native=stats(native), controlled=stats(new), **report)
        print("RELABEL", game, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[game]["controlled"].items()}),
              "absorbed", report["absorbed"], "live frames changed", round(report["share_of_live_frames_changed"], 4), flush=True)
    keys = ("spells", "per_live_minute", "mean_s", "median_s", "share_lt_1s", "share_lt_3s", "dead_share")
    mean = {kind: {k: float(np.mean([rows[g][kind][k] for g in GAMES])) for k in keys} for kind in ("native", "controlled")}
    name = "relabel_report.json" if variant == "main" else f"relabel_report_{variant}.json"
    (ROOT / name).write_text(json.dumps(dict(variant=variant, rule=rule, target=TARGET, six_game_mean=mean, games=rows), indent=1))
    print("RELABEL MEAN", variant, json.dumps({k: {a: round(b, 4) for a, b in v.items()} for k, v in mean.items()}), flush=True)


def controlled_labels(game, variant="main"):
    return np.load(labels_dir(variant) / f"{game}.npz")["labels"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["calibrate", "diagnostic", "write"])
    parser.add_argument("--variant", choices=list(VARIANTS), default="main")
    args = parser.parse_args()
    if args.step == "write":
        write(args.variant)
    else:
        {"calibrate": calibrate, "diagnostic": diagnostic}[args.step]()
