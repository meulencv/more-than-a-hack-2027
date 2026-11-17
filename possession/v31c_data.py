"""V31c data: LABEL-BLIND ball views of the open Metrica games + the compromise target M.

Two defects of the in-domain training of V30d are removed here.

(a) Target. V30d trained either on the "consistent" or on the "alternative" reading of the
    possession definition and slid along that semantic axis. The target here is the compromise
    labelling M that scores >= 0.85 under BOTH readings (data/work/review_v30b/checks.py):
        Metrica: possession.v27_lab.duration_rule(alternative reference, absorb_s=2.0, island_s=0, edge_s=0)
        DFL:     the same rule applied to the native labels
    i.e. every possession shorter than 2.0 s that is sandwiched between two possessions of the
    other team is absorbed. The limit 2.0 s is fixed a priori (ABSORB_S), never selected.

(b) Ball. On the open Metrica games the ball is missing exactly while the ball is dead, so a
    network learns "the ball vanished => dead ball starts". V30d placed synthetic sightings as a
    function of the reference labels. Here NOTHING is placed, moved, removed or restored as a
    function of the labels:
      1. blind_fill: every ball gap inside a half (defined by the NaN structure only) is filled
         between the last and the next sighting with one of four label-blind paths drawn at
         random (straight line / travel then rest / rest then travel / carried by the player
         nearest to the next sighting), plus coloured noise;
      2. the ball is removed again by a random alternating process that ignores the labels and
         draws its gap lengths from the challenge match (video-filled ball: FILLED profile;
         raw tracking: the original recipe's profile inside sensor_view);
      3. the usual sensor degradations of the players (possession.v6_data.sensor_view).
    Features built from ball availability are not added anywhere; the availability the networks
    see on these views is statistically independent of the labels by construction.

Training views of one game (seeds differ from every evaluation seed):
    0  blind fill + light ball noise + random removal, share U(0.10, 0.30), FILLED gap profile
    1  blind fill + 'moderate' (noise 0.9 m, 30% removed, raw-tracking gap profile)
    2  blind fill + 'stress'   (noise 1.1 m, 8 s blocks removed, identity swaps)
    3  blind fill + moderate player noise + random removal, share U(0.20, 0.45), FILLED gap profile

    .venv/bin/python -m possession.v31c_data --game 1 --view 0
    .venv/bin/python -m possession.v31c_data --profile        # data/work/v31c/filled_profile.npz
    .venv/bin/python -m possession.v31c_data --stats          # target statistics (DFL + Metrica)

No challenge label is read: the challenge contributes ball-availability run lengths only.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np

ROOT = Path("data/work/v31c")
REFERENCES = Path("data/work/v27")
GAMES = (1, 2, 3)
ABSORB_S = 2.0
VIEWS = {0: dict(mode="clean", ball_sigma=0.3, removal=(0.10, 0.30)),
         1: dict(mode="moderate", ball_sigma=0.0, removal=None),
         2: dict(mode="stress", ball_sigma=0.0, removal=None),
         3: dict(mode="players", ball_sigma=0.9, removal=(0.20, 0.45))}
KINDS = ("line", "travel_rest", "rest_travel", "carried")
SHORT = 10         # gaps shorter than this many frames are always a straight line


# -- targets -----------------------------------------------------------------------------------

def compromise(labels, blocks, absorb_s=ABSORB_S):
    """M: sandwiched possessions shorter than absorb_s absorbed. labels: 25 Hz codes, -1 = not graded."""
    from . import v27_core as c
    from .v27_lab import duration_rule
    segs = duration_rule(c.segments(labels, blocks), absorb_s=absorb_s, island_s=0, edge_s=0)
    out = labels.copy()
    for a, z, k in segs:
        out[a:z] = k
    if not np.array_equal(out == 2, labels == 2) or not np.array_equal(out < 0, labels < 0):
        raise ValueError("The duration rule changed stoppages or the graded window")
    return out


def references(game):
    meta = np.load(REFERENCES / f"metrica_{game}.npz")
    return {name: meta[name] for name in ("consistent", "alternative")}, [tuple(int(v) for v in b) for b in meta["blocks"]]


def metrica_target(game, absorb_s=ABSORB_S):
    refs, blocks = references(game)
    return compromise(refs["alternative"].astype(np.int8), blocks, absorb_s)


def dfl_target(full_labels, blocks, absorb_s=ABSORB_S):
    return compromise(np.asarray(full_labels).astype(np.int8), [tuple(int(v) for v in b) for b in blocks], absorb_s)


# -- label-blind ball --------------------------------------------------------------------------

def gaps(seen):
    d = np.diff(np.r_[0, (~seen).astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def blind_fill(match, seed):
    """Copy of `match` whose ball has no gap inside the halves. Inputs: ball NaN structure, ball
    sightings and player positions. The reference labels are not an argument of this function."""
    from .degrade import _ar_noise
    rng = np.random.default_rng(seed)
    ball = match.ball.astype(np.float64).copy()
    players = match.players.astype(np.float64)
    counts = dict.fromkeys(KINDS, 0)
    filled = np.zeros(len(ball), dtype=bool)
    for _, first, last in match.blocks:
        b = ball[first:last + 1]
        seen = np.isfinite(b).all(axis=1)
        if not seen.any():
            continue
        index = np.flatnonzero(seen)
        for a, z in gaps(seen):
            draw = rng.random(3)                                 # always drawn: one stream per gap
            kind = KINDS[int(rng.integers(len(KINDS)))]
            n = z - a
            if a == 0 or z == len(b):                            # no sighting on one side: hold the other
                b[a:z] = b[index[0]] if a == 0 else b[index[-1]]
                filled[first + a:first + z] = True
                continue
            p0, p1 = b[a - 1].copy(), b[z].copy()
            t = (np.arange(1, n + 1) / (n + 1))[:, None]
            if n < SHORT or kind == "line":
                kind, path = "line", p0 + t * (p1 - p0)
            elif kind == "travel_rest":                          # reaches the next sighting early, then lies there
                u = 0.15 + 0.6 * draw[0]
                path = p0 + np.clip(t / u, 0, 1) * (p1 - p0)
            elif kind == "rest_travel":                          # lies where it was last seen, then travels
                u = 0.15 + 0.6 * draw[0]
                path = p0 + np.clip((t - u) / (1 - u), 0, 1) * (p1 - p0)
            else:                                                # carried by the player nearest to the next sighting
                near = np.linalg.norm(players[first + z] - p1, axis=1)
                if np.isfinite(near).any():
                    carrier = int(np.nanargmin(near))
                    track = players[first + a:first + z, carrier]
                    line = p0 + t * (p1 - p0)
                    track = np.where(np.isfinite(track), track, line)
                    w_in = np.clip(t / (0.1 + 0.3 * draw[1]), 0, 1)
                    w_out = np.clip((1 - t) / (0.05 + 0.15 * draw[2]), 0, 1)
                    path = (1 - w_in) * p0 + w_in * track
                    path = (1 - w_out) * p1 + w_out * path
                else:
                    kind, path = "line", p0 + t * (p1 - p0)
            counts[kind] += 1
            b[a:z] = path + _ar_noise(rng, n, 0.25, 12) * np.minimum(1, np.minimum(t, 1 - t) * n / 5)
            filled[first + a:first + z] = True
    return match.copy_with(ball=ball.astype(np.float32)), dict(filled_frames=int(filled.sum()), kinds=counts)


def filled_profile():
    """Run lengths of ball missing / seen of the challenge inputs used at inference (video-filled ball)."""
    path = ROOT / "filled_profile.npz"
    if not path.exists():
        from . import v29_final as final
        from .degrade import challenge_profile
        path.parent.mkdir(parents=True, exist_ok=True)
        match, report = final.filled_match(ROOT / "chal")
        p = challenge_profile(match)
        np.savez(path.with_suffix(".part.npz"), missing_runs=p.missing_runs, seen_runs=p.seen_runs,
                 missing_share=report["ball_missing_final"])
        path.with_suffix(".part.npz").replace(path)
    saved = np.load(path)
    return saved["missing_runs"], saved["seen_runs"], float(saved["missing_share"])


def random_removal(match, seed, share_range, ball_sigma, player_sigma=0.0):
    """Label-blind removal with the gap lengths of the video-filled challenge ball (+ noise)."""
    from .degrade import _alternating_mask, _ar_noise
    rng = np.random.default_rng(seed)
    missing_runs, seen_runs, _ = filled_profile()
    ball = match.ball.astype(np.float64).copy()
    players = match.players.astype(np.float64).copy()
    share = float(rng.uniform(*share_range))
    for _, first, last in match.blocks:
        n = last - first + 1
        if ball_sigma:
            ball[first:last + 1] += _ar_noise(rng, n, ball_sigma, 15)
        ball[first:last + 1][_alternating_mask(rng, n, missing_runs, seen_runs, share)] = np.nan
        if player_sigma:
            for j in range(players.shape[1]):
                players[first:last + 1, j] += _ar_noise(rng, n, player_sigma, 12)
    return match.copy_with(players=players.astype(np.float32), ball=ball.astype(np.float32)), share


def view_match(game, view):
    """The degraded match of one training view (no label is read)."""
    from .sources import external_match, challenge_match
    from .degrade import challenge_profile
    from .v6_data import sensor_view
    spec = VIEWS[view]
    source = external_match(game)
    seeds = dict(fill=8000 + 100 * game + view, sensor=9000 + 100 * game + view)
    match, report = blind_fill(source, seeds["fill"])
    if spec["mode"] in ("moderate", "stress"):
        match = sensor_view(match, spec["mode"], seeds["sensor"], challenge_profile(challenge_match()))
    else:
        match, share = random_removal(match, seeds["sensor"], spec["removal"], spec["ball_sigma"],
                                      0.12 if spec["mode"] == "players" else 0.0)
        report["removal_share_drawn"] = share
    report["seeds"] = seeds
    return source, match, report


def vanish_alignment(seen, labels, k=2):
    """Diagnostic only: share of reference D starts within k frames of a ball-vanish event."""
    vanish = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
    starts = np.flatnonzero((labels[1:] == 2) & (labels[:-1] != 2) & (labels[:-1] >= 0)) + 1
    if not len(vanish) or not len(starts):
        return 0.0
    i = np.searchsorted(vanish, starts)
    d = np.minimum(np.abs(vanish[np.clip(i - 1, 0, len(vanish) - 1)] - starts), np.abs(vanish[np.clip(i, 0, len(vanish) - 1)] - starts))
    return float((d <= k).mean())


def record_path(game, view):
    return ROOT / "records" / f"train_game{game}_v{view}.joblib"


def build(game, view):
    from .v12_compare import match_record
    path = record_path(game, view)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    refs, blocks = references(game)
    source, match, report = view_match(game, view)
    if blocks != source.blocks:
        raise ValueError("Reference blocks differ from the match")
    r = match_record(match)
    seen = np.isfinite(match.ball).all(axis=1)
    source_seen = np.isfinite(source.ball).all(axis=1)
    target = metrica_target(game)
    y = refs["consistent"]
    labels = target[r["frames"]]
    if labels.min() < 0 or labels.max() > 2:
        raise ValueError("Ungraded frame on the 5 Hz timeline")
    r.update(n_frames=len(match.ball), ball_seen=seen, labels_M=labels.astype(np.int8),
             labels_A=refs["alternative"][r["frames"]].astype(np.int8), labels_C=refs["consistent"][r["frames"]].astype(np.int8),
             game=game, view=view, spec=dict(VIEWS[view]), fill=report, absorb_s=ABSORB_S,
             seen_given_dead=float(seen[y == 2].mean()), seen_given_live=float(seen[(y == 0) | (y == 1)].mean()),
             reference_Dstart_within2_of_vanish=vanish_alignment(seen, y),
             source_Dstart_within2_of_vanish=vanish_alignment(source_seen, y))
    temporary = path.with_suffix(".part")
    joblib.dump(r, temporary, compress=1)
    temporary.replace(path)
    print("V31C RECORD", game, view, r["direct"].shape, "P(seen|dead)", round(r["seen_given_dead"], 3), "P(seen|live)",
          round(r["seen_given_live"], 3), "ref D starts within 2 fr of a vanish: source", round(r["source_Dstart_within2_of_vanish"], 3),
          "view", round(r["reference_Dstart_within2_of_vanish"], 3), json.dumps(report, default=float), flush=True)
    return path


def record(game, view):
    return joblib.load(record_path(game, view))


def stats():
    """How much the 2.0 s rule changes each target (also at 1.5 and 3 s; reported, not selected)."""
    from . import v6_data, v10_data, v27_core as c
    out = {}
    for game in GAMES:
        refs, blocks = references(game)
        alt = refs["alternative"].astype(np.int8)
        row = dict(alternative=c.duration_stats(np.where(alt >= 0, alt, 3), blocks)["spells"],
                   consistent=c.duration_stats(np.where(refs["consistent"] >= 0, refs["consistent"], 3), blocks)["spells"])
        for limit in (1.5, 2.0, 3.0):
            m = compromise(alt, blocks, limit)
            row[f"M_{limit}"] = dict(spells=c.duration_stats(np.where(m >= 0, m, 3), blocks)["spells"], frames_changed=int((m != alt).sum()))
        out[f"metrica_{game}"] = row
    native = v6_data.protocol()
    for game in native["native_train"] + native["native_development"]:
        r = v10_data.record("train" if game in native["native_train"] else "development", game, "clean")
        full, blocks = r["full_labels"].astype(np.int8), [tuple(int(v) for v in b) for b in r["blocks"]]
        row = dict(native=c.duration_stats(np.where(full >= 0, full, 3), blocks)["spells"])
        for limit in (1.5, 2.0, 3.0):
            m = compromise(full, blocks, limit)
            row[f"M_{limit}"] = dict(spells=c.duration_stats(np.where(m >= 0, m, 3), blocks)["spells"], frames_changed=int((m != full).sum()))
        out[f"dfl_{game}"] = row
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "target_stats.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, choices=GAMES)
    parser.add_argument("--view", type=int, choices=list(VIEWS))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    if args.profile:
        print("FILLED PROFILE", [round(float(v), 3) for v in (filled_profile()[0].mean(), filled_profile()[1].mean(), filled_profile()[2])])
    elif args.stats:
        stats()
    else:
        for game in ([args.game] if args.game else GAMES):
            for view in ([args.view] if args.view is not None else VIEWS):
                build(game, view)
