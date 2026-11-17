"""V30d data: degraded network inputs of the three open Metrica games, for IN-DOMAIN training.

Why the Metrica games cannot be used as they are: on their tracking the ball is missing almost
exactly when the ball is dead (P(ball seen | dead) = 0.00-0.13, P(seen | live) = 0.93-0.96), a
data artefact of the provider's export that a network would learn in a minute and that does not
hold on the challenge match (video-filled ball: seen on 74% of graded frames with ~46% of them
dead). Every training view therefore first SYNTHESISES dead-ball sightings (the ball placed at
the restart spot, carried there by the nearest player, or fetched from where it was last seen),
then applies the label-blind sensor degradations of the original recipe
(possession.v6_data.sensor_view: clean / moderate / stress; challenge sensor profile).

Training views of one game (all seeds differ from the evaluation seeds 907 + game):
    0  sightings on a random 50-100% of the stoppages          + no extra sensor loss
    1  sightings on a random 50-100% of the stoppages          + moderate
    2  sightings on a random 50-100% of the stoppages          + stress
    3  sightings on every stoppage, from its first frame        + moderate   (state-blind ball)
Evaluation-only extra condition of one game ("sight", a transfer check, never trained on):
    sightings on 75% of the stoppages (seed 5000 + game) + moderate with the evaluation seed.

Labels are only used to place the synthetic sightings (external games; allowed, as in
possession.degrade.degrade) and as training targets. The challenge match contributes its
unlabeled sensor profile only (ball-missing run lengths), as in the original recipe.

    .venv/bin/python -m possession.v30d_data --game 1 --view 0      # one training view
    .venv/bin/python -m possession.v30d_data --game 1 --sight       # the extra evaluation record
    .venv/bin/python -m possession.v30d_data                        # everything (sequential)
"""
import argparse
from pathlib import Path
import joblib
import numpy as np

ROOT = Path("data/work/v30d")
REFERENCES = Path("data/work/v27")
GAMES = (1, 2, 3)
VIEWS = {0: dict(p_fill=(0.5, 1.0), p_from_start=0.4, mode="clean"),
         1: dict(p_fill=(0.5, 1.0), p_from_start=0.4, mode="moderate"),
         2: dict(p_fill=(0.5, 1.0), p_from_start=0.4, mode="stress"),
         3: dict(p_fill=(1.0, 1.0), p_from_start=1.0, mode="moderate")}
SIGHT = dict(p_fill=(0.75, 0.75), p_from_start=0.4, mode="moderate")
LOOK = 50          # frames searched for the restart spot / the last seen ball


def references(game):
    meta = np.load(REFERENCES / f"metrica_{game}.npz")
    return {name: meta[name] for name in ("consistent", "alternative")}, [tuple(int(v) for v in b) for b in meta["blocks"]]


def dead_runs(dead, blocks):
    """[(start, stop)] half-open runs of dead ball, cut at the halves."""
    out = []
    for _, first, last in blocks:
        d = np.diff(np.r_[0, dead[first:last + 1].astype(np.int8), 0])
        out += [(int(a) + first, int(z) + first) for a, z in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return out


def sighted(match, dead, blocks, seed, p_fill, p_from_start):
    """Copy of `match` with a synthetic ball during stoppages (real observations are kept)."""
    rng = np.random.default_rng(seed)
    ball = match.ball.astype(np.float64).copy()
    players = match.players.astype(np.float64)
    seen = np.isfinite(ball).all(axis=1)
    share = rng.uniform(*p_fill)
    n = len(ball)
    filled = 0
    for start, stop in dead_runs(dead, blocks):
        draw = rng.random(4)                                    # always drawn: same stream whatever happens
        kind = int(rng.integers(3))
        offset = rng.normal(0, 1, 2)
        if draw[0] > share or stop - start < 5:
            continue
        ahead = np.flatnonzero(seen[stop:min(n, stop + LOOK)])
        if not len(ahead):
            continue
        spot = match.ball[stop + ahead[0]].astype(np.float64)
        first = start if draw[1] < p_from_start else start + int((stop - start) * 0.8 * draw[2])
        length = stop - first
        if kind == 0:                                           # placed for the restart
            path = np.repeat((spot + 0.3 * offset)[None], length, axis=0)
        elif kind == 1:                                         # carried by the nearest player
            near = np.linalg.norm(players[first] - spot, axis=1)
            if np.isfinite(near).any():
                carrier = int(np.nanargmin(near))
                path = players[first:stop, carrier]
                path = np.where(np.isfinite(path), path, spot) + 0.5 * offset
            else:
                path = np.repeat(spot[None], length, axis=0)
        else:                                                   # fetched from where it was last seen
            back = np.flatnonzero(seen[max(0, first - LOOK):first])
            origin = match.ball[max(0, first - LOOK) + back[-1]].astype(np.float64) if len(back) else spot
            travel = max(1, int(length * (0.4 + 0.4 * draw[3])))
            t = np.clip(np.arange(length) / travel, 0, 1)[:, None]
            path = origin + t * (spot + 0.3 * offset - origin)
        hole = ~seen[first:stop]
        ball[first:stop][hole] = path[hole]
        filled += int(hole.sum())
    return match.copy_with(ball=ball.astype(np.float32)), filled


def build(game, view=None):
    """One record (possession.v12_compare.match_record layout + the two 5 Hz training targets)."""
    from .sources import external_match, challenge_match
    from .degrade import challenge_profile
    from .v6_data import sensor_view
    from .v12_compare import match_record
    path = record_path(game, view)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = SIGHT if view is None else VIEWS[view]
    refs, blocks = references(game)
    source = external_match(game)
    if blocks != source.blocks:
        raise ValueError("Reference blocks differ from the match")
    seed = 5000 + game if view is None else 3000 + 100 * game + view
    sensor_seed = 907 + game if view is None else 4000 + 100 * game + view
    match, filled = sighted(source, refs["consistent"] == 2, blocks, seed, spec["p_fill"], spec["p_from_start"])
    match = sensor_view(match, spec["mode"], sensor_seed, challenge_profile(challenge_match()))
    r = match_record(match)
    seen = np.isfinite(match.ball).all(axis=1)
    y = refs["consistent"]
    r.update(n_frames=len(match.ball), ball_seen=seen, labels_C=refs["consistent"][r["frames"]].astype(np.int8),
             labels_A=refs["alternative"][r["frames"]].astype(np.int8), game=game, view=view, spec=dict(spec),
             seeds=dict(sighting=seed, sensor=sensor_seed), sighting_frames=filled,
             seen_given_dead=float(seen[y == 2].mean()), seen_given_live=float(seen[(y == 0) | (y == 1)].mean()))
    temporary = path.with_suffix(".part")
    joblib.dump(r, temporary, compress=1)
    temporary.replace(path)
    print("V30D RECORD", game, "sight" if view is None else view, r["direct"].shape, "P(seen|dead)", round(r["seen_given_dead"], 3),
          "P(seen|live)", round(r["seen_given_live"], 3), flush=True)
    return path


def record_path(game, view=None):
    return ROOT / "records" / (f"eval_game{game}_sight.joblib" if view is None else f"train_game{game}_v{view}.joblib")


def record(game, view=None):
    return joblib.load(record_path(game, view))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, choices=GAMES)
    parser.add_argument("--view", type=int, choices=list(VIEWS))
    parser.add_argument("--sight", action="store_true")
    args = parser.parse_args()
    for game in ([args.game] if args.game else GAMES):
        if args.sight:
            build(game, None)
        elif args.view is not None:
            build(game, args.view)
        else:
            for view in VIEWS:
                build(game, view)
            build(game, None)
