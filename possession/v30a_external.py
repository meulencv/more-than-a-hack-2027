"""V30a: convention fingerprints of the EXTERNAL references (never reads the challenge labels).

The case study of the public window (possession/v30a_cases.py) answers four yes/no questions
about how the challenge ground truth is written. This script asks the same questions of each
external reference construction, so that the closest one can be chosen as a discrete option:

  restart   at the first live frame after a stoppage, has the ball already been travelling
            (it was kicked while the label still said dead), or does the label go live with the
            ball still at rest (label goes live at the kick)?
  stoppage  at the first dead frame, is the ball still well inside the pitch (whistle, or a
            late call) or at / beyond a line?
  switch    at a direct H<->A switch, is a player of the gaining team at the ball (switch at
            its touch) or is the ball loose / in flight?
  short     do possessions shorter than a second and a half exist, and how common are they?

    .venv/bin/python -m possession.v30a_external
"""
import json
from pathlib import Path

import numpy as np

OUT = Path("data/work/v30a")
FPS = 25


def runs(labels):
    edges = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    return [(int(a), int(b - 1), int(labels[a])) for a, b in zip(edges[:-1], edges[1:])]


def fingerprint(ball, players, team, labels, pitch):
    """ball (n,2) metres, players (n,slots,2), labels 0/1/2/-1."""
    segs = runs(labels)
    moved_before, in_field, at_ball, loose, durations = [], [], [], [], []
    for i, (a, b, code) in enumerate(segs):
        if code in (0, 1):
            durations.append((b - a + 1) / FPS)
        if i == 0:
            continue
        previous = segs[i - 1][2]
        if previous == 2 and code in (0, 1) and a >= 10:
            p0, p1 = ball[a - 10], ball[a]
            if np.isfinite(p0).all() and np.isfinite(p1).all():
                moved_before.append(float(np.hypot(*(p1 - p0))) > 2.0)
        if previous in (0, 1) and code == 2:
            p = ball[a]
            if np.isfinite(p).all():
                margin = min(p[0], pitch[0] - p[0], p[1], pitch[1] - p[1])
                in_field.append(margin > 1.0)
        if previous in (0, 1) and code in (0, 1):
            p = ball[a]
            if np.isfinite(p).all():
                d = np.sqrt(((players[a] - p) ** 2).sum(axis=1))
                gain = d[team == code]
                gain = float(np.nanmin(gain)) if np.isfinite(gain).any() else np.nan
                if np.isfinite(gain):
                    at_ball.append(gain < 1.5)
                    loose.append(gain > 3.0)
    live = sum(durations) / 60.0
    durations = np.array(durations)
    return dict(
        restarts=len(moved_before),
        restart_ball_already_travelling=round(float(np.mean(moved_before)), 3),
        stoppages=len(in_field),
        stoppage_ball_inside_pitch=round(float(np.mean(in_field)), 3),
        switches=len(at_ball),
        switch_gainer_at_ball=round(float(np.mean(at_ball)), 3),
        switch_ball_loose=round(float(np.mean(loose)), 3),
        possessions=len(durations),
        possessions_per_live_minute=round(len(durations) / live, 2),
        share_shorter_than_1_5s=round(float(np.mean(durations < 1.5)), 3),
        share_shorter_than_4s=round(float(np.mean(durations < 4.0)), 3))


def main():
    from possession.sources import external_match
    from possession.sportec_native import external_native_match
    from possession import v13_cv
    rows = []
    for game in (1, 2, 3):
        match = external_match(game)
        with np.load("data/work/v27/metrica_{}.npz".format(game)) as saved:
            for reading in ("consistent", "alternative"):
                row = dict(source="metrica sample {} {}".format(game, reading))
                row.update(fingerprint(match.ball, match.players, match.team, saved[reading],
                                       match.pitch))
                rows.append(row)
    for game in v13_cv.settings()["games"]:
        match = external_native_match(game)
        row = dict(source="DFL native {}".format(game))
        row.update(fingerprint(match.ball, match.players, match.team, match.meta["labels"],
                               match.pitch))
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "external_conventions.json").write_text(json.dumps(rows, indent=1))
    keys = [k for k in rows[0] if k != "source"]
    print("{:34s}".format("source") + " ".join("{:>9s}".format(k[:9]) for k in keys))
    for row in rows:
        print("{:34s}".format(row["source"]) + " ".join("{:9}".format(row[k]) for k in keys))
    print("columns:", keys)


if __name__ == "__main__":
    main()
