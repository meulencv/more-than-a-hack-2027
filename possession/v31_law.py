"""Kick-off law: a label-free correction of the first possession of the second half.

Law 8 of the game: the second half is started by the team that did NOT take the kick-off of the
first half. The decoder does not know this; on the challenge match it gives the first 1.3 s of
the second half to the team that also kicked off the first half, then switches. This step

  1. reads who kicked off the first half from the decoded labels (the first possession) and
     checks it against the tracking (the team of the player nearest to the centre spot during
     the second before the published kick-off);
  2. does the same check for the second half (the team on the centre spot must be the other one);
  3. only if law and tracking agree, and the first decoded possession of the second half is a
     short one (<= 3 s) of the wrong team immediately followed by the right team, gives those
     frames to the right team;
  4. labels graded frames before the published second-half kick-off as dead ball (FAQ.md: "does
     not kick off until 113,742. Those 2,934 frames are graded and all of them are D").

Nothing is fitted and no label is read: the inputs are the published kick-off frames
(starter/load_smart_tagging.py), the automatic tracking and the pipeline's own decode.
"""
from __future__ import annotations

import numpy as np

from . import v27_core as c

FPS = 25
LIMIT = 3 * FPS            # only a possession this short may be reassigned
WINDOW = FPS               # centre-spot check: the second before the kick-off


def centre_spot_team(match, frame, window=WINDOW):
    """(team, median distance in m) of the player nearest to the centre spot before `frame`."""
    centre = np.array(match.pitch, dtype=np.float64) / 2
    d = np.linalg.norm(match.players[frame - window:frame + 1] - centre, axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    nearest = d.argmin(axis=1)
    teams = match.team[nearest]
    return int(np.round(teams.mean())), float(np.median(d.min(axis=1)))


def kickoff_law(codes, match, blocks, kickoffs, limit=LIMIT):
    """(codes after, report). `codes`: 0 H, 1 A, 2 D per video frame; `kickoffs`: the two
    published kick-off frames; `blocks`: [(half, first graded frame, last graded frame)]."""
    out = np.array(codes, copy=True)
    first, second = sorted(int(k) for k in kickoffs)
    ordered = sorted(blocks, key=lambda b: b[1])
    team1, d1 = centre_spot_team(match, first)
    team2, d2 = centre_spot_team(match, second)
    segs = c.segments(out, ordered)
    decoded_first = next(s for s in segs if s[2] < 2)
    j = next(i for i, (a, z, k) in enumerate(segs) if z > second and k < 2)
    a, z, k = segs[j]
    report = dict(first_half_kickoff_team="HA"[team1], second_half_centre_spot_team="HA"[team2],
                  centre_spot_distance_m=[round(d1, 2), round(d2, 2)], law_and_tracking_agree=bool(team1 != team2),
                  first_half_decoded_first_team="HA"[decoded_first[2]],
                  second_half_decoded_first_possession=dict(start=int(a), stop=int(z), team="HA"[k]),
                  reassigned_frames=0, dead_frames_before_kickoff=0)
    agree = team1 != team2 and decoded_first[2] == team1
    if agree and k != team2 and z - a <= limit and j + 1 < len(segs) and segs[j + 1][2] == team2:
        out[a:z] = team2
        report["reassigned_frames"] = int(z - a)
    start = ordered[1][1]
    early = np.flatnonzero(out[start:second] != 2) + start
    out[early] = 2
    report["dead_frames_before_kickoff"] = int(len(early))
    return out, report
