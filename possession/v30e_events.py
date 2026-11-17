"""V30e: physical events of the challenge match and how the pipeline's stoppage boundaries sit
relative to them. Label-free: no ground truth, no scorer.

    .venv/bin/python -m possession.v30e_events            # data/work/v30e/events_report.json

Events (all from data/work/v30e/features):
    whistle    referee whistle heard in BOTH recordings (dual tone >= 12 dB, >= 3 frames)
    crossing   the filled ball goes from inside the pitch to more than 0.3 m outside the lines
    kick       the filled ball rests (< 1.5 m/s) for >= 1 s and then moves at > 6 m/s

For each, the offset to the pipeline's decoded stoppage onset / restart, next to a control in
which the events are moved in time by 1 to 4 minutes (200 draws): a real relation shows as many
more boundaries close to the event than in the control.
"""
import json
from pathlib import Path
import numpy as np
from games import GAME, half_video_blocks
from . import v27_core as c
from . import v30e_features as F

ROOT = Path("data/work/v30e")
FPS = 25
OUTSIDE_M = 0.3
REST_FRAMES = 25
KICK_SPEED = 6.0
DRAWS = 200


def graded_mask(n):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def whistles():
    return np.array([w["start"] for w in json.loads((F.OUT / "events.json").read_text())["whistles"]], dtype=np.int64)


def crossings(ball=None, graded=None):
    """First frame of each excursion of the seen ball beyond the lines (>= 3 frames outside,
    seen inside within the previous 10 frames, excursions at least 2 s apart)."""
    ball = ball or F.load_group("ball")
    outside = np.nan_to_num(ball["b_outside_m"], nan=-99.0) > OUTSIDE_M
    inside = (ball["b_seen"] > 0) & ~outside
    graded = graded_mask(len(outside)) if graded is None else graded
    starts = np.flatnonzero(outside[1:] & ~outside[:-1]) + 1
    keep = []
    for t in starts:
        if graded[t] and outside[t:t + 3].all() and inside[max(0, t - 10):t].any() and (not keep or t - keep[-1] > 2 * FPS):
            keep.append(int(t))
    return np.array(keep, dtype=np.int64)


def kicks(ball=None):
    """Last resting frame before the ball leaves at speed (speed is a +-5 frame chord, so the
    contact is about 3 to 8 frames later)."""
    ball = ball or F.load_group("ball")
    rest, speed = ball["b_rest_frames"], np.nan_to_num(ball["b_speed"], nan=0.0)
    graded = graded_mask(len(rest))
    t = np.flatnonzero((rest[:-8] >= REST_FRAMES) & (speed[8:] > KICK_SPEED))
    t = t[graded[t]]
    return t[np.r_[True, np.diff(t) > FPS]] if len(t) else t


def stoppages(codes):
    """(onsets, restarts) of the decoded stoppages that have a possession before / after."""
    segs = c.segments(codes, list(half_video_blocks(GAME)))
    onsets, restarts = [], []
    for i, (a, z, k) in enumerate(segs):
        if k != 2:
            continue
        if i > 0 and segs[i - 1][1] == a:
            onsets.append(a)
        if i + 1 < len(segs) and segs[i + 1][0] == z:
            restarts.append(z)
    return np.array(onsets), np.array(restarts)


def offsets(events, boundaries, reach):
    """boundary - event for the boundary nearest to each event, kept when |.| <= reach."""
    if not len(events) or not len(boundaries):
        return np.array([])
    j = np.clip(np.searchsorted(boundaries, events), 1, len(boundaries) - 1)
    d = np.where(np.abs(boundaries[j] - events) < np.abs(boundaries[j - 1] - events), boundaries[j] - events, boundaries[j - 1] - events)
    return d[np.abs(d) <= reach]


def relation(events, boundaries, graded, reach=5 * FPS, close=2 * FPS, seed=0):
    d = offsets(events, boundaries, reach)
    rng = np.random.default_rng(seed)
    control_reach, control_close = [], []
    for _ in range(DRAWS):
        moved = events + rng.integers(60 * FPS, 240 * FPS) * rng.choice([-1, 1])
        moved = moved[(moved >= 0) & (moved < len(graded))]
        moved = moved[graded[moved]]
        e = offsets(moved, boundaries, reach)
        control_reach.append(len(e) / max(1, len(moved)))
        control_close.append((np.abs(e) <= close).sum() / max(1, len(moved)))
    row = dict(events=int(len(events)), boundary_within_5s=int(len(d)), share_within_5s=float(len(d) / max(1, len(events))),
               share_within_2s=float((np.abs(d) <= close).sum() / max(1, len(events))),
               control_share_within_5s=float(np.mean(control_reach)), control_share_within_2s=float(np.mean(control_close)))
    if len(d):
        row.update(offset_median_frames=float(np.median(d)), offset_quartiles=[float(v) for v in np.percentile(d, [25, 75])],
                   boundary_after_event=int((d > 0).sum()), boundary_before_event=int((d < 0).sum()),
                   median_when_after=float(np.median(d[d > 0])) if (d > 0).any() else None,
                   median_when_before=float(np.median(d[d < 0])) if (d < 0).any() else None,
                   offsets_frames=np.sort(d).astype(int).tolist())
    return row


def report(codes, name="submission"):
    graded = graded_mask(len(codes))
    onsets, restarts = stoppages(codes)
    ball = F.load_group("ball")
    w, x, k = whistles(), crossings(ball), kicks(ball)
    out = dict(candidate=name, stoppages=int(len(onsets)),
               whistle_vs_onset=relation(w, onsets, graded), whistle_vs_restart=relation(w, restarts, graded),
               crossing_vs_onset=relation(x, onsets, graded), kick_vs_restart=relation(k, restarts, graded),
               kick_vs_onset_should_be_unrelated=relation(k, onsets, graded),
               whistles_in_decoded_play=int((codes[w] < 2).sum()), whistles_in_decoded_stoppage=int((codes[w] == 2).sum()),
               crossings_in_decoded_play=int((codes[x] < 2).sum()), kicks_in_decoded_stoppage=int((codes[k] == 2).sum()))
    return out


def main():
    if not (ROOT / "base.npz").exists():
        from . import v30e_lab
        v30e_lab.base()
    base = np.load(ROOT / "base.npz")
    result = dict(decoder_before_verifier=report(base["before"], "V24 fusion + decoder"), submission=report(base["after"], "v29_final"))
    for name, row in result.items():
        for key, value in row.items():
            if isinstance(value, dict):
                print("EVENTS", name, key, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in value.items() if a != "offsets_frames"}, flush=True)
            else:
                print("EVENTS", name, key, value, flush=True)
    (ROOT / "events_report.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
