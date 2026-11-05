"""Approximate frame-level H/A/D references for OTHER games, derived from their event data.

These references are an approximation of the challenge's definition, not an equivalent of it:

* The challenge ground truth comes from corrected tracking; here we only have Metrica's
  published event annotations (who did what, when). Boundaries are therefore approximate to
  roughly the event timing precision (a few frames to ~1 s).
* LIVE possession: a team keeps possession from the start of one of its CONTROL events
  (pass, carry, shot, set piece, recovery, ball lost [it had the ball]) until the start of a
  CONTROL event by the opponent. Challenges, aerial duels and deflections (CHALLENGE events)
  never change possession by themselves, matching the challenge's "controlled possession" rule.
* DEAD ball: from the event that stopped play (ball out, foul, offside, goal, ...) until the
  next restart (SET PIECE, or a pass tagged GOAL KICK in games 1-2, which have no goal-kick set
  piece type). If an unstopped live event sits inside that interval the dead interval starts
  after it instead.
* Frames before the first kick-off of a period, after its last event, and between periods are
  not labelled (-1), exactly like the ungraded frames of the challenge.

Codes: 0 = home (H), 1 = away (A), 2 = dead (D), -1 = not labelled.
"""
import numpy as np

CONTROL = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST"}
TOUCH_FRAMES = 5  # 0.2 s: recovery and loss in practically the same contact
STOPPERS = ("OFFSIDE", "HAND BALL", "END HALF", "GOAL", "OUT", "FAULT", "WOODWORK")


def _is_restart(row):
    return row.type == "SET PIECE" or (row.type == "PASS" and "GOAL KICK" in row.subtype)


def _is_stop(row):
    if row.type in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if row.type in ("SHOT", "BALL LOST", "CHALLENGE"):
        return any(word in row.subtype for word in STOPPERS) and "ADVANTAGE" not in row.subtype
    return False


def _one_touch(rows, i, max_frames=TOUCH_FRAMES):
    """A recovery immediately followed by the same team losing the ball is a touch.

    Example: an intercepting header that goes straight to nobody/the opponent. The challenge
    rules say a simple touch without control and a deflection do not change possession. Only
    events of that team before the next opponent control event are inspected.
    """
    team = rows[i].team
    if rows[i].type not in ("RECOVERY", "BALL LOST"):
        return False
    for row in rows[i:]:
        if row.start - rows[i].start > max_frames:
            return False
        if row.team != team and row.type in CONTROL:
            return True
        if row.team == team and row.type in ("PASS", "CARRY", "SHOT", "SET PIECE"):
            return False
        if row.team == team and row.type == "BALL LOST":
            return True
    return False


def event_reference(events, n):
    labels = np.full(n, -1, dtype=np.int8)
    blocks = []
    events = events.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    for period in sorted(events.period.unique()):
        ev = events[events.period == period].reset_index(drop=True)
        kick = ev[(ev.type == "SET PIECE") & ev.subtype.str.contains("KICK OFF")]
        if kick.empty:
            continue
        first = int(kick.start.iloc[0])
        last = int(min(n - 1, max(ev.end.max(), ev.start.max())))
        ev = ev[ev.start >= first].reset_index(drop=True)
        blocks.append((int(period), first, last))
        team = np.full(last - first + 1, -1, dtype=np.int8)
        current = int(ev.team.iloc[0])
        cursor = first
        rows = list(ev.itertuples())
        for i, row in enumerate(rows):
            if row.type in CONTROL and row.team != current and not _one_touch(rows, i):
                team[cursor - first:row.start - first] = current
                cursor, current = row.start, int(row.team)
        team[cursor - first:] = current
        dead = np.zeros_like(team, dtype=bool)
        restart_rows = [r for r in ev.itertuples() if _is_restart(r)]
        previous_restart = first
        for restart in restart_rows:
            if restart.start <= first:
                continue
            window = ev[(ev.start >= previous_restart) & (ev.start < restart.start)]
            previous_restart = restart.start
            if window.empty:
                continue
            stops = [r for r in window.itertuples() if _is_stop(r)]
            if stops:
                stop = stops[-1]
                stop_frame = stop.end if stop.type in ("BALL OUT", "SHOT") else stop.start
                later = window[(window.start > stop_frame)
                               & window.type.isin(["PASS", "CARRY", "SHOT"])]
                if not later.empty:
                    stop_frame = int(later.end.max())
            else:
                live = window[window.type.isin(["PASS", "CARRY", "SHOT", "BALL LOST",
                                                "CHALLENGE", "RECOVERY"])]
                if live.empty:
                    continue
                stop_frame = int(live.end.max())
            stop_frame = max(int(stop_frame), first)
            if restart.start - stop_frame >= 2:
                dead[stop_frame - first:restart.start - first] = True
        block = np.where(dead, 2, team)
        labels[first:last + 1] = block
    return labels, blocks
