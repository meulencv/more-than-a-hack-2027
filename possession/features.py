"""Physical observations and soft tactical priors, with no hand-labelled examples."""
import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d
from games import GAMES, GAME


def interpolate_short_gaps(ball_xy, max_gap, max_speed, fps):
    out = ball_xy.copy()
    valid = np.isfinite(out).all(axis=1)
    anchors = np.flatnonzero(valid)
    for left, right in zip(anchors[:-1], anchors[1:]):
        gap = right - left - 1
        if 0 < gap <= max_gap:
            speed = np.linalg.norm(out[right] - out[left]) * fps / (right - left)
            if speed <= max_speed:
                alpha = np.arange(1, gap + 1)[:, None] / (gap + 1)
                out[left + 1:right] = out[left] * (1 - alpha) + out[right] * alpha
    return out


def tactical_prior(events, n):
    evidence = np.zeros((n, 2), dtype=np.float32)
    set_piece = np.zeros(n, dtype=np.float32)
    for event in events:
        start, end = max(0, event["start_frame"]), min(n, event["end_frame"])
        if start >= end:
            continue
        code = (event["code"] or "").upper()
        # Set-piece windows include preparation AND the restart: never hard-label D.
        if code == "SET PIECES":
            set_piece[start:end] = 1
            continue
        if code == "GOALS":
            continue
        name = event["tags"].get("Team", "").lower()
        if name == "barcelona":
            team = 0
        elif name in ("atletic club", "athletic club"):
            team = 1
        else:
            continue
        if code.startswith("DEFENDING") or code == "DEFENSIVE TRANSITION":
            team = 1 - team
        # Correlated overlapping tags must not count as independent votes.
        evidence[start:end, team] = 1
    return evidence[:, 0] - evidence[:, 1], set_piece


def compute(players, ball, slots, events, config):
    fps = config["fps"]
    n = len(ball)
    pitch = np.array(GAMES[GAME]["pitch"])
    observed = np.isfinite(ball[:, :2]).all(axis=1)
    xy = interpolate_short_gaps(ball[:, :2] * pitch,
                               config["max_interpolation_frames"],
                               config["max_ball_speed_mps"], fps)
    visible = np.isfinite(xy).all(axis=1)
    speed = np.full(n, np.nan)
    lag = max(1, round(fps * 0.2))
    speed[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=1) * fps / lag
    distance = np.linalg.norm(players[:, :, :2] * pitch - xy[:, None, :], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    home = distance[:, slots == 0].min(axis=1)
    away = distance[:, slots == 1].min(axis=1)
    nearest = np.minimum(home, away)
    margin = np.zeros(n)
    good = visible & np.isfinite(nearest)
    margin[good] = np.tanh((away[good] - home[good]) / config["control_radius_m"])
    confidence = np.where(good, np.exp(-nearest / 10), 0)
    # A fast airborne/pass observation is weaker evidence of CONTROL than a reception.
    confidence *= np.where(np.isfinite(speed), 1 / (1 + (np.nan_to_num(speed) / 12) ** 2), 0.5)
    tag, set_piece = tactical_prior(events, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        motion = np.nanmedian(players[:, :, 2], axis=1)
    motion = np.nan_to_num(motion, nan=0.0)
    slow = np.isfinite(speed) & (speed < config["stationary_speed_mps"])
    still = uniform_filter1d(slow.astype(float), size=fps, mode="nearest")
    missing = uniform_filter1d((~visible).astype(float), size=fps, mode="nearest")
    low_motion = np.clip((1.5 - uniform_filter1d(motion, fps)) / 1.5, 0, 1)
    outside = visible & ((xy[:, 0] < -0.5) | (xy[:, 0] > pitch[0] + 0.5)
                         | (xy[:, 1] < -0.5) | (xy[:, 1] > pitch[1] + 0.5))
    team_score = (config["team_evidence_weight"] * margin * confidence
                  + config["tag_evidence_weight"] * tag)
    alive = np.maximum(confidence, abs(tag) * 0.8)
    # Missing ball alone is almost neutral, not a dead-ball detector.
    dead = (-0.5 + 1.3 * still + 1.0 * missing * low_motion
            + 2.0 * outside + 0.25 * set_piece - 0.8 * alive)
    emission = np.column_stack((team_score, -team_score, dead))
    diagnostics = {"observed_ball": observed, "interpolated_ball": visible & ~observed,
                   "nearest_distance_m": nearest, "ball_speed_mps": speed,
                   "median_player_speed_mps": motion, "team_tag": tag,
                   "set_piece_window": set_piece, "emission": emission}
    return emission, diagnostics
