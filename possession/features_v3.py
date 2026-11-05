"""Team-symmetric v3 observations, identical code path for external games and the challenge.

Design rules (see docs/MEMORIA_TECNICA.md):
* only positions are used; speeds are recomputed from positions at the same lag everywhere;
* visibility is an absolute count of tracked players (not a fraction of slots);
* every feature has a mirrored counterpart, so swapping teams is an exact transformation
  (used for augmentation and for a symmetric prediction at inference);
* nothing depends on frame numbers, team names, Smart Tagging or any label of the challenge.

Ball-dependent features are NaN when the ball is not tracked. The gradient-boosting model
handles NaN natively, so "ball missing" is never hard-coded as "dead ball".
"""
import warnings
import numpy as np
import pandas as pd

FPS = 25
LAG = 5                      # 0.2 s for speeds
STEP = 5                     # features are sampled at 5 Hz
WINDOWS_S = (1, 3, 8)


def attack_direction(players, team, first, last):
    """+1 if the home team attacks towards +x in this block, else -1.

    Uses the deepest player of each team (normally the goalkeeper): the team whose deepest
    player sits nearer x=0 defends that goal. Robust to a dominant team camping in the
    opponent's half, which would fool a centroid rule.
    """
    x = players[first:last + 1, :, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home_min = np.nanmedian(np.nanmin(np.where(team == 0, x, np.nan), axis=1))
        away_min = np.nanmedian(np.nanmin(np.where(team == 1, x, np.nan), axis=1))
        home_max = np.nanmedian(np.nanmax(np.where(team == 0, x, np.nan), axis=1))
        away_max = np.nanmedian(np.nanmax(np.where(team == 1, x, np.nan), axis=1))
    # home defends left goal if its deepest-left player is further left AND its most advanced
    # player is less advanced to the right than the away keeper.
    score = (away_min - home_min) + (away_max - home_max)
    return 1 if score > 0 else -1


def _team_stats(X, Y, mask):
    """Per-frame centroid, spread and line height for players selected by mask."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xs = np.where(mask, X, np.nan)
        ys = np.where(mask, Y, np.nan)
        cx, cy = np.nanmean(xs, axis=1), np.nanmean(ys, axis=1)
        sx, sy = np.nanstd(xs, axis=1), np.nanstd(ys, axis=1)
        srt = np.sort(xs, axis=1)            # NaN sorted last
        count = np.isfinite(xs).sum(axis=1)
    return cx, cy, sx, sy, srt, count


def frame_signals(match, swap=False):
    """Raw per-frame signals in the home-attacking frame (home attacks +X)."""
    team = match.team if not swap else (1 - match.team).astype(np.int8)
    L, W = match.pitch
    n = len(match.ball)
    P = match.players.astype(np.float64)
    B = match.ball.astype(np.float64)
    direction = np.zeros(n)
    for _, first, last in match.blocks:
        direction[first:last + 1] = attack_direction(P, team, first, last)
    direction[direction == 0] = 1
    X = np.where(direction[:, None] > 0, P[:, :, 0], L - P[:, :, 0])
    Y = np.abs(P[:, :, 1] - W / 2)
    bX = np.where(direction > 0, B[:, 0], L - B[:, 0])
    bY = np.abs(B[:, 1] - W / 2)
    seen = np.isfinite(bX)
    home, away = team == 0, team == 1
    tracked = np.isfinite(X)
    s = {}
    s["ball_seen"] = seen.astype(float)
    s["ball_x"] = bX
    s["ball_y"] = bY
    outside = np.maximum.reduce([-bX, bX - L, bY - W / 2, np.zeros(n)])
    s["ball_outside_m"] = np.where(seen, outside, np.nan)
    v = np.full((n, 2), np.nan)
    v[LAG:] = (B[LAG:] - B[:-LAG]) * FPS / LAG
    speed = np.linalg.norm(v, axis=1)
    speed[speed > 45] = np.nan
    s["ball_speed"] = speed
    s["ball_vx"] = np.where(direction > 0, v[:, 0], -v[:, 0])
    dist = np.sqrt((P[:, :, 0] - B[:, None, 0]) ** 2 + (P[:, :, 1] - B[:, None, 1]) ** 2)
    for name, sel in (("h", home), ("a", away)):
        d = np.sort(np.where(np.isfinite(dist[:, sel]), dist[:, sel], np.inf), axis=1)
        d1 = d[:, 0] if d.shape[1] else np.full(n, np.inf)
        d2 = d[:, 1] if d.shape[1] > 1 else np.full(n, np.inf)
        s[f"d1_{name}"] = np.where(seen & np.isfinite(d1), np.minimum(d1, 60), np.nan)
        s[f"d2_{name}"] = np.where(seen & np.isfinite(d2), np.minimum(d2, 60), np.nan)
        s[f"n3_{name}"] = np.where(seen, (d < 3).sum(axis=1), np.nan)
        s[f"n8_{name}"] = np.where(seen, (d < 8).sum(axis=1), np.nan)
    s["margin"] = np.clip(s["d1_a"] - s["d1_h"], -15, 15)
    s["near_min"] = np.fmin(s["d1_h"], s["d1_a"])
    # Velocity of players from positions (identical lag in every dataset).
    pv = np.full(P.shape, np.nan)
    pv[LAG:] = (P[LAG:] - P[:-LAG]) * FPS / LAG
    pspeed = np.linalg.norm(pv, axis=2)
    bad = pspeed > 12
    pspeed[bad] = np.nan
    pvx = np.where(direction[:, None] > 0, pv[:, :, 0], -pv[:, :, 0])
    pvx[bad] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s["speed_med"] = np.nanmedian(pspeed, axis=1)
        s["speed_p90"] = np.nanpercentile(pspeed, 90, axis=1)
        for name, sel, sign in (("h", home, 1), ("a", away, -1)):
            s[f"speed_{name}"] = np.nanmean(pspeed[:, sel], axis=1)
            # forward = towards the goal this team attacks
            s[f"fwd_{name}"] = sign * np.nanmean(pvx[:, sel], axis=1)
    stats = {}
    for name, sel in (("h", home), ("a", away)):
        cx, cy, sx, sy, srt, count = _team_stats(X, Y, sel[None, :] & tracked)
        stats[name] = (cx, srt, count)
        s[f"count_{name}"] = count.astype(float)
        s[f"sx_{name}"], s[f"sy_{name}"] = sx, sy
        s[f"cy_{name}"] = cy
    # Express each team's x in ITS OWN attacking direction for mirrored features.
    s["cx_h"] = stats["h"][0]
    s["cx_a"] = L - stats["a"][0]
    srt_h, srt_a = stats["h"][1], stats["a"][1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # second-deepest player (first outfield line) of each team, own direction
        s["line_h"] = srt_h[:, 1]
        cnt_a = stats["a"][2]
        idx = np.clip(cnt_a - 2, 0, None)
        s["line_a"] = L - np.take_along_axis(srt_a, idx[:, None], axis=1)[:, 0]
        s["line_a"][cnt_a < 2] = np.nan
        s["line_h"][stats["h"][2] < 2] = np.nan
        # ball distance to each team's centroid (who surrounds the ball)
        s["ball_to_c_h"] = np.where(seen, np.abs(bX - stats["h"][0]), np.nan)
        s["ball_to_c_a"] = np.where(seen, np.abs(bX - stats["a"][0]), np.nan)
    s["ball_x_a"] = L - bX           # ball position in away attacking direction
    s["ball_vx_a"] = -s["ball_vx"]
    return pd.DataFrame(s)


def _gap_features(signal, seen, cap_s=60.0):
    """Carry the last/next observed value across ball-missing gaps, with elapsed time."""
    n = len(signal)
    idx = np.arange(n)
    last = np.where(seen, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(seen, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    last_val = np.where(last >= 0, signal[np.clip(last, 0, n - 1)], np.nan)
    next_val = np.where(nxt < n, signal[np.clip(nxt, 0, n - 1)], np.nan)
    since = np.where(last >= 0, (idx - last) / FPS, cap_s).clip(0, cap_s)
    until = np.where(nxt < n, (nxt - idx) / FPS, cap_s).clip(0, cap_s)
    return last_val, next_val, since, until


def block_features(sig, first, last):
    """Windowed context features for one block (never crossing a half boundary)."""
    frame = sig.iloc[first:last + 1].reset_index(drop=True)
    out = {}
    seen = frame["ball_seen"].to_numpy() > 0
    margin = frame["margin"].to_numpy()
    smooth_margin = pd.Series(margin).rolling(5, center=True, min_periods=1).mean().to_numpy()
    lv, nv, since, until = _gap_features(smooth_margin, seen)
    out["margin_last"], out["margin_next"] = lv, nv
    out["ball_since_s"], out["ball_until_s"] = since, until
    bx = frame["ball_x"].to_numpy()
    lx, nx, _, _ = _gap_features(bx, seen)
    out["ball_x_last"], out["ball_x_next"] = lx, nx
    lo, no, _, _ = _gap_features(frame["ball_outside_m"].to_numpy(), seen)
    out["outside_last"], out["outside_next"] = lo, no
    keys = ["margin", "ball_seen", "ball_speed", "speed_med", "speed_p90", "fwd_h", "fwd_a",
            "near_min", "n3_h", "n3_a", "sx_h", "sx_a", "cx_h", "cx_a", "ball_vx"]
    for w in WINDOWS_S:
        size = w * FPS
        for k in keys:
            col = frame[k]
            out[f"{k}_c{w}"] = col.rolling(size, center=True, min_periods=1).mean().to_numpy()
            if w in (1, 3) and k in ("margin", "ball_seen", "speed_med", "fwd_h", "fwd_a",
                                       "ball_speed", "near_min"):
                out[f"{k}_p{w}"] = col.rolling(size, min_periods=1).mean().to_numpy()
                out[f"{k}_f{w}"] = col[::-1].rolling(size, min_periods=1).mean().to_numpy()[::-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        still = (frame["ball_speed"] < 0.5).astype(float).where(frame["ball_speed"].notna())
        out["ball_still_c3"] = still.rolling(3 * FPS, center=True, min_periods=1).mean().to_numpy()
        slow = (frame["speed_med"] < 1.0).astype(float)
        out["slow_c8"] = slow.rolling(8 * FPS, center=True, min_periods=1).mean().to_numpy()
        out["slow_c20"] = slow.rolling(20 * FPS, center=True, min_periods=1).mean().to_numpy()
        out["seen_c20"] = frame["ball_seen"].rolling(20 * FPS, center=True,
                                                     min_periods=1).mean().to_numpy()
    feats = pd.concat([frame, pd.DataFrame(out)], axis=1)
    return feats


def feature_table(match, step=STEP, swap=False):
    """(features DataFrame at 5 Hz, frame index array). Frames span every block."""
    sig = frame_signals(match, swap=swap)
    parts, frames = [], []
    for _, first, last in match.blocks:
        feats = block_features(sig, first, last)
        pick = np.arange(0, last - first + 1, step)
        parts.append(feats.iloc[pick].reset_index(drop=True))
        frames.append(first + pick)
    table = pd.concat(parts, ignore_index=True).astype(np.float32)
    return table, np.concatenate(frames)
