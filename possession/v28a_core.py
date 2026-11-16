"""V28a building blocks: a physical description of every predicted possession, from tracking.

The V27 verifier (possession.v27_core / v27_lab) decides whether a short predicted possession
is removed using only the frozen network output. Here the same candidate removals get a second
block of features computed from the tracking itself (ball kinematics, who is next to the ball,
whether the ball is brought under control or flies through), so they exist on the challenge
match as well. Nothing here reads any ground truth of the challenge match.

Codes: 0 = H, 1 = A, 2 = D, 3 = not graded.
"""
from pathlib import Path
import numpy as np
from . import v27_core as c

FPS = c.FPS
ROOT = Path("data/work/v28a")
HALF = 5                       # velocity chord half-width (frames): v(t) = (x[t+5] - x[t-5]) / 0.4 s
CHORD = 12                     # heading chord (frames) for direction changes
CLOSE_M = 2.5                  # "next to the ball" (challenge median nearest-player distance 1.9 m)
FREE_M = 3.5                   # nobody near: ball in flight / loose
CAP_S = 60.0


# -- per-frame signals -----------------------------------------------------------------------

def _shift_difference(x, lag):
    """x[t + lag] - x[t - lag], NaN where either end is outside the array."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    out[lag:-lag] = x[2 * lag:] - x[:-2 * lag]
    return out


def signals(ball, players, team, pitch):
    """Per-frame physical signals of one match (all NaN where the ball is not tracked).

    ball (n, 2) and players (n, slots, 2) in metres, team (slots,) 0/1, pitch (length, width).
    """
    ball = np.asarray(ball, dtype=np.float64)
    players = np.asarray(players, dtype=np.float32)
    team = np.asarray(team)
    n = len(ball)
    seen = np.isfinite(ball).all(1)
    velocity = _shift_difference(ball, HALF) * (FPS / (2.0 * HALF))
    speed = np.linalg.norm(velocity, axis=1)
    distance = np.full((n, 2), np.nan)
    relative = np.full((n, 2), np.nan)
    step = 20000
    for t in (0, 1):
        slots = np.flatnonzero(team == t)
        for first in range(0, n, step):
            rows = slice(first, min(n, first + step))
            xy = players[rows][:, slots].astype(np.float64)
            d = np.linalg.norm(xy - ball[rows][:, None, :], axis=2)
            d[~np.isfinite(d)] = np.inf
            j = d.argmin(1)
            best = d[np.arange(len(j)), j]
            distance[rows, t] = np.where(np.isfinite(best), best, np.nan)
            lo = np.clip(np.arange(rows.start, rows.stop) - HALF, 0, n - 1)
            hi = np.clip(np.arange(rows.start, rows.stop) + HALF, 0, n - 1)
            pv = (players[hi, slots[j]].astype(np.float64) - players[lo, slots[j]].astype(np.float64)) * (FPS / (2.0 * HALF))
            relative[rows, t] = np.linalg.norm(velocity[rows] - pv, axis=1)
    relative[~np.isfinite(distance)] = np.nan
    # Direction change at t: angle between the chord that arrives and the chord that leaves,
    # only when the ball travels at least 5 m/s on both (tracking noise is ~1 m).
    arrive = np.full((n, 2), np.nan)
    leave = np.full((n, 2), np.nan)
    arrive[CHORD:] = ball[CHORD:] - ball[:-CHORD]
    leave[:-CHORD] = ball[CHORD:] - ball[:-CHORD]
    la, ll = np.linalg.norm(arrive, axis=1), np.linalg.norm(leave, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (arrive * leave).sum(1) / (la * ll)
    fast = (la > 5.0 * CHORD / FPS) & (ll > 5.0 * CHORD / FPS)
    turn = np.where(fast, np.degrees(np.arccos(np.clip(cosine, -1, 1))), 0.0)
    turn[~np.isfinite(cosine)] = np.nan
    jump = np.abs(_shift_difference(speed, 6))             # speed change over ~0.5 s
    event = ((turn > 45.0) | (jump > 6.0)).astype(np.float64)
    event[~np.isfinite(turn) & ~np.isfinite(jump)] = np.nan
    # keep one event per ~0.4 s
    marks = np.zeros(n)
    last = -10 ** 9
    for t in np.flatnonzero(event == 1):
        if t - last >= 10:
            marks[t] = 1
            last = t
    marks[~seen] = np.nan
    length, width = pitch
    touchline = np.minimum(ball[:, 1], width - ball[:, 1])
    goalline = np.minimum(ball[:, 0], length - ball[:, 0])
    return dict(seen=seen, ball=ball, speed=speed, distance=distance, relative=relative, turn=turn,
                event=marks, touchline=touchline, goalline=goalline)


# -- per-candidate physical features ---------------------------------------------------------

PHYSICAL = ["seen_in", "near_own_share", "d_own_med", "d_own_min", "d_opp_med", "d_opp_min", "margin_med",
            "own_close_s", "opp_close_s", "own_carry_share", "opp_carry_share", "free_share",
            "speed_mean", "speed_max", "speed_min", "path_m", "disp_m", "straight", "events", "events_per_s",
            "line_dev_m", "through_cos", "speed_before", "speed_after", "touch_own", "touch_opp",
            "pre_seen", "pre_near_opp", "pre_d_opp_med", "post_seen", "post_near_opp", "post_d_opp_med",
            "post_opp_first_close_s", "pre_own_close_s", "post_own_close_s", "touchline_m", "goalline_m"]
STRUCTURE = ["prev2_s", "next2_s", "prev2_same", "next2_same", "since_dead_s", "to_dead_s", "merged_s",
             "prev_conf", "next_conf", "prev_live", "next_live"]
SHAPE = ["own_dominant_s", "own_dominant_share", "own_rolling_max", "new_max", "new_dominant_share",
         "margin_centre", "margin_before", "margin_after", "edge_peaks"]
FEATURES = list(c.FEATURES) + PHYSICAL + STRUCTURE + SHAPE


def _longest(mask):
    if not mask.any():
        return 0.0
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return float((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max()) / FPS


def _nanstat(function, values, default=np.nan):
    values = values[np.isfinite(values)]
    return float(function(values)) if len(values) else default


def physical_features(a, z, k, s):
    """The spell [a, z) of team k described from tracking; NaN when the ball is not tracked."""
    o = 1 - k
    n = len(s["seen"])
    inside = slice(a, z)
    seen = s["seen"][inside]
    d_own, d_opp = s["distance"][inside, k], s["distance"][inside, o]
    valid = seen & np.isfinite(d_own) & np.isfinite(d_opp)
    row = dict.fromkeys(PHYSICAL, np.nan)
    row["seen_in"] = float(seen.mean())
    if valid.any():
        own_near = valid & (d_own < d_opp)
        opp_near = valid & (d_opp < d_own)
        relative_own, relative_opp = s["relative"][inside, k], s["relative"][inside, o]
        with np.errstate(invalid="ignore"):
            row.update(near_own_share=float(own_near.sum() / valid.sum()),
                       d_own_med=float(np.median(d_own[valid])), d_own_min=float(d_own[valid].min()),
                       d_opp_med=float(np.median(d_opp[valid])), d_opp_min=float(d_opp[valid].min()),
                       margin_med=float(np.median((d_opp - d_own)[valid])),
                       own_close_s=_longest(own_near & (d_own < CLOSE_M)),
                       opp_close_s=_longest(opp_near & (d_opp < CLOSE_M)),
                       own_carry_share=float((valid & (d_own < CLOSE_M) & (relative_own < 3.0)).sum() / valid.sum()),
                       opp_carry_share=float((valid & (d_opp < CLOSE_M) & (relative_opp < 3.0)).sum() / valid.sum()),
                       free_share=float((valid & (np.minimum(d_own, d_opp) > FREE_M)).sum() / valid.sum()))
            event = s["event"][inside] == 1
            row["touch_own"] = float((event & own_near & (d_own < FREE_M)).sum())
            row["touch_opp"] = float((event & opp_near & (d_opp < FREE_M)).sum())
    speed = s["speed"][inside]
    row["speed_mean"] = _nanstat(np.mean, speed)
    row["speed_max"] = _nanstat(np.max, speed)
    row["speed_min"] = _nanstat(np.min, speed)
    ball = s["ball"][inside]
    if seen.sum() >= 2:
        sampled = ball[::5]
        steps = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
        steps = steps[np.isfinite(steps)]
        if len(steps):
            row["path_m"] = float(steps.sum())
        tracked = ball[seen]
        chord = tracked[-1] - tracked[0]
        row["disp_m"] = float(np.linalg.norm(chord))
        if np.isfinite(row["path_m"]) and row["path_m"] > 0:
            row["straight"] = float(min(1.0, row["disp_m"] / row["path_m"]))
        if row["disp_m"] > 1e-6:
            offset = tracked - tracked[0]
            row["line_dev_m"] = float(np.abs(offset[:, 0] * chord[1] - offset[:, 1] * chord[0]).max() / row["disp_m"])
        events = s["event"][inside]
        row["events"] = float(np.nansum(events))
        row["events_per_s"] = row["events"] / (seen.sum() / FPS)
    # Does the ball go straight through the spell? Heading just before versus just after.
    a0, z1 = a - CHORD, z + CHORD
    if a0 >= 0 and z1 < n:
        before = s["ball"][a] - s["ball"][a0]
        after = s["ball"][z1] - s["ball"][min(z, n - 1)]
        if np.isfinite(before).all() and np.isfinite(after).all():
            lb, lf = np.linalg.norm(before), np.linalg.norm(after)
            row["speed_before"], row["speed_after"] = float(lb * FPS / CHORD), float(lf * FPS / CHORD)
            if lb > 0.5 and lf > 0.5:
                row["through_cos"] = float(before @ after / (lb * lf))
    for name, window in (("pre", slice(max(0, a - 2 * FPS), a)), ("post", slice(z, min(n, z + 2 * FPS)))):
        if window.stop <= window.start:
            continue
        w_own, w_opp = s["distance"][window, k], s["distance"][window, o]
        ok = np.isfinite(w_own) & np.isfinite(w_opp)
        row[name + "_seen"] = float(s["seen"][window].mean())
        if ok.any():
            row[name + "_near_opp"] = float((ok & (w_opp < w_own)).sum() / ok.sum())
            row[name + "_d_opp_med"] = float(np.median(w_opp[ok]))
            row[name + "_own_close_s"] = _longest(ok & (w_own < w_opp) & (w_own < CLOSE_M))
    after = slice(z, min(n, z + 5 * FPS))
    if after.stop > after.start:
        w_own, w_opp = s["distance"][after, k], s["distance"][after, o]
        ok = np.isfinite(w_own) & np.isfinite(w_opp)
        if ok.any():
            hit = np.flatnonzero(ok & (w_opp < w_own) & (w_opp < CLOSE_M))
            row["post_opp_first_close_s"] = float(hit[0]) / FPS if len(hit) else 5.0
    row["touchline_m"] = _nanstat(np.mean, s["touchline"][inside])
    row["goalline_m"] = _nanstat(np.mean, s["goalline"][inside])
    return [row[name] for name in PHYSICAL]


def structure_features(segs, i, new, p):
    a, z, k = segs[i]
    previous = segs[i - 1] if i > 0 and segs[i - 1][1] == a else None
    following = segs[i + 1] if i + 1 < len(segs) and segs[i + 1][0] == z else None
    previous2 = segs[i - 2] if previous and i > 1 and segs[i - 2][1] == previous[0] else None
    following2 = segs[i + 2] if following and i + 2 < len(segs) and segs[i + 2][0] == following[1] else None

    def duration(seg):
        return min(CAP_S, (seg[1] - seg[0]) / FPS) if seg else CAP_S

    # live time since the last stoppage (or the start of the half) and until the next one
    j = i
    while j > 0 and segs[j - 1][1] == segs[j][0] and segs[j - 1][2] != 2:
        j -= 1
    since = min(CAP_S, (a - segs[j][0]) / FPS)
    j = i
    while j + 1 < len(segs) and segs[j + 1][0] == segs[j][1] and segs[j + 1][2] != 2:
        j += 1
    until = min(CAP_S, (segs[j][1] - z) / FPS)
    merged = z - a
    if previous and previous[2] == new:
        merged += previous[1] - previous[0]
    if following and following[2] == new:
        merged += following[1] - following[0]

    def confidence(seg, window):
        if not seg or window.stop <= window.start:
            return np.nan
        return float(p[window, seg[2]].mean())

    return [duration(previous2), duration(following2),
            float(previous2 is not None and previous2[2] == k), float(following2 is not None and following2[2] == k),
            since, until, min(CAP_S, merged / FPS),
            confidence(previous, slice(max(previous[0], a - 5 * FPS), a) if previous else slice(0, 0)),
            confidence(following, slice(z, min(following[1], z + 5 * FPS)) if following else slice(0, 0)),
            float(previous is not None and previous[2] < 2), float(following is not None and following[2] < 2)]


def shape_features(a, z, k, new, p, edge):
    """How the network output behaves inside the spell (beyond the V27 means)."""
    inside = p[a:z]
    top = inside.argmax(1)
    own = inside[:, k]
    window = min(FPS, len(own))
    rolling = np.convolve(own, np.ones(window) / window, mode="valid")
    centre = (a + z) // 2
    before = slice(max(0, a - FPS), a)
    after = slice(z, min(len(p), z + FPS))

    def margin(window_):
        if window_.stop <= window_.start:
            return np.nan
        return float(np.nanmean(p[window_, new] - p[window_, k]))

    high = edge[a:z] > 0.5
    return [_longest(top == k), float((top == k).mean()), float(rolling.max()), float(inside[:, new].max()),
            float((top == new).mean()), float(p[centre, new] - p[centre, k]), margin(before), margin(after),
            float(np.sum(np.diff(np.r_[0, high.astype(np.int8)]) == 1))]


def candidate_features(segs, cands, p, edge, s):
    """V27 network-output features + physical + structure + shape, one row per candidate removal."""
    base = c.candidate_features(segs, cands, p, edge)
    extra = np.array([physical_features(segs[i][0], segs[i][1], segs[i][2], s) + structure_features(segs, i, new, p)
                      + shape_features(segs[i][0], segs[i][1], segs[i][2], new, p, edge)
                      for i, _, new in cands], dtype=np.float64).reshape(len(cands), len(FEATURES) - len(c.FEATURES))
    return np.hstack([base, extra])


# -- models: every one returns a score whose natural decision boundary is 0 ------------------

GBM = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=25,
           l2_regularization=1.0, early_stopping=False, random_state=0)      # = possession.v27_lab.MODEL

# A-priori signs (+1: more of it -> removing the possession is MORE likely to help).
MONOTONE = dict(duration_s=-1, own_mean=-1, own_max=-1, own_p10=-1, team_share_mean=-1, team_share_max=-1,
                log_ratio_mean=-1, log_ratio_sum=-1, decoder_gain=-1, new_mean=1,
                near_own_share=-1, own_close_s=-1, own_carry_share=-1, margin_med=-1, touch_own=-1,
                d_own_med=1, d_own_min=1, opp_carry_share=1, opp_close_s=1)


class Model:
    """Small, regularised scorers of a candidate removal. Score > 0 means "remove".

    gbm       gradient-boosted regression of the exact spellF1 gain (the V27 recipe)
    gbm_mono  the same with monotone constraints on the features that have an a-priori sign
    gbc       gain-weighted classification of the sign of the gain (score = P - 0.5)
    logit     the same with a logistic regression (median imputation + missing flags)
    ridge     linear regression of the gain
    blend / blend_mono   gain regressor + purity-gap regressor, each divided by the standard
              deviation of its training target and added (two views of one question)
    """

    def __init__(self, kind="gbm", columns=None, parameters=None, live_only=False):
        self.kind = kind
        self.columns = list(range(len(FEATURES))) if columns is None else list(columns)
        self.parameters = dict(GBM, **(parameters or {}))
        self.live_only = live_only

    def _view(self, x):
        """live_only: the physical block is used only for a possession between two possessions
        (T-X-T). Next to a stoppage it is blanked, because there the open Metrica games have no
        ball track at all while the challenge tracker often keeps one (a difference of the data
        sources, not of the football)."""
        if getattr(self, "live_only", False):
            x = x.copy()
            first = len(c.FEATURES)
            rows = x[:, FEATURES.index("is_absorb")] == 0
            x[rows, first:first + len(PHYSICAL)] = np.nan
        return x[:, self.columns]

    def _regressor(self, x, y, monotone):
        from sklearn.ensemble import HistGradientBoostingRegressor
        extra = dict(monotonic_cst=[MONOTONE.get(FEATURES[j], 0) for j in self.columns]) if monotone else {}
        return HistGradientBoostingRegressor(**self.parameters, **extra).fit(x, y)

    def fit(self, x, gain, purity=None):
        x = self._view(x)
        monotone = self.kind.endswith("_mono")
        if self.kind in ("gbm", "gbm_mono"):
            self.parts = [(self._regressor(x, gain, monotone), 1.0)]
        elif self.kind in ("blend", "blend_mono"):
            self.parts = [(self._regressor(x, gain, monotone), float(np.std(gain))),
                          (self._regressor(x, purity, monotone), float(np.std(purity)))]
        elif self.kind == "gbc":
            from sklearn.ensemble import HistGradientBoostingClassifier
            keep = gain != 0
            weight = np.abs(gain[keep])
            self.model = HistGradientBoostingClassifier(**self.parameters).fit(
                x[keep], gain[keep] > 0, sample_weight=weight / weight.mean())
        elif self.kind in ("logit", "ridge"):
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.center = np.nanmedian(x, axis=0)
            self.flags = np.flatnonzero((~np.isfinite(x)).any(0))
            z = self._linear(x)
            if self.kind == "logit":
                keep = gain != 0
                weight = np.abs(gain[keep])
                self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
                self.model.fit(z[keep], gain[keep] > 0, logisticregression__sample_weight=weight / weight.mean())
            else:
                self.model = make_pipeline(StandardScaler(), Ridge(alpha=30.0)).fit(z, gain)
        else:
            raise ValueError("Unknown model kind")
        return self

    def _linear(self, x):
        missing = ~np.isfinite(x)
        return np.hstack([np.where(missing, self.center, x), missing[:, self.flags].astype(float)])

    def predict(self, x):
        x = self._view(x)
        if self.kind in ("gbm", "gbm_mono", "blend", "blend_mono"):
            return sum(model.predict(x) / scale for model, scale in self.parts)
        if self.kind == "gbc":
            return self.model.predict_proba(x)[:, 1] - 0.5
        if self.kind == "logit":
            return self.model.predict_proba(self._linear(x))[:, 1] - 0.5
        return self.model.predict(self._linear(x))



# -- tracking of the evaluation conditions and of the challenge match ------------------------

def metrica_signals(game, mode, profile=None):
    """Signals of one open Metrica game under the same sensor view the cached ensemble saw."""
    from .sources import external_match, challenge_match
    from .degrade import challenge_profile
    from .v6_data import sensor_view
    source = external_match(game)
    if mode != "clean" and profile is None:
        profile = challenge_profile(challenge_match())
    view = sensor_view(source, mode, 907 + game, profile)      # seeds of possession.v16_transfer
    return signals(view.ball, view.players, view.team, view.pitch)


def challenge_signals():
    from .sources import challenge_match
    match = challenge_match()
    return signals(match.ball, match.players, match.team, match.pitch)


def cached_signals(game, mode):
    path = ROOT / f"signals_{game}_{mode}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m possession.v28a_lab signals`")
    return dict(np.load(path))
