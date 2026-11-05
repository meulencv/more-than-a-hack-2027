"""Make clean external tracking look like the challenge's automatic tracking.

Calibration uses ONLY unlabeled input statistics of the challenge tracking (run lengths of
ball visibility, nearest player-to-ball distance, number of tracked players). It never reads
challenge labels. Labels of the external game are used to decide where dead-ball sightings may
be synthesised, which is part of building external training/validation data.

Known approximation: the true joint law of "ball missing" and "game state" in the challenge is
unknown. We therefore randomise its strength per seed instead of fitting it.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import ks_2samp


@dataclass
class Profile:
    missing_runs: np.ndarray     # challenge ball-missing run lengths (frames), graded frames
    seen_runs: np.ndarray        # challenge ball-seen run lengths (frames)
    nearest_q: np.ndarray        # challenge nearest player-ball distance sample (m)
    player_missing: float        # fraction of the 22 expected players not tracked


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def nearest_distance(match, frames=None):
    players, ball = match.players, match.ball
    if frames is not None:
        players, ball = players[frames], ball[frames]
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d[~np.isfinite(d)] = np.inf
    near = d.min(axis=1)
    return near[np.isfinite(near)]


def challenge_profile(match):
    frames = np.concatenate([np.arange(a, z + 1) for _, a, z in match.blocks])
    seen = np.isfinite(match.ball[frames, 0])
    missing_runs, seen_runs = [], []
    for _, a, z in match.blocks:
        s = np.isfinite(match.ball[a:z + 1, 0])
        missing_runs.append(_runs(~s))
        seen_runs.append(_runs(s))
    visible = np.isfinite(match.players[frames, :, 0]).sum(axis=1)
    near = nearest_distance(match, frames[seen])
    return Profile(np.concatenate(missing_runs), np.concatenate(seen_runs),
                   np.sort(near[::7]), float(np.clip(1 - visible.mean() / 22, 0, 1)))


def _ar_noise(rng, n, sigma, tau_frames):
    rho = np.exp(-1.0 / tau_frames)
    eps = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), size=(n, 2))
    out = np.empty((n, 2))
    out[0] = rng.normal(0, sigma, 2)
    # first-order IIR filter, vectorised through scipy for speed
    from scipy.signal import lfilter
    out[1:] = lfilter([1.0], [1.0, -rho], eps[1:], axis=0, zi=(rho * out[0])[None, :])[0]
    return out


def _alternating_mask(rng, n, missing_runs, seen_runs, p_missing):
    k = missing_runs.mean() * (1 - p_missing) / (p_missing * seen_runs.mean())
    mask = np.zeros(n, dtype=bool)
    t, missing = 0, rng.random() < p_missing
    while t < n:
        length = int(max(1, (rng.choice(missing_runs) if missing else rng.choice(seen_runs) * k)))
        if missing:
            mask[t:t + length] = True
        t += length
        missing = not missing
    return mask


def degrade(match, profile, seed, sigma=1.5, tau_s=0.6, p_live=None, p_dead_sighting=None):
    """Return a degraded copy of an external match (labels unchanged)."""
    rng = np.random.default_rng(seed)
    n = len(match.ball)
    labels = match.meta["labels"]
    p_live = rng.uniform(0.2, 0.4) if p_live is None else p_live
    p_dead_sighting = rng.uniform(0.25, 0.75) if p_dead_sighting is None else p_dead_sighting
    ball = match.ball.astype(np.float64).copy()
    players = match.players.astype(np.float64).copy()
    # 1. Dead-ball sightings: in clean external data the ball is usually NaN when dead.
    #    The challenge tracker can see a ball being placed or carried for a restart.
    dead = labels == 2
    d = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for start, end in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if rng.random() > p_dead_sighting or end >= n:
            continue
        spot = ball[end]
        if not np.isfinite(spot).all():
            continue
        first = start + int((end - start) * rng.uniform(0.0, 0.8))
        if rng.random() < 0.5:
            ball[first:end] = spot + rng.normal(0, 0.3, 2)
        else:
            near = np.linalg.norm(players[first] - spot, axis=1)
            if np.isfinite(near).any():
                carrier = int(np.nanargmin(near))
                path = players[first:end, carrier]
                path = np.where(np.isfinite(path), path, spot)
                ball[first:end] = path + rng.normal(0, 0.5, 2)
    # 2. Detection offset: projected ball vs player feet, aerial balls, calibration error.
    ball += _ar_noise(rng, n, sigma, tau_s * 25) + rng.normal(0, 0.12, size=(n, 2))
    # 3. Long ball-missing runs with the challenge's run-length structure.
    ball[_alternating_mask(rng, n, profile.missing_runs, profile.seen_runs, p_live)] = np.nan
    # 4. Player tracking dropouts: per player, short alternating gaps.
    slots = players.shape[1]
    for j in range(slots):
        if not np.isfinite(players[:, j, 0]).any():
            continue
        gaps = _alternating_mask(rng, n, np.array([25.0, 50, 100]), np.array([250.0, 500, 1000]),
                                 max(0.01, profile.player_missing))
        players[gaps, j] = np.nan
    players += _ar_noise(rng, n, 0.15, 12.0)[:, None, :]
    return match.copy_with(ball=ball.astype(np.float32), players=players.astype(np.float32),
                           meta={**match.meta, "degradation": {
                               "seed": int(seed), "sigma": sigma, "p_live": float(p_live),
                               "p_dead_sighting": float(p_dead_sighting)}})


def calibrate_sigma(external, profile, grid=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), seed=11):
    """Choose ball noise so the nearest-player distance matches the challenge (KS distance)."""
    results = {}
    for sigma in grid:
        degraded = degrade(external, profile, seed, sigma=sigma)
        near = nearest_distance(degraded)
        results[sigma] = float(ks_2samp(near[::7], profile.nearest_q).statistic)
    best = min(results, key=results.get)
    return best, results
