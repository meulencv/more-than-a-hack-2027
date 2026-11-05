"""v3 estimator: degraded external training, symmetric probabilities, segment decoding.

Pipeline (identical for external validation games and for the challenge):
    tracking -> features_v3 (5 Hz) -> gradient boosting P(H, A, D) averaged with the
    team-swapped input -> temporal_v3 Viterbi -> one label per frame inside each block.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from .degrade import degrade
from .features_v3 import feature_table, STEP
from .temporal_v3 import decode

CONFIG_PATH = Path(__file__).with_name("config_v3.json")
SWAP = np.array([1, 0, 2])


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())


def degraded_copy(match, profile, seed, cfg):
    rng = np.random.default_rng(10_000 + seed)
    sigma = rng.uniform(*cfg["degrade"]["sigma_range"])
    p_live = rng.uniform(*cfg["degrade"]["p_live_range"])
    return degrade(match, profile, seed, sigma=sigma, p_live=p_live)


def training_rows(match, profile, seeds, cfg):
    """Feature rows + labels for one external game, both team orientations."""
    xs, ys = [], []
    for seed in seeds:
        d = degraded_copy(match, profile, seed, cfg)
        labels = d.meta["labels"]
        for swap in (False, True):
            x, frames = feature_table(d, swap=swap)
            y = labels[frames].astype(int)
            keep = y >= 0
            if swap:
                y = np.where(y >= 0, SWAP[np.clip(y, 0, 2)], y)
            xs.append(x[keep])
            ys.append(y[keep])
    import pandas as pd
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def fit(x, y, cfg):
    params = dict(cfg["model"])
    model = HistGradientBoostingClassifier(**params, random_state=cfg["seed"],
                                           early_stopping=False)
    model.fit(x, y)
    return model


def symmetric_proba(model, match):
    x, frames = feature_table(match)
    xs, frames_s = feature_table(match, swap=True)
    if not np.array_equal(frames, frames_s):
        raise RuntimeError("Swapped feature table is misaligned")
    p = model.predict_proba(x)
    ps = model.predict_proba(xs)[:, SWAP]
    return (p + ps) / 2, frames


def stage2_refine(codes, proba_5hz, frames_5hz, stage2_cfg):
    """Refine candidate segments: bridge short deflections without control and false dead gaps."""
    if not stage2_cfg or not stage2_cfg.get("enabled", False):
        return codes
    refined = codes.copy()
    n = len(codes)
    boundaries = np.r_[0, np.flatnonzero(refined[1:] != refined[:-1]) + 1, n]
    segs = [(boundaries[i], boundaries[i + 1], refined[boundaries[i]])
            for i in range(len(boundaries) - 1)]

    max_to_s = stage2_cfg.get("max_turnover_duration_s", 3.0)
    to_thresh = stage2_cfg.get("turnover_conviction_threshold", 0.70)
    max_d_s = stage2_cfg.get("max_dead_duration_s", 2.5)
    d_thresh = stage2_cfg.get("dead_conviction_threshold", 0.65)

    for i in range(1, len(segs) - 1):
        prev_s, prev_e, prev_c = segs[i - 1]
        cur_s, cur_e, cur_c = segs[i]
        next_s, next_e, next_c = segs[i + 1]
        dur_s = (cur_e - cur_s) / 25.0

        # Deflection / contested bridge: H -> A -> H or A -> H -> A
        if prev_c in (0, 1) and next_c == prev_c and cur_c == (1 - prev_c):
            if dur_s < max_to_s:
                sel = (frames_5hz >= cur_s) & (frames_5hz < cur_e)
                if sel.any() and proba_5hz[sel, cur_c].mean() < to_thresh:
                    refined[cur_s:cur_e] = prev_c

        # Dead sandwich: Live -> D -> Live where D is short (< max_d_s)
        if prev_c in (0, 1) and next_c == prev_c and cur_c == 2:
            if dur_s < max_d_s:
                sel = (frames_5hz >= cur_s) & (frames_5hz < cur_e)
                if sel.any() and proba_5hz[sel, 2].mean() < d_thresh:
                    refined[cur_s:cur_e] = prev_c

    return refined


def decode_blocks(proba, frames, match, dec, stage2_cfg=None):
    """Frame-level codes (0 H, 1 A, 2 D, 3 X outside blocks)."""
    n = len(match.ball)
    codes = np.full(n, 3, dtype=np.int8)
    logp = dec["emission_weight"] * np.log(np.clip(proba, 1e-4, 1.0))
    for _, first, last in match.blocks:
        sel = (frames >= first) & (frames <= last)
        states = decode(logp[sel], dec["min_steps"], dec["cost_turnover"],
                        dec["cost_to_dead"], dec["cost_to_live"])
        f = frames[sel]
        stop = np.r_[f[1:], last + 1]
        codes[first:last + 1] = np.repeat(states, stop - f)
    if stage2_cfg and stage2_cfg.get("enabled", False):
        codes = stage2_refine(codes, proba, frames, stage2_cfg)
    return codes


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = ["load_config", "training_rows", "fit", "symmetric_proba", "decode_blocks", "STEP"]
