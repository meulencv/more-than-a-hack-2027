"""Hierarchical H/A/D estimator: live/dead and conditional team control.

The three-class v3 baseline is preserved. No public challenge labels are read.
"""
import numpy as np
from .v3 import fit


def train(x, y, cfg):
    alive = y != 2
    return {"dead": fit(x, (y == 2).astype(int), cfg),
            "team": fit(x[alive], y[alive], cfg)}


def probabilities(models, x, mirrored):
    dead = (models["dead"].predict_proba(x)[:, 1]
            + models["dead"].predict_proba(mirrored)[:, 1]) / 2
    home = (models["team"].predict_proba(x)[:, 0]
            + models["team"].predict_proba(mirrored)[:, 1]) / 2
    return np.column_stack(((1 - dead) * home, (1 - dead) * (1 - home), dead))
