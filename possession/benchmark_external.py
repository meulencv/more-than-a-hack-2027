"""Benchmark on external games: evaluates full possession metrics (spellF1, macroF1).

Compares:
1. Reference baseline (closest player + smooth window: 0, 25, 50, 125 frames)
2. Proposed v3 model under Leave-One-Game-Out cross-validation across games 1, 2, 3.

Evaluations run on degraded external tracking (using profile calibrated from challenge
unlabeled tracking noise/occlusion statistics).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import evaluation.possession_metrics as pm
from .sources import external_match, challenge_match
from .degrade import challenge_profile, degrade
from .features_v3 import feature_table
from .v3 import load_config, training_rows, fit, symmetric_proba, decode_blocks


def run_baseline(match, window=50):
    """Replicates starter/baseline.py nearest player + window majority smoothing."""
    players = match.players  # (n, slots, 2)
    ball = match.ball        # (n, 2)
    team = match.team        # (slots,)
    n = len(ball)
    
    seen = np.isfinite(ball[:, 0])
    dists = np.linalg.norm(players - ball[:, None, :], axis=2)
    dists[~np.isfinite(dists)] = np.inf
    
    home_d = dists[:, team == 0].min(axis=1)
    away_d = dists[:, team == 1].min(axis=1)
    
    codes = np.full(n, -1, dtype=np.int8)
    valid_dist = seen & (np.isfinite(home_d) | np.isfinite(away_d))
    codes[valid_dist] = np.where(home_d[valid_dist] <= away_d[valid_dist], 0, 1)
    
    # Smooth within each block separately
    out = np.full(n, 3, dtype=np.int8)  # 3 is X
    for _, first, last in match.blocks:
        sub = codes[first:last + 1]
        m = len(sub)
        if window <= 0:
            pred = np.where(sub == 0, 0, np.where(sub == 1, 1, 2)).astype(np.int8)
        else:
            h_cum = np.r_[0, np.cumsum(sub == 0)]
            a_cum = np.r_[0, np.cumsum(sub == 1)]
            idx = np.arange(m)
            lo = np.maximum(0, idx - window)
            hi = np.minimum(m, idx + window + 1)
            h_cnt = h_cum[hi] - h_cum[lo]
            a_cnt = a_cum[hi] - a_cum[lo]
            has_seen = (h_cnt + a_cnt) > 0
            pred = np.full(m, 2, dtype=np.int8)
            pred[has_seen] = np.where(h_cnt[has_seen] >= a_cnt[has_seen], 0, 1)
        out[first:last + 1] = pred
    return out


def evaluate_prediction(gt_labels, pred_codes):
    """Computes challenge metrics via official evaluation/possession_metrics.py."""
    # Convert gt_labels (-1 -> 3 for ungraded)
    gt = np.where(gt_labels >= 0, gt_labels, 3).astype(np.int8)
    pred = pred_codes.astype(np.int8)
    res = pm.evaluate(gt, pred, bootstrap_n=0)
    return {
        "macro_f1": float(res["macro_f1"]),
        "spell_f1_mean": float(res["spell_f1_mean"]),
        "spell_f1_headline": float(res["spell_f1_at_headline"]),
        "accuracy": float(res["accuracy"]),
        "alive_dead_f1": float(res["alive_dead_f1"]),
        "n_pred_spells": int(res["n_pred_spells"]),
        "n_gt_spells": int(res["n_gt_spells"]),
        "mean_spell_duration_s": float(res["mean_spell_duration_s"]),
        "gt_mean_spell_duration_s": float(res["gt_mean_spell_duration_s"])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_v3.json"))
    parser.add_argument("--output", type=Path, default=Path("data/work/external_benchmark.json"))
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    print("Computing challenge profile from unlabeled tracking...", flush=True)
    c_match = challenge_match()
    profile = challenge_profile(c_match)
    
    print("Loading external games 1, 2, 3...", flush=True)
    matches = {g: external_match(g) for g in (1, 2, 3)}
    
    # Degrade test matches with a fixed benchmark seed
    eval_seed = 42
    test_degraded = {g: degrade(matches[g], profile, seed=eval_seed + g,
                                sigma=1.1, p_live=0.35, p_dead_sighting=0.5)
                     for g in (1, 2, 3)}
    
    results = {"baselines": {}, "leave_one_out": {}}
    
    print("\n--- Evaluating Baselines on Degraded External Games ---", flush=True)
    for g in (1, 2, 3):
        results["baselines"][f"game_{g}"] = {}
        for w in (0, 25, 50, 125):
            pred = run_baseline(test_degraded[g], window=w)
            ev = evaluate_prediction(matches[g].meta["labels"], pred)
            results["baselines"][f"game_{g}"][f"smooth_{w}"] = ev
            print(f"Game {g} baseline smooth{w:3d} -> spellF1: {ev['spell_f1_mean']:.4f}, macroF1: {ev['macro_f1']:.4f}, spells: {ev['n_pred_spells']} (GT: {ev['n_gt_spells']})")

    print("\n--- Running Leave-One-Game-Out CV for v3 ---", flush=True)
    for test_game in (1, 2, 3):
        train_games = [g for g in (1, 2, 3) if g != test_game]
        print(f"\nTraining on Games {train_games}, Testing on Game {test_game}...", flush=True)
        
        train_dfs, train_ys = [], []
        for tg in train_games:
            df, y = training_rows(matches[tg], profile, cfg["training_seeds"], cfg)
            train_dfs.append(df)
            train_ys.append(y)
        
        X_train = pd.concat(train_dfs, ignore_index=True)
        y_train = np.concatenate(train_ys)
        print(f"  Training set size: {len(y_train)} samples across {X_train.shape[1]} features", flush=True)
        
        model = fit(X_train, y_train, cfg)
        
        # Test on held-out degraded game
        proba, frames = symmetric_proba(model, test_degraded[test_game])
        pred_codes = decode_blocks(proba, frames, test_degraded[test_game], cfg["decoder"],
                                   stage2_cfg=cfg.get("stage2"))
        
        ev = evaluate_prediction(matches[test_game].meta["labels"], pred_codes)
        results["leave_one_out"][f"test_game_{test_game}"] = ev
        print(f"  v3 Test Game {test_game} -> spellF1: {ev['spell_f1_mean']:.4f}, macroF1: {ev['macro_f1']:.4f}, alive_dead_f1: {ev['alive_dead_f1']:.4f}, spells: {ev['n_pred_spells']} (GT: {ev['n_gt_spells']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nFull benchmark saved to {args.output}")


if __name__ == "__main__":
    main()
