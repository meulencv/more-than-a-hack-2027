"""State-blind tracking corruption stress test; no challenge labels are read.

This is a robustness diagnostic, not an unbiased leaderboard. Clean external ball
availability is already state-correlated. Added corruption never uses event labels.
"""
import json
from pathlib import Path
import joblib
import numpy as np
from .sources import external_match
from .degrade import _ar_noise
from .v3 import symmetric_proba, decode_blocks, load_config
from .benchmark_external import evaluate_prediction
from .audit_reference import alternative_reference


def blind_corruption(match, seed=907, long_gaps=True):
    """Only observed arrays and slot team membership determine output observations."""
    rng = np.random.default_rng(seed)
    n = len(match.ball)
    ball = match.ball.astype(float).copy()
    players = match.players.astype(float).copy()
    ball += _ar_noise(rng, n, 1.1, 15)
    for slot in range(players.shape[1]):
        players[:, slot] += _ar_noise(rng, n, 0.35, 12)
    if long_gaps:
        # Blocks of 8 seconds chosen without looking at play state or possession.
        block = 8 * 25
        missing = np.repeat(rng.random((n + block - 1) // block) < 0.4, block)[:n]
        ball[missing] = np.nan
    # Reacquisition swaps track identity within each team every 20 seconds. Team
    # geometry is preserved, while velocity computation must survive fragmented IDs.
    for first in range(0, n, 20 * 25):
        last = min(first + 20 * 25, n)
        for team in (0, 1):
            slots = np.flatnonzero(match.team == team)
            players[first:last, slots] = players[first:last, rng.permutation(slots)]
    return match.copy_with(players=players.astype(np.float32), ball=ball.astype(np.float32))


def main():
    root = Path("data/work/audit_stress")
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path("possession/config_external_selected.json"))
    report = []
    for game in (1, 2, 3):
        source = external_match(game)
        model = joblib.load(Path("data/work/hierarchical_v4") / f"fold_{game}.joblib")["baseline"]
        corrected, _ = alternative_reference(source.meta["events"], len(source.ball))
        match = blind_corruption(source, 907 + game)
        p, frames = symmetric_proba(model, match)
        pred = decode_blocks(p, frames, match, cfg["decoder"], stage2_cfg=cfg.get("stage2"))
        record = {"game": game, "model": "frozen v3 leave-one-game-out",
                  "corruption": "label-blind long gaps + tracking ID switches + independent player noise",
                  "original_weak_reference": evaluate_prediction(source.meta["labels"], pred),
                  "alternative_weak_reference": evaluate_prediction(corrected, pred)}
        report.append(record)
        print(json.dumps(record), flush=True)
        (root / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
