"""Compare frozen TCN folds with the exact state-blind stress used for v3."""
import json
from pathlib import Path
import numpy as np
from .sources import external_match
from .sequence_run import load_sequence_model, match_probabilities
from .audit_stress import blind_corruption
from .audit_reference import alternative_reference
from .v3 import load_config, decode_blocks
from .benchmark_external import evaluate_prediction


def main():
    root = Path("data/work/audit_stress")
    cfg = load_config(Path("possession/config_external_selected.json"))
    results = []
    for game in (1, 2, 3):
        source = external_match(game)
        match = blind_corruption(source, 907 + game)
        corrected, _ = alternative_reference(source.meta["events"], len(source.ball))
        model, norm = load_sequence_model(Path("data/work/sequence_tcn") / f"fold_{game}.pt", "mps")
        p, frames = match_probabilities(model, norm, match, "mps")
        pred = decode_blocks(p, frames, match, cfg["decoder"], stage2_cfg=cfg.get("stage2"))
        record = {"game": game, "model": "frozen TCN leave-one-game-out, 500 steps",
                  "original_weak_reference": evaluate_prediction(source.meta["labels"], pred),
                  "alternative_weak_reference": evaluate_prediction(corrected, pred)}
        results.append(record)
        np.savez_compressed(root / f"tcn_probabilities_{game}.npz", probabilities=p, frames=frames)
        print(json.dumps(record), flush=True)
        (root / "tcn_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
