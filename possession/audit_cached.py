"""Re-score the exact same frozen predictions against two weak references."""
import json
from pathlib import Path
import numpy as np
from .sources import external_match
from .v3 import decode_blocks, load_config
from .benchmark_external import evaluate_prediction
from .audit_reference import alternative_reference


def main():
    cache = Path("data/work/hierarchical_v4")
    out = Path("data/work/audit_reference")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path("possession/config_external_selected.json"))
    results = []
    for game in (1, 2, 3):
        match = external_match(game)
        corrected, _ = alternative_reference(match.meta["events"], len(match.ball))
        for seed in (42 + game, 202711 + game):
            data = np.load(cache / f"probabilities_{game}_{seed}.npz")
            pred = decode_blocks(data["baseline"], data["frames"], match,
                                 cfg["decoder"], stage2_cfg=cfg.get("stage2"))
            record = {"game": game, "seed": seed,
                      "original": evaluate_prediction(match.meta["labels"], pred),
                      "alternative": evaluate_prediction(corrected, pred)}
            print(json.dumps(record), flush=True)
            results.append(record)
    means = {ref: float(np.mean([r[ref]["spell_f1_mean"] for r in results]))
             for ref in ("original", "alternative")}
    (out / "cached_scores.json").write_text(json.dumps({"means": means, "cases": results}, indent=2))
    print("Mean spellF1:", means, flush=True)


if __name__ == "__main__":
    main()
