"""Train on external games (1, 2, 3) and generate the v3 full-match submission.

Strictly adheres to challenge rules:
* Training uses ONLY external games (Metrica Sample Games 1, 2, 3).
* No challenge labels or public 5-minute ground truth are ever read.
* Produces deterministic output (fixed random seeds, single-thread or fixed threading).

Usage:
    OMP_NUM_THREADS=8 .venv/bin/python -m possession.run_v3
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from starter.baseline import write_submission
from .sources import external_match, challenge_match
from .degrade import challenge_profile
from .v3 import load_config, training_rows, fit, symmetric_proba, decode_blocks, file_sha256

LABELS_MAP = np.array(["H", "A", "D", "X"], dtype="<U1")


def train_external(cfg, profile, model_path, data_dir=None):
    print("Training final v3 model on external games 1, 2, and 3...", flush=True)
    matches = [external_match(g, root=data_dir) if data_dir else external_match(g) for g in (1, 2, 3)]
    dfs, ys = [], []
    for m in matches:
        df, y = training_rows(m, profile, cfg["training_seeds"], cfg)
        dfs.append(df)
        ys.append(y)
    X = pd.concat(dfs, ignore_index=True)
    y = np.concatenate(ys)
    print(f"Total training data: {len(y)} samples, {X.shape[1]} features", flush=True)
    model = fit(X, y, cfg)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved trained model to {model_path}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_v3.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/temporal_v3.csv"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/work/temporal_v3"))
    parser.add_argument("--model", type=Path, help="Pretrained model path (if omitted, trains fresh)")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    print("Loading challenge tactical tracking...", flush=True)
    c_match = challenge_match(cache_dir=args.work_dir.parent / "cache")
    profile = challenge_profile(c_match)

    model_path = args.model if args.model else args.work_dir / "model.joblib"
    if args.model and args.model.exists():
        print(f"Loading existing model from {args.model}...", flush=True)
        model = joblib.load(args.model)
    else:
        model = train_external(cfg, profile, model_path)

    print("Extracting features and predicting symmetric probabilities on challenge match...", flush=True)
    proba, frames = symmetric_proba(model, c_match)

    print("Decoding temporal states...", flush=True)
    pred_codes = decode_blocks(proba, frames, c_match, cfg["decoder"], stage2_cfg=cfg.get("stage2"))
    labels = LABELS_MAP[pred_codes]

    print(f"Writing submission to {args.output}...", flush=True)
    write_submission(str(args.output), labels)

    # Save diagnostics
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    segments = [{"start": int(a), "end_exclusive": int(b), "label": str(labels[a])}
                for a, b in zip(boundaries[:-1], boundaries[1:])]
    (args.work_dir / "segments.json").write_text(json.dumps(segments, indent=2))

    report = {
        "config": cfg,
        "training_data": "Metrica Sample Games 1, 2, 3 (external)",
        "ground_truth_read": False,
        "model_sha256": file_sha256(model_path),
        "submission_sha256": file_sha256(args.output),
        "label_counts": {lbl: int((labels == lbl).sum()) for lbl in "HADX"},
        "possessions": int(sum(s["label"] in "HA" for s in segments)),
        "mean_possession_duration_s": float(np.mean([s["end_exclusive"] - s["start"] for s in segments if s["label"] in "HA"]) / 25.0) if any(s["label"] in "HA" for s in segments) else 0.0,
        "dead_segments": int(sum(s["label"] == "D" for s in segments)),
        "mean_dead_duration_s": float(np.mean([s["end_exclusive"] - s["start"] for s in segments if s["label"] == "D"]) / 25.0) if any(s["label"] == "D" for s in segments) else 0.0
    }
    (args.work_dir / "run.json").write_text(json.dumps(report, indent=2))
    print("Run completed successfully:\n", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
