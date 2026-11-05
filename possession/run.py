"""Generate a deterministic full-match submission without accessing ground truth.

    python -m possession.run
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from games import GAME, half_video_blocks
from starter.baseline import write_submission
from starter.load_smart_tagging import load
from .data import load_tracking
from .features import compute
from .temporal import decode


def predict(emission, config, blocks):
    labels = np.full(len(emission), "X", dtype="<U1")
    step = config["step_frames"]
    for _, first, last in blocks:
        evidence = emission[first:last + 1]
        starts = np.arange(0, len(evidence), step)
        sizes = np.minimum(step, len(evidence) - starts)
        downsampled = np.add.reduceat(evidence, starts, axis=0) / sizes[:, None]
        states = decode(downsampled, config["minimum_state_steps"], config["switch_cost"])
        labels[first:last + 1] = np.repeat(np.array(["H", "A", "D"])[states], sizes)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/temporal_v1.csv"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/work/temporal_v1"))
    parser.add_argument("--dead-model", type=Path,
                        help="Optional classifier trained on other matches; use an explicit output/work-dir")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    print("Loading automatic tracking (no ground truth)...", flush=True)
    players, ball, slots = load_tracking(cache_dir=args.work_dir.parent / "cache")
    emission, features = compute(players, ball, slots, load(), config)
    if args.dead_model:
        if args.output == Path("evaluation/results/temporal_v1.csv"):
            parser.error("Specify a separate --output when using an external model")
        import joblib
        from .external_training import feature_matrix
        classifier = joblib.load(args.dead_model)
        dead_probability = np.clip(classifier.predict_proba(feature_matrix(players, ball))[:, 1],
                                   0.005, 0.995)
        # Team evidence is conditional on a live ball; external classifier estimates
        # live/dead from motion and visibility without using any tactical tags.
        emission[:, :2] += np.log1p(-dead_probability)[:, None]
        emission[:, 2] = np.log(dead_probability)
        features["dead_probability"] = dead_probability
    labels = predict(emission, config, half_video_blocks(GAME))
    write_submission(str(args.output), labels)
    np.savez_compressed(args.work_dir / "features.npz", **features)
    # Contiguous runs for diagnosis only. Not corrections or hand annotations.
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    segments = [{"start": int(a), "end_exclusive": int(b), "label": str(labels[a])}
                for a, b in zip(boundaries[:-1], boundaries[1:])]
    (args.work_dir / "segments.json").write_text(json.dumps(segments, indent=2))
    report = {"config": config,
              "training_labels": "external Metrica game 1 event-derived examples" if args.dead_model else "none",
              "dead_model_sha256": hashlib.sha256(args.dead_model.read_bytes()).hexdigest() if args.dead_model else None,
              "ground_truth_read": False,
              "submission_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "label_counts": {label: int((labels == label).sum()) for label in "HADX"},
              "possessions": sum(s["label"] in "HA" for s in segments),
              "ball_observed_fraction": float(features["observed_ball"].mean())}
    (args.work_dir / "run.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
