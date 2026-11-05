"""Train a ball-live/dead classifier exclusively on two other Metrica games.

Event-derived labels are weak supervision, NOT frame-level possession ground truth.
Game 1 trains; game 2 is held out. No Barcelona/Athletic labels are ever loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score

FEATURE_NAMES = ["ball_seen_1s", "ball_seen_3s", "ball_speed_mps", "ball_still_1s",
                 "ball_still_3s", "players_median_speed_1s", "players_median_speed_3s",
                 "players_p75_speed_1s", "players_visible", "nearest_distance_m",
                 "ball_outside", "ball_missing"]


def feature_matrix(players, ball, fps=25):
    """Identical observations for external data and competition tracking. No tags."""
    pitch = np.array([105.3, 68.0])
    xy = ball[:, :2] * pitch
    seen = np.isfinite(xy).all(axis=1)
    speed = np.full(len(ball), np.nan)
    lag = max(1, round(0.2 * fps))
    speed[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=1) * fps / lag
    speed[speed > 45] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(players[:, :, 2], axis=1)
        p75 = np.nanpercentile(players[:, :, 2], 75, axis=1)
    med, p75 = np.nan_to_num(med, nan=0), np.nan_to_num(p75, nan=0)
    dist = np.linalg.norm(players[:, :, :2] * pitch - xy[:, None], axis=2)
    dist[~np.isfinite(dist)] = np.inf
    nearest = dist.min(axis=1)
    nearest[~np.isfinite(nearest)] = np.nan
    outside = seen & ((xy[:, 0] < -0.5) | (xy[:, 0] > pitch[0] + 0.5)
                      | (xy[:, 1] < -0.5) | (xy[:, 1] > pitch[1] + 0.5))
    def avg(x, seconds):
        return uniform_filter1d(np.asarray(x, dtype=float), max(1, round(seconds * fps)))
    still = np.isfinite(speed) & (speed < 0.4)
    return np.column_stack([
        avg(seen, 1), avg(seen, 3), speed, avg(still, 1), avg(still, 3),
        avg(med, 1), avg(med, 3), avg(p75, 1),
        np.isfinite(players[:, :, 0]).mean(axis=1), nearest, outside, ~seen,
    ]).astype(np.float32)


def read_game(root, game):
    home = pd.read_csv(root / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv", skiprows=2)
    away = pd.read_csv(root / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv", skiprows=2)
    if not np.array_equal(home["Frame"], away["Frame"]):
        raise ValueError("External home/away frames are not aligned")
    n = len(home)
    xy = np.concatenate([frame.iloc[:, 3:-2].to_numpy(dtype=np.float32).reshape(n, -1, 2)
                         for frame in (home, away)], axis=1)
    players = np.full((*xy.shape[:2], 3), np.nan, dtype=np.float32)
    players[:, :, :2] = xy
    delta = np.linalg.norm((xy[5:] - xy[:-5]) * [105.3, 68], axis=2) * 5
    # Cross-half jumps and tracking glitches are not physical speed observations.
    delta[delta > 15] = np.nan
    players[5:, :, 2] = delta
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    ball[:, :2] = home.iloc[:, -2:].to_numpy(dtype=np.float32)
    events = pd.read_csv(root / f"Sample_Game_{game}_RawEventsData.csv")
    frames = home["Frame"].to_numpy(dtype=int)
    if not np.array_equal(frames, np.arange(1, n + 1)):
        raise ValueError("External frames do not start at 1 and increase by 1")
    return players, ball, weak_labels(events, n)


def weak_labels(events, n):
    """Live: interior of passes. Dead: unambiguous ball-out to next restart gaps.

    Excludes duels, ambiguous fouls/advantage, handovers and boundaries. Nothing
    is labelled by a person in this code. Source event annotation is by Metrica.
    """
    labels = np.full(n, -1, dtype=np.int8)
    live = np.zeros(n, dtype=bool)
    for _, row in events[events.Type == "PASS"].iterrows():
        start, end = int(row["Start Frame"]) - 1, int(row["End Frame"])
        if end - start > 10:
            live[max(0, start + 5):min(n, end - 5)] = True
    labels[live] = 0
    restarts = events[events.Type == "SET PIECE"].sort_values("Start Frame")
    for _, row in events[events.Type == "BALL OUT"].iterrows():
        start = max(int(row["Start Frame"]), int(row["End Frame"])) - 1
        following = restarts[(restarts.Period == row.Period) & (restarts["Start Frame"] > start + 1)]
        if following.empty:
            continue
        end = int(following.iloc[0]["Start Frame"]) - 1
        if 25 <= end - start <= 25 * 120 and not live[max(start, 0):min(end, n)].any():
            labels[max(start + 10, 0):min(end - 10, n)] = 1
    return labels


def occlude(ball, seed):
    """Synthetic contiguous dropout: 40% missing in 1-second blocks on average."""
    rng = np.random.default_rng(seed)
    mask = np.repeat(rng.random((len(ball) + 24) // 25) < 0.4, 25)[:len(ball)]
    augmented = ball.copy()
    augmented[mask] = np.nan
    return augmented


def metrics(model, x, y):
    keep = (y >= 0) & (np.arange(len(y)) % 5 == 0)
    pred = model.predict(x[keep])
    return {"samples": int(keep.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(y[keep], pred)),
            "classes": classification_report(y[keep], pred, output_dict=True, zero_division=0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/external/metrica_sample"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/work/external_deadball"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    p1, b1, y1 = read_game(args.data_dir, 1)
    p2, b2, y2 = read_game(args.data_dir, 2)
    keep = (y1 >= 0) & (np.arange(len(y1)) % 5 == 0)
    train_x = np.concatenate([feature_matrix(p1, b)[keep]
                              for b in (b1, occlude(b1, 2027), occlude(b1, 2028))])
    train_y = np.tile(y1[keep], 3)
    model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
                                          learning_rate=0.08, l2_regularization=1,
                                          class_weight="balanced", random_state=2027,
                                          early_stopping=False)
    print("Training on external game 1 only:", len(train_y), "examples", flush=True)
    model.fit(train_x, train_y)
    joblib.dump(model, args.output_dir / "model.joblib")
    report = {"source": "https://github.com/metrica-sports/sample-data",
              "revision": "e706dd506b360d69d9d123d5b8026e7294b13996",
              "label_type": "weak event-derived labels; not possession ground truth",
              "label_provider": "Metrica Sports; individual annotators not disclosed in source",
              "training_game": 1, "validation_game": 2,
              "challenge_ground_truth_read": False,
              "feature_names": FEATURE_NAMES,
              "training_examples": int(len(train_y)),
              "validation_clean": metrics(model, feature_matrix(p2, b2), y2),
              "validation_dropout": metrics(model, feature_matrix(p2, occlude(b2, 2029)), y2),
              "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in args.data_dir.glob("*.csv")}}
    (args.output_dir / "training.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("training_examples", "validation_clean", "validation_dropout")}, indent=2))


if __name__ == "__main__":
    main()
