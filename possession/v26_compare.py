"""Compare two submissions frame by frame (no ground truth): what V26 changed and where."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from games import GAME, half_video_blocks
from .data import load_tracking
from .v24_camera_fusion import spells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=Path("evaluation/results/v24_camera_fusion.csv"))
    parser.add_argument("--new", type=Path, default=Path("evaluation/results/v26_video_ball.csv"))
    parser.add_argument("--video-ball", type=Path, default=Path("data/work/v26/video_ball.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/work/v26/compare.json"))
    args = parser.parse_args()
    blocks = list(half_video_blocks(GAME))
    code = {"H": 0, "A": 1, "D": 2, "X": 3}
    old = pd.read_csv(args.old)["label"].map(code).to_numpy()
    new = pd.read_csv(args.new)["label"].map(code).to_numpy()
    graded = np.zeros(len(old), dtype=bool)
    for _, a, z in blocks:
        graded[a:z + 1] = True
    _, ball, _ = load_tracking(cache_dir="data/work/cache")
    tracked = np.isfinite(ball[:, 0])
    source = np.load(args.video_ball)["source"]
    changed = graded & (old != new)
    names = "HAD"
    moves = {names[i] + "->" + names[j]: int((changed & (old == i) & (new == j)).sum())
             for i in range(3) for j in range(3) if i != j}
    report = dict(
        old=dict(csv=str(args.old), spells=spells(old, blocks), dead_share=float((old[graded] == 2).mean())),
        new=dict(csv=str(args.new), spells=spells(new, blocks), dead_share=float((new[graded] == 2).mean())),
        graded_frames=int(graded.sum()), changed_frames=int(changed.sum()),
        changed_share=float(changed.sum() / graded.sum()),
        changed_where=dict(tracking_ball_present=int((changed & tracked).sum()),
                           tracking_ball_missing=int((changed & ~tracked).sum()),
                           of_which_video_ball_filled=int((changed & ~tracked & (source >= 2)).sum()),
                           of_which_still_no_ball=int((changed & ~tracked & (source < 2)).sum())),
        changed_share_by_stratum=dict(
            tracking_ball_present=float(changed[graded & tracked].mean()),
            video_ball_filled=float(changed[graded & ~tracked & (source >= 2)].mean()),
            still_no_ball=float(changed[graded & ~tracked & (source < 2)].mean())),
        moves=moves, ground_truth_read=False)
    args.output.write_text(json.dumps(report, indent=2))
    print("V26 COMPARE", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
