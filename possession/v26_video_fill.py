"""V26: the frozen four-network ensemble re-run on a ball track completed from the video.

Order of ball sources, none of them overwriting an earlier one:
1. the tactical automatic tracking;
2. the self-supervised video detector (possession.v26_ball / v26_track), tactical image -> pitch;
3. the broadcast tracking through the automatic camera alignment (unchanged V11 logic).

The networks, their weights and the decoder are untouched: they simply see a ball on more
frames, which is the situation they were trained in. The resulting probabilities feed the
camera-view fusion (possession.v24_camera_fusion). No ground truth is read.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .sources import challenge_match
from .camera_sync import fused_ball
from .competitive_bundle import CompetitiveBundle, camera_match, read_alignment, align
from .v12_compare import match_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("docs/FINAL_SELECTION.json"))
    parser.add_argument("--video-ball", type=Path, default=Path("data/work/v26/video_ball.npz"))
    parser.add_argument("--alignment", type=Path, default=Path("data/work/v11/challenge_alignment.npz"),
                        help="alignment.npz of the frozen run")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--work-dir", type=Path, default=Path("data/work/v26/run"))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    selection = json.loads(args.selection.read_text())
    reference = challenge_match()
    other, valid = camera_match(reference, args.work_dir)
    alignment = align(reference, other, valid, args.work_dir, args.alignment)
    saved = np.load(args.video_ball)
    video = saved["ball"] * np.array(reference.pitch, dtype=np.float32)      # stored normalised, used in metres
    ball = reference.ball.copy()
    from_video = (~np.isfinite(ball[:, 0]) & np.isfinite(video[:, 0]) & (saved["source"] >= 2)
                  & (saved["confidence"] >= args.min_confidence))
    ball[from_video] = video[from_video]
    match, from_broadcast = fused_ball(reference.copy_with(ball=ball), other, alignment)
    bundle = CompetitiveBundle(selection, args.device)
    record = match_record(match)
    probabilities, edges = bundle.probabilities(record)
    np.savez_compressed(args.work_dir / "probabilities.npz", probabilities=probabilities, edges=edges,
                        frames=record["frames"])
    graded = np.zeros(len(ball), dtype=bool)
    for _, a, z in reference.blocks:
        graded[a:z + 1] = True
    report = dict(ball_missing_tracking=float((~np.isfinite(reference.ball[graded, 0])).mean()),
                  ball_missing_final=float((~np.isfinite(match.ball[graded, 0])).mean()),
                  from_video=int((from_video & graded).sum()), from_broadcast=int((from_broadcast & graded).sum()),
                  challenge_labels_read=False)
    (args.work_dir / "run.json").write_text(json.dumps(report, indent=2))
    print("V26 VIDEO FILL", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
