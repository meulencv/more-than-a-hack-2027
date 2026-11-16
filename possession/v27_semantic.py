"""V27: object-level semantic adapter (spell verifier) on top of the frozen ensemble / V24 fusion.

Why. The four networks learned DFL's native possession labels (~9 possessions per live minute,
half of them under 3 s). The challenge ground truth is Metrica's controlled possession; in
Metrica's own open games an event-derived reference with the challenge's touch rule has ~4.2
possessions per live minute of mean ~14 s, which is also what DATA.md publishes for the public
window (17 possessions, mean 14.2 s). The decoded ensemble sits in between and over-fragments.

What. After the unchanged decoder, every predicted possession of at most 8 s next to another
state is a candidate for removal (T-X-T -> surrounding team; next to a stoppage -> D or the
neighbouring team). A small gradient-boosting regressor predicts the spellF1 change of each
removal from features of the frozen output only (duration, context, probabilities, boundary
scores, decoder margin). It was fitted on the three open Metrica games:

* mode="robust" (default, adopted): target = min(gain under the touch-rule reference, gain under
  the alternative reference). A possession is removed only if that is predicted to help under
  BOTH readings of "possession"; threshold 0 is the natural decision boundary, not tuned.
* mode="consistent" (semantic bet, NOT adopted): target = gain under the touch-rule reference.

No challenge ground truth is read here or anywhere in the fitting (possession.v27_lab).

    .venv/bin/python -m possession.v27_semantic \
        --probabilities data/work/final_cpu/probabilities.npz \
        --alignment data/work/v24/replay_bundle/alignment.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from . import v27_core as c

MODES = ("robust", "consistent")


def load_verifier(mode="robust", root=c.ROOT):
    import joblib
    if mode not in MODES:
        raise ValueError("Unknown verifier mode")
    path = Path(root) / f"verifier_{mode}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m possession.v27_lab table` and `... fit`")
    saved = joblib.load(path)
    if saved["features"] != c.FEATURES:
        raise ValueError("Verifier was fitted with another feature list")
    return saved


def semantic_adapter(probabilities, edges, frames, blocks, n_frames, mode="robust", threshold=0.0,
                     verifier=None):
    """Frozen-ensemble (or fused) probabilities -> (codes after the adapter, codes before, report).

    probabilities (n, 3) H/A/D, edges (n,), frames (n,) as written by competitive_bundle /
    v24_camera_fusion; blocks = [(half, first_frame, last_frame)] inclusive; codes use 0/1/2 and
    3 outside the halves. `before` is exactly v14_resolution.decode_fine(..., 'probability').
    """
    saved = verifier or load_verifier(mode)
    model, max_frames = saved["model"], saved["max_frames"]
    interpolated = c.interpolate(np.asarray(probabilities), np.asarray(edges), np.asarray(frames), blocks)
    before = c.decode(interpolated, n_frames)[0]
    p, edge = c.dense(interpolated, n_frames)
    segs = c.segments(before, blocks)

    def score(current, cands):
        return model.predict(c.candidate_features(current, cands, p, edge))

    new, applied = c.apply_scored(segs, score, threshold, max_frames)
    after = c.to_codes(new, n_frames)
    graded = np.concatenate([np.arange(a, z + 1) for _, a, z in blocks])
    kinds = {}
    for action, _ in applied:
        kinds[action] = kinds.get(action, 0) + 1
    report = dict(mode=saved["target"], threshold=threshold, removals=len(applied), removals_by_kind=kinds,
                  removed_seconds_median=float(np.median([d for _, d in applied])) if applied else 0.0,
                  before=c.duration_stats(before, blocks), after=c.duration_stats(after, blocks),
                  graded_frames=int(len(graded)), graded_frames_changed=int((before[graded] != after[graded]).sum()),
                  dead_frames_added=int(((after[graded] == 2) & (before[graded] != 2)).sum()))
    return after, before, report


def fused_probabilities(probabilities, edges, frames, alignment_path):
    """The V24 camera fusion, imported unchanged: teacher decode -> camera student -> fusion."""
    from types import SimpleNamespace
    from games import GAME, GAMES, half_video_blocks, source_frames
    from . import v24_camera_fusion as v24
    from .v14_resolution import decode_fine
    blocks = list(half_video_blocks(GAME))
    n = source_frames("tactical")
    match = SimpleNamespace(ball=np.empty((n, 2)), blocks=blocks)
    teacher = decode_fine(probabilities, edges, frames, match, "probability")
    offsets = v24.half_offsets(json.loads(Path(alignment_path).read_text())["runs"], blocks)
    features = v24.build_features(blocks, offsets, tuple(GAMES[GAME]["pitch"]))
    dead = (teacher[frames] == 2).astype(np.int8)
    student, prior = v24.cross_fitted_student(features, dead, frames, blocks)
    fused = v24.fuse(probabilities, student, prior)
    fused[v24.before_published_kickoff(frames, blocks)] = [5e-5, 5e-5, 1 - 1e-4]
    return fused, blocks, n


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from starter.baseline import write_submission
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, default=Path("data/work/final_cpu/probabilities.npz"))
    parser.add_argument("--alignment", type=Path, default=Path("data/work/v24/replay_bundle/alignment.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/v27_semantic.csv"))
    parser.add_argument("--mode", choices=MODES, default="robust")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--no-camera-fusion", action="store_true", help="probabilities are already fused")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    saved = np.load(args.probabilities)
    probabilities, edges, frames = saved["probabilities"], saved["edges"], saved["frames"]
    if args.no_camera_fusion:
        from games import GAME, half_video_blocks, source_frames
        fused, blocks, n = probabilities, list(half_video_blocks(GAME)), source_frames("tactical")
    else:
        fused, blocks, n = fused_probabilities(probabilities, edges, frames, args.alignment)
    after, before, report = semantic_adapter(fused, edges, frames, blocks, n, args.mode, args.threshold)
    labels = np.array(["H", "A", "D", "X"])[after]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(args.output), labels)
    report.update(csv=str(args.output), csv_sha256=digest(args.output), rows=int(len(labels)),
                  probabilities=str(args.probabilities), camera_fusion=not args.no_camera_fusion,
                  counts={k: int((labels == k).sum()) for k in "HADX"}, challenge_labels_read=False)
    target = args.report or c.ROOT / f"run_{args.mode}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print("V27 SEMANTIC ADAPTER", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
