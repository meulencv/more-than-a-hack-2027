"""V24: fuse the frozen four-network ensemble with a label-free "camera operator" view.

Why. The four networks were trained on another provider's tracking and transfer worst on
dead ball: on the challenge tracking a stoppage often has no ball at all, or a ball being
carried to the restart that looks like play. The two recordings carry an independent cue
that the networks never see: what the camera operators did. The TV director cuts away from
the main camera (replay, close-up -> no broadcast homography) mostly while the ball is dead,
and both cameras pan and zoom differently during a stoppage.

How (no ground truth of the challenge game is read, public or private):

1. Teacher = frozen ensemble probabilities, decoded exactly as the frozen candidate.
2. The broadcast timeline is mapped onto the tactical one with the per-half frame offset
   found by the automatic constellation alignment (possession.camera_sync). The recording
   is continuous inside a half, so one offset per half covers every frame, including the
   cut-away stretches that the alignment itself cannot match.
3. Student = gradient boosting on camera-only features (broadcast cut state and its timing,
   pan speed, zoom, focus point of both cameras). It is trained on the teacher's own D /
   not-D decisions (algorithmic pseudo-labels) and predicts each ninth of a half from a
   model fitted on the other nine ninths+tenths (cross-fitting), so it cannot copy the
   teacher's decision on the stretch it is scoring.
   Graded frames before the published second-half kick-off are dead ball by the
   organisers' own documentation (FAQ.md) and are set to D.
4. The two views are combined as independent evidence on dead-vs-alive:
       logit P(D) = logit P_teacher(D) + logit P_student(D) - logit prior(D)
   H and A keep the teacher's ratio. The fused probabilities go through the unchanged
   25 Hz decoder.

Every constant below was fixed a priori; none was tuned on the challenge labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from games import GAME, GAMES, half_video_blocks, source_frames
from starter.load_homography import load as load_homography
from starter.baseline import write_submission
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
from .v14_resolution import decode_fine

FPS = 25
IMAGE = (1920.0, 1080.0)
CAP = 1500                      # 60 s: beyond this a run boundary is simply "far"
FOLDS_PER_HALF = 5
STUDENT = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=300,
               l2_regularization=1.0, early_stopping=False, random_state=0)
CLIP = (0.02, 0.98)             # student confidence is bounded: it never vetoes on its own
FEATURES = ["b_valid", "b_since_change", "b_until_change", "b_previous_run", "b_next_run",
            "b_reprojection", "b_focus_x", "b_focus_y", "b_log_width", "b_pan_2s",
            "b_zoom_rate_2s", "t_focus_x", "t_focus_y", "t_log_width", "t_pan_05s", "t_pan_2s",
            "t_pan_6s", "t_zoom_rate_2s", "focus_gap"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def camera_view(matrices, valid, pitch):
    """Pitch point at the image centre (m) and log of the pitch width seen at mid-height (m)."""
    n = len(matrices)
    focus = np.full((n, 2), np.nan)
    log_width = np.full(n, np.nan)
    usable = valid & np.isfinite(matrices).all(axis=(1, 2))
    usable[usable] = np.abs(np.linalg.det(matrices[usable])) > 1e-9
    index = np.flatnonzero(usable)
    inverse = np.linalg.inv(matrices[index])

    def unproject(px, py):
        v = inverse @ np.array([px, py, 1.0])
        return v[:, :2] / v[:, 2:3] * np.asarray(pitch), v[:, 2]

    centre, wc = unproject(IMAGE[0] / 2, IMAGE[1] / 2)
    left, wl = unproject(0.0, IMAGE[1] / 2)
    right, wr = unproject(IMAGE[0], IMAGE[1] / 2)
    focus[index] = np.clip(centre, -60.0, 170.0)
    same_side = (np.sign(wl) == np.sign(wc)) & (np.sign(wr) == np.sign(wc))
    width = np.where(same_side, np.linalg.norm(right - left, axis=1), np.nan)
    log_width[index] = np.log(np.clip(width, 2.0, 400.0))
    return focus, log_width


def centred_rate(x, window):
    """|x(t+h) - x(t-h)| per second over a centred window; NaN at edges and across gaps."""
    x = x.reshape(len(x), -1)
    h = max(1, window // 2)
    out = np.full(len(x), np.nan)
    out[h:-h] = np.linalg.norm(x[2 * h:] - x[:-2 * h], axis=1) / (2 * h / FPS)
    return out


def half_offsets(runs, blocks):
    """One broadcast->tactical frame offset per half, from the automatic alignment runs."""
    offsets = {}
    for half, first, last in blocks:
        values = []
        for run in runs:
            target = run["source_first"] - run["offset_frames"]
            if first <= target <= last:
                values += [run["offset_frames"]] * int(run["accepted_frames"])
        if not values:
            raise ValueError("No accepted camera alignment in half {}".format(half))
        offsets[half] = int(np.median(values))
    return offsets


def run_position(flag):
    """For a boolean sequence: frames since / until the current run's edges, neighbour lengths."""
    n = len(flag)
    change = np.flatnonzero(np.diff(flag.astype(np.int8)) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]
    lengths = ends - starts
    run = np.repeat(np.arange(len(starts)), lengths)
    position = np.arange(n)
    since = position - starts[run]
    until = ends[run] - 1 - position
    previous = np.r_[CAP, lengths[:-1]][run]
    following = np.r_[lengths[1:], CAP][run]
    return [np.minimum(v, CAP).astype(np.float64) for v in (since, until, previous, following)]


def build_features(blocks, offsets, pitch):
    """(n_tactical_frames, len(FEATURES)) camera-only features; NaN outside the two halves."""
    n = source_frames("tactical")
    t_matrix, _, t_valid = load_homography(source="tactical")
    b_matrix, b_error, b_valid = load_homography(source="broadcast")
    t_focus, t_width = camera_view(t_matrix, t_valid, pitch)
    b_focus, b_width = camera_view(b_matrix, b_valid, pitch)
    b_series = dict(b_reprojection=np.where(b_valid, b_error, np.nan), b_focus_x=b_focus[:, 0],
                    b_focus_y=b_focus[:, 1], b_log_width=b_width,
                    b_pan_2s=centred_rate(b_focus, 2 * FPS),
                    b_zoom_rate_2s=centred_rate(b_width, 2 * FPS))
    t_series = dict(t_focus_x=t_focus[:, 0], t_focus_y=t_focus[:, 1], t_log_width=t_width,
                    t_pan_05s=centred_rate(t_focus, FPS // 2), t_pan_2s=centred_rate(t_focus, 2 * FPS),
                    t_pan_6s=centred_rate(t_focus, 6 * FPS),
                    t_zoom_rate_2s=centred_rate(t_width, 2 * FPS))
    table = {name: np.full(n, np.nan) for name in FEATURES}
    used = -1                   # last broadcast frame already assigned to an earlier half
    for half, first, last in blocks:
        tactical = np.arange(first, last + 1)
        source = tactical + offsets[half]
        # A broadcast frame belongs to one half only: the TV recording skips most of the
        # interval, so the start of a later half has no footage and counts as "cut away".
        inside = (source > used) & (source < len(b_valid))
        used = int(source[-1])
        seen = np.zeros(len(tactical), dtype=bool)
        seen[inside] = b_valid[source[inside]]
        table["b_valid"][tactical] = seen
        for name, values in zip(FEATURES[1:5], run_position(seen)):
            table[name][tactical] = values
        for name, values in b_series.items():
            table[name][tactical[inside]] = values[source[inside]]
        for name, values in t_series.items():
            table[name][tactical] = values[tactical]
    table["focus_gap"] = np.hypot(table["b_focus_x"] - table["t_focus_x"],
                                  table["b_focus_y"] - table["t_focus_y"])
    return np.column_stack([table[name] for name in FEATURES])


def cross_fitted_student(features, dead, frames, blocks):
    """Out-of-fold P(teacher says D | camera view) on the 5 Hz grid."""
    fold = np.full(len(frames), -1)
    for k, (_, first, last) in enumerate(blocks):
        inside = (frames >= first) & (frames <= last)
        edges = np.linspace(first, last + 1, FOLDS_PER_HALF + 1)
        fold[inside] = k * FOLDS_PER_HALF + np.clip(
            np.searchsorted(edges, frames[inside], side="right") - 1, 0, FOLDS_PER_HALF - 1)
    if (fold < 0).any():
        raise ValueError("Emission frame outside the graded halves")
    x = features[frames]
    student = np.full(len(frames), np.nan)
    prior = np.full(len(frames), np.nan)
    for k in np.unique(fold):
        held = fold == k
        model = HistGradientBoostingClassifier(**STUDENT).fit(x[~held], dead[~held])
        student[held] = model.predict_proba(x[held])[:, 1]
        prior[held] = dead[~held].mean()
    return student, prior


def fuse(probabilities, student, prior):
    logit = lambda p: np.log(p) - np.log1p(-p)
    teacher_dead = np.clip(probabilities[:, 2].astype(np.float64), 1e-4, 1 - 1e-4)
    score = logit(teacher_dead) + logit(np.clip(student, *CLIP)) - logit(prior)
    dead = 1.0 / (1.0 + np.exp(-score))
    alive = np.clip(probabilities[:, :2].astype(np.float64), 1e-6, None)
    alive = alive / alive.sum(axis=1, keepdims=True) * (1.0 - dead)[:, None]
    return np.column_stack([alive, dead])


def before_published_kickoff(frames, blocks):
    """Graded frames that precede a half's published kick-off (DATA.md / FAQ.md): dead ball."""
    kickoffs = sorted(KICKOFF_VIDEO_FRAMES.values())
    mask = np.zeros(len(frames), dtype=bool)
    for (_, first, last), kickoff in zip(sorted(blocks, key=lambda b: b[1]), kickoffs):
        if not first <= kickoff <= last:
            raise ValueError("Published kick-off outside its half")
        mask |= (frames >= first) & (frames < kickoff)
    return mask


def spells(codes, blocks):
    count = 0
    for _, first, last in blocks:
        c = codes[first:last + 1]
        start = np.r_[True, c[1:] != c[:-1]]
        count += int((start & (c < 2)).sum())
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, default=Path("data/work/final_cpu/probabilities.npz"))
    parser.add_argument("--alignment", type=Path, default=Path("data/work/v11/challenge_alignment.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/v24_camera_fusion.csv"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/work/v24"))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    blocks = list(half_video_blocks(GAME))
    pitch = tuple(GAMES[GAME]["pitch"])
    match = SimpleNamespace(ball=np.empty((source_frames("tactical"), 2)), blocks=blocks)
    saved = np.load(args.probabilities)
    probabilities, edges, frames = saved["probabilities"], saved["edges"], saved["frames"]
    teacher = decode_fine(probabilities, edges, frames, match, "probability")
    offsets = half_offsets(json.loads(args.alignment.read_text())["runs"], blocks)
    features = build_features(blocks, offsets, pitch)
    dead = (teacher[frames] == 2).astype(np.int8)
    student, prior = cross_fitted_student(features, dead, frames, blocks)
    fused = fuse(probabilities, student, prior)
    waiting = before_published_kickoff(frames, blocks)
    fused[waiting] = [5e-5, 5e-5, 1 - 1e-4]
    codes = decode_fine(fused, edges, frames, match, "probability")
    labels = np.array(["H", "A", "D", "X"])[codes]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(args.output), labels)
    graded = np.concatenate([np.arange(a, z + 1) for _, a, z in blocks])
    ranks = np.argsort(np.argsort(student))
    auc = (ranks[dead == 1].mean() - (dead.sum() - 1) / 2) / max(1, (dead == 0).sum())
    report = dict(
        csv=str(args.output), csv_sha256=digest(args.output), rows=len(labels),
        offsets_by_half={str(k): v for k, v in offsets.items()}, student=STUDENT, clip=CLIP,
        folds=FOLDS_PER_HALF * len(blocks), features=FEATURES,
        student_auc_against_teacher=float(auc),
        teacher=dict(spells=spells(teacher, blocks), dead_share=float((teacher[graded] == 2).mean())),
        fused=dict(spells=spells(codes, blocks), dead_share=float((codes[graded] == 2).mean())),
        changed_frames=int((teacher[graded] != codes[graded]).sum()),
        frames_before_published_kickoff=int(waiting.sum()) * int(np.diff(frames[:2])[0]),
        counts={k: int((labels == k).sum()) for k in "HADX"},
        challenge_labels_read=False)
    np.savez_compressed(args.work_dir / "fusion.npz", student=student, prior=prior, fused=fused, frames=frames)
    (args.work_dir / "run.json").write_text(json.dumps(report, indent=2))
    print("V24 CAMERA FUSION", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
