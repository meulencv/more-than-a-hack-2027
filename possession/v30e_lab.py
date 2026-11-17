"""V30e lab: what do sound and image add to the stoppage decisions of the challenge pipeline?

Everything here is label-free on the challenge match: no file under evaluation/gt is read and
the official scorer is never run. "Teacher" means the pipeline's own decisions.

    .venv/bin/python -m possession.v30e_lab base          # the submission from the cached probabilities (must reproduce v29_final.csv)
    .venv/bin/python -m possession.v30e_lab students      # cross-fitted students per modality (+ controls)
    .venv/bin/python -m possession.v30e_lab information   # does a modality move the ball-blind teacher towards the ball-filled one?
    .venv/bin/python -m possession.v30e_lab audit         # ball / player physics of the frames the camera fusion changes

Students. As in V24, a gradient-boosting student predicts the teacher's dead / not-dead decision
from one modality and is scored out of fold (each fifth of a half is predicted by a model that
never saw it). AUC against the teacher says how much of the teacher's decision the modality
recovers; the control repeats it with the features moved in time by two minutes.

Information. The frozen networks were run twice: without the video ball (T0,
data/work/final_cpu) and with it (T1, the submission's base, ball seen on 15 points more of the
graded frames). T1 is not the truth, but it is the same model with strictly more evidence. A
modality that carries real stoppage information should, when fused into T0, move T0's decode
TOWARDS T1's, most of all on the frames where T0 had no ball. Features that contain the ball
itself are excluded from this test.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from games import GAME, half_video_blocks, source_frames
from . import v27_core as c
from . import v24_camera_fusion as v24
from . import v30e_features as F
from .v14_resolution import decode_fine

ROOT = Path("data/work/v30e")
PROBABILITIES = Path("data/work/v26/run_f2/probabilities.npz")
PROBABILITIES_BLIND = Path("data/work/final_cpu/probabilities.npz")
SETS = {
    "camera": ("camera",),
    "audio": ("audio",),
    "audio_no_whistle": ("audio",),
    "whistle_only": ("audio",),
    "video": ("video",),
    "audio+video": ("audio", "video"),
    "camera+audio": ("camera", "audio"),
    "camera+video": ("camera", "video"),
    "camera+audio+video": ("camera", "audio", "video"),
    "ball": ("ball",),
    "kinematics": ("kinematics",),
    "ball+kinematics": ("ball", "kinematics"),
    "everything": ("camera", "audio", "video", "ball", "kinematics"),
}
INDEPENDENT = ("camera", "audio", "video", "audio+video", "camera+audio", "camera+video", "camera+audio+video")
SHIFT_ROWS = 600                # control: features moved by 600 rows of the 5 Hz grid = 2 minutes


# -- data ------------------------------------------------------------------------------------

def blocks():
    return list(half_video_blocks(GAME))


def load_triple(path=PROBABILITIES):
    with np.load(path) as saved:
        return saved["probabilities"], saved["edges"], saved["frames"]


def teacher_codes(triple, n):
    match = SimpleNamespace(ball=np.empty((n, 2)), blocks=blocks())
    return decode_fine(triple[0], triple[1], triple[2], match, "probability")


def feature_table():
    groups = {name: F.load_group(name) for name in F.GROUPS}
    return groups


def matrix(groups, key, frames):
    names, columns = [], []
    for group in SETS[key]:
        for name, values in groups[group].items():
            if key == "audio_no_whistle" and "whistle" in name:
                continue
            if key == "whistle_only" and "whistle" not in name:
                continue
            names.append(name)
            columns.append(values[frames])
    return np.column_stack(columns).astype(np.float32), names


def fold_index(frames, per_half=v24.FOLDS_PER_HALF):
    fold = np.full(len(frames), -1)
    for k, (_, first, last) in enumerate(blocks()):
        inside = (frames >= first) & (frames <= last)
        edges = np.linspace(first, last + 1, per_half + 1)
        fold[inside] = k * per_half + np.clip(np.searchsorted(edges, frames[inside], side="right") - 1, 0, per_half - 1)
    return fold


def cross_fit(x, y, fold, weight=None):
    """Out-of-fold student P(teacher says D) and the training prior of each fold."""
    student = np.full(len(y), np.nan)
    prior = np.full(len(y), np.nan)
    for k in np.unique(fold):
        held = fold == k
        w = None if weight is None else weight[~held]
        model = HistGradientBoostingClassifier(**v24.STUDENT).fit(x[~held], y[~held], sample_weight=w)
        student[held] = model.predict_proba(x[held])[:, 1]
        prior[held] = y[~held].mean()
    return student, prior


def auc(y, score):
    y = np.asarray(y).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    ranks = np.argsort(np.argsort(score, kind="stable"), kind="stable")
    return float((ranks[y].mean() - (y.sum() - 1) / 2) / (~y).sum())


def shifted(x, frames, rows=SHIFT_ROWS):
    """Features rolled in time inside each half (the control: same marginals, wrong moment)."""
    out = x.copy()
    for _, first, last in blocks():
        inside = np.flatnonzero((frames >= first) & (frames <= last))
        out[inside] = np.roll(x[inside], rows, axis=0)
    return out


def interior(codes, frames, margin=50):
    """5 Hz rows farther than `margin` frames from any change of the decoded dead/alive state."""
    dead = codes == 2
    change = np.flatnonzero(dead[1:] != dead[:-1]) + 1
    if not len(change):
        return np.ones(len(frames), dtype=bool)
    j = np.clip(np.searchsorted(change, frames), 1, len(change) - 1)
    return np.minimum(np.abs(change[j] - frames), np.abs(change[j - 1] - frames)) > margin


# -- students --------------------------------------------------------------------------------

def students(args):
    n = source_frames("tactical")
    triple = load_triple()
    frames = triple[2]
    codes = teacher_codes(triple, n)
    dead = (codes[frames] == 2).astype(np.int8)
    confident = interior(codes, frames) & ((triple[0][:, 2] > 0.9) | (triple[0][:, 2] < 0.1))
    near = ~interior(codes, frames, 75)
    groups = feature_table()
    fold = fold_index(frames)
    half = np.array([1 if f <= blocks()[0][2] else 2 for f in frames])
    report = dict(rows=len(frames), dead_share=float(dead.mean()), confident_share=float(confident.mean()), sets={})
    saved = {}
    for key in SETS:
        x, names = matrix(groups, key, frames)
        s, prior = cross_fit(x, dead, fold)
        control, _ = cross_fit(shifted(x, frames), dead, fold)
        row = dict(features=len(names), auc=auc(dead, s), auc_confident_frames=auc(dead[confident], s[confident]),
                   auc_within_3s_of_a_boundary=auc(dead[near], s[near]), auc_half_1=auc(dead[half == 1], s[half == 1]),
                   auc_half_2=auc(dead[half == 2], s[half == 2]), auc_control_shifted_2min=auc(dead, control),
                   log_loss=float(-np.mean(np.where(dead == 1, np.log(np.clip(s, 1e-6, 1)), np.log(np.clip(1 - s, 1e-6, 1))))),
                   log_loss_prior=float(-np.mean(np.where(dead == 1, np.log(prior), np.log(1 - prior)))))
        if key in ("camera", "camera+audio+video", "audio", "video"):
            for per_half in (3, 8):
                other, _ = cross_fit(x, dead, fold_index(frames, per_half))
                row[f"auc_{per_half}_folds_per_half"] = auc(dead, other)
                row[f"correlation_with_5_folds_{per_half}"] = float(np.corrcoef(s, other)[0, 1])
            trained_confident, _ = cross_fit(x, dead, fold, confident.astype(np.float64) + 1e-3)
            row["auc_trained_on_confident_frames_only"] = auc(dead, trained_confident)
        report["sets"][key] = row
        saved["student_" + key.replace("+", "_")] = s.astype(np.float32)
        print("STUDENT", f"{key:<22}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    # single features: which ones carry the signal (one-feature students would be slow; use rank AUC)
    single = {}
    for group in ("audio", "video"):
        for name, values in groups[group].items():
            v = values[frames]
            ok = np.isfinite(v)
            a = auc(dead[ok], v[ok])
            single[name] = dict(auc_raw=round(max(a, 1 - a), 4), direction="higher when dead" if a >= 0.5 else "lower when dead",
                                finite=round(float(ok.mean()), 3))
    report["single_feature_rank_auc"] = dict(sorted(single.items(), key=lambda item: -item[1]["auc_raw"]))
    for name, row in list(report["single_feature_rank_auc"].items())[:16]:
        print("SINGLE", f"{name:<28}", row, flush=True)
    np.savez_compressed(ROOT / "students.npz", frames=frames, dead=dead, fold=fold, **saved)
    (ROOT / "students.json").write_text(json.dumps(report, indent=1))


# -- information test ------------------------------------------------------------------------

def fuse_and_decode(triple, student, prior, n):
    fused = v24.fuse(triple[0], student, prior)
    fused[v24.before_published_kickoff(triple[2], blocks())] = [5e-5, 5e-5, 1 - 1e-4]
    return teacher_codes((fused, triple[1], triple[2]), n), fused


def agreement(codes, target, mask):
    return float(((codes[mask] == 2) == (target[mask] == 2)).mean())


def pseudo_spell_f1(codes, target):
    b = blocks()
    return c.spell_f1(c.spells_of(c.segments(target, b)), c.spells_of(c.segments(codes, b)))


def information(args):
    n = source_frames("tactical")
    blind, filled = load_triple(PROBABILITIES_BLIND), load_triple()
    if not np.array_equal(blind[2], filled[2]):
        raise ValueError("The two runs are on different grids")
    frames = filled[2]
    groups = feature_table()
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks():
        graded[first:last + 1] = True
    kickoff_rule = np.zeros(n, dtype=bool)
    waiting = v24.before_published_kickoff(np.arange(n), blocks())
    kickoff_rule[waiting] = True
    scored = graded & ~kickoff_rule                           # the published-kick-off rule is not evidence of anything
    video_ball = groups["ball"]["b_source_video"] > 0
    no_ball_before = scored & (groups["ball"]["b_seen_tracking"] == 0)
    filled_by_video = scored & video_ball
    target = teacher_codes(filled, n)
    target[kickoff_rule & graded] = 2
    base = teacher_codes(blind, n)
    base[kickoff_rule & graded] = 2
    differs = scored & ((base == 2) != (target == 2))
    report = dict(frames_scored=int(scored.sum()), frames_filled_by_video=int(filled_by_video.sum()),
                  frames_without_tracking_ball=int(no_ball_before.sum()), frames_where_blind_and_filled_teacher_differ=int(differs.sum()),
                  blind_teacher=dict(agreement_all=agreement(base, target, scored), agreement_filled_by_video=agreement(base, target, filled_by_video),
                                     agreement_no_tracking_ball=agreement(base, target, no_ball_before),
                                     pseudo_spell_f1=pseudo_spell_f1(base, target)), sets={})
    print("INFORMATION blind teacher", report["blind_teacher"], "differ", report["frames_where_blind_and_filled_teacher_differ"], flush=True)
    dead = (teacher_codes(blind, n)[frames] == 2).astype(np.int8)
    fold = fold_index(frames)
    for key in (args.sets or INDEPENDENT + ("audio_no_whistle", "whistle_only")):
        x, _ = matrix(groups, key, frames)
        row = {}
        for tag, values in (("fused", x), ("control_shifted_2min", shifted(x, frames))):
            student, prior = cross_fit(values, dead, fold)
            codes, _ = fuse_and_decode(blind, student, prior, n)
            # threshold-free: on the 5 Hz rows where the two teachers disagree, does the student
            # (trained on the BLIND teacher, out of fold) rank the rows the filled teacher calls dead higher?
            rows = differs[frames]
            row_auc = auc(target[frames][rows] == 2, student[rows])
            fixed = int((differs & ((codes == 2) == (target == 2))).sum())
            broken = int((scored & ~differs & ((codes == 2) != (target == 2))).sum())
            row[tag] = dict(agreement_all=agreement(codes, target, scored), agreement_filled_by_video=agreement(codes, target, filled_by_video),
                            agreement_no_tracking_ball=agreement(codes, target, no_ball_before),
                            disagreement_frames_fixed=fixed, agreement_frames_broken=broken, net_frames=fixed - broken,
                            frames_changed_by_fusion=int((scored & ((codes == 2) != (base == 2))).sum()),
                            student_auc_for_filled_teacher_on_disagreement_rows=row_auc, disagreement_rows=int(rows.sum()),
                            pseudo_spell_f1=pseudo_spell_f1(codes, target))
        report["sets"][key] = row
        print("INFORMATION", f"{key:<20}", {t: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for t, r in row.items()}, flush=True)
    (ROOT / "information.json").write_text(json.dumps(report, indent=1))


def base(args=None):
    """The submission's pipeline on the cached probabilities -> data/work/v30e/base.npz.

    teacher = decode of the frozen ensemble alone; before = V24 fusion + decoder; after = + the
    w = 0.5 spell verifier, which must equal evaluation/results/v29_final.csv.
    """
    import joblib
    import pandas as pd
    from . import v29_final as final
    n = source_frames("tactical")
    triple = load_triple()
    fused, b, _ = final.fused(triple)
    after, before, _ = final.verify(fused, triple[1], triple[2], b, n, joblib.load("artifacts/verifier/frozen_v27_w0.50.joblib"))
    reference = pd.read_csv("evaluation/results/v29_final.csv")["label"].map(dict(H=0, A=1, D=2, X=3)).to_numpy()
    same = bool(np.array_equal(reference, after))
    print("BASE reproduces evaluation/results/v29_final.csv:", same, flush=True)
    if not same:
        raise RuntimeError("The cached probabilities do not reproduce the submission")
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "base.npz", fused=fused, probabilities=triple[0], edges=triple[1], frames=triple[2],
                        before=before, after=after, teacher=teacher_codes(triple, n))


def audit(args=None):
    """What do the frames that the camera fusion (and the verifier) change look like physically?

    Mixture reading: with r_dead / r_live the rate of 'ball faster than 5 m/s' in frames both
    decodes call dead / live, a group with rate r is (r - r_dead) / (r_live - r_dead) live.
    CAVEAT: the frozen ensemble sees the same ball, so its own dead calls are conditioned on a
    slow or absent ball; the reading is suggestive for the vetoed group, not a proof.
    """
    if not (ROOT / "base.npz").exists():
        base()
    saved = np.load(ROOT / "base.npz")
    teacher, fused, final_codes = saved["teacher"], saved["before"], saved["after"]
    ball, kin = F.load_group("ball"), F.load_group("kinematics")
    graded = np.zeros(len(teacher), dtype=bool)
    for _, first, last in blocks():
        graded[first:last + 1] = True
    graded[v24.before_published_kickoff(np.arange(len(teacher)), blocks())] = False
    seen = ball["b_seen"] > 0
    speed = ball["b_speed"]
    groups = {"both dead": (teacher == 2) & (fused == 2), "both live": (teacher < 2) & (fused < 2),
              "dead only after the camera fusion": (teacher < 2) & (fused == 2),
              "dead only before the camera fusion (vetoed)": (teacher == 2) & (fused < 2),
              "made dead by the verifier": (fused < 2) & (final_codes == 2)}
    rows = {}
    for name, mask in groups.items():
        m = mask & graded
        s = m & seen & np.isfinite(speed)
        rows[name] = dict(frames=int(m.sum()), ball_seen=float(seen[m].mean()), ball_fast_given_seen=float((speed[s] > 5).mean()),
                          ball_at_rest_given_seen=float((speed[s] < 1.5).mean()), ball_beyond_lines_given_seen=float((ball["b_outside_m"][s] > 0.3).mean()),
                          players_mean_speed=float(np.nanmean(kin["k_speed_mean"][m])), share_standing=float(np.nanmean(kin["k_share_standing"][m])))
    dead, live = rows["both dead"]["ball_fast_given_seen"], rows["both live"]["ball_fast_given_seen"]
    for name, row in rows.items():
        row["live_share_by_ball_speed_mixture"] = float(np.clip((row["ball_fast_given_seen"] - dead) / (live - dead), 0, 1))
        print("AUDIT", f"{name:<44}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    (ROOT / "fusion_physics_audit.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    steps = dict(base=base, students=students, information=information, audit=audit)
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=list(steps))
    parser.add_argument("--sets", nargs="+", choices=list(SETS), default=None, help="information: only these feature sets")
    args = parser.parse_args()
    steps[args.step](args)
