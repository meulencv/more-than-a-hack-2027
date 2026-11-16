"""V28a: physically informed spell verifier on top of the frozen ensemble / V24 fusion.

Same pipeline and same action as the V27 verifier (possession.v27_semantic): after the unchanged
decoder, every predicted possession of at most 8 s next to another state may be removed
(T-X-T -> the surrounding team; next to a stoppage -> D or the neighbouring team). What changes
is what the small model sees and what it is fitted to:

* features: the 23 network-output features of V27 + a physical description of the spell
  computed from the tracking (ball kinematics, how long and how close the spell's team is to
  the ball versus the opponent, straight fly-through versus control, what happens just before
  and after, where on the pitch). Structure features (second neighbours, live time since /
  until a stoppage) and posterior-shape features were also built and tested; they did not
  help held-out and the adopted model does not use them;
* model (possession.v28a_lab, leave-one-game-out on Metrica's open games): two small
  gradient-boosted regressors, one of the exact spellF1 change of the removal and one of the
  "purity gap" (does the reference give the spell's frames to the other code), added after
  dividing each by the standard deviation of its target; a removal is applied when the sum is
  positive (0 is the natural boundary, not tuned).
* adopted variant `physical_live_blend`: V27 features + physical features, the physical block
  being used only for a possession between two possessions (T-X-T). Next to a stoppage the open
  Metrica games have no ball track while the challenge tracker often keeps one, so there the
  physics would be read out of distribution. `--verifier verifier_full.joblib` is the variant
  that uses the physics everywhere (best held-out mean, not adopted for that reason).

No challenge ground truth is read here or in the fitting. The tracking of the challenge match
is an allowed automatic asset.

    .venv/bin/python -m possession.v28a_verifier \
        --probabilities data/work/final_cpu/probabilities.npz \
        --alignment data/work/v24/replay_bundle/alignment.json \
        --output evaluation/results/v28a_verifier.csv

Importable: `apply_to_probabilities(path)` (whole pipeline from a probabilities file) and
`physical_adapter(probabilities, edges, frames, blocks, n_frames, signals)` (any match).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import v27_core as c
from . import v28a_core as k28

LABELS = np.array(["H", "A", "D", "X"])


def load_verifier(root=k28.ROOT, name="verifier.joblib"):
    """name: 'verifier.joblib' (adopted: physics for T-X-T only) or 'verifier_full.joblib'."""
    import joblib
    path = Path(root) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the steps of possession.v28a_lab (signals, table, logo, fit)")
    saved = joblib.load(path)
    if saved["features"] != k28.FEATURES:
        raise ValueError("Verifier was fitted with another feature list")
    return saved


def physical_adapter(probabilities, edges, frames, blocks, n_frames, signals, threshold=0.0, verifier=None, passes=3):
    """Probabilities (frozen ensemble or fused) + tracking signals -> (codes after, codes before, report).

    probabilities (n, 3) H/A/D, edges (n,), frames (n,) as written by competitive_bundle /
    v24_camera_fusion; blocks = [(half, first_frame, last_frame)] inclusive; signals =
    possession.v28a_core.signals(ball, players, team, pitch) of the same match, indexed by the
    same frames. `before` is exactly v14_resolution.decode_fine(..., 'probability').
    """
    saved = verifier or load_verifier()
    model, max_frames = saved["model"], saved["max_frames"]
    interpolated = c.interpolate(np.asarray(probabilities), np.asarray(edges), np.asarray(frames), blocks)
    before = c.decode(interpolated, n_frames)[0]
    p, edge = c.dense(interpolated, n_frames)
    segs = c.segments(before, blocks)

    def score(current, cands):
        return model.predict(k28.candidate_features(current, cands, p, edge, signals))

    new, applied = c.apply_scored(segs, score, threshold, max_frames, passes)
    after = c.to_codes(new, n_frames)
    graded = np.concatenate([np.arange(a, z + 1) for _, a, z in blocks])
    kinds = {}
    for action, _ in applied:
        kinds[action] = kinds.get(action, 0) + 1
    report = dict(variant=saved["variant"], threshold=threshold, removals=len(applied), removals_by_kind=kinds,
                  removed_seconds_median=float(np.median([d for _, d in applied])) if applied else 0.0,
                  before=c.duration_stats(before, blocks), after=c.duration_stats(after, blocks),
                  graded_frames=int(len(graded)), graded_frames_changed=int((before[graded] != after[graded]).sum()),
                  dead_frames_added=int(((after[graded] == 2) & (before[graded] != 2)).sum()),
                  ball_tracked_share_graded=float(signals["seen"][graded].mean()))
    return after, before, report


def challenge_signals(cache=k28.ROOT / "signals_challenge.npz"):
    """Per-frame tracking signals of the challenge match (cached; label-free)."""
    cache = Path(cache)
    if cache.exists():
        return dict(np.load(cache))
    s = k28.challenge_signals()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **s)
    return s


def apply_to_probabilities(probabilities_path="data/work/final_cpu/probabilities.npz",
                           alignment_path="data/work/v24/replay_bundle/alignment.json",
                           camera_fusion=True, threshold=0.0, verifier=None):
    """Challenge match: probabilities file -> (codes after, codes before, report, blocks).

    The file holds `probabilities`, `edges`, `frames` (competitive_bundle format). With
    camera_fusion=True it goes through the unchanged V24 fusion exactly as V27 does
    (possession.v27_semantic.fused_probabilities); pass False if it is already fused.
    """
    from .v27_semantic import fused_probabilities
    saved = np.load(probabilities_path)
    probabilities, edges, frames = saved["probabilities"], saved["edges"], saved["frames"]
    if camera_fusion:
        fused, blocks, n = fused_probabilities(probabilities, edges, frames, alignment_path)
    else:
        from games import GAME, half_video_blocks, source_frames
        fused, blocks, n = probabilities, list(half_video_blocks(GAME)), source_frames("tactical")
    signals = challenge_signals()
    if len(signals["seen"]) != n:
        raise ValueError("Tracking and submission lengths differ")
    after, before, report = physical_adapter(fused, edges, frames, blocks, n, signals, threshold, verifier)
    report.update(probabilities=str(probabilities_path), camera_fusion=bool(camera_fusion), challenge_labels_read=False)
    return after, before, report, blocks


def challenge_candidates(probabilities_path="data/work/final_cpu/probabilities.npz",
                         alignment_path="data/work/v24/replay_bundle/alignment.json"):
    """Feature rows of the candidate removals on the challenge match's baseline decode."""
    from .v27_semantic import fused_probabilities
    saved = np.load(probabilities_path)
    fused, blocks, n = fused_probabilities(saved["probabilities"], saved["edges"], saved["frames"], alignment_path)
    interpolated = c.interpolate(fused, saved["edges"], saved["frames"], blocks)
    p, edge = c.dense(interpolated, n)
    segs = c.segments(c.decode(interpolated, n)[0], blocks)
    return k28.candidate_features(segs, c.candidates(segs, load_verifier()["max_frames"]), p, edge, challenge_signals())


def codes_of(path):
    import pandas as pd
    return pd.read_csv(path)["label"].map(dict(H=0, A=1, D=2, X=3)).to_numpy().astype(np.int8)


def tagged_phase_disagreement(codes):
    """Label-free check: inside Metrica's automatic BUILD UP / PROGRESSION phases (which name the
    attacking team), the share of frames labelled as the OTHER team."""
    from starter.load_smart_tagging import load
    teams = {"Barcelona": 0, "Atletic Club": 1}
    out = {}
    for family in ("BUILD UP", "PROGRESSION"):
        other = total = 0
        for e in load():
            team = teams.get(e["tags"].get("Team"), -1)
            if (e["code"] or "").strip().upper() != family or team < 0:
                continue
            window = codes[int(e["start_frame"]):int(e["end_frame"])]
            window = window[window < 3]
            other += int((window == 1 - team).sum())
            total += int(len(window))
        out[family] = round(other / total, 4) if total else None
    return out


def compare(codes, blocks, names=("v24_camera_fusion", "v27_semantic", "v27_semantic_bet")):
    """Statistics of the new labels next to the earlier candidates (predictions only)."""
    graded = np.concatenate([np.arange(a, z + 1) for _, a, z in blocks])
    out = dict(v28a_verifier=dict(c.duration_stats(codes, blocks), other_team_share_in_tagged_phases=tagged_phase_disagreement(codes)))
    for name in names:
        path = Path("evaluation/results") / f"{name}.csv"
        if not path.exists():
            continue
        other = codes_of(path)
        out[name] = dict(c.duration_stats(other, blocks), other_team_share_in_tagged_phases=tagged_phase_disagreement(other),
                         graded_frames_different_from_v28a=int((other[graded] != codes[graded]).sum()),
                         share_different_from_v28a=float((other[graded] != codes[graded]).mean()))
    return out


def main():
    from starter.baseline import write_submission
    from .v27_semantic import digest
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, default=Path("data/work/final_cpu/probabilities.npz"))
    parser.add_argument("--alignment", type=Path, default=Path("data/work/v24/replay_bundle/alignment.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/v28a_verifier.csv"))
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--no-camera-fusion", action="store_true", help="probabilities are already fused")
    parser.add_argument("--verifier", default="verifier.joblib", help="verifier.joblib (adopted) or verifier_full.joblib")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    args.report = args.report or k28.ROOT / f"run_{args.output.stem}.json"
    after, before, report, blocks = apply_to_probabilities(args.probabilities, args.alignment, not args.no_camera_fusion,
                                                           args.threshold, load_verifier(name=args.verifier))
    labels = LABELS[after]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(args.output), labels)
    report.update(csv=str(args.output), csv_sha256=digest(args.output), rows=int(len(labels)),
                  counts={k: int((labels == k).sum()) for k in "HADX"}, comparison=compare(after, blocks))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print("V28A VERIFIER", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
