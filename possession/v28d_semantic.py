"""V28d: controlled-possession networks + frozen networks + refit spell verifier.

Why. The frozen ensemble learned DFL native labels (a possession per touch, ~8.7 per live
minute). V28d retrains the same four networks, same recipe, on DFL labels converted to
controlled possession by an algorithmic tracking rule (possession.v28d_relabel). The retrained
networks answer "who has CONTROLLED possession"; the frozen ones still localise touches and
stoppages sharply. The two ensembles are averaged with equal weight (8 networks; weights fixed
a priori, not tuned), decoded with the unchanged decoder, passed through the unchanged V24
camera fusion, and finally the V27 spell verifier is applied -- refit on the blended outputs of
the three open Metrica games (same features, model and candidate rule as possession.v27_lab).

Pipeline for a new probabilities file pair:

    blended = 0.5 * frozen + 0.5 * retrained            (probabilities and boundary scores)
    fused   = possession.v27_semantic.fused_probabilities(blended, ...)      # V24, unchanged
    codes   = possession.v27_semantic.semantic_adapter(fused, ..., verifier=load_verifier())

Commands:

    .venv/bin/python -m possession.v28d_semantic fit         # verifier on the three Metrica games
    .venv/bin/python -m possession.v28d_semantic apply \
        --frozen data/work/final_cpu/probabilities.npz \
        --retrained data/work/v28d/challenge/probabilities.npz \
        --alignment data/work/v24/replay_bundle/alignment.json \
        --save-blend data/work/v28d/challenge/blended_probabilities.npz \
        --output evaluation/results/v28d_retrained.csv

No challenge ground truth is read here or anywhere in possession.v28d_*.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from . import v27_core as c

ROOT = Path("data/work/v28d")
WEIGHTS = (0.5, 0.5)          # frozen, retrained: fixed a priori
BLEND_TAG = "blend"
MODES = ("consistent", "robust")


def blend_arrays(sources, weights=WEIGHTS):
    """sources: [(probabilities, edges, frames)] on the same timeline -> blended triple."""
    frames = np.asarray(sources[0][2])
    if any(not np.array_equal(frames, s[2]) for s in sources):
        raise ValueError("The probability files are not on the same timeline")
    if len(weights) != len(sources) or not np.isclose(sum(weights), 1.0) or min(weights) < 0:
        raise ValueError("Invalid blend weights")
    probabilities = sum(w * np.asarray(s[0], dtype=np.float64) for w, s in zip(weights, sources))
    edges = sum(w * np.asarray(s[1], dtype=np.float64) for w, s in zip(weights, sources))
    return probabilities, edges, frames


def load_triple(path):
    saved = np.load(path)
    return saved["probabilities"], saved["edges"], saved["frames"]


def verifier_path(mode="consistent", tag=BLEND_TAG):
    return ROOT / f"verifier_{tag}_{mode}.joblib"


def fit_verifiers(tag=BLEND_TAG):
    """Final verifiers on the three Metrica games, on this ensemble's own (blended) outputs."""
    import joblib
    from . import v27_lab as lab
    from .v28d_eval import conditions, candidate_table
    table = candidate_table(conditions(tag))
    for mode in MODES:
        model = lab.fit(*lab.training(table, [1, 2, 3], mode))
        joblib.dump(dict(model=model, features=c.FEATURES, max_frames=lab.MAX_FRAMES, target=mode, parameters=lab.MODEL,
                         ensemble=tag, weights=list(WEIGHTS)), verifier_path(mode, tag))
        print("V28D VERIFIER FITTED", tag, mode, flush=True)


def load_verifier(mode="consistent", tag=BLEND_TAG):
    import joblib
    path = verifier_path(mode, tag)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m possession.v28d_semantic fit`")
    saved = joblib.load(path)
    if saved["features"] != c.FEATURES:
        raise ValueError("Verifier was fitted with another feature list")
    return saved


def controlled_adapter(frozen, retrained, alignment=None, mode="consistent", threshold=0.0, camera_fusion=True,
                       blocks=None, n_frames=None, verifier=None):
    """Re-apply entry point.

    frozen / retrained: (probabilities, edges, frames) triples or .npz paths written by
    possession.competitive_bundle for the frozen selection (docs/FINAL_SELECTION.json) and the
    retrained selection (data/work/v28d/selection.json) on the SAME match. With camera_fusion the
    challenge match's V24 fusion is applied (alignment = alignment.json of the bundle run);
    otherwise pass blocks and n_frames. Returns (codes after, codes before the verifier, report).
    """
    from . import v27_semantic as v27
    triples = [load_triple(s) if isinstance(s, (str, Path)) else s for s in (frozen, retrained)]
    probabilities, edges, frames = blend_arrays(triples)
    if camera_fusion:
        probabilities, blocks, n_frames = v27.fused_probabilities(probabilities, edges, frames, alignment)
    elif blocks is None or n_frames is None:
        raise ValueError("Without camera fusion, blocks and n_frames are required")
    after, before, report = v27.semantic_adapter(probabilities, edges, frames, blocks, n_frames, mode, threshold,
                                                 verifier=verifier or load_verifier(mode))
    report.update(weights=list(WEIGHTS), camera_fusion=bool(camera_fusion))
    return after, before, report, blocks


def codes_of(path):
    import pandas as pd
    return pd.read_csv(path)["label"].map(dict(H=0, A=1, D=2, X=3)).to_numpy().astype(np.int8)


def tagged_phase_disagreement(codes):
    """Label-free check on the challenge match: inside Metrica's automatic BUILD UP / PROGRESSION
    Smart Tagging phases (which name the attacking team), share of frames given to the OTHER team."""
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
    """Statistics of the new labels next to the earlier candidates (predictions only, no ground truth)."""
    graded = np.concatenate([np.arange(a, z + 1) for _, a, z in blocks])
    out = dict(v28d=dict(c.duration_stats(codes, blocks), other_team_share_in_tagged_phases=tagged_phase_disagreement(codes)))
    for name in names:
        path = Path("evaluation/results") / f"{name}.csv"
        if not path.exists():
            continue
        other = codes_of(path)
        out[name] = dict(c.duration_stats(other, blocks), other_team_share_in_tagged_phases=tagged_phase_disagreement(other),
                         frame_agreement_with_v28d=float((other[graded] == codes[graded]).mean()))
    return out


def apply(args):
    from starter.baseline import write_submission
    after, before, report, blocks = controlled_adapter(args.frozen, args.retrained, args.alignment, args.mode, args.threshold)
    if args.save_blend is not None:        # the blended (pre-fusion) probabilities, same layout as competitive_bundle's file
        p, e, f = blend_arrays([load_triple(args.frozen), load_triple(args.retrained)])
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_blend, probabilities=p.astype(np.float32), edges=e.astype(np.float32), frames=f)
    labels = np.array(["H", "A", "D", "X"])[after]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(args.output), labels)
    report.update(csv=str(args.output), csv_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), rows=int(len(labels)),
                  frozen=str(args.frozen), retrained=str(args.retrained), counts={k: int((labels == k).sum()) for k in "HADX"},
                  comparison=compare(after, blocks), challenge_labels_read=False)
    target = args.report or ROOT / f"run_{args.output.stem}.json"
    target.write_text(json.dumps(report, indent=2))
    print("V28D", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["fit", "apply"])
    parser.add_argument("--frozen", type=Path, default=Path("data/work/final_cpu/probabilities.npz"))
    parser.add_argument("--retrained", type=Path, default=ROOT / "challenge" / "probabilities.npz")
    parser.add_argument("--alignment", type=Path, default=Path("data/work/v24/replay_bundle/alignment.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/v28d_retrained.csv"))
    parser.add_argument("--mode", choices=MODES, default="consistent")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--save-blend", type=Path, help="also write the blended probabilities (.npz)")
    args = parser.parse_args()
    if args.step == "fit":
        fit_verifiers()
    else:
        apply(args)
