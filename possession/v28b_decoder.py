"""V28b: object-level decoder on top of the frozen ensemble / V24 fusion.

What. The frame-level Viterbi (possession.v14_resolution.decode_fine) is replaced by decoding on
a segment lattice (possession.v28b_core):

1. ten explicit-duration (semi-Markov) decoders, each with calibrated emissions (team
   temperature, per-frame evidence floors, stoppage bias), its own switch costs and a duration
   prior per state and context. The duration priors come from the consistent references of the
   open Metrica sample games, and the ten parameter sets are the best of a fixed-seed random
   search scored with the official spellF1 on those games;
2. minimum-Bayes-risk combination: the ten decodes are the hypotheses and the output is the
   segmentation with the highest expected spellF1 against them (expected IoU-matched possessions
   minus the Dinkelbach price per predicted possession);
3. optionally the V27 spell verifier, re-fitted on the candidate removals of that consensus.

family="duration" re-decides stoppages and teams; family="team" keeps every stoppage of the
baseline decode and only re-decides the team (more conservative under a shift of P(D)).

No challenge ground truth is read here or anywhere in the fitting (possession.v28b_lab).

    .venv/bin/python -m possession.v28b_decoder \
        --probabilities data/work/final_cpu/probabilities.npz \
        --alignment data/work/v24/replay_bundle/alignment.json \
        --output evaluation/results/v28b_decoder.csv
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from . import v27_core as c
from . import v28b_core as b

ROOT = Path("data/work/v28b")
FAMILIES = ("duration", "team")


def load_model(family="duration", root=ROOT):
    import joblib
    if family not in FAMILIES:
        raise ValueError("Unknown decoder family")
    path = Path(root) / f"decoder_{family}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m possession.v28b_lab search --family {family}` and `... fit`")
    saved = joblib.load(path)
    if saved["features"] != c.FEATURES:
        raise ValueError("Verifier was fitted with another feature list")
    return saved


def object_decoder(probabilities, edges, frames, blocks, n_frames, family="duration", verify=True, model=None):
    """Probabilities -> (codes after V28b, codes of the baseline decode, report).

    probabilities (n, 3) H/A/D, edges (n,), frames (n,) as written by competitive_bundle /
    v24_camera_fusion; blocks = [(half, first_frame, last_frame)] inclusive; codes use 0/1/2 and
    3 outside the halves. The baseline is exactly v14_resolution.decode_fine(..., 'probability').
    """
    saved = model or load_model(family)
    interpolated = c.interpolate(np.asarray(probabilities), np.asarray(edges), np.asarray(frames), blocks)
    lattice = b.build_lattice(interpolated, n_frames, saved["lattice_configs"])
    before = b.decode(lattice, n_frames)
    decodes = [b.decode(lattice, n_frames, par, saved["prior"]) for par in saved["configs"]]
    consensus, infos = b.combine(lattice, n_frames, decodes)
    after, applied = consensus, []
    if verify:
        p, edge = c.dense(interpolated, n_frames)

        def score(current, cands):
            return saved["verifier"].predict(c.candidate_features(current, cands, p, edge))

        segs, applied = c.apply_scored(c.segments(consensus, blocks), score, saved["threshold"], saved["max_frames"])
        after = c.to_codes(segs, n_frames)
    graded = np.concatenate([np.arange(a, z + 1) for _, a, z in blocks])
    kinds = {}
    for action, _ in applied:
        kinds[action] = kinds.get(action, 0) + 1
    report = dict(family=saved["family"], decoders=len(decodes), verifier=bool(verify), threshold=saved["threshold"],
                  lattice_atoms=[int(len(h["bounds"]) - 1) for h in lattice],
                  decoder_spells=[c.duration_stats(d, blocks)["spells"] for d in decodes],
                  expected_spell_f1_among_decoders=[round(i["expected_f1"], 4) for i in infos],
                  verifier_removals=len(applied), verifier_removals_by_kind=kinds,
                  before=c.duration_stats(before, blocks), consensus=c.duration_stats(consensus, blocks),
                  after=c.duration_stats(after, blocks), graded_frames=int(len(graded)),
                  graded_frames_changed=int((before[graded] != after[graded]).sum()),
                  dead_frames_added=int(((after[graded] == 2) & (before[graded] != 2)).sum()),
                  dead_frames_removed=int(((after[graded] != 2) & (before[graded] == 2)).sum()))
    return after, before, report


def decode_file(probabilities_path="data/work/final_cpu/probabilities.npz",
                alignment_path="data/work/v24/replay_bundle/alignment.json", family="duration", verify=True,
                camera_fusion=True, model=None):
    """Re-apply V28b to another probabilities file of the challenge match.

    The file holds `probabilities`, `edges`, `frames` (competitive_bundle format). With
    camera_fusion=True it goes through the unchanged V24 fusion exactly as V27 does
    (possession.v27_semantic.fused_probabilities); pass False if it is already fused.
    Returns (codes, baseline codes, report, blocks).
    """
    saved = np.load(probabilities_path)
    probabilities, edges, frames = saved["probabilities"], saved["edges"], saved["frames"]
    if camera_fusion:
        from .v27_semantic import fused_probabilities
        probabilities, blocks, n = fused_probabilities(probabilities, edges, frames, alignment_path)
    else:
        from games import GAME, half_video_blocks, source_frames
        blocks, n = list(half_video_blocks(GAME)), source_frames("tactical")
    after, before, report = object_decoder(probabilities, edges, frames, blocks, n, family, verify, model)
    report.update(probabilities=str(probabilities_path), camera_fusion=bool(camera_fusion), challenge_labels_read=False)
    return after, before, report, blocks


def codes_of(path):
    import pandas as pd
    return pd.read_csv(path)["label"].map(dict(H=0, A=1, D=2, X=3)).to_numpy().astype(np.int8)


def tagged_phase_disagreement(codes):
    """Label-free check: inside Metrica's automatic BUILD UP / PROGRESSION phases (which name the
    attacking team), the share of graded frames labelled as the OTHER team."""
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
    out = dict(v28b=dict(c.duration_stats(codes, blocks), other_team_share_in_tagged_phases=tagged_phase_disagreement(codes)))
    for name in names:
        path = Path("evaluation/results") / f"{name}.csv"
        if not path.exists():
            continue
        other = codes_of(path)
        differ = other[graded] != codes[graded]
        out[name] = dict(c.duration_stats(other, blocks), other_team_share_in_tagged_phases=tagged_phase_disagreement(other),
                         graded_frames_different_from_v28b=int(differ.sum()), share_different_from_v28b=float(differ.mean()),
                         live_dead_frames_different=int(((other[graded] == 2) != (codes[graded] == 2)).sum()))
    return out


def validate(path, n_frames):
    import evaluation.possession_metrics as pm
    problem = pm.validate_submission(pm.read_label_csv(path), n_frames)
    if problem:
        raise ValueError(f"{path} is not a valid submission: {problem}")


def main():
    from starter.baseline import write_submission
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, default=Path("data/work/final_cpu/probabilities.npz"))
    parser.add_argument("--alignment", type=Path, default=Path("data/work/v24/replay_bundle/alignment.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/v28b_decoder.csv"))
    parser.add_argument("--family", choices=FAMILIES, default="duration")
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument("--no-camera-fusion", action="store_true", help="probabilities are already fused")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    after, _, report, blocks = decode_file(args.probabilities, args.alignment, args.family, not args.no_verifier,
                                           not args.no_camera_fusion)
    labels = np.array(["H", "A", "D", "X"])[after]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(args.output), labels)
    validate(args.output, len(labels))
    report.update(csv=str(args.output), csv_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  rows=int(len(labels)), counts={k: int((labels == k).sum()) for k in "HADX"},
                  comparison=compare(after, blocks))
    target = args.report or ROOT / f"run_{args.output.stem}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print("V28B OBJECT DECODER", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
