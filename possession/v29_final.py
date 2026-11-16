"""V29: one pipeline for the challenge match, from the tracking + the video-filled ball to the CSVs.

    tracking + data/work/v26/video_ball_f2.npz (+ broadcast ball through the V11 alignment)
      -> frozen four-network ensemble          (must equal data/work/v26/run_f2/probabilities.npz)
      -> retrained four-network ensemble (V28d checkpoints) ON THE SAME FILLED BALL
      -> base "blend"  = 0.5 frozen + 0.5 retrained         base "frozen" = frozen only
      -> unchanged V24 camera fusion (possession.v27_semantic.fused_probabilities)
      -> unchanged 25 Hz decoder
      -> spell verifier fitted on THAT base's outputs on the three open Metrica games
         (possession.v29_lab; target = w * gain_consistent + (1 - w) * gain_alternative)

Candidates (CSV = <output-dir>/v29_<name>.csv):
    hedge         blend base,  V27 feature set, w = 0.7     (study-time prior, informed by the public window's published summary; NOT submitted -- the submission is the frozen base at w = 0.5, see REPRODUCE.md)
    bet           blend base,  V27 feature set, w = 1.0
    safe          frozen base, V27 feature set, robust (min) target; no retrained networks
    hedge_frozen  frozen base, V27 feature set, w = 0.7     (extra: the hedge on the base that
                  is not itself committed to the consistent reading, see v29_lab hedge)

    .venv/bin/python -m possession.v29_final run                 # networks on CPU (default), ~4 min
    .venv/bin/python -m possession.v29_final run --device mps    # same bytes are NOT guaranteed
    .venv/bin/python -m possession.v29_final shift               # covariate check of the verifier inputs

An existing CSV is never replaced: if the file exists the new bytes must be identical.
No ground truth of the challenge match is read anywhere (public or private).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from . import v27_core as c
from . import v28a_core as k28
from . import v29_lab as lab

ROOT = lab.ROOT
FROZEN_SELECTION = Path("docs/FINAL_SELECTION.json")
RETRAINED_SELECTION = Path("data/work/v28d/selection.json")
VIDEO_BALL = Path("data/work/v26/video_ball_f2.npz")
ALIGNMENT = Path("data/work/v11/challenge_alignment.npz")          # .json next to it feeds the V24 fusion
FROZEN_REFERENCE = Path("data/work/v26/run_f2/probabilities.npz")
FAMILY = "v27"
CANDIDATES = {
    "hedge": dict(base="blend", family=FAMILY, target="w0.70"),
    "bet": dict(base="blend", family=FAMILY, target="w1.00"),
    "safe": dict(base="frozen", family="v27", target="robust"),
    "hedge_frozen": dict(base="frozen", family=FAMILY, target="w0.70"),
}
COMPARE = ("v26b_video_ball", "v27_semantic_bet", "v24_camera_fusion", "v28d_retrained")
LABELS = np.array(["H", "A", "D", "X"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- inputs and networks ---------------------------------------------------------------------

def filled_match(work, video_ball=VIDEO_BALL, alignment=ALIGNMENT):
    """The challenge tracking with the ball completed exactly as possession.v26_video_fill does:
    tracking ball, then the video detector, then the broadcast ball through the alignment."""
    from .sources import challenge_match
    from .camera_sync import fused_ball
    from .competitive_bundle import camera_match, read_alignment
    reference = challenge_match()
    other, _ = camera_match(reference, work)
    saved = np.load(video_ball)
    video = saved["ball"] * np.array(reference.pitch, dtype=np.float32)
    ball = reference.ball.copy()
    from_video = ~np.isfinite(ball[:, 0]) & np.isfinite(video[:, 0]) & (saved["source"] >= 2)
    ball[from_video] = video[from_video]
    match, from_broadcast = fused_ball(reference.copy_with(ball=ball), other, read_alignment(Path(alignment)))
    graded = np.zeros(len(ball), dtype=bool)
    for _, a, z in reference.blocks:
        graded[a:z + 1] = True
    report = dict(ball_missing_tracking=float((~np.isfinite(reference.ball[graded, 0])).mean()),
                  ball_missing_final=float((~np.isfinite(match.ball[graded, 0])).mean()),
                  from_video=int((from_video & graded).sum()), from_broadcast=int((from_broadcast & graded).sum()))
    return match, report


def check_fingerprints(selection):
    from .competitive_bundle import input_signature, digest as file_digest
    if selection.get("input_files_sha256") != input_signature():
        raise ValueError("Benchmark input fingerprint changed")
    if selection.get("alignment_algorithm_sha256") != file_digest("possession/camera_sync.py"):
        raise ValueError("Changed alignment algorithm")


def network_outputs(work, device="cpu"):
    """(frozen triple, retrained triple, filled match, report); both ensembles see one record."""
    import torch
    from .competitive_bundle import CompetitiveBundle
    from .v12_compare import match_record
    torch.set_num_threads(6)
    started = time.time()
    match, report = filled_match(work)
    record = match_record(match)
    report["seconds_inputs"] = round(time.time() - started, 1)
    out = []
    for name, path in (("frozen", FROZEN_SELECTION), ("retrained", RETRAINED_SELECTION)):
        started = time.time()
        selection = json.loads(path.read_text())
        check_fingerprints(selection)
        p, e = CompetitiveBundle(selection, device).probabilities(record)
        np.savez_compressed(work / f"{name}_probabilities.npz", probabilities=p, edges=e, frames=record["frames"])
        out.append((p, e, record["frames"]))
        report[f"seconds_{name}_networks"] = round(time.time() - started, 1)
    reference = np.load(FROZEN_REFERENCE)
    p, e, frames = out[0]
    report["frozen_reproduces_run_f2"] = bool(np.array_equal(frames, reference["frames"]) and np.array_equal(p, reference["probabilities"])
                                              and np.array_equal(e, reference["edges"]))
    report["frozen_max_abs_difference_to_run_f2"] = float(max(np.abs(p - reference["probabilities"]).max(), np.abs(e - reference["edges"]).max()))
    report["device"] = device
    return out[0], out[1], match, report


def bases(frozen, retrained):
    from .v28d_semantic import blend_arrays
    return dict(frozen=tuple(np.asarray(v) for v in frozen), blend=blend_arrays([frozen, retrained]))


def fused(triple, alignment=ALIGNMENT):
    """Unchanged V24 camera fusion -> (fused probabilities, blocks, n_frames)."""
    from .v27_semantic import fused_probabilities
    probabilities, edges, frames = triple
    return fused_probabilities(probabilities, edges, frames, Path(alignment).with_suffix(".json"))


def challenge_signals(match):
    """V28a physical signals from the SAME filled ball the networks saw."""
    return k28.signals(match.ball, match.players, match.team, match.pitch)


# -- verifier --------------------------------------------------------------------------------

def load_verifier(base, family, target):
    import joblib
    path = lab.verifier_path(base, family, target)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m possession.v29_lab table` and `... fit`")
    saved = joblib.load(path)
    if saved["features"] != k28.FEATURES or saved["base"] != base:
        raise ValueError("Verifier does not belong to this base / feature list")
    return saved


def decoded(probabilities, edges, frames, blocks, n_frames):
    interpolated = c.interpolate(np.asarray(probabilities), np.asarray(edges), np.asarray(frames), blocks)
    codes = c.decode(interpolated, n_frames)[0]
    p, edge = c.dense(interpolated, n_frames)
    return codes, p, edge


def verify(probabilities, edges, frames, blocks, n_frames, saved, signals=None, threshold=0.0):
    """Fused probabilities -> (codes after, codes before, report). `before` is the plain decode."""
    model = saved["model"]
    if lab.needs_physics(model.columns) and signals is None:
        raise ValueError("This verifier needs the tracking signals")
    before, p, edge = decoded(probabilities, edges, frames, blocks, n_frames)
    segs = c.segments(before, blocks)
    cands = c.candidates(segs, saved["max_frames"])
    first = model.predict(lab.features(segs, cands, p, edge, signals if lab.needs_physics(model.columns) else None))
    new, applied = lab.apply(model, segs, p, edge, signals, threshold)
    after = c.to_codes(new, n_frames)
    kinds = {}
    for action, _ in applied:
        kinds[action] = kinds.get(action, 0) + 1
    by_action = {}
    for action in c.ACTIONS:
        rows = np.array([a == action for _, a, _ in cands])
        by_action[action] = dict(candidates=int(rows.sum()), score_positive=int((first[rows] >= threshold).sum()))
    graded = np.concatenate([np.arange(a, z + 1) for _, a, z in blocks])
    report = dict(base=saved["base"], family=saved["family"], target=saved["target"], threshold=threshold,
                  removals=len(applied), removals_by_kind=kinds,
                  removed_seconds_median=float(np.median([d for _, d in applied])) if applied else 0.0,
                  candidates_first_pass=len(cands), candidates_by_action=by_action,
                  share_of_candidates_scored_positive=float((first >= threshold).mean()) if len(cands) else 0.0,
                  before=c.duration_stats(before, blocks), after=c.duration_stats(after, blocks),
                  graded_frames=int(len(graded)), graded_frames_changed_by_verifier=int((before[graded] != after[graded]).sum()),
                  dead_frames_added=int(((after[graded] == 2) & (before[graded] != 2)).sum()))
    return after, before, report


def codes_of(path):
    import pandas as pd
    return pd.read_csv(path)["label"].map(dict(H=0, A=1, D=2, X=3)).to_numpy().astype(np.int8)


def write_csv(path, codes):
    """Write the submission; an existing file is kept and must already hold the same bytes."""
    from starter.baseline import write_submission
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    write_submission(str(temporary), LABELS[codes])
    new = digest(temporary)
    if path.exists():
        if digest(path) != new:
            raise RuntimeError(f"{path} exists with other bytes; kept. New result left at {temporary}")
        temporary.unlink()
    else:
        temporary.replace(path)
    return new


def run(args):
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.reuse_networks:
        from .v28d_semantic import load_triple
        frozen, retrained = load_triple(work / "frozen_probabilities.npz"), load_triple(work / "retrained_probabilities.npz")
        match, report = None, dict(networks="reused from " + str(work))
    else:
        frozen, retrained, match, report = network_outputs(work, args.device)
        if args.device == "cpu" and not report["frozen_reproduces_run_f2"]:
            raise RuntimeError("Frozen ensemble on the filled ball does not reproduce " + str(FROZEN_REFERENCE))
    triples = bases(frozen, retrained)
    np.savez_compressed(work / "blend_probabilities.npz", probabilities=triples["blend"][0], edges=triples["blend"][1], frames=triples["blend"][2])
    selected = {name: dict(CANDIDATES[name], family=CANDIDATES[name]["family"] if name == "safe" else args.family)
                for name in args.candidates}
    verifiers = {name: load_verifier(**spec) for name, spec in selected.items()}
    signals = None
    if any(lab.needs_physics(v["model"].columns) for v in verifiers.values()):
        if match is None:
            match, _ = filled_match(work)
        signals = challenge_signals(match)
    fusion = {}
    for base in sorted({spec["base"] for spec in selected.values()}):
        tick = time.time()
        probabilities, blocks, n = fused(triples[base])
        fusion[base] = (probabilities, triples[base][1], triples[base][2], blocks, n)
        np.savez_compressed(work / f"fused_{base}.npz", probabilities=probabilities, edges=triples[base][1], frames=triples[base][2])
        report[f"seconds_fusion_{base}"] = round(time.time() - tick, 1)
    others = {name: codes_of(Path("evaluation/results") / f"{name}.csv") for name in COMPARE
              if (Path("evaluation/results") / f"{name}.csv").exists()}
    report["candidates"] = {}
    for name, spec in selected.items():
        probabilities, edges, frames, blocks, n = fusion[spec["base"]]
        after, before, row = verify(probabilities, edges, frames, blocks, n, verifiers[name], signals)
        graded = np.concatenate([np.arange(a, z + 1) for _, a, z in blocks])
        path = args.output_dir / f"v29_{name}.csv"
        row.update(csv=str(path), csv_sha256=write_csv(path, after), rows=int(len(after)),
                   counts={k: int((LABELS[after] == k).sum()) for k in "HADX"},
                   graded_frames_different_from={other: int((codes[graded] != after[graded]).sum()) for other, codes in others.items()},
                   decoder_only_different_from={other: int((codes[graded] != before[graded]).sum()) for other, codes in others.items()})
        report["candidates"][name] = row
    report.update(seconds_total=round(time.time() - started, 1), challenge_labels_read=False)
    (work / "run.json").write_text(json.dumps(report, indent=2))
    print("V29", json.dumps(report, indent=2), flush=True)


# -- label-free checks -----------------------------------------------------------------------

def challenge_candidates(work, base):
    """Feature rows (V27 + physical) of the candidate removals on the challenge decode of a base."""
    from .v28d_semantic import load_triple
    triple = load_triple(work / ("frozen_probabilities.npz" if base == "frozen" else "blend_probabilities.npz"))
    probabilities, blocks, n = fused(triple)
    codes, p, edge = decoded(probabilities, triple[1], triple[2], blocks, n)
    segs = c.segments(codes, blocks)
    cands = c.candidates(segs, lab.MAX_FRAMES)
    match, _ = filled_match(work)
    return lab.features(segs, cands, p, edge, challenge_signals(match)), np.array([c.ACTIONS.index(a) for _, a, _ in cands])


def domain_auc(x, y):
    """Cross-validated AUC of 'is this candidate from the challenge match?' (0.5 = no shift)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    scores = np.zeros(len(y))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=20,
                                               early_stopping=False, random_state=0).fit(x[train], y[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
    return float(roc_auc_score(y, scores))


def shift(args):
    """Verifier inputs on the challenge match (filled ball) next to the Metrica conditions."""
    absorb = k28.FEATURES.index("is_absorb")
    report = {}
    for base in lab.BASES:
        x, actions = challenge_candidates(args.work_dir, base)
        table = lab.load_table(base)
        reference = {m: np.concatenate([table[f"x_{g}_{m}"] for g in lab.GAMES]) for m in c.MODES}
        everything = np.concatenate(list(reference.values()))
        rows = dict(candidates=dict(challenge=len(x), **{m: round(len(v) / 3, 1) for m, v in reference.items()}),
                    share_absorb=dict(challenge=float(x[:, absorb].mean()), **{m: float(v[:, absorb].mean()) for m, v in reference.items()}))
        auc = {}
        for name, columns, only_absorb in (("v27_features_all_candidates", lab.V, False), ("v27_features_T-X-T", lab.V, True),
                                           ("physical_features_T-X-T", lab.P_NO_SEEN, True)):
            a, b = (x[x[:, absorb] == 1], everything[everything[:, absorb] == 1]) if only_absorb else (x, everything)
            auc[name] = dict(vs_all_conditions=round(domain_auc(np.vstack([a, b])[:, columns], np.r_[np.ones(len(a)), np.zeros(len(b))]), 3))
            for mode, values in reference.items():
                v = values[values[:, absorb] == 1] if only_absorb else values
                auc[name]["vs_" + mode] = round(domain_auc(np.vstack([a, v])[:, columns], np.r_[np.ones(len(a)), np.zeros(len(v))]), 3)
        # yardstick: how well can one Metrica game be told from the two others (same condition mix)?
        yard = {}
        for name, columns, only_absorb in (("v27_features_all_candidates", lab.V, False), ("physical_features_T-X-T", lab.P_NO_SEEN, True)):
            values = []
            for game in lab.GAMES:
                parts = [(np.concatenate([table[f"x_{g}_{m}"] for m in c.MODES]), g == game) for g in lab.GAMES]
                xx = np.vstack([p for p, _ in parts])
                yy = np.concatenate([np.full(len(p), float(flag)) for p, flag in parts])
                keep = xx[:, absorb] == 1 if only_absorb else np.ones(len(xx), dtype=bool)
                values.append(domain_auc(xx[keep][:, columns], yy[keep]))
            yard[name] = [round(v, 3) for v in values]
        rows["domain_auc"] = auc
        rows["domain_auc_one_metrica_game_vs_other_two"] = yard
        features = {}
        for group, flag in (("T-X-T", 1.0), ("next_to_stoppage", 0.0)):
            for j in lab.V + lab.PHYSICAL:
                name = k28.FEATURES[j]
                cell = {}
                for source, values in [("challenge", x)] + list(reference.items()):
                    v = values[values[:, absorb] == flag][:, j]
                    ok = np.isfinite(v)
                    cell[source] = dict(median=round(float(np.median(v[ok])), 3) if ok.any() else None,
                                        q25=round(float(np.percentile(v[ok], 25)), 3) if ok.any() else None,
                                        q75=round(float(np.percentile(v[ok], 75)), 3) if ok.any() else None, finite=round(float(ok.mean()), 3))
                pooled = everything[everything[:, absorb] == flag][:, j]
                pooled = pooled[np.isfinite(pooled)]
                mine = x[x[:, absorb] == flag][:, j]
                mine = mine[np.isfinite(mine)]
                if len(pooled) and len(mine):
                    grid = np.sort(np.concatenate([pooled, mine]))
                    cell["ks_vs_metrica"] = round(float(np.abs(np.searchsorted(np.sort(mine), grid, side="right") / len(mine)
                                                               - np.searchsorted(np.sort(pooled), grid, side="right") / len(pooled)).max()), 3)
                features[f"{group}/{name}"] = cell
        rows["features"] = features
        report[base] = rows
        print("SHIFT", base, json.dumps({k: v for k, v in rows.items() if k != "features"}), flush=True)
        ranked = sorted(((v.get("ks_vs_metrica", 0.0), k) for k, v in features.items()), reverse=True)[:14]
        for ks, key in ranked:
            cell = features[key]
            print("SHIFT", base, f"{key:<40} KS {ks:.2f}", {s: (cell[s]["median"], cell[s]["finite"]) for s in ("challenge",) + c.MODES}, flush=True)
    (ROOT / "shift.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["run", "shift"])
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "run")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--candidates", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--family", choices=list(lab.FAMILIES), default=FAMILY,
                        help="verifier family of hedge / bet / hedge_frozen (study only; the shipped CSVs use the default)")
    parser.add_argument("--reuse-networks", action="store_true", help="reuse the probabilities saved in --work-dir")
    args = parser.parse_args()
    if args.step == "run":
        run(args)
    else:
        shift(args)
