"""V26 hide-and-fill ablation: does the video ball move the frozen networks back towards what
they say with the tracking's ball?

No ground truth is read. The pseudo-reference is the frozen run itself (R: tracking ball +
broadcast fill, data/work/final_cpu/probabilities.npz, decoded with the unchanged decoder).
Inside the detector's held-out minutes the tracking's ball is hidden in simulated gaps whose
lengths are drawn from the real gaps, and the bundle is re-run three ways:

    G   hidden frames have no ball at all (broadcast fill removed there too)
    F   hidden frames are filled by the video track, shipped configuration (flight filter on)
    F2  as F with the flight filter off

The tracker never sees the hidden ball: it gets the tracking's ball only outside the
simulated gaps, exactly as on a real gap. Metrics are computed on the held-out minutes only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import evaluation.possession_metrics as pm
from .sources import challenge_match
from .camera_sync import fused_ball
from .competitive_bundle import CompetitiveBundle, camera_match, read_alignment
from .v12_compare import match_record
from .v14_resolution import decode_fine
from .video_ball import ROOT, HELD_BLOCK, held_out
from . import v26_track

HIDE_SHARE = 0.40


def simulated_gaps(seen, graded, done, rng):
    """Boolean mask of simulated gaps inside held-out minutes, lengths drawn from the real gaps."""
    n = len(seen)
    frames = np.arange(n)
    held = held_out(frames) & graded & done
    missing = graded & ~seen
    edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(np.int8), 0]))
    lengths = edges[1::2] - edges[::2]
    starts = np.flatnonzero(held & seen)
    target = HIDE_SHARE * len(starts)
    gap = np.zeros(n, dtype=bool)
    drawn = []
    while (gap & held & seen).sum() < target:
        length = int(rng.choice(lengths))
        start = int(rng.choice(starts))
        if gap[start]:
            continue
        stop = min(start + length, (start // HELD_BLOCK + 1) * HELD_BLOCK)       # stay inside the held-out minute
        gap[start:stop] = True
        drawn.append(stop - start)
    gap &= held
    return gap, held, lengths, np.array(drawn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("docs/FINAL_SELECTION.json"))
    parser.add_argument("--reference", type=Path, default=Path("data/work/final_cpu/probabilities.npz"))
    parser.add_argument("--alignment", type=Path, default=Path("data/work/v11/challenge_alignment.npz"))
    parser.add_argument("--detections", default="detections_net3.npz")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2601, 2602, 2603])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "ablation.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    work = ROOT / "run"
    operating = json.loads((ROOT / "video_ball.json").read_text())["operating_point"]
    threshold, min_weight = operating["threshold"], operating["min_weight"]
    selection = json.loads(args.selection.read_text())
    reference = challenge_match()
    other, _ = camera_match(reference, work)
    alignment = read_alignment(args.alignment)
    match_r, _ = fused_ball(reference, other, alignment)
    bundle = CompetitiveBundle(selection, "cpu")
    saved = np.load(args.reference)
    p_r, e_r, grid = saved["probabilities"], saved["edges"], saved["frames"]
    codes_r = decode_fine(p_r, e_r, grid, reference, "probability")
    state = v26_track.prepare(ROOT / args.detections)
    n, atd, done, graded, geometry = state["n"], state["atd"], state["done"], state["graded"], state["geometry"]
    seen = np.isfinite(atd[:, 0])

    def run(ball):
        probabilities, boundaries = bundle.probabilities(match_record(match_r.copy_with(ball=ball.astype(np.float32))))
        return probabilities, decode_fine(probabilities, boundaries, grid, reference, "probability")

    def video(gap, flights):
        hidden = (~seen | gap) & done
        known = np.where(gap[:, None], np.nan, atd)
        table = v26_track.chunk_table(state["tracklets"], threshold, min_weight, hidden, known, geometry)
        xy, _, kind = v26_track.track(state["tracklets"], table, hidden, known, n, geometry, filter_flights=flights)
        return xy, kind > 0

    rows = []
    for seed in args.seeds:
        gap, held, real_lengths, drawn = simulated_gaps(seen, graded, done, np.random.default_rng(seed))
        hidden = gap & np.isfinite(match_r.ball[:, 0])               # frames that lose a ball R had
        ball_g = match_r.ball.astype(np.float64).copy()
        ball_g[gap] = np.nan
        truth = np.where(held, codes_r, 3).astype(np.int8)
        on_grid = hidden[grid]
        row = dict(seed=seed, heldout_frames=int(held.sum()), heldout_frames_with_tracked_ball=int((held & seen).sum()),
                   hidden_frames=int(hidden.sum()), hidden_share_of_tracked=float((gap & seen).sum() / (held & seen).sum()),
                   simulated_gaps=int(len(drawn)), simulated_median_length=float(np.median(drawn)),
                   real_median_length=float(np.median(real_lengths)), variants={})
        filled_f = None
        for name, flights in (("G", None), ("F", True), ("F2", False)):
            ball = ball_g.copy()
            filled = np.zeros(n, dtype=bool)
            position = {}
            if flights is not None:
                xy, has = video(gap, flights)
                filled = hidden & has
                ball[filled] = xy[filled]
                error = np.linalg.norm(xy[filled & seen] - atd[filled & seen], axis=1)
                position = dict(filled_share_of_hidden=float(filled.sum() / hidden.sum()),
                                within_3m_of_tracking=float((error <= 3.0).mean()),
                                far_over_5m_of_tracking=float((error > 5.0).mean()),
                                median_error_m=float(np.median(error)))
                if name == "F":
                    filled_f = filled
            probabilities, codes = run(ball)
            scores = pm.evaluate(truth, codes, bootstrap_n=0)
            agree = codes == codes_r
            entry = dict(spell_f1=float(scores["spell_f1_mean"]), macro_f1=float(scores["macro_f1"]),
                         accuracy=float(scores["accuracy"]), n_reference_spells=int(scores["n_gt_spells"]),
                         n_spells=int(scores["n_pred_spells"]),
                         agreement_heldout=float(agree[held].mean()), agreement_hidden=float(agree[hidden].mean()),
                         abs_change_h_minus_a_hidden=float(np.abs((probabilities[on_grid, 0] - probabilities[on_grid, 1])
                                                                  - (p_r[on_grid, 0] - p_r[on_grid, 1])).mean()),
                         abs_change_dead_hidden=float(np.abs(probabilities[on_grid, 2] - p_r[on_grid, 2]).mean()),
                         **position)
            row["variants"][name] = entry
            row["_codes_" + name] = codes
            print("ABLATION seed", seed, name, json.dumps(entry), flush=True)
        # Agreement on hidden frames split by what the shipped video track did there.
        for name in ("G", "F", "F2"):
            agree = row.pop("_codes_" + name) == codes_r
            row["variants"][name]["agreement_hidden_video_filled"] = float(agree[filled_f].mean())
            row["variants"][name]["agreement_hidden_left_empty"] = float(agree[hidden & ~filled_f].mean())
        row["hidden_video_filled_frames"] = int(filled_f.sum())
        row["hidden_left_empty_frames"] = int((hidden & ~filled_f).sum())
        rows.append(row)
    summary = {}
    for name in ("G", "F", "F2"):
        summary[name] = {}
        for key in rows[0]["variants"][name]:
            values = np.array([r["variants"][name][key] for r in rows], dtype=np.float64)
            summary[name][key] = dict(mean=float(values.mean()), min=float(values.min()), max=float(values.max()))
    recovered = {}
    for key in ("spell_f1", "macro_f1", "agreement_hidden", "agreement_hidden_video_filled"):
        for name in ("F", "F2"):
            per_seed = [(r["variants"][name][key] - r["variants"]["G"][key]) / max(1.0 - r["variants"]["G"][key], 1e-9)
                        for r in rows]
            recovered[name + "_" + key] = dict(mean=float(np.mean(per_seed)), min=float(np.min(per_seed)),
                                               max=float(np.max(per_seed)))
    report = dict(threshold=threshold, min_weight=min_weight, detections=args.detections, seeds=args.seeds,
                  hide_share_target=HIDE_SHARE, summary=summary, share_of_loss_recovered=recovered, per_seed=rows,
                  pseudo_reference="frozen run decoded labels (not the challenge ground truth)", ground_truth_read=False)
    args.output.write_text(json.dumps(report, indent=2))
    print("ABLATION SUMMARY", json.dumps(dict(summary=summary, share_of_loss_recovered=recovered), indent=2), flush=True)


if __name__ == "__main__":
    main()
