"""V31c on the challenge match: M-target, label-blind-ball networks through the path of possession.v29_final.

    tracking + video-filled ball (possession.v29_final.filled_match, unchanged)
      -> one network record (possession.v12_compare.match_record)
      -> frozen ensemble (sanity: must reproduce data/work/v26/run_f2/probabilities.npz on CPU)
      -> V31c final ensembles, two seeds = eight networks (data/work/v31c/final/M and M_seed10: six
         DFL games + the three open Metrica games in label-blind views, possession.v31c_train --held 0)
      -> base of the recipe (probability blend) -> unchanged V24 camera fusion -> 25 Hz decoder
      -> spell verifier w0.5 fitted on the base's OUT-OF-FOLD outputs on the three Metrica games
         (possession.v31c_eval; every game predicted by networks that did not train on it)

Recipes (CSV = evaluation/results/v31c_<name>.csv):
    m2_frozen   base 0.5 M2 + 0.5 frozen,  verifier w = 0.5
    m2          base M2 (both seeds),      verifier w = 0.5
    (decoder_* = the same bases without the verifier, diagnostics)
    ab2 / ab2_frozen   the CONTROL networks (alternative target, label-blind ball), verifier w = 0.5

NO RECIPE MET THE ADOPTION BAR (see data/work/v31c/score.json), so nothing is written to
evaluation/results by default use; the control recipe was run once as a label-free trial:

    .venv/bin/python -m possession.v31c_final --recipes ab2 ab2_frozen decoder_ab2 \
        --work-dir data/work/v31c/challenge_trial --output-dir data/work/v31c/trial_csv

Also prints the label-free challenge diagnostics of data/work/review_v30d/challenge_diag.py
(share of predicted D starts within 2 frames of a ball-vanish event, against chance).
An existing CSV is never replaced by other bytes. No ground truth of the challenge is read.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from . import v27_core as c
from . import v29_final as final
from . import v29_lab as lab
from . import v31c_eval, v31c_train

ROOT = Path("data/work/v31c")
RECIPES = {"m2_frozen": dict(base="M2+frozen", w=0.5), "m2": dict(base="M2", w=0.5),
           # control networks (alternative target on the label-blind views); label-free TRIAL only
           "ab2": dict(base="Ab2", w=0.5), "ab2_frozen": dict(base="Ab2+frozen", w=0.5), "decoder_ab2": dict(base="Ab2", w=None),
           "m_frozen_1seed": dict(base="M+frozen", w=0.5), "m_1seed": dict(base="M", w=0.5),
           "decoder_m2_frozen": dict(base="M2+frozen", w=None), "decoder_m2": dict(base="M2", w=None)}
REFERENCE_CSV = Path("evaluation/results/v29_final.csv")


def selections():
    return {"frozen": final.FROZEN_SELECTION, "M": v31c_train.selection_path("M", 0), "M_seed10": v31c_train.selection_path("M", 0, 10),
            "Ab": v31c_train.selection_path("A", 0), "Ab_seed10": v31c_train.selection_path("A", 0, 10)}


def network_outputs(work, tags, device="cpu"):
    import torch
    from .competitive_bundle import CompetitiveBundle
    from .v12_compare import match_record
    torch.set_num_threads(6)
    match, report = final.filled_match(work)
    record = match_record(match)
    out = {}
    for tag in tags:
        selection = json.loads(Path(selections()[tag]).read_text())
        final.check_fingerprints(selection)
        p, e = CompetitiveBundle(selection, device).probabilities(record)
        np.savez_compressed(work / f"{tag}_probabilities.npz", probabilities=p, edges=e, frames=record["frames"])
        out[tag] = (p, e, record["frames"])
    reference = np.load(final.FROZEN_REFERENCE)
    p, e, frames = out["frozen"]
    report["frozen_reproduces_run_f2"] = bool(np.array_equal(frames, reference["frames"]) and np.array_equal(p, reference["probabilities"])
                                              and np.array_equal(e, reference["edges"]))
    report["frozen_max_abs_difference_to_run_f2"] = float(max(np.abs(p - reference["probabilities"]).max(), np.abs(e - reference["edges"]).max()))
    report["device"] = device
    seen = np.isfinite(match.ball).all(axis=1)
    return out, report, seen, match.blocks


def blended(outputs, base):
    parts = v31c_eval.BASES[base]
    frames = outputs[next(iter(parts))][2]
    return (sum(w * np.asarray(outputs[tag][0], dtype=np.float64) for tag, w in parts.items()),
            sum(w * np.asarray(outputs[tag][1], dtype=np.float64) for tag, w in parts.items()), frames)


def final_verifier(base, w):
    """All three games; the table comes from the out-of-fold network outputs of the base (standard conditions)."""
    ev = v31c_eval.patched()
    table = ev.table_of([k for k in ev.conditions(base) if k.mode in v31c_eval.STANDARD])
    model = ev.fit_verifier(table, list(v31c_eval.GAMES), w)
    return dict(model=model, max_frames=lab.MAX_FRAMES, base=base, family="v27", target=f"w{w:.2f}")


def diagnostics(codes, seen, blocks, seed=0):
    """Label-free: where predicted D starts sit relative to ball-vanish events (reviewer's challenge_diag)."""
    graded = np.zeros(len(seen), dtype=bool)
    for _, a, z in blocks:
        graded[a:z + 1] = True
    vanish = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
    starts = np.flatnonzero((codes[1:] == 2) & (codes[:-1] != 2) & graded[1:] & graded[:-1]) + 1
    rng = np.random.default_rng(seed)
    row = dict(n_D_starts=int(len(starts)), P_D_given_ball_missing=float((codes[graded & ~seen] == 2).mean()),
               P_D_given_ball_seen=float((codes[graded & seen] == 2).mean()))
    for k in (2, 5, 12):
        chance = np.mean([(v31c_eval.near(starts + rng.integers(-250, 251, len(starts)), vanish) <= k).mean() for _ in range(200)])
        row[f"Dstart_within{k}_of_vanish"] = float((v31c_eval.near(starts, vanish) <= k).mean())
        row[f"Dstart_within{k}_of_vanish_chance"] = float(chance)
    return row


def run(args):
    import joblib
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tags = sorted({"frozen"} | {tag for name in args.recipes for tag in v31c_eval.BASES[RECIPES[name]["base"]]})
    outputs, report, seen, match_blocks = network_outputs(work, tags, args.device)
    if args.device == "cpu" and not report["frozen_reproduces_run_f2"]:
        raise RuntimeError("Frozen ensemble on the filled ball does not reproduce " + str(final.FROZEN_REFERENCE))
    reference = final.codes_of(REFERENCE_CSV)
    report["v29_final_diagnostics"] = diagnostics(reference, seen, match_blocks)
    report["recipes"] = {}
    for name in args.recipes:
        spec = RECIPES[name]
        triple = blended(outputs, spec["base"])
        probabilities, blocks, n = final.fused(triple)
        np.savez_compressed(work / f"fused_{spec['base']}.npz", probabilities=probabilities, edges=triple[1], frames=triple[2])
        if spec["w"] is None:
            before = final.decoded(probabilities, triple[1], triple[2], blocks, n)[0]
            after, row = before, dict(removals=0, after=c.duration_stats(before, blocks))
        else:
            saved = final_verifier(spec["base"], spec["w"])
            joblib.dump(saved, work / f"verifier_{name}.joblib")
            after, before, row = final.verify(probabilities, triple[1], triple[2], blocks, n, saved)
        graded = np.concatenate([np.arange(a, z + 1) for _, a, z in blocks])
        path = args.output_dir / f"v31c_{name}.csv"
        row.update(csv=str(path), csv_sha256=final.write_csv(path, after), rows=int(len(after)),
                   counts={k: int((final.LABELS[after] == k).sum()) for k in "HADX"},
                   graded_frames_changed_vs_v29_final=int((reference[graded] != after[graded]).sum()),
                   share_changed_vs_v29_final=float((reference[graded] != after[graded]).mean()),
                   decoder_only_changed_vs_v29_final=int((reference[graded] != before[graded]).sum()),
                   v29_final=c.duration_stats(reference, blocks), diagnostics=diagnostics(after, seen, match_blocks))
        report["recipes"][name] = row
        print("V31C", name, json.dumps({k: row[k] for k in ("removals", "after", "graded_frames_changed_vs_v29_final",
                                                             "share_changed_vs_v29_final", "diagnostics")}, default=float), flush=True)
    report.update(seconds_total=round(time.time() - started, 1), challenge_labels_read=False)
    (work / "run.json").write_text(json.dumps(report, indent=2, default=float))
    print("V31C RUN", json.dumps({k: v for k, v in report.items() if k != "recipes"}, default=float), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", nargs="+", choices=list(RECIPES), default=["m2_frozen", "m2"])
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "challenge")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    run(parser.parse_args())
