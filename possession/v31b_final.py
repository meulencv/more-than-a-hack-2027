"""V31b on the challenge match: networks trained on the compromise target, path of possession.v29_final.

    tracking + video-filled ball (possession.v29_final.filled_match, unchanged)
      -> one network record (possession.v12_compare.match_record)
      -> frozen ensemble (sanity: must reproduce data/work/v26/run_f2/probabilities.npz on CPU)
      -> V31b ensembles from data/work/v31b/final/<run>/ (six DFL games only; possession.v31b_train)
      -> base of the recipe (probability blend, possession.v31b_eval.BASES)
      -> unchanged V24 camera fusion -> 25 Hz decoder
      -> spell verifier (w = 0.5) fitted on the base's outputs on the three open Metrica games
         (possession.v31b_eval: standard conditions; no network ever trained on those games)

    .venv/bin/python -m possession.v31b_final freeze --runs m2 m2_seed10      # copy checkpoints -> data/work/v31b/final
    .venv/bin/python -m possession.v31b_final run --base m2x2_blend --name m2x2_blend
        -> evaluation/results/v31b_<name>.csv
    dry run (no adoption, nothing under evaluation/results, checkpoints not frozen):
    .venv/bin/python -m possession.v31b_final run --base m2x2_blend --unfrozen --output-dir data/work/v31b/challenge/trial

An existing CSV is never replaced by other bytes. No ground truth of the challenge is read.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from . import v27_core as c
from . import v29_final as final
from . import v29_lab as lab
from . import v30d_eval as e30
from . import v31b_eval, v31b_train

ROOT = Path("data/work/v31b")
FINAL = ROOT / "final"
REFERENCE_CSV = Path("evaluation/results/v29_final.csv")


def final_selection(run):
    return FINAL / f"selection_{run}.json"


def freeze(runs):
    """Copy the trained checkpoints to data/work/v31b/final and write selections that point there."""
    for run in runs:
        variant, _, seed = run.partition("_seed")
        source = json.loads(v31b_train.selection_path(variant, int(seed or 0)).read_text())
        target = FINAL / run
        target.mkdir(parents=True, exist_ok=True)
        for spec in source["components"]:
            path = target / Path(spec["path"]).name
            if not path.exists():
                shutil.copyfile(spec["path"], path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                raise RuntimeError(f"{path} differs from the evaluated checkpoint")
            spec["path"] = str(path)
        final_selection(run).write_text(json.dumps(source, indent=2))
        print("V31B FROZEN", final_selection(run), flush=True)


UNFROZEN = False        # dry run: read the evaluated checkpoints in data/work/v31b/checkpoints directly


def selection_of(tag):
    if tag == "frozen":
        return final.FROZEN_SELECTION
    if tag in v31b_eval.SELECTIONS:
        return Path(v31b_eval.SELECTIONS[tag])
    if UNFROZEN:
        variant, _, seed = tag.partition("_seed")
        return v31b_train.selection_path(variant, int(seed or 0))
    return final_selection(tag)


def network_outputs(work, tags, device="cpu"):
    import torch
    from .competitive_bundle import CompetitiveBundle
    from .v12_compare import match_record
    torch.set_num_threads(6)
    match, report = final.filled_match(work)
    record = match_record(match)
    out = {}
    for tag in tags:
        selection = json.loads(Path(selection_of(tag)).read_text())
        final.check_fingerprints(selection)
        for spec in selection["components"]:
            if hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest() != spec["sha256"]:
                raise RuntimeError(f"Checkpoint {spec['path']} changed")
        p, e = CompetitiveBundle(selection, device).probabilities(record)
        np.savez_compressed(work / f"{tag}_probabilities.npz", probabilities=p, edges=e, frames=record["frames"])
        out[tag] = (p, e, record["frames"])
    reference = np.load(final.FROZEN_REFERENCE)
    p, e, frames = out["frozen"]
    report["frozen_reproduces_run_f2"] = bool(np.array_equal(frames, reference["frames"]) and np.array_equal(p, reference["probabilities"])
                                              and np.array_equal(e, reference["edges"]))
    report["device"] = device
    seen = np.isfinite(match.ball).all(axis=1)
    return out, report, seen, match.blocks


def blended(outputs, base):
    parts = v31b_eval.BASES[base]
    frames = outputs[next(iter(parts))][2]
    return (sum(w * np.asarray(outputs[tag][0], dtype=np.float64) for tag, w in parts.items()),
            sum(w * np.asarray(outputs[tag][1], dtype=np.float64) for tag, w in parts.items()), frames)


def final_verifier(base, w):
    """All three Metrica games, standard conditions, on the base's own outputs."""
    table = e30.table_of(v31b_eval.conditions(base, v31b_eval.STANDARD))
    model = e30.fit_verifier(table, list(v31b_eval.GAMES), w)
    return dict(model=model, max_frames=lab.MAX_FRAMES, base=base, family="v27", target=f"w{w:.2f}")


def vanish_diagnostic(codes, seen, blocks):
    """Share of predicted D starts within k frames of a ball-vanish event (reviewer's challenge_diag)."""
    graded = np.zeros(len(seen), dtype=bool)
    for _, a, z in blocks:
        graded[a:z + 1] = True
    vanish = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
    starts = np.flatnonzero((codes[1:] == 2) & (codes[:-1] != 2) & graded[1:] & graded[:-1]) + 1
    i = np.searchsorted(vanish, starts)
    distance = np.minimum(np.abs(vanish[np.clip(i - 1, 0, len(vanish) - 1)] - starts), np.abs(vanish[np.clip(i, 0, len(vanish) - 1)] - starts))
    return dict(d_starts=int(len(starts)), **{f"within{k}_of_vanish": round(float((distance <= k).mean()), 3) for k in (2, 5, 12)},
                p_dead_given_ball_seen=round(float((codes[graded & seen] == 2).mean()), 3),
                p_dead_given_ball_missing=round(float((codes[graded & ~seen] == 2).mean()), 3))


def run(args):
    import joblib
    global UNFROZEN
    UNFROZEN = args.unfrozen
    if UNFROZEN and args.output_dir == Path("evaluation/results"):
        raise ValueError("A dry run (--unfrozen) must not write to evaluation/results")
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tags = sorted({"frozen"} | set(v31b_eval.BASES[args.base]))
    outputs, report, seen, blocks0 = network_outputs(work, tags, args.device)
    if args.device == "cpu" and not report["frozen_reproduces_run_f2"]:
        raise RuntimeError("Frozen ensemble on the filled ball does not reproduce " + str(final.FROZEN_REFERENCE))
    reference = final.codes_of(REFERENCE_CSV)
    triple = blended(outputs, args.base)
    probabilities, blocks, n = final.fused(triple)
    np.savez_compressed(work / f"fused_{args.base}.npz", probabilities=probabilities, edges=triple[1], frames=triple[2])
    saved = final_verifier(args.base, args.w)
    joblib.dump(saved, work / f"verifier_{args.name}.joblib")
    after, before, row = final.verify(probabilities, triple[1], triple[2], blocks, n, saved)
    graded = np.concatenate([np.arange(a, z + 1) for _, a, z in blocks])
    path = args.output_dir / f"v31b_{args.name}.csv"
    row.update(csv=str(path), csv_sha256=final.write_csv(path, after), rows=int(len(after)),
               counts={k: int((final.LABELS[after] == k).sum()) for k in "HADX"},
               graded_frames_changed_vs_v29_final=int((reference[graded] != after[graded]).sum()),
               share_changed_vs_v29_final=float((reference[graded] != after[graded]).mean()),
               decoder_only_changed_vs_v29_final=int((reference[graded] != before[graded]).sum()),
               v29_final=c.duration_stats(reference, blocks),
               d_start_diagnostic=dict(candidate=vanish_diagnostic(after, seen, blocks), v29_final=vanish_diagnostic(reference, seen, blocks)))
    report.update(recipe=dict(base=args.base, parts=v31b_eval.BASES[args.base], w=args.w, name=args.name), result=row,
                  seconds_total=round(time.time() - started, 1), challenge_labels_read=False)
    (work / f"run_{args.name}.json").write_text(json.dumps(report, indent=2, default=float))
    print("V31B RUN", json.dumps({k: row[k] for k in ("removals", "after", "v29_final", "graded_frames_changed_vs_v29_final", "share_changed_vs_v29_final",
                                                      "d_start_diagnostic", "csv", "csv_sha256")}, default=float), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["freeze", "run"])
    parser.add_argument("--runs", nargs="+", default=["m2", "m2_seed10"])
    parser.add_argument("--base", default="m2x2_blend")
    parser.add_argument("--name")
    parser.add_argument("--w", type=float, default=0.5)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "challenge")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--unfrozen", action="store_true", help="dry run from data/work/v31b/checkpoints (never to evaluation/results)")
    args = parser.parse_args()
    if args.step == "freeze":
        freeze(args.runs)
    else:
        args.name = args.name or args.base
        run(args)
