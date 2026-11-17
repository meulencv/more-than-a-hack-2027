"""V30d on the challenge match: in-domain networks through the path of possession.v29_final.

    tracking + video-filled ball (possession.v29_final.filled_match, unchanged)
      -> one network record (possession.v12_compare.match_record)
      -> frozen ensemble (sanity: must reproduce data/work/v26/run_f2/probabilities.npz on CPU)
      -> V30d final ensembles C and A, two seeds each = eight networks per target
         (data/work/v30d/final/{C,A} and data/work/v30d/checkpoints/{C,A}_h0_seed10: six DFL games
         + the three open Metrica games, possession.v30d_train --held 0 [--seed-offset 10])
      -> base of the recipe (probability blend) -> unchanged V24 camera fusion -> 25 Hz decoder
      -> spell verifier fitted on the base's OUT-OF-FOLD outputs on the three Metrica games
         (possession.v30d_eval; every game predicted by networks that did not train on it)

Recipes (CSV = evaluation/results/v30d_<name>.csv):
    consistent   base C2 (both seeds of C),   verifier w = 1.0
    alternative  base A2 (both seeds of A),   verifier w = 0.0
    hedged       base 0.5 C2 + 0.5 A2,        verifier w = 0.5

    .venv/bin/python -m possession.v30d_final --recipes consistent alternative hedged

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
from . import v30d_eval, v30d_train

ROOT = Path("data/work/v30d")
RECIPES = {"consistent": dict(base="C2", w=1.0), "alternative": dict(base="A2", w=0.0), "hedged": dict(base="CA2", w=0.5),
           # studied alternatives (see the report; written only when asked for)
           "hedged_c25": dict(base="C25A75_2", w=0.5), "hedged_a": dict(base="A2", w=0.5),
           "consistent_1seed": dict(base="C", w=1.0), "alternative_1seed": dict(base="A", w=0.0), "hedged_1seed": dict(base="CA", w=0.5),
           "decoder_c": dict(base="C2", w=None), "decoder_a": dict(base="A2", w=None)}
REFERENCE_CSV = Path("evaluation/results/v29_final.csv")


def selections():
    return {"frozen": final.FROZEN_SELECTION, "v28d": final.RETRAINED_SELECTION,
            "C": v30d_train.selection_path("C", 0), "A": v30d_train.selection_path("A", 0),
            "C_seed10": v30d_train.selection_path("C", 0, 10), "A_seed10": v30d_train.selection_path("A", 0, 10)}


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
    return out, report


def blended(outputs, base):
    parts = v30d_eval.BASES[base]
    frames = outputs[next(iter(parts))][2]
    return (sum(w * np.asarray(outputs[tag][0], dtype=np.float64) for tag, w in parts.items()),
            sum(w * np.asarray(outputs[tag][1], dtype=np.float64) for tag, w in parts.items()), frames)


def final_verifier(base, w):
    """All three games; the table comes from the out-of-fold network outputs of the base."""
    table = v30d_eval.table_of(v30d_eval.conditions(base))
    model = v30d_eval.fit_verifier(table, list(v30d_eval.GAMES), w)
    return dict(model=model, max_frames=lab.MAX_FRAMES, base=base, family="v27", target=f"w{w:.2f}")


def run(args):
    import joblib
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tags = sorted({"frozen"} | {tag for name in args.recipes for tag in v30d_eval.BASES[RECIPES[name]["base"]]})
    outputs, report = network_outputs(work, tags, args.device)
    if args.device == "cpu" and not report["frozen_reproduces_run_f2"]:
        raise RuntimeError("Frozen ensemble on the filled ball does not reproduce " + str(final.FROZEN_REFERENCE))
    reference = final.codes_of(REFERENCE_CSV)
    report["recipes"] = {}
    for name in args.recipes:
        spec = RECIPES[name]
        triple = blended(outputs, spec["base"])
        probabilities, blocks, n = final.fused(triple)
        np.savez_compressed(work / f"fused_{spec['base']}.npz", probabilities=probabilities, edges=triple[1], frames=triple[2])
        if spec["w"] is None:                                    # decoder only (diagnostic)
            before = final.decoded(probabilities, triple[1], triple[2], blocks, n)[0]
            after, row = before, dict(removals=0, after=c.duration_stats(before, blocks))
        else:
            saved = final_verifier(spec["base"], spec["w"])
            joblib.dump(saved, work / f"verifier_{name}.joblib")
            after, before, row = final.verify(probabilities, triple[1], triple[2], blocks, n, saved)
        graded = np.concatenate([np.arange(a, z + 1) for _, a, z in blocks])
        path = args.output_dir / f"v30d_{name}.csv"
        row.update(csv=str(path), csv_sha256=final.write_csv(path, after), rows=int(len(after)),
                   counts={k: int((final.LABELS[after] == k).sum()) for k in "HADX"},
                   graded_frames_changed_vs_v29_final=int((reference[graded] != after[graded]).sum()),
                   share_changed_vs_v29_final=float((reference[graded] != after[graded]).mean()),
                   decoder_only_changed_vs_v29_final=int((reference[graded] != before[graded]).sum()),
                   v29_final=c.duration_stats(reference, blocks))
        report["recipes"][name] = row
        print("V30D", name, json.dumps({k: row[k] for k in ("removals", "after", "graded_frames_changed_vs_v29_final", "share_changed_vs_v29_final")}), flush=True)
    report.update(seconds_total=round(time.time() - started, 1), challenge_labels_read=False)
    (work / "run.json").write_text(json.dumps(report, indent=2, default=float))
    print("V30D RUN", json.dumps({k: v for k, v in report.items() if k != "recipes"}, default=float), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", nargs="+", choices=list(RECIPES), default=["consistent", "alternative", "hedged"])
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "challenge")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    run(parser.parse_args())
