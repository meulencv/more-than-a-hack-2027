"""V31b step 1: DFL native per-frame possession -> the compromise target "M" (a-priori rule).

M (see data/work/review_v30b/checks.py, part C): the touch-level reading of possession with every
possession shorter than ABSORB_S that is sandwiched between two possessions of the other team
absorbed into them (possession.v27_lab.duration_rule, absorb_s = 2.0, island_s = 0, edge_s = 0).
On the open Metrica games that single labelling scores >= 0.85 under BOTH readings of the
definition. The same rule is applied here to the DFL/Sportec native labels at 25 Hz:

    m2   absorb_s = 2.0     (the a-priori constant; main target "M_dfl")
    m3   absorb_s = 3.0     (ONE additional a-priori variant "M_dfl3", because DFL native labels
                             are more fragmented than the Metrica touch-level reference)

Nothing is matched to Metrica statistics, no tracking is read (pure duration rule on the label
sequence), stoppage (D) and unlabelled frames are never modified. No Metrica label and no
challenge label is used to build the targets; `report` only prints the Metrica M statistics next
to the DFL ones.

    .venv/bin/python -m possession.v31b_relabel write      # data/work/v31b/labels_{m2,m3}/{game}.npz
    .venv/bin/python -m possession.v31b_relabel report     # data/work/v31b/relabel_report.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import v27_core as c
from .v27_lab import duration_rule

ROOT = Path("data/work/v31b")
GAMES = ("J03WMX", "J03WN1", "J03WOH", "J03WPY", "J03WOY", "J03WQQ")
VARIANTS = {"m2": 2.0, "m3": 3.0}
KEYS = ("spells", "per_live_minute", "mean_s", "median_s", "share_lt_1s", "share_lt_3s", "dead_share")


def labels_dir(variant):
    return ROOT / f"labels_{variant}"


def compromise(labels, absorb_s):
    """labels: per-frame codes (0 H, 1 A, 2 D, negative = not graded). Returns the M labelling."""
    labels = np.asarray(labels)
    codes = np.where(labels >= 0, labels, 3).astype(np.int8)
    blocks = c.graded_blocks(labels)
    segs = duration_rule(c.segments(codes, blocks), absorb_s=absorb_s, island_s=0, edge_s=0)
    new = c.to_codes(segs, len(codes))
    out = labels.copy()
    graded = labels >= 0
    out[graded] = new[graded]
    if not np.array_equal(out == 2, labels == 2) or not np.array_equal(out < 0, labels < 0):
        raise ValueError("The rule must not touch stoppages or unknown frames")
    return out


def stats(labels):
    return c.duration_stats(np.where(labels >= 0, labels, 3).astype(np.int8), c.graded_blocks(labels))


def native_labels(game):
    from .sportec_native import external_native_match
    return external_native_match(game).meta["labels"]


def write():
    for variant, absorb_s in VARIANTS.items():
        labels_dir(variant).mkdir(parents=True, exist_ok=True)
    for game in GAMES:
        native = native_labels(game)
        for variant, absorb_s in VARIANTS.items():
            new = compromise(native, absorb_s)
            np.savez_compressed(labels_dir(variant) / f"{game}.npz", labels=new, native=native, absorb_s=absorb_s)
            live = (native == 0) | (native == 1)
            print("V31B RELABEL", variant, game, {k: round(float(v), 3) for k, v in stats(new).items()},
                  "live frames changed", round(float((new != native)[live].mean()), 4), flush=True)


def target_labels(game, variant):
    return np.load(labels_dir(variant) / f"{game}.npz")["labels"]


def report():
    """DFL native / M_dfl / M_dfl3 / V28d 'C' statistics next to the Metrica references and M."""
    rows = {"dfl": {}, "metrica": {}}
    for game in GAMES:
        saved = {v: np.load(labels_dir(v) / f"{game}.npz") for v in VARIANTS}
        native = saved["m2"]["native"]
        row = dict(native=stats(native))
        for variant in VARIANTS:
            row[variant] = stats(saved[variant]["labels"])
            live = (native == 0) | (native == 1)
            row[variant]["live_frames_changed"] = float((saved[variant]["labels"] != native)[live].mean())
        v28d = Path("data/work/v28d/labels") / f"{game}.npz"
        if v28d.exists():
            row["v28d_C"] = stats(np.load(v28d)["labels"])
        rows["dfl"][game] = row
    for game in (1, 2, 3):
        meta = np.load(c.ROOT / f"metrica_{game}.npz")
        row = dict(consistent=stats(meta["consistent"]), alternative=stats(meta["alternative"]))
        gt = {r: c.spells_of(c.segments(np.where(meta[r] >= 0, meta[r], 3).astype(np.int8), [tuple(int(v) for v in b) for b in meta["blocks"]]))
              for r in ("consistent", "alternative")}
        for limit in (1.5, 2.0, 3.0):
            m = compromise(meta["alternative"], limit)
            row[f"M_{limit}"] = dict(stats(m), **{"spell_f1_vs_" + r: c.official(meta[r], np.where(m >= 0, m, 3).astype(np.int8))["spell_f1"]
                                                    for r in gt})
        rows["metrica"][str(game)] = row
    mean = {}
    for side, names in (("dfl", ["native", "m2", "m3", "v28d_C"]), ("metrica", ["consistent", "alternative", "M_1.5", "M_2.0", "M_3.0"])):
        for name in names:
            values = [r[name] for r in rows[side].values() if name in r]
            mean[f"{side} {name}"] = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "relabel_report.json").write_text(json.dumps(dict(variants=VARIANTS, games=rows, mean=mean), indent=1))
    for side in rows:
        for game, row in rows[side].items():
            for name, v in row.items():
                print(f"{side:<8}{game:<8}{name:<12}" + " ".join(f"{k} {v[k]:.3f}" for k in v if k in ("per_live_minute", "share_lt_1s", "share_lt_3s", "mean_s",
                      "live_frames_changed", "spell_f1_vs_consistent", "spell_f1_vs_alternative")), flush=True)
    for name, v in mean.items():
        print(f"MEAN {name:<22}" + " ".join(f"{k} {v[k]:.3f}" for k in v if k in ("per_live_minute", "share_lt_1s", "share_lt_3s", "mean_s", "live_frames_changed",
              "spell_f1_vs_consistent", "spell_f1_vs_alternative")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["write", "report"])
    {"write": write, "report": report}[parser.parse_args().step]()
