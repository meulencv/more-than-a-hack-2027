"""V28d step 2: retrain the four networks of the frozen ensemble on controlled-possession labels.

The training recipe is the frozen one (possession.v13_cv.fit_components: same six DFL games,
same clean/moderate/stress views, same architectures, seeds, steps and optimiser); the ONLY
change is the target: DFL native per-frame labels are replaced by possession.v28d_relabel's
controlled-possession labels (stoppages untouched). The original training functions are called
unchanged; the labels are swapped where the records are read.

    .venv/bin/python -m possession.v28d_train --component standard --device cpu
    .venv/bin/python -m possession.v28d_train --component physics --device cpu
    .venv/bin/python -m possession.v28d_train --component trajectory2027 --device mps
    .venv/bin/python -m possession.v28d_train --component trajectory2028 --device mps
    .venv/bin/python -m possession.v28d_train --selection      # data/work/v28d/selection.json
    (add --variant c08 | c20 | native for the sensitivity variants / the native-label control)

No Metrica game and no challenge label is read here.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from . import v6_data, v10_data, v10_relational, v13_cv
from .v6_temporal import train as train_temporal
from .v28d_relabel import GAMES, VARIANTS, controlled_labels

ROOT = Path("data/work/v28d")
COMPONENTS = {"standard": (123, 2027, 800), "physics": (156, 2027, 800),
              "trajectory2027": (None, 2027, 3000), "trajectory2028": (None, 2028, 3000)}


VARIANT = "main"          # which relabelling the records carry; set once per process by fit()


def run_name(variant="main", seed_offset=0):
    return ("" if variant == "main" else f"_{variant}") + (f"_seed{seed_offset}" if seed_offset else "")


def checkpoint_dir(variant="main", seed_offset=0):
    return ROOT / f"checkpoints{run_name(variant, seed_offset)}"


def selection_path(variant="main", seed_offset=0):
    return ROOT / f"selection{run_name(variant, seed_offset)}.json"


def relabelled_record(role, game=None, mode=None):
    """v10_data.record with the 5 Hz training target replaced by controlled possession."""
    r = v10_data.record(role, game, mode)
    if role == "unlabeled_challenge":
        raise ValueError("The challenge match is not a training record here")
    full = controlled_labels(game, VARIANT) if VARIANT != "native" else r["full_labels"]
    if len(full) != len(r["full_labels"]) or not np.array_equal(full == 2, r["full_labels"] == 2) \
            or not np.array_equal(full < 0, r["full_labels"] < 0):
        raise ValueError("Relabelled timeline differs from the native one")
    r = dict(r)
    r["native_labels"] = r["labels"]
    r["labels"] = full[r["frames"]]
    r["full_labels"] = full
    return r


def role_of(game):
    return "train" if game in v6_data.protocol()["native_train"] else "development"


def training_rows(games, width):
    """Same as v13_cv.training_rows, on relabelled records."""
    xx, yy = [], []
    for game in games:
        for mode in ("clean", "moderate", "stress"):
            r = relabelled_record(role_of(game), game, mode)
            for key in ("direct", "mirror"):
                for a, b in r["slices"]:
                    y = r["labels"][a:b]
                    xx.append(r[key][a:b, :width])
                    yy.append(y if key == "direct" else np.where(y >= 0, np.array([1, 0, 2])[np.clip(y, 0, 2)], y))
    return xx, yy


def fit(name, device, steps=None, variant="main", seed_offset=0):
    global VARIANT
    VARIANT = variant
    width, seed, default_steps = COMPONENTS[name]
    seed += seed_offset
    steps = steps or default_steps
    games = list(GAMES)
    if games != v13_cv.settings()["games"]:
        raise ValueError("Training games differ from the frozen protocol")
    path = checkpoint_dir(variant, seed_offset) / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"{path} exists; preserve it")
    torch.set_num_threads(6)
    if width:
        xx, yy = training_rows(games, width)
        net, norm = train_temporal(xx, yy, kind="boundary", steps=steps, seed=seed, device=device)
    else:
        original = v10_relational.data
        v10_relational.data = SimpleNamespace(record=relabelled_record)     # this process only
        try:
            net, norm = v10_relational.fit("trajectory", seed=seed, steps=steps, device=device, cosine=True, games=games)
        finally:
            v10_relational.data = original
    checkpoint = dict(state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()}, mean=norm.mean, std=norm.std,
                      kind="boundary" if width else "trajectory", steps=steps, seed=seed, train_games=games,
                      labels=f"v28d {variant}", relabel_rule=VARIANTS.get(variant), device=device,
                      relabel_sha256=hashlib.sha256(Path("possession/v28d_relabel.py").read_bytes()).hexdigest())
    temporary = path.with_suffix(".part")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    print("V28D COMPONENT", name, "steps", steps, "device", device, flush=True)


def write_selection(variant="main", seed_offset=0):
    """A CompetitiveBundle selection that points to the retrained checkpoints (same weights/decoder)."""
    frozen = json.loads(Path("docs/FINAL_SELECTION.json").read_text())
    selection = dict(frozen)
    selection["name"] = "V28d: frozen four-network recipe retrained on controlled-possession DFL labels"
    selection["components"] = []
    for spec in frozen["components"]:
        path = checkpoint_dir(variant, seed_offset) / Path(spec["path"]).name
        selection["components"].append(dict(spec, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    selection["relabel_variant"] = variant
    selection["relabel_rule"] = VARIANTS.get(variant)
    selection["seed_offset"] = seed_offset
    selection_path(variant, seed_offset).write_text(json.dumps(selection, indent=2))
    print("V28D SELECTION", json.dumps(selection["components"], indent=1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=list(COMPONENTS))
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--selection", action="store_true")
    parser.add_argument("--variant", choices=list(VARIANTS) + ["native"], default="main",
                        help="relabelling variant; 'native' = unchanged DFL labels (retraining-noise control)")
    parser.add_argument("--seed-offset", type=int, default=0, help="seed replicate (frozen seeds + offset)")
    args = parser.parse_args()
    if args.component:
        fit(args.component, args.device, args.steps, args.variant, args.seed_offset)
    if args.selection:
        write_selection(args.variant, args.seed_offset)
