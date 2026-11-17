"""V31b step 2: retrain the four networks of the frozen ensemble on the compromise target M.

Exactly the V28d recipe (possession.v28d_train: the six DFL games, clean/moderate/stress views,
same architectures, seeds, steps, optimiser and augmentations as the frozen ensemble); the ONLY
change is the target: DFL native per-frame labels with possession.v31b_relabel's a-priori
duration rule applied (m2: T-X-T possessions shorter than 2 s absorbed; m3: shorter than 3 s).
No Metrica game and no challenge label is read: all three Metrica games stay held out.

    .venv/bin/python -m possession.v31b_train --variant m2 --component standard --device cpu
    .venv/bin/python -m possession.v31b_train --variant m2 --component physics --device cpu
    .venv/bin/python -m possession.v31b_train --variant m2 --component trajectory2027 --device mps
    .venv/bin/python -m possession.v31b_train --variant m2 --component trajectory2028 --device mps
    .venv/bin/python -m possession.v31b_train --variant m2 --selection     # data/work/v31b/selection_m2.json
    (add --seed-offset 10 for the second seed; --variant m3 for the 3 s variant)
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from . import v10_relational, v13_cv
from . import v28d_train as t
from .v6_temporal import train as train_temporal
from .v31b_relabel import GAMES, VARIANTS, target_labels

ROOT = Path("data/work/v31b")
COMPONENTS = t.COMPONENTS


def run_name(variant, seed_offset=0):
    return variant + (f"_seed{seed_offset}" if seed_offset else "")


def checkpoint_dir(variant, seed_offset=0):
    return ROOT / "checkpoints" / run_name(variant, seed_offset)


def selection_path(variant, seed_offset=0):
    return ROOT / f"selection_{run_name(variant, seed_offset)}.json"


def fit(name, device, variant, seed_offset=0, steps=None):
    if variant not in VARIANTS:
        raise ValueError("Unknown target")
    t.VARIANT = variant                           # v28d_train.relabelled_record reads these two names
    t.controlled_labels = target_labels
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
    torch.set_num_threads(4)
    if width:
        xx, yy = t.training_rows(games, width)
        net, norm = train_temporal(xx, yy, kind="boundary", steps=steps, seed=seed, device=device)
    else:
        original = v10_relational.data
        v10_relational.data = SimpleNamespace(record=t.relabelled_record)     # this process only
        try:
            net, norm = v10_relational.fit("trajectory", seed=seed, steps=steps, device=device, cosine=True, games=games)
        finally:
            v10_relational.data = original
    checkpoint = dict(state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()}, mean=norm.mean, std=norm.std,
                      kind="boundary" if width else "trajectory", steps=steps, seed=seed, train_games=games,
                      labels=f"v31b {variant}", relabel_rule=dict(rule="v27_lab.duration_rule", absorb_s=VARIANTS[variant], island_s=0, edge_s=0),
                      device=device, relabel_sha256=hashlib.sha256(Path("possession/v31b_relabel.py").read_bytes()).hexdigest())
    temporary = path.with_suffix(".part")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    print("V31B COMPONENT", variant, name, "seed", seed, "steps", steps, "device", device, flush=True)


def write_selection(variant, seed_offset=0):
    """A CompetitiveBundle selection that points to the retrained checkpoints (same weights/decoder)."""
    frozen = json.loads(Path("docs/FINAL_SELECTION.json").read_text())
    selection = dict(frozen)
    selection["name"] = f"V31b: frozen four-network recipe retrained on DFL labels moved to the compromise target ({variant})"
    selection["components"] = []
    for spec in frozen["components"]:
        path = checkpoint_dir(variant, seed_offset) / Path(spec["path"]).name
        selection["components"].append(dict(spec, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    selection["relabel_variant"] = variant
    selection["relabel_rule"] = dict(rule="v27_lab.duration_rule", absorb_s=VARIANTS[variant], island_s=0, edge_s=0)
    selection["seed_offset"] = seed_offset
    selection_path(variant, seed_offset).write_text(json.dumps(selection, indent=2))
    print("V31B SELECTION", selection_path(variant, seed_offset), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=list(COMPONENTS))
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--selection", action="store_true")
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--seed-offset", type=int, default=0, help="seed replicate (frozen seeds + offset)")
    args = parser.parse_args()
    if args.component:
        fit(args.component, args.device, args.variant, args.seed_offset, args.steps)
    if args.selection:
        write_selection(args.variant, args.seed_offset)
