"""V31c: the four networks of the frozen ensemble trained IN-DOMAIN on the compromise target M
with a LABEL-BLIND ball on the Metrica games.

The frozen recipe is kept (architectures, seeds, steps, optimiser, the six DFL games in their
clean / moderate / stress views; the original training functions are called unchanged). As in
V30d the open Metrica sample games are added (each sequence repeated METRICA_REPEAT times), but

* the target is M (possession.v31c_data): sandwiched possessions shorter than 2.0 s absorbed, on
  the DFL native labels and on the Metrica "alternative" reference alike;
* the Metrica views are the label-blind ones of possession.v31c_data (blind fill of every gap,
  random removal with the challenge gap lengths): ball availability carries no label timing.

Targets (--target):
    M     absorb 2.0 s   (the a-priori compromise; the candidate)
    M15   absorb 1.5 s   (sensitivity only)
    M30   absorb 3.0 s   (sensitivity only)
    A     absorb 0       (control: native DFL + alternative Metrica reference on the SAME
                          label-blind views; isolates the effect of the target from the ball)

Leave-one-game-out: --held g trains without Metrica game g; --held 0 trains on all three
(final checkpoints for the challenge match, data/work/v31c/final).

    .venv/bin/python -m possession.v31c_train --target M --held 1 --all
    .venv/bin/python -m possession.v31c_train --target M --held 1 --seed-offset 10 --all
    .venv/bin/python -m possession.v31c_train --target M --held 1 --component standard --device cpu

No challenge label is read. The challenge match contributes nothing to training here.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from . import v6_data, v10_data, v10_relational, v13_cv, v31c_data
from .v6_temporal import train as train_temporal, segment_weights, boundary_targets

ROOT = Path("data/work/v31c")
COMPONENTS = {"standard": (123, 2027, 800), "physics": (156, 2027, 800),
              "trajectory2027": (None, 2027, 3000), "trajectory2028": (None, 2028, 3000)}
TARGETS = {"M": 2.0, "M15": 1.5, "M30": 3.0, "A": 0.0}
METRICA_REPEAT = 2
SWAP = np.array([1, 0, 2])
ABSORB = None             # set once per process by fit()
_dfl_cache = {}


def run_name(target, held, seed_offset=0):
    return f"{target}_h{held}" + (f"_seed{seed_offset}" if seed_offset else "")


def checkpoint_dir(target, held, seed_offset=0):
    if held == 0:
        return ROOT / "final" / (target + (f"_seed{seed_offset}" if seed_offset else ""))
    return ROOT / "checkpoints" / run_name(target, held, seed_offset)


def selection_path(target, held, seed_offset=0):
    return ROOT / "selections" / f"{run_name(target, held, seed_offset)}.json"


def dfl_games():
    return v13_cv.settings()["games"]


def dfl_record(role, game=None, mode=None):
    """v10_data.record with the 5 Hz training target replaced by M (absorb ABSORB seconds)."""
    if role == "unlabeled_challenge":
        raise ValueError("The challenge match is not a training record here")
    r = dict(v10_data.record(role, game, mode))
    if not np.array_equal(r["labels"], r["full_labels"][r["frames"]]):
        raise ValueError("5 Hz labels are not the sampled 25 Hz labels")
    if ABSORB:
        if game not in _dfl_cache:
            _dfl_cache[game] = v31c_data.dfl_target(r["full_labels"], r["blocks"], ABSORB)
        full = _dfl_cache[game]
        r["native_labels"] = r["labels"]
        r["labels"] = full[r["frames"]]
        r["full_labels"] = full
    return r


def role_of(game):
    return "train" if game in v6_data.protocol()["native_train"] else "development"


def metrica_games(held):
    return [g for g in v31c_data.GAMES if g != held]


def metrica_records(held):
    out = []
    for game in metrica_games(held):
        target = v31c_data.metrica_target(game, ABSORB) if ABSORB else None
        for view in v31c_data.VIEWS:
            r = v31c_data.record(game, view)
            y = target[r["frames"]] if ABSORB else r["labels_A"]
            if ABSORB == v31c_data.ABSORB_S and not np.array_equal(y, r["labels_M"]):
                raise ValueError("Stored target differs from the rule")
            out.append((r, y.astype(np.int64)))
    return out


def mirrored(y):
    return np.where(y >= 0, SWAP[np.clip(y, 0, 2)], y)


def temporal_rows(held, width, repeat=METRICA_REPEAT):
    """DFL rows as possession.v13_cv / v28d_train build them (target M) + the Metrica rows, repeated."""
    xx, yy = [], []
    for game in dfl_games():
        for mode in ("clean", "moderate", "stress"):
            r = dfl_record(role_of(game), game, mode)
            for key in ("direct", "mirror"):
                for a, b in r["slices"]:
                    y = r["labels"][a:b]
                    xx.append(r[key][a:b, :width])
                    yy.append(y if key == "direct" else mirrored(y))
    native = len(xx)
    for r, labels in metrica_records(held):
        for a, b in r["slices"]:
            y = labels[a:b]
            for _ in range(repeat):
                xx += [r["direct"][a:b, :width], r["mirror"][a:b, :width]]
                yy += [y, mirrored(y)]
    return xx, yy, native


def relational_sequences(held, repeat=METRICA_REPEAT):
    """Replacement of possession.v10_relational.sequences: DFL samples + Metrica samples."""
    original = v10_relational.sequences

    def sequences(games=None):
        output, normalizer = original(games)
        for r, labels in metrica_records(held):
            for a, b in r["slices"]:
                y = labels[a:b]
                item = dict(nodes=r["nodes"][a:b], ball=r["ball_tokens"][a:b], team=r["team"], y=y,
                            segments=segment_weights(y), edges=boundary_targets(y),
                            changes=np.flatnonzero(y[1:] != y[:-1]) + 1)
                output += [item] * repeat
        return output, normalizer
    return sequences


def fit(name, target, held, device, seed_offset=0, steps=None):
    global ABSORB
    ABSORB = TARGETS[target]
    width, seed, default_steps = COMPONENTS[name]
    seed += seed_offset
    steps = steps or default_steps
    path = checkpoint_dir(target, held, seed_offset) / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"{path} exists; preserve it")
    torch.set_num_threads(4)
    games = list(dfl_games())
    if width:
        xx, yy, native = temporal_rows(held, width)
        share = 1 - native / len(xx)
        net, norm = train_temporal(xx, yy, kind="boundary", steps=steps, seed=seed, device=device)
    else:
        saved = v10_relational.data, v10_relational.sequences
        v10_relational.data = SimpleNamespace(record=dfl_record)                  # this process only
        v10_relational.sequences = relational_sequences(held)
        try:
            count = len(v10_relational.sequences(games)[0])
            share = 1 - 36 / count
            net, norm = v10_relational.fit("trajectory", seed=seed, steps=steps, device=device, cosine=True, games=games)
        finally:
            v10_relational.data, v10_relational.sequences = saved
    checkpoint = dict(state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()}, mean=norm.mean, std=norm.std,
                      kind="boundary" if width else "trajectory", steps=steps, seed=seed, train_games=games,
                      metrica_train_games=metrica_games(held), metrica_views=list(v31c_data.VIEWS), metrica_repeat=METRICA_REPEAT,
                      metrica_sequence_share=share, target=target, absorb_s=ABSORB, device=device,
                      data_sha256=hashlib.sha256(Path("possession/v31c_data.py").read_bytes()).hexdigest())
    temporary = path.with_suffix(".part")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    print("V31C COMPONENT", name, target, "held", held, "steps", steps, "device", device, "metrica share", round(share, 3), flush=True)


def write_selection(target, held, seed_offset=0):
    frozen = json.loads(Path("docs/FINAL_SELECTION.json").read_text())
    selection = dict(frozen)
    selection["name"] = f"V31c: frozen four-network recipe trained in-domain, label-blind ball, target {target}, Metrica game held out: {held or 'none'}"
    selection["components"] = []
    for spec in frozen["components"]:
        path = checkpoint_dir(target, held, seed_offset) / Path(spec["path"]).name
        selection["components"].append(dict(spec, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    selection.update(target=target, absorb_s=TARGETS[target], held_out_metrica_game=held, seed_offset=seed_offset,
                     metrica_repeat=METRICA_REPEAT)
    path = selection_path(target, held, seed_offset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2))
    print("V31C SELECTION", path, flush=True)


def fit_all(target, held, seed_offset=0):
    """The four components as four processes (two on CPU, two on MPS), then the selection."""
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, (width, _, _) in COMPONENTS.items():
        if (checkpoint_dir(target, held, seed_offset) / f"{name}.pt").exists():
            continue
        command = [sys.executable, "-m", "possession.v31c_train", "--target", target, "--held", str(held), "--component", name,
                   "--device", "cpu" if width else "mps", "--seed-offset", str(seed_offset)]
        log = open(logs / f"train_{run_name(target, held, seed_offset)}_{name}.log", "w")
        jobs.append((name, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)))
    failed = [name for name, job in jobs if job.wait() != 0]
    if failed:
        raise RuntimeError(f"Training failed: {failed}")
    write_selection(target, held, seed_offset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=list(TARGETS), required=True)
    parser.add_argument("--held", type=int, choices=[0, 1, 2, 3], required=True, help="Metrica game left out (0 = none, final)")
    parser.add_argument("--component", choices=list(COMPONENTS))
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--selection", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        fit_all(args.target, args.held, args.seed_offset)
    if args.component:
        fit(args.component, args.target, args.held, args.device, args.seed_offset, args.steps)
    if args.selection:
        write_selection(args.target, args.held, args.seed_offset)
