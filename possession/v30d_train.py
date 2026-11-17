"""V30d: the four networks of the frozen ensemble trained IN-DOMAIN, for both readings.

The frozen recipe is kept (possession.v13_cv.fit_components: architectures, seeds, steps,
optimiser, the six DFL games in clean/moderate/stress views; the original training functions
are called unchanged). Two things change:

* the provider's own games are added: the open Metrica sample games, in the four degraded
  views of possession.v30d_data (synthetic dead-ball sightings + sensor degradation), each
  sequence repeated METRICA_REPEAT times so that 2 (or 3) games out of 8 (9) weigh about as
  much as the six DFL games in the uniform sequence sampler of the training loops;
* two target sets:
      C "consistent"   DFL relabelled to controlled possession (V28d labels, data/work/v28d/labels)
                       + Metrica "consistent" reference (one-touch interceptions are not possessions)
      A "alternative"  DFL native labels + Metrica "alternative" reference (they are)

Leave-one-game-out: --held g trains without Metrica game g (the networks that are scored on g).
--held 0 trains on all three (final checkpoints for the challenge match, data/work/v30d/final).

    .venv/bin/python -m possession.v30d_train --target C --held 1 --component standard --device cpu
    .venv/bin/python -m possession.v30d_train --target C --held 1 --component trajectory2027 --device mps
    .venv/bin/python -m possession.v30d_train --target C --held 1 --selection
    .venv/bin/python -m possession.v30d_train --target C --held 1 --all      # the four components + selection

No challenge label is read. The challenge match contributes nothing to training here.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from . import v10_relational, v13_cv, v28d_train, v30d_data
from .v6_temporal import train as train_temporal, segment_weights, boundary_targets
from .v28d_relabel import GAMES as DFL_GAMES

ROOT = Path("data/work/v30d")
COMPONENTS = v28d_train.COMPONENTS
TARGETS = {"C": dict(dfl_variant="main", metrica="labels_C"), "A": dict(dfl_variant="native", metrica="labels_A")}
METRICA_REPEAT = 2
SWAP = np.array([1, 0, 2])


def run_name(target, held, seed_offset=0, repeat=METRICA_REPEAT):
    return f"{target}_h{held}" + (f"_seed{seed_offset}" if seed_offset else "") + (f"_r{repeat}" if repeat != METRICA_REPEAT else "")


def checkpoint_dir(target, held, seed_offset=0, repeat=METRICA_REPEAT):
    if held == 0 and seed_offset == 0 and repeat == METRICA_REPEAT:
        return ROOT / "final" / target
    return ROOT / "checkpoints" / run_name(target, held, seed_offset, repeat)


def selection_path(target, held, seed_offset=0, repeat=METRICA_REPEAT):
    return ROOT / "selections" / f"{run_name(target, held, seed_offset, repeat)}.json"


def metrica_games(held):
    return [g for g in v30d_data.GAMES if g != held]


def metrica_records(held):
    return [v30d_data.record(game, view) for game in metrica_games(held) for view in v30d_data.VIEWS]


def mirrored(y):
    return np.where(y >= 0, SWAP[np.clip(y, 0, 2)], y)


def temporal_rows(target, held, width, repeat):
    """DFL rows exactly as V28d builds them (+ native control) and the Metrica rows, repeated."""
    v28d_train.VARIANT = TARGETS[target]["dfl_variant"]
    xx, yy = v28d_train.training_rows(list(DFL_GAMES), width)
    native = len(xx)
    for r in metrica_records(held):
        labels = r[TARGETS[target]["metrica"]]
        for a, b in r["slices"]:
            y = labels[a:b].astype(np.int64)
            for _ in range(repeat):
                xx += [r["direct"][a:b, :width], r["mirror"][a:b, :width]]
                yy += [y, mirrored(y)]
    return xx, yy, native


def relational_sequences(target, held, repeat):
    """Replacement of possession.v10_relational.sequences: DFL samples + Metrica samples."""
    original = v10_relational.sequences

    def sequences(games=None):
        output, normalizer = original(games)
        for r in metrica_records(held):
            labels = r[TARGETS[target]["metrica"]]
            for a, b in r["slices"]:
                y = labels[a:b].astype(np.int64)
                item = dict(nodes=r["nodes"][a:b], ball=r["ball_tokens"][a:b], team=r["team"], y=y,
                            segments=segment_weights(y), edges=boundary_targets(y),
                            changes=np.flatnonzero(y[1:] != y[:-1]) + 1)
                output += [item] * repeat
        return output, normalizer
    return sequences


def fit(name, target, held, device, seed_offset=0, repeat=METRICA_REPEAT, steps=None):
    width, seed, default_steps = COMPONENTS[name]
    seed += seed_offset
    steps = steps or default_steps
    if list(DFL_GAMES) != v13_cv.settings()["games"]:
        raise ValueError("DFL games differ from the frozen protocol")
    path = checkpoint_dir(target, held, seed_offset, repeat) / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"{path} exists; preserve it")
    torch.set_num_threads(4)
    v28d_train.VARIANT = TARGETS[target]["dfl_variant"]
    if width:
        xx, yy, native = temporal_rows(target, held, width, repeat)
        share = 1 - native / len(xx)
        net, norm = train_temporal(xx, yy, kind="boundary", steps=steps, seed=seed, device=device)
    else:
        from types import SimpleNamespace
        saved = v10_relational.data, v10_relational.sequences
        v10_relational.data = SimpleNamespace(record=v28d_train.relabelled_record)      # this process only
        v10_relational.sequences = relational_sequences(target, held, repeat)
        try:
            count = len(v10_relational.sequences(list(DFL_GAMES))[0])
            share = 1 - 36 / count
            net, norm = v10_relational.fit("trajectory", seed=seed, steps=steps, device=device, cosine=True, games=list(DFL_GAMES))
        finally:
            v10_relational.data, v10_relational.sequences = saved
    checkpoint = dict(state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()}, mean=norm.mean, std=norm.std,
                      kind="boundary" if width else "trajectory", steps=steps, seed=seed, train_games=list(DFL_GAMES),
                      metrica_train_games=metrica_games(held), metrica_views=list(v30d_data.VIEWS), metrica_repeat=repeat,
                      metrica_sequence_share=share, target=target, labels=TARGETS[target], device=device,
                      data_sha256=hashlib.sha256(Path("possession/v30d_data.py").read_bytes()).hexdigest())
    temporary = path.with_suffix(".part")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    print("V30D COMPONENT", name, target, "held", held, "steps", steps, "device", device, "metrica share", round(share, 3), flush=True)


def write_selection(target, held, seed_offset=0, repeat=METRICA_REPEAT):
    frozen = json.loads(Path("docs/FINAL_SELECTION.json").read_text())
    selection = dict(frozen)
    selection["name"] = f"V30d: frozen four-network recipe trained in-domain, target {target}, Metrica game held out: {held or 'none'}"
    selection["components"] = []
    for spec in frozen["components"]:
        path = checkpoint_dir(target, held, seed_offset, repeat) / Path(spec["path"]).name
        selection["components"].append(dict(spec, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    selection.update(target=target, held_out_metrica_game=held, seed_offset=seed_offset, metrica_repeat=repeat)
    path = selection_path(target, held, seed_offset, repeat)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2))
    print("V30D SELECTION", path, flush=True)


def fit_all(target, held, seed_offset=0, repeat=METRICA_REPEAT):
    """The four components as four processes (two on CPU, two on MPS), then the selection."""
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, (width, _, _) in COMPONENTS.items():
        if (checkpoint_dir(target, held, seed_offset, repeat) / f"{name}.pt").exists():
            continue
        command = [sys.executable, "-m", "possession.v30d_train", "--target", target, "--held", str(held), "--component", name,
                   "--device", "cpu" if width else "mps", "--seed-offset", str(seed_offset), "--repeat", str(repeat)]
        log = open(logs / f"train_{run_name(target, held, seed_offset, repeat)}_{name}.log", "w")
        jobs.append((name, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)))
    failed = [name for name, job in jobs if job.wait() != 0]
    if failed:
        raise RuntimeError(f"Training failed: {failed}")
    write_selection(target, held, seed_offset, repeat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=list(TARGETS), required=True)
    parser.add_argument("--held", type=int, choices=[0, 1, 2, 3], required=True, help="Metrica game left out (0 = none, final)")
    parser.add_argument("--component", choices=list(COMPONENTS))
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=METRICA_REPEAT, help="copies of each Metrica sequence in the sampler")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--selection", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        fit_all(args.target, args.held, args.seed_offset, args.repeat)
    if args.component:
        fit(args.component, args.target, args.held, args.device, args.seed_offset, args.repeat, args.steps)
    if args.selection:
        write_selection(args.target, args.held, args.seed_offset, args.repeat)
