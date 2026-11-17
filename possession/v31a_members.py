"""V31a: the four frozen networks' outputs kept SEPARATE (the caches only hold their mean).

Same checkpoints, same cached inputs and same per-network view averaging as
possession.competitive_bundle.CompetitiveBundle.probabilities; the only difference is that the
four probability tracks are saved before they are averaged, so a second stage / verifier can
see where the networks disagree. Nothing is retrained and no label is read. CPU.

    .venv/bin/python -m possession.v31a_members predict           # 9 standard + 9 label-blind conditions
        -> data/work/v31a/members/game{g}_{mode}.npz   members (4, T, 3), edges (T,), frames (T,)
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path("data/work/v31a/members")
GAMES = (1, 2, 3)
MODES = ("clean", "moderate", "stress")
BLIND = ("interp", "interpclean", "blind3")
SELECTION = Path("docs/FINAL_SELECTION.json")


def member_outputs(bundle, r):
    """[(name, probabilities, boundary or None)] per network, exactly as the bundle computes them."""
    from .v10_relational import predict as relational_predict
    from .v6_temporal import predict as temporal_predict
    from .v6_experiment import subsequences
    out = []
    for spec, net, norm in bundle.models:
        if spec["kind"] == "trajectory":
            p, e, _ = relational_predict(net, norm, r, bundle.device)
        else:
            width = len(norm.mean)
            a, e = temporal_predict(net, norm, subsequences(r["direct"][:, :width], r["slices"]), bundle.device)
            b, f = temporal_predict(net, norm, subsequences(r["mirror"][:, :width], r["slices"]), bundle.device)
            p = (np.concatenate(a) + np.concatenate(b)[:, [1, 0, 2]]) / 2
            e = (np.concatenate(e) + np.concatenate(f)) / 2
        out.append((Path(spec["path"]).stem, p, e if spec["boundary_head"] else None))
    return out


def record_of(game, mode):
    import joblib
    if mode in BLIND:
        return joblib.load(f"data/work/review_v30d/records/game{game}_{mode}.joblib")
    from .v28d_data import metrica_record
    return metrica_record(game, mode)


def cached_mean(game, mode):
    if mode in BLIND:
        return Path(f"data/work/review_v30d/outputs/frozen/game{game}_{mode}.npz")
    return Path(f"data/work/v16/game{game}_{mode}_final.npz")


def path(game, mode):
    return ROOT / f"game{game}_{mode}.npz"


def predict(threads=6, modes=MODES + BLIND):
    import torch
    from .competitive_bundle import CompetitiveBundle
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    bundle = CompetitiveBundle(json.loads(SELECTION.read_text()), "cpu")
    ROOT.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        for game in GAMES:
            if path(game, mode).exists():
                continue
            r = record_of(game, mode)
            members = member_outputs(bundle, r)
            p = np.stack([m[1] for m in members]).astype(np.float32)
            edges = np.mean([m[2] for m in members if m[2] is not None], axis=0).astype(np.float32)
            saved = np.load(cached_mean(game, mode))
            if not np.array_equal(saved["frames"], r["frames"]):
                raise ValueError("Different timelines")
            gap = float(np.abs(p.mean(0) - saved["probabilities"]).max())
            temporary = path(game, mode).with_suffix(".part.npz")
            np.savez_compressed(temporary, members=p, edges=edges, frames=r["frames"], names=np.array([m[0] for m in members]))
            temporary.replace(path(game, mode))
            print("MEMBERS", game, mode, p.shape, "max |mean - cached ensemble|", round(gap, 5), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["predict"])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--modes", nargs="*", default=list(MODES + BLIND))
    args = parser.parse_args()
    predict(args.threads, tuple(args.modes))
