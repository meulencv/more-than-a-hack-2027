"""V31a test-time augmentation of the FROZEN four-network ensemble (no retraining).

What inference already averages (possession.competitive_bundle.CompetitiveBundle.probabilities):
    * the team-swap view, for all four networks (tabular networks: r['mirror'], trajectory
      networks: possession.v10_relational.mirror = swap teams + point reflection);
    * the attack direction is canonicalised per half (features_v3.attack_direction), so a
      180-degree rotation of the pitch gives the same input, and an x flip equals a y flip.
The only symmetry left is therefore ONE view: the mirror across the long axis of the pitch
(y -> pitch_y - y). `check` verifies those statements numerically; `predict` runs the frozen
checkpoints on the y-mirrored open Metrica games (same degradations, same seeds as
possession.v16_transfer / v28d_data: the degraded match is mirrored, not re-degraded) on CPU;
`blend` averages identity and mirror.

    .venv/bin/python -m possession.v31a_tta check
    .venv/bin/python -m possession.v31a_tta predict            # data/work/v31a/tta/yflip/game{g}_{mode}.npz
    .venv/bin/python -m possession.v31a_tta predict_blind      # reviewer's label-blind conditions, mirrored in record space
    .venv/bin/python -m possession.v31a_tta blend              # data/work/v31a/tta/avg/game{g}_{mode}.npz

Identity view = data/work/v16/game{g}_{mode}_final.npz (the cached frozen outputs every
other study uses). No label is read; CPU only.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path("data/work/v31a/tta")
GAMES = (1, 2, 3)
MODES = ("clean", "moderate", "stress")
SELECTION = Path("docs/FINAL_SELECTION.json")


def flipped(match, x=False, y=False):
    players = match.players.astype(np.float32).copy()
    ball = match.ball.astype(np.float32).copy()
    if x:
        players[:, :, 0] = match.pitch[0] - players[:, :, 0]
        ball[:, 0] = match.pitch[0] - ball[:, 0]
    if y:
        players[:, :, 1] = match.pitch[1] - players[:, :, 1]
        ball[:, 1] = match.pitch[1] - ball[:, 1]
    return match.copy_with(players=players, ball=ball)


def degraded(game, mode, profile=None):
    from .sources import external_match, challenge_match
    from .degrade import challenge_profile
    from .v6_data import sensor_view
    profile = profile or challenge_profile(challenge_match())
    return sensor_view(external_match(game), mode, 907 + game, profile)


def path(tag, game, mode):
    if tag == "identity":
        return Path(f"data/work/v16/game{game}_{mode}_final.npz")
    return ROOT / tag / f"game{game}_{mode}.npz"


Y_NODES = [1, 3, 6, 9]            # possession.v10_data.raw_table: y, vy, y - ball_y, vy - ball_vy
Y_BALL = [1, 3]                   # ball tokens: y, vy


def flip_record(r):
    """The y mirror applied to a cached network input (possession.v12_compare.match_record).

    Exact: the tabular features of the two tabular networks do not depend on the sign of y
    (`check`: identity vs y differs by float noise only), the trajectory tokens change sign in
    their y columns. Lets any cached record (the reviewer's label-blind conditions, the
    challenge match) be mirrored without rebuilding it."""
    out = dict(r)
    nodes = r["nodes"].copy()
    nodes[:, :, Y_NODES] *= -1
    ball = r["ball_tokens"].copy()
    ball[:, Y_BALL] *= -1
    out["nodes"], out["ball_tokens"] = nodes, ball
    return out


def check():
    """x flip == y flip and both == identity at the network input (so one extra view exists)."""
    from .v12_compare import match_record
    from .v28d_data import metrica_record
    match = degraded(1, "clean")
    first = match.blocks[0][1]
    short = match.copy_with(blocks=[(1, first, first + 25 * 120)])
    records = {name: match_record(flipped(short, **kw)) for name, kw in
               (("identity", {}), ("x", dict(x=True)), ("y", dict(y=True)), ("both", dict(x=True, y=True)))}
    out = {}
    for a, b in (("identity", "both"), ("x", "y"), ("identity", "y")):
        out[f"{a}_vs_{b}"] = {key: float(np.nanmax(np.abs(records[a][key].astype(np.float64) - records[b][key].astype(np.float64))))
                              for key in ("direct", "mirror", "nodes", "ball_tokens")}
    shortcut = flip_record(records["identity"])
    out["record_flip_vs_y"] = {key: float(np.nanmax(np.abs(shortcut[key].astype(np.float64) - records["y"][key].astype(np.float64))))
                               for key in ("direct", "mirror", "nodes", "ball_tokens")}
    cached = metrica_record(1, "clean")
    n = len(records["identity"]["frames"])
    out["identity_vs_cached_record"] = {key: float(np.nanmax(np.abs(records["identity"][key].astype(np.float64) - cached[key][:n].astype(np.float64))))
                                        for key in ("nodes", "ball_tokens")}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


def predict(games=GAMES, modes=MODES, threads=6):
    import torch
    from .sources import challenge_match
    from .degrade import challenge_profile
    from .v12_compare import match_record
    from .competitive_bundle import CompetitiveBundle
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    bundle = CompetitiveBundle(json.loads(SELECTION.read_text()), "cpu")
    profile = challenge_profile(challenge_match())
    for game in games:
        for mode in modes:
            target = path("yflip", game, mode)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            start = time.time()
            r = match_record(flipped(degraded(game, mode, profile), y=True))
            built = time.time() - start
            p, e = bundle.probabilities(r)
            temporary = target.with_suffix(".part.npz")
            np.savez_compressed(temporary, probabilities=p, edges=e, frames=r["frames"])
            temporary.replace(target)
            print("TTA yflip", game, mode, p.shape, f"record {built:.0f}s total {time.time() - start:.0f}s", flush=True)


BLIND = ("interp", "interpclean", "blind3")
REVIEW = Path("data/work/review_v30d")


def predict_blind(threads=6):
    """The reviewer's label-blind ball conditions (cached records), mirrored in record space."""
    import joblib
    import torch
    from .competitive_bundle import CompetitiveBundle
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    bundle = CompetitiveBundle(json.loads(SELECTION.read_text()), "cpu")
    for game in GAMES:
        for mode in BLIND:
            target = ROOT / "yflip_blind" / f"game{game}_{mode}.npz"
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            r = flip_record(joblib.load(REVIEW / "records" / f"game{game}_{mode}.joblib"))
            p, e = bundle.probabilities(r)
            temporary = target.with_suffix(".part.npz")
            np.savez_compressed(temporary, probabilities=p, edges=e, frames=r["frames"])
            temporary.replace(target)
            print("TTA yflip blind", game, mode, p.shape, flush=True)


def blend():
    pairs = [(path("identity", g, m), path("yflip", g, m), path("avg", g, m), g, m) for g in GAMES for m in MODES]
    pairs += [(REVIEW / "outputs" / "frozen" / f"game{g}_{m}.npz", ROOT / "yflip_blind" / f"game{g}_{m}.npz",
               ROOT / "avg_blind" / f"game{g}_{m}.npz", g, m) for g in GAMES for m in BLIND]
    for first, second, target, game, mode in pairs:
        if True:
            if not second.exists():
                continue
            a, b = np.load(first), np.load(second)
            if not np.array_equal(a["frames"], b["frames"]):
                raise ValueError("Different timelines")
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, probabilities=(a["probabilities"].astype(np.float64) + b["probabilities"]) / 2,
                                edges=(a["edges"].astype(np.float64) + b["edges"]) / 2, frames=a["frames"])
            d = np.abs(a["probabilities"] - b["probabilities"])
            print("BLEND", game, mode, "mean |dp|", round(float(d.mean()), 4), "argmax changed",
                  round(float((a["probabilities"].argmax(1) != b["probabilities"].argmax(1)).mean()), 4), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["check", "predict", "predict_blind", "blend"])
    parser.add_argument("--games", type=int, nargs="*", default=list(GAMES))
    parser.add_argument("--modes", nargs="*", default=list(MODES))
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.step == "check":
        check()
    elif args.step == "predict_blind":
        predict_blind(args.threads)
    elif args.step == "predict":
        predict(args.games, args.modes, args.threads)
    else:
        blend()
