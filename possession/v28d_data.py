"""V28d data: cached network inputs for the three open Metrica games (external test only).

Same construction as possession.v16_transfer (sensor_view seed 907+game, challenge sensor
profile, v12_compare.match_record), saved once so that every retrained network is evaluated on
byte-identical inputs. The Metrica games are never used for training.

    .venv/bin/python -m possession.v28d_data --game 1 --mode clean     # one condition
    .venv/bin/python -m possession.v28d_data                            # all nine (sequential)
"""
import argparse
from pathlib import Path
import joblib
import numpy as np

ROOT = Path("data/work/v28d")
MODES = ("clean", "moderate", "stress")


def metrica_path(game, mode):
    return ROOT / "metrica_records" / f"game{game}_{mode}.joblib"


def build_metrica(game, mode):
    from .sources import external_match, challenge_match
    from .degrade import challenge_profile
    from .v6_data import sensor_view
    from .v12_compare import match_record
    path = metrica_path(game, mode)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = challenge_profile(challenge_match())
    source = external_match(game)
    match = sensor_view(source, mode, 907 + game, profile)
    r = match_record(match)
    r["n_frames"] = len(match.ball)
    r["ball_seen"] = np.isfinite(match.ball).all(axis=1)
    temporary = path.with_suffix(".part")
    joblib.dump(r, temporary, compress=1)
    temporary.replace(path)
    print("METRICA RECORD", game, mode, r["direct"].shape, flush=True)
    return path


def metrica_record(game, mode):
    return joblib.load(metrica_path(game, mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, choices=[1, 2, 3])
    parser.add_argument("--mode", choices=MODES)
    args = parser.parse_args()
    for game in ([args.game] if args.game else [1, 2, 3]):
        for mode in ([args.mode] if args.mode else MODES):
            build_metrica(game, mode)
