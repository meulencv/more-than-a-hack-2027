"""Strict frame-indexed loading of the automatic tracking; cache is local/private."""
from pathlib import Path
import numpy as np
from games import asset_paths, source_frames
from starter.baseline import team_of_each_track


def load_tracking(source="tactical", cache_dir=None):
    paths = asset_paths(source=source)
    slots = team_of_each_track(paths["atd_meta"])
    n = source_frames(source)
    raw = Path(paths["atd_raw"])
    signature = np.array([raw.stat().st_size, raw.stat().st_mtime_ns])
    cache = Path(cache_dir) / (source + "_tracking.npz") if cache_dir else None
    if cache and cache.exists():
        with np.load(cache) as saved:
            if np.array_equal(saved["signature"], signature):
                return saved["players"], saved["ball"], slots
    players = np.full((n, len(slots), 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    count = 0
    with raw.open() as handle:
        for count, line in enumerate(handle, 1):
            parts = line.strip().split(":")
            if len(parts) != 3 or int(parts[0]) != count:
                raise ValueError("Tracking has a malformed or unordered frame at {}".format(count))
            values = np.fromstring(parts[1].replace(";", ","), sep=",", dtype=np.float32)
            if values.size != len(slots) * 3:
                raise ValueError("Unexpected track slot count at {}".format(count))
            players[count - 1] = values.reshape(-1, 3)
            ball[count - 1] = np.fromstring(parts[2], sep=",", dtype=np.float32)
    if count != n:
        raise ValueError("Expected {} records; found {}".format(n, count))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, signature=signature, players=players, ball=ball)
    return players, ball, slots
