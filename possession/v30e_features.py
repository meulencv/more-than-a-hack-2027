"""V30e: per-frame multimodal features of the challenge match on the tactical timeline.

    .venv/bin/python -m possession.v30e_features build

Needs data/work/v30e/audio_{tactical,broadcast}.npz (possession.v30e_audio extract) and
data/work/v30e/video_{tactical,broadcast}.npz (possession.v30e_video). Writes, under
data/work/v30e/features/:

    audio.npz, video.npz, ball.npz, kinematics.npz, camera.npz   one float32 array per feature,
                                                                 length 199319 (row = tactical video frame)
    events.json                                                   whistles, scoreboard, broadcast cuts
    README.md

Nothing here reads a possession label of the challenge match (public or private), and no
feature uses an output of the possession networks.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from games import GAME, GAMES, half_video_blocks, source_frames

ROOT = Path("data/work/v30e")
OUT = ROOT / "features"
FPS = 25
CAP = 1500
ALIGNMENT = Path("data/work/v11/challenge_alignment.json")
WHISTLE_DB = 12.0               # dual-tone prominence that both recordings must reach
WHISTLE_MIN = 3                 # frames above it
WHISTLE_GAP = 8                 # blasts closer than this are one whistle
GROUPS = ("audio", "video", "ball", "kinematics", "camera")


# -- small helpers ---------------------------------------------------------------------------

def blocks_and_offsets():
    from . import v24_camera_fusion as v24
    blocks = list(half_video_blocks(GAME))
    return blocks, v24.half_offsets(json.loads(ALIGNMENT.read_text())["runs"], blocks)


def rolling_mean(x, window):
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    c = np.cumsum(np.r_[0.0, np.where(ok, x, 0.0)])
    k = np.cumsum(np.r_[0, ok.astype(np.int64)])
    h = window // 2
    lo = np.clip(np.arange(len(x)) - h, 0, len(x))
    hi = np.clip(np.arange(len(x)) + h + 1, 0, len(x))
    count = k[hi] - k[lo]
    return np.where(count > 0, (c[hi] - c[lo]) / np.maximum(count, 1), np.nan)


def rolling_median_coarse(x, window, step=25):
    """Running median on a coarse grid (step frames), linearly interpolated back."""
    grid = np.arange(0, len(x), step)
    h = window // 2
    values = np.array([np.nanmedian(x[max(0, g - h):g + h + 1]) if np.isfinite(x[max(0, g - h):g + h + 1]).any() else np.nan for g in grid])
    ok = np.isfinite(values)
    return np.interp(np.arange(len(x)), grid[ok], values[ok]) if ok.any() else np.full(len(x), np.nan)


def since_until(marks, n):
    """Frames since the last / until the next marked frame, capped at CAP."""
    marks = np.sort(np.asarray(marks, dtype=np.int64))
    t = np.arange(n)
    if not len(marks):
        return np.full(n, CAP, np.float32), np.full(n, CAP, np.float32)
    j = np.searchsorted(marks, t, side="right")
    since = np.where(j > 0, t - marks[np.clip(j - 1, 0, len(marks) - 1)], CAP)
    k = np.searchsorted(marks, t, side="left")
    until = np.where(k < len(marks), marks[np.clip(k, 0, len(marks) - 1)] - t, CAP)
    return np.minimum(since, CAP).astype(np.float32), np.minimum(until, CAP).astype(np.float32)


def to_tactical(values, blocks, offsets, n, lag=None, fill=np.nan):
    """A per-broadcast-frame array on the tactical timeline (one offset per half, as V24)."""
    out = np.full((n,) + values.shape[1:], fill, dtype=np.float32)
    used = -1
    for half, first, last in blocks:
        tactical = np.arange(first, last + 1)
        source = tactical + offsets[half] + (0 if lag is None else lag[half])
        inside = (source > used) & (source < len(values)) & (source >= 0)
        used = int(source[-1])
        out[tactical[inside]] = values[source[inside]]
    return out


def events_of(x, threshold, gap, minimum):
    """Runs of x > threshold (merged over short gaps): [(start, stop, peak, frames above)]."""
    loud = np.nan_to_num(x, nan=-99.0) > threshold
    d = np.diff(np.r_[0, loud.astype(np.int8), 0])
    runs = []
    for a, z in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if runs and a - runs[-1][1] < gap:
            runs[-1][1] = int(z)
        else:
            runs.append([int(a), int(z)])
    return [(a, z, float(np.nanmax(x[a:z])), int(loud[a:z].sum())) for a, z in runs if loud[a:z].sum() >= minimum]


# -- audio -----------------------------------------------------------------------------------

def dual_tone(spectrum_db):
    """Prominence (dB) of the referee whistle's TWO simultaneous tones in the 3.3-4.7 kHz band.

    Both published kick-offs are preceded by a blast with peaks near 3.85-3.92 kHz and
    4.27-4.31 kHz (bins 24-28 and 43-47 of the 65-bin band) in both recordings; crowd whistles
    are single tones. The pair may move together by +-4 bins (86 Hz) between blasts. The score
    is the weaker of the two peaks over the median of the band away from them.
    """
    s = spectrum_db.astype(np.float32)
    best = np.full(len(s), -99.0, dtype=np.float32)
    for shift in range(-4, 5):
        a = s[:, 24 + shift:29 + shift].max(axis=1)
        b = s[:, 43 + shift:48 + shift].max(axis=1)
        floor = np.median(s[:, np.r_[np.arange(4, 19 + shift), np.arange(33 + shift, 40 + shift), np.arange(52 + shift, 63)]], axis=1)
        best = np.maximum(best, np.minimum(a, b) - floor)
    return best


def per_frame(columns, n, how="max"):
    """Two half-frame columns per video frame -> one value per video frame."""
    columns = columns[:2 * n]
    pairs = columns.reshape(n, 2, *columns.shape[1:])
    return pairs.max(axis=1) if how == "max" else pairs.mean(axis=1)


def audio_lag(wt, wb_raw, blocks, offsets, reach=20):
    """Extra frames by which the broadcast SOUND trails the tactical sound, per half.

    The picture offset comes from the player constellations; the microphones are elsewhere, so
    the sound may differ by a few frames. Found by correlating the two whistle detectors.
    """
    lag, detail = {}, {}
    for half, first, last in blocks:
        x = np.clip(wt[first:last + 1] - 8.0, 0, None)
        scores = []
        for shift in range(-reach, reach + 1):
            source = np.arange(first, last + 1) + offsets[half] + shift
            ok = (source >= 0) & (source < len(wb_raw))
            y = np.zeros(len(x))
            y[ok] = np.clip(wb_raw[source[ok]] - 8.0, 0, None)
            scores.append(float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12)))
        k = int(np.argmax(scores))
        lag[half] = k - reach
        detail[str(half)] = dict(lag_frames=k - reach, correlation=round(scores[k], 3), correlation_at_zero=round(scores[reach], 3),
                                 correlation_at_edges=[round(scores[0], 3), round(scores[-1], 3)])
    return lag, detail


def build_audio(blocks, offsets, n):
    t = np.load(ROOT / "audio_tactical.npz")
    b = np.load(ROOT / "audio_broadcast.npz")
    nb = source_frames("broadcast")
    wt = per_frame(dual_tone(t["whistle_spectrum_db"]), n)
    wb_raw = per_frame(dual_tone(b["whistle_spectrum_db"]), nb)
    lag, lag_detail = audio_lag(wt, wb_raw, blocks, offsets)
    wb = to_tactical(wb_raw, blocks, offsets, n, lag)
    both = np.where(np.isfinite(wb), np.minimum(wt, wb), np.nan)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    whistles = [dict(start=a, stop=z, peak_db=round(p, 1), frames=k, tactical_db=round(float(wt[a:z].max()), 1),
                     broadcast_db=round(float(np.nanmax(wb[a:z])), 1))
                for a, z, p, k in events_of(both, WHISTLE_DB, WHISTLE_GAP, WHISTLE_MIN) if graded[a]]
    starts = [w["start"] for w in whistles]
    since, until = since_until(starts, n)
    out = dict(a_whistle_tactical_db=wt, a_whistle_broadcast_db=wb, a_whistle_both_db=both,
               a_whistle_both_max_2s=np.array([np.nan]), a_since_whistle=since, a_until_whistle=until)
    padded = np.nan_to_num(both, nan=-99.0)
    best = padded.copy()
    for shift in range(1, 26):                       # max over +-1 s
        best[shift:] = np.maximum(best[shift:], padded[:-shift])
        best[:-shift] = np.maximum(best[:-shift], padded[shift:])
    out["a_whistle_both_max_2s"] = np.where(best > -90, best, np.nan)
    for tag, source, mapper in (("t", t, lambda v: v), ("b", b, lambda v: to_tactical(v, blocks, offsets, n, lag))):
        frames = n if tag == "t" else nb
        level = per_frame(source["level_db"], frames, "mean")
        bands = per_frame(source["band_db"], frames, "mean")
        flux = per_frame(source["flux"], frames, "mean")
        low, mid, high = bands[:, 0], bands[:, 1:4].mean(axis=1), bands[:, 5:7].mean(axis=1)
        base = rolling_median_coarse(level, 120 * FPS)
        series = dict(level_1s=rolling_mean(level, FPS) - base, level_5s=rolling_mean(level, 5 * FPS) - base,
                      level_20s=rolling_mean(level, 20 * FPS) - base,
                      low_minus_mid_3s=rolling_mean(low - mid, 3 * FPS), high_minus_mid_3s=rolling_mean(high - mid, 3 * FPS),
                      flux_1s=rolling_mean(flux, FPS), flux_5s=rolling_mean(flux, 5 * FPS))
        slope = np.full(frames, np.nan)
        smooth = rolling_mean(level, 2 * FPS)
        slope[50:-50] = smooth[100:] - smooth[:-100]          # level 2 s ahead minus 2 s behind
        series["level_rise_4s"] = slope
        for name, values in series.items():
            out[f"a_{tag}_{name}"] = mapper(np.asarray(values, dtype=np.float32))
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}, dict(whistles=whistles, audio_lag=lag_detail)


# -- video -----------------------------------------------------------------------------------

def scoreboard(board, every):
    """Clock visibility and score changes from the tactical scoreboard crop (one row per second).

    Text is white on a dark translucent plate. The score occupies columns 128-200; a change of
    the score is a change of its binarised glyphs that persists.
    """
    clock = (board[:, 8:44, 14:100] > 170).mean(axis=(1, 2))
    clock_on = clock > 0.04
    glyph = board[:, 8:44, 128:200] > 170
    states, labels = [], np.zeros(len(board), dtype=np.int64)
    for i, g in enumerate(glyph):
        for j, reference in enumerate(states):
            union = (g | reference).sum()
            if union and (g & reference).sum() / union > 0.80:
                labels[i] = j
                break
        else:
            states.append(g)
            labels[i] = len(states) - 1
    # keep states that last at least 60 s; a change is the first second of a new lasting state
    changes, current = [], None
    counts = np.bincount(labels)
    lasting = set(np.flatnonzero(counts >= 60).tolist())
    for i, label in enumerate(labels):
        if label in lasting and label != current:
            if current is not None:
                changes.append(i * every)
            current = label
    edges = np.flatnonzero(np.diff(clock_on.astype(np.int8)) != 0) + 1
    runs = [(int(a * every), int(z * every)) for a, z in zip(np.r_[0, edges], np.r_[edges, len(clock_on)]) if clock_on[a] and z - a >= 60]
    # does the clock ever freeze while shown? (identical clock crop for >= 3 consecutive seconds)
    same = (np.abs(board[1:, 8:44, 14:100].astype(np.int16) - board[:-1, 8:44, 14:100]) > 40).mean(axis=(1, 2)) < 0.002
    frozen = 0
    run = 0
    for flag, on in zip(same, clock_on[1:]):
        run = run + 1 if (flag and on) else 0
        frozen = max(frozen, run)
    return clock_on, labels, dict(score_change_frames=changes, clock_shown_runs=runs, distinct_lasting_score_states=len(lasting),
                                  longest_frozen_clock_seconds=int(frozen))


def build_video(blocks, offsets, n):
    t = np.load(ROOT / "video_tactical.npz")
    b = np.load(ROOT / "video_broadcast.npz")
    out = {}
    grid = t["diff_grid"].astype(np.float64)
    out["v_t_diff_pitch_1s"] = np.log1p(rolling_mean(t["diff_pitch"], FPS))
    out["v_t_diff_pitch_5s"] = np.log1p(rolling_mean(t["diff_pitch"], 5 * FPS))
    out["v_t_diff_peak_ratio_1s"] = rolling_mean(grid.max(axis=1) / (grid.mean(axis=1) + 0.05), FPS)
    out["v_t_green"] = t["green"]
    clock_on, labels, board_report = scoreboard(t["board"], int(t["board_every"]))
    every = int(t["board_every"])
    out["v_t_clock_on"] = np.repeat(clock_on.astype(np.float32), every)[:n]
    since, _ = since_until(board_report["score_change_frames"], n)
    out["v_t_since_score_change"] = since
    # broadcast: hard cuts, grass share, scoreboard on screen
    diff = b["diff_all"].astype(np.float64)
    local = rolling_median_coarse(diff, 10 * FPS, step=5)
    cut = np.flatnonzero((diff > 12.0) & (diff > 4.0 * (local + 0.5)))
    cut = cut[np.r_[True, np.diff(cut) > 5]] if len(cut) else cut
    board = b["board"].astype(np.float32)
    template = np.median(board[::200], axis=0)
    centred = board - board.mean(axis=(1, 2), keepdims=True)
    reference = template - template.mean()
    presence = (centred * reference).sum(axis=(1, 2)) / (np.sqrt((centred ** 2).sum(axis=(1, 2)) * (reference ** 2).sum()) + 1e-6)
    cut_since, cut_until = since_until(cut, len(diff))
    series = dict(v_b_diff_1s=np.log1p(rolling_mean(diff, FPS)), v_b_green=b["green"], v_b_green_3s=rolling_mean(b["green"], 3 * FPS),
                  v_b_board=presence, v_b_board_3s=rolling_mean(presence, 3 * FPS), v_b_since_cut=cut_since, v_b_until_cut=cut_until,
                  v_b_cuts_20s=rolling_mean(np.isin(np.arange(len(diff)), cut).astype(np.float64), 20 * FPS) * 20 * FPS)
    for name, values in series.items():
        out[name] = to_tactical(np.asarray(values, dtype=np.float32), blocks, offsets, n)
    cut_tactical = []
    for half, first, last in blocks:
        inside = cut - offsets[half]
        cut_tactical += inside[(inside >= first) & (inside <= last)].tolist()
    report = dict(scoreboard=board_report, broadcast_cuts=len(cut), broadcast_cut_frames_tactical=sorted(int(v) for v in cut_tactical),
                  broadcast_scoreboard_on_share=float((presence > 0.6).mean()))
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}, report


# -- ball and players ------------------------------------------------------------------------

def run_length(flag):
    """Length so far of the current run of True (0 where False)."""
    out = np.zeros(len(flag), dtype=np.float32)
    count = 0
    for i, f in enumerate(flag):
        count = count + 1 if f else 0
        out[i] = count
    return out


def build_ball_and_players(n):
    from . import v29_final as final
    from . import v28a_core as k28
    from .sources import challenge_match
    match, fill = final.filled_match(ROOT)
    raw = challenge_match()
    s = k28.signals(match.ball, match.players, match.team, match.pitch)
    length, width = match.pitch
    ball = np.asarray(match.ball, dtype=np.float64)
    seen = np.isfinite(ball).all(axis=1)
    outside = np.maximum.reduce([-ball[:, 0], ball[:, 0] - length, -ball[:, 1], ball[:, 1] - width])
    speed = s["speed"]
    slow = seen & np.isfinite(speed) & (speed < 1.5)
    saved = np.load("data/work/v26/video_ball_f2.npz")
    peaks = np.load("data/work/v26/detections_net3.npz")["peaks"]
    ball_out = dict(
        b_seen=seen.astype(np.float32), b_seen_tracking=np.isfinite(raw.ball).all(axis=1).astype(np.float32),
        b_source_video=(seen & ~np.isfinite(raw.ball).all(axis=1)).astype(np.float32),
        b_speed=speed, b_speed_1s=rolling_mean(speed, FPS), b_outside_m=outside, b_touchline_m=s["touchline"], b_goalline_m=s["goalline"],
        b_rest_frames=np.minimum(run_length(slow), CAP), b_unseen_frames=np.minimum(run_length(~seen), CAP),
        b_seen_share_3s=rolling_mean(seen.astype(np.float64), 3 * FPS),
        b_nearest_player_m=np.nanmin(s["distance"], axis=1, initial=np.inf, where=np.isfinite(s["distance"])),
        b_detector_peaks=(peaks[:, :, 2] >= 0.4).sum(axis=1).astype(np.float32),
        b_detector_best=np.nan_to_num(peaks[:, :, 2], nan=0.0).max(axis=1), b_video_confidence=saved["confidence"])
    ball_out["b_nearest_player_m"] = np.where(np.isfinite(ball_out["b_nearest_player_m"]), ball_out["b_nearest_player_m"], np.nan)
    players = np.asarray(match.players, dtype=np.float32)
    velocity = np.full(players.shape, np.nan, dtype=np.float32)
    velocity[5:-5] = (players[10:] - players[:-10]) * (FPS / 10.0)
    pace = np.linalg.norm(velocity, axis=2)
    tracked = np.isfinite(pace)
    count = np.maximum(tracked.sum(axis=1), 1)
    pace0 = np.where(tracked, pace, 0.0)
    kin = dict(k_players=np.isfinite(players).all(axis=2).sum(axis=1).astype(np.float32),
               k_speed_mean=pace0.sum(axis=1) / count, k_share_standing=((pace < 1.0) & tracked).sum(axis=1) / count,
               k_share_walking=((pace < 2.0) & tracked).sum(axis=1) / count, k_share_running=((pace > 4.0) & tracked).sum(axis=1) / count,
               k_speed_max=np.where(tracked, pace, -1).max(axis=1))
    spread = []
    for side in (0, 1):
        xy = players[:, match.team == side]
        centre = np.nanmean(xy, axis=1, keepdims=True)
        spread.append(np.nanmean(np.linalg.norm(xy - centre, axis=2), axis=1))
        kin[f"k_centroid_speed_{side}"] = np.r_[np.zeros(5), np.linalg.norm(centre[10:, 0] - centre[:-10, 0], axis=1) * (FPS / 10.0), np.zeros(5)]
    kin["k_spread_mean"] = (spread[0] + spread[1]) / 2
    kin["k_speed_mean_3s"] = rolling_mean(kin["k_speed_mean"], 3 * FPS)
    kin["k_share_standing_3s"] = rolling_mean(kin["k_share_standing"], 3 * FPS)
    invalid = kin["k_players"] < 6
    for name in list(kin):
        if name != "k_players":
            kin[name] = np.where(invalid, np.nan, kin[name])
    return ({k: np.asarray(v, dtype=np.float32) for k, v in ball_out.items()}, {k: np.asarray(v, dtype=np.float32) for k, v in kin.items()},
            dict(ball_fill=fill))


def build_camera(blocks, offsets):
    from . import v24_camera_fusion as v24
    x = v24.build_features(blocks, offsets, tuple(GAMES[GAME]["pitch"]))
    return {"c_" + name: x[:, j].astype(np.float32) for j, name in enumerate(v24.FEATURES)}


README = """# data/work/v30e/features

Per-frame multimodal features of the challenge match (BAR-ATH), written by
`python -m possession.v30e_features build`. Every array has 199,319 rows: row i is frame i of
the tactical video (the submission index). float32, NaN where a signal does not exist (outside
the broadcast footage, ball not seen, ...). Graded frames: 11185-90651 and 110808-190197.

| file | prefix | content |
| --- | --- | --- |
| audio.npz | a_ | referee-whistle dual-tone prominence in each recording and in both (dB), frames since / until a whistle, crowd level over a 120 s running median at 1 / 5 / 20 s, level rise over 4 s, spectral tilt, spectral flux; `a_t_*` tactical sound, `a_b_*` broadcast sound mapped through the camera alignment plus the measured audio lag |
| video.npz | v_ | tactical frame-difference energy on the pitch (log, 1 s and 5 s) and its spatial peak ratio, grass share, burnt-in clock on screen, frames since the score changed; broadcast frame-difference, grass share, scoreboard-on-screen correlation, frames since / until a hard cut, cuts per 20 s |
| ball.npz | b_ | the filled ball the networks saw (tracking + video detector + broadcast): seen, source, speed, metres outside the lines, distance to touchline / goal line, frames at rest, frames unseen, nearest player, detector peak count |
| kinematics.npz | k_ | players: mean speed, share standing / walking / running, spread, centroid speed (control group; the networks see the tracking already) |
| camera.npz | c_ | the 19 camera-operator features of V24 (unchanged) |
| events.json | | whistles (start, stop, peak dB per recording), audio lag per half, scoreboard score changes and clock runs, broadcast hard cuts |

No possession label of the challenge match and no network output enters any of these.

```python
import numpy as np
audio = np.load("data/work/v30e/features/audio.npz")
x = np.column_stack([audio[k] for k in audio.files])      # (199319, n_features)
```
"""


def load_group(name):
    with np.load(OUT / f"{name}.npz") as saved:
        return {k: saved[k] for k in saved.files}


def build(args):
    """Build the requested groups; events.json is merged so groups can be rebuilt one at a time."""
    OUT.mkdir(parents=True, exist_ok=True)
    blocks, offsets = blocks_and_offsets()
    n = source_frames("tactical")
    path = OUT / "events.json"
    events = json.loads(path.read_text()) if path.exists() else {}
    events.update(offsets_by_half={str(k): v for k, v in offsets.items()}, challenge_labels_read=False)
    tables = {}
    if "audio" in args.groups:
        tables["audio"], report = build_audio(blocks, offsets, n)
        events.update(report)
    if "video" in args.groups:
        tables["video"], report = build_video(blocks, offsets, n)
        events.update(report)
    if "ball" in args.groups or "kinematics" in args.groups:
        tables["ball"], tables["kinematics"], report = build_ball_and_players(n)
        events.update(report)
    if "camera" in args.groups:
        tables["camera"] = build_camera(blocks, offsets)
    for name, table in tables.items():
        for key, values in table.items():
            if values.shape != (n,):
                raise ValueError(f"{key} has shape {values.shape}")
        np.savez_compressed(OUT / f"{name}.npz", **table)
        print("FEATURES", name, len(table), list(table), flush=True)
    path.write_text(json.dumps(events, indent=1))
    (OUT / "README.md").write_text(README)
    print("EVENTS", json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in events.items() if k != "broadcast_cut_frames_tactical"})[:1500], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["build"])
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    args = parser.parse_args()
    dict(build=build)[args.step](args)
