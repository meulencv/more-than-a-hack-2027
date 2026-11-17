"""V30a: qualitative case study of the labelling conventions in the 5 public minutes.

SCOPE AND DISCLOSURE. This is the only module that reads the public ground truth
(evaluation/gt/BAR-ATH_possession_gt_public.csv, video frames 11185-18684). It is used for the one
purpose CHALLENGE.md states: to understand how the definition is applied. Nothing here fits, tunes,
selects or scores anything; it produces no labels and no numeric constants. Every invocation is
appended to data/work/v30a/LOG.md so that exactly what was looked at can be disclosed.

Commands (all deterministic):

    .venv/bin/python -m possession.v30a_cases segments
        the 21 label segments of the window.
    .venv/bin/python -m possession.v30a_cases contacts
        ball-contact candidates from the automatic tracking (direction or speed change of the ball
        next to a player) with the public label at that frame, to find opponent touches that do
        and do not change the label.
    .venv/bin/python -m possession.v30a_cases pressure
        stretches of each possession where an opponent is tracked near the ball (the logged run used
        2.5 m), to
        choose what to inspect on video.
    .venv/bin/python -m possession.v30a_cases whistles
        tonal blasts of the tactical audio inside the window (possession/audio_whistle.py), to
        relate stoppage onsets and restarts to the referee's whistle.
    .venv/bin/python -m possession.v30a_cases table FIRST LAST [--step N]
        per-frame tracking summary: label, ball position, ball speed, nearest player of each team.
    .venv/bin/python -m possession.v30a_cases sheet FIRST LAST [--step N] [--crop W H] [--name S]
        contact sheet of the tactical video, cropped around the ball, one tile every N frames,
        each tile stamped with the frame number and the public label.
"""
import argparse
import datetime
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import asset_paths  # noqa: E402
from possession.data import load_tracking  # noqa: E402
from starter import load_homography  # noqa: E402

OUT = ROOT / "data/work/v30a"
WINDOW = (11185, 18684)
PITCH = np.array([105.3, 68.0])
GT = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
BALL = ROOT / "data/work/v26/video_ball_f2.npz"


def log(text):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "LOG.md").open("a") as handle:
        handle.write("- {} `{}`\n".format(datetime.datetime.now().isoformat(timespec="seconds"),
                                         text))


def labels():
    values = pd.read_csv(GT)["label"].values.astype(str)
    outside = np.ones(len(values), dtype=bool)
    outside[WINDOW[0]:WINDOW[1] + 1] = False
    if not np.all(values[outside] == "X"):
        raise ValueError("public ground truth has labels outside the declared window")
    return values


def segments(values):
    rows, start = [], WINDOW[0]
    for frame in range(WINDOW[0] + 1, WINDOW[1] + 2):
        if frame > WINDOW[1] or values[frame] != values[start]:
            rows.append((start, frame - 1, values[start]))
            start = frame
    return rows


def load_scene():
    players, tracked, slots = load_tracking(cache_dir=str(ROOT / "data/work/cache"))
    with np.load(BALL) as saved:
        ball, source = saved["ball"].astype(float), saved["source"].astype(int)
    return players, ball, source, slots, np.isfinite(tracked[:, 0])


def nearest(players, ball, slots, frame):
    """Distance in metres from the ball to the nearest tracked player of each team."""
    out = []
    for team in (0, 1):
        xy = players[frame][slots == team, :2].astype(float)
        d = np.sqrt((((xy - ball[frame]) * PITCH) ** 2).sum(axis=1))
        out.append(float(np.nanmin(d)) if np.isfinite(d).any() else np.nan)
    return out


def velocity(ball, frame, lag=2):
    a, b = ball[frame - lag], ball[frame + lag]
    return (b - a) * PITCH * 25.0 / (2 * lag)


def command_segments(args):
    values = labels()
    for first, last, label in segments(values):
        print("{:6d} {:6d} {} {:5d} frames {:6.2f} s".format(
            first, last, label, last - first + 1, (last - first + 1) / 25.0))


def command_table(args):
    values = labels()
    players, ball, source, slots, _ = load_scene()
    print(" frame gt src    x_m    y_m  speed  dHome  dAway")
    for frame in range(args.first, args.last + 1, args.step):
        dh, da = nearest(players, ball, slots, frame)
        v = velocity(ball, frame)
        print("{:6d} {}  {:2d} {:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f}".format(
            frame, values[frame], source[frame], ball[frame, 0] * PITCH[0],
            ball[frame, 1] * PITCH[1], float(np.hypot(*v)), dh, da))


def command_contacts(args):
    """Ball-contact candidates: the ball changes velocity while a player is next to it."""
    values = labels()
    players, ball, source, slots, _ = load_scene()
    rows = []
    for frame in range(WINDOW[0] - 50, WINDOW[1] + 50):
        before = (ball[frame] - ball[frame - 3]) * PITCH * 25.0 / 3
        after = (ball[frame + 3] - ball[frame]) * PITCH * 25.0 / 3
        if not (np.isfinite(before).all() and np.isfinite(after).all()):
            continue
        change = float(np.hypot(*(after - before)))
        dh, da = nearest(players, ball, slots, frame)
        if change > 4.0 and min(dh, da) < 2.0:
            rows.append((frame, change, dh, da))
    # keep the strongest candidate of each cluster of nearby frames
    kept, cluster = [], []
    for row in rows:
        if cluster and row[0] - cluster[-1][0] > 6:
            kept.append(max(cluster, key=lambda r: r[1]))
            cluster = []
        cluster.append(row)
    if cluster:
        kept.append(max(cluster, key=lambda r: r[1]))
    print(" frame gt  near  change  dHome  dAway   note")
    for frame, change, dh, da in kept:
        label = values[frame] if WINDOW[0] <= frame <= WINDOW[1] else "X"
        near = "H" if dh < da else "A"
        contested = abs(dh - da) < 1.0
        note = ""
        if label in "HA" and near != label:
            note = "OPPONENT NEAR" + (" (both close)" if contested else "")
        print("{:6d} {}   {}   {:6.1f} {:6.1f} {:6.1f}   {}".format(
            frame, label, near, change, dh, da, note))


def command_sheet(args):
    values = labels()
    players, ball, source, slots, tracked = load_scene()
    matrices, _, valid = load_homography.load()
    frames = list(range(args.first, args.last + 1, args.step))
    for extra in (args.more.split(",") if args.more else []):
        a, b = extra.split("-")
        frames += list(range(int(a), int(b) + 1, args.step))
    # centre of each tile: the ball, linearly interpolated through gaps
    known = np.flatnonzero(np.isfinite(ball[:, 0]))
    cw, ch = args.crop
    scale = args.scale
    tiles = []
    capture = cv2.VideoCapture(asset_paths()["video"])
    capture.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    current = frames[0]
    for frame in frames:
        if frame < current or frame - current > 200:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            current = frame
        while current < frame:
            capture.grab()
            current += 1
        ok, image = capture.read()
        current += 1
        if not ok:
            raise IOError("cannot read frame {}".format(frame))
        if args.fixed is not None:
            bx, by = args.fixed
        else:
            bx = np.interp(frame, known, ball[known, 0])
            by = np.interp(frame, known, ball[known, 1])
        px, py = load_homography.project(matrices[frame], bx, by)
        px, py = float(px) + args.offset[0], float(py) + args.offset[1]
        if args.wide:
            cw, ch, px, py = image.shape[1], image.shape[0], image.shape[1] / 2, image.shape[0] / 2
        x0 = int(np.clip(px - cw / 2, 0, image.shape[1] - cw))
        y0 = int(np.clip(py - ch / 2 - (0 if args.wide else ch * 0.1), 0, image.shape[0] - ch))
        tile = image[y0:y0 + ch, x0:x0 + cw].copy()
        tile = cv2.resize(tile, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
        if np.isfinite(ball[frame, 0]) and not args.clean and args.offset == [0.0, 0.0]:
            colour = (0, 255, 255) if tracked[frame] else (255, 255, 0)
            centre = (int((px - x0) * scale), int((py - y0) * scale))
            cv2.circle(tile, centre, int(9 * scale), colour, 1, cv2.LINE_AA)
        label = values[frame] if WINDOW[0] <= frame <= WINDOW[1] else "X"
        colour = {"H": (255, 80, 80), "A": (200, 200, 255), "D": (0, 0, 0), "X": (128, 128, 128)}
        cv2.rectangle(tile, (0, 0), (150, 26), colour[label], -1)
        text = (255, 255, 255) if label in "HDX" else (0, 0, 0)
        cv2.putText(tile, "{} {}".format(frame, label), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    text, 2, cv2.LINE_AA)
        tiles.append(tile)
    columns = args.columns
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    sheet = np.vstack(rows)
    name = args.name or "sheet_{}_{}_s{}".format(args.first, args.last, args.step)
    path = OUT / "sheets" / (name + ".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(path)


def command_pressure(args):
    """Stretches inside each possession where an opponent is tracked near the ball.

    Used only to decide which parts of a possession to inspect on video for opponent touches
    that do not change the label; stretches where every opponent is far from the ball are skipped.
    """
    values = labels()
    players, ball, source, slots, _ = load_scene()
    for first, last, label in segments(values):
        if label == "D":
            continue
        opponent = 1 if label == "H" else 0
        flag = np.zeros(last - first + 1, dtype=bool)
        for frame in range(first, last + 1):
            if not np.isfinite(ball[frame, 0]):
                flag[frame - first] = True
                continue
            flag[frame - first] = nearest(players, ball, slots, frame)[opponent] < args.radius
        grown = np.convolve(flag, np.ones(2 * args.grow + 1), mode="same") > 0
        runs, start = [], None
        for i, value in enumerate(grown):
            if value and start is None:
                start = i
            if start is not None and (not value or i == len(grown) - 1):
                runs.append((first + start, first + i))
                start = None
        print("{} {}-{}: {} of {} frames; {}".format(
            label, first, last, int(grown.sum()), len(grown),
            " ".join("{}-{}".format(a, b) for a, b in runs)))


def command_whistles(args):
    """Tonal blasts in the whistle band of the tactical audio inside the window, with the label.

    Uses possession/audio_whistle.py unchanged (referee template from the two kick-off whistles).
    Only used to describe whether a stoppage begins at the whistle or at another moment.
    """
    from possession import audio_whistle
    values = labels()
    cache = OUT / "tone_track.npz"
    if cache.exists():
        with np.load(cache) as saved:
            tone, pitch = saved["tone"], saved["pitch"]
    else:
        tone, pitch = audio_whistle.tone_track(asset_paths()["video"], len(values))
        np.savez_compressed(cache, tone=tone, pitch=pitch)
    blocks = [(1, 11185, 90651), (2, 110808, 190197)]
    referee, template = audio_whistle.referee_whistles(tone, pitch, blocks, [11185, 113742])
    referee = set(int(x) for x in referee)
    print("template", template)
    print(" start length tone pitch spread referee label_at_start label_after")
    for event in audio_whistle.tone_events(tone, pitch, WINDOW[0] - 100, WINDOW[1] + 1):
        start = event["start"]
        end = start + event["length"]
        print("{:6d} {:4d} {:5.1f} {:6.0f} {:6.0f} {:>3} {} {}".format(
            start, event["length"], event["tone"], event["pitch"], event["spread"],
            "REF" if start in referee else "-", values[start] if start >= WINDOW[0] else "X",
            values[min(end + 25, WINDOW[1])]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("segments")
    sub.add_parser("contacts")
    table = sub.add_parser("table")
    table.add_argument("first", type=int)
    table.add_argument("last", type=int)
    table.add_argument("--step", type=int, default=1)
    sheet = sub.add_parser("sheet")
    sheet.add_argument("first", type=int)
    sheet.add_argument("last", type=int)
    sheet.add_argument("--step", type=int, default=3)
    sheet.add_argument("--crop", type=int, nargs=2, default=[320, 240])
    sheet.add_argument("--scale", type=float, default=1.5)
    sheet.add_argument("--columns", type=int, default=4)
    sheet.add_argument("--fixed", type=float, nargs=2, default=None,
                       help="fixed pitch point (normalised) to centre on instead of the ball")
    sheet.add_argument("--offset", type=float, nargs=2, default=[0.0, 0.0],
                       help="shift of the crop centre in native pixels")
    sheet.add_argument("--more", default=None, help="further frame ranges, e.g. 100-140,300-320")
    sheet.add_argument("--wide", action="store_true", help="whole frame instead of a crop")
    sheet.add_argument("--clean", action="store_true", help="do not draw the ball marker")
    sheet.add_argument("--name", default=None)
    pressure = sub.add_parser("pressure")
    pressure.add_argument("--radius", type=float, default=4.0)
    pressure.add_argument("--grow", type=int, default=8)
    sub.add_parser("whistles")
    args = parser.parse_args()
    log(" ".join(sys.argv[1:]))
    {"segments": command_segments, "contacts": command_contacts, "table": command_table,
     "sheet": command_sheet, "pressure": command_pressure,
     "whistles": command_whistles}[args.command](args)


if __name__ == "__main__":
    main()
