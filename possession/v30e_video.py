"""V30e: cheap per-frame image statistics of the two recordings (no labels, no network).

One ffmpeg pass per recording, decoded at 160x90 RGB plus a full-resolution crop of the
burnt-in scoreboard. Per video frame it stores

    diff_all      mean |gray(t) - gray(t-1)| over the whole 160x90 image
    diff_pitch    the same over the lower 3/4 of the image (stands and adverts excluded)
    diff_grid     the same on a 3 x 4 grid of cells (camera pan gives a uniform field, players
                  moving give a localised one)
    green         share of pixels that are grass-coloured (wide shot vs close-up / crowd shot)
    luma          mean brightness
    board         the scoreboard crop (tactical: every 25th frame at full resolution, for the
                  clock / score reading; broadcast: every frame at quarter resolution, for
                  "is the scoreboard on screen", which the TV director removes during replays)

    .venv/bin/python -m possession.v30e_video --source tactical
    .venv/bin/python -m possession.v30e_video --source broadcast

Output: data/work/v30e/video_<source>.npz. Deterministic for a given ffmpeg build (software
decode, bilinear scaling, integer arithmetic afterwards).
"""
import argparse
import subprocess
import time
from pathlib import Path
import numpy as np
from games import asset_paths, source_frames

ROOT = Path("data/work/v30e")
W, H = 160, 90
# scoreboard rectangles in the 1920x1080 image: (width, height, x, y)
BOARD = dict(tactical=(216, 52, 122, 0), broadcast=(212, 116, 95, 56))
BOARD_OUT = dict(tactical=(216, 52), broadcast=(53, 29))
BOARD_EVERY = dict(tactical=25, broadcast=1)
CHUNK = 500


def reader(command, frame_bytes):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=frame_bytes * 64)
    try:
        while True:
            raw = process.stdout.read(frame_bytes * CHUNK)
            if not raw:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(-1, frame_bytes)
    finally:
        process.stdout.close()
        process.wait()


def statistics(blocks, n):
    """Per-frame statistics from an iterator of (m, W*H*3) uint8 blocks."""
    out = dict(diff_all=np.zeros(n, np.float32), diff_pitch=np.zeros(n, np.float32), diff_grid=np.zeros((n, 12), np.float32),
               green=np.zeros(n, np.float32), luma=np.zeros(n, np.float32))
    previous = None
    k = 0
    started = time.time()
    for block in blocks:
        rgb = block.reshape(-1, H, W, 3).astype(np.int16)
        gray = (rgb[..., 0] * 77 + rgb[..., 1] * 150 + rgb[..., 2] * 29) >> 8
        m = len(gray)
        if k + m > n:
            raise ValueError(f"more than {n} frames decoded")
        before = np.concatenate([gray[:1] if previous is None else previous[None], gray[:-1]])
        d = np.abs(gray - before).astype(np.float32)
        previous = gray[-1]
        out["diff_all"][k:k + m] = d.mean(axis=(1, 2))
        out["diff_pitch"][k:k + m] = d[:, H // 4:].mean(axis=(1, 2))
        cells = d[:, H - 69:].reshape(m, 3, 23, 4, 40).mean(axis=(2, 4))
        out["diff_grid"][k:k + m] = cells.reshape(m, 12)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        out["green"][k:k + m] = ((g > r + 12) & (g > b + 12)).mean(axis=(1, 2))
        out["luma"][k:k + m] = gray.mean(axis=(1, 2))
        k += m
        if (k // CHUNK) % 40 == 0:
            print("frames", k, n, round(k / (time.time() - started), 1), "fps", flush=True)
    return out, k


def one_pass(video, source, n, limit=None):
    """One decode, two outputs: 160x90 RGB to the pipe, the scoreboard crop to a temporary file."""
    w, h, x, y = BOARD[source]
    ow, oh = BOARD_OUT[source]
    every = BOARD_EVERY[source]
    crop = (f"select='not(mod(n\\,{every}))'," if every > 1 else "") + f"crop={w}:{h}:{x}:{y}"
    if (ow, oh) != (w, h):
        crop += f",scale={ow}:{oh}:flags=area"
    temporary = ROOT / f"board_{source}.gray"
    graph = f"[0:v]split=2[a][b];[a]scale={W}:{H}:flags=bilinear[small];[b]{crop}[board]"
    count = ["-frames:v", str(limit)] if limit else []
    count_board = ["-frames:v", str(limit // every)] if limit else []
    command = (["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an", "-filter_complex", graph,
                "-map", "[board]", "-fps_mode", "passthrough", "-pix_fmt", "gray", "-f", "rawvideo"] + count_board + [str(temporary),
                "-map", "[small]", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo"] + count + ["-"])
    out, k = statistics(reader(command, W * H * 3), n)
    board = np.fromfile(temporary, dtype=np.uint8).reshape(-1, oh, ow)
    temporary.unlink()
    return out, k, board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["tactical", "broadcast"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    video = asset_paths(source=args.source)["video"]
    n = args.limit or source_frames(args.source)
    started = time.time()
    out, k, board = one_pass(video, args.source, n, args.limit or None)
    # games.source_frames counts the rows of the provider's files; the broadcast video stream
    # holds one frame less. The missing last row keeps its zeros and is reported.
    if not n - 1 <= k <= n:
        raise ValueError(f"decoded {k} frames, expected {n}")
    print("decoded", k, "of", n, "frames; board", board.shape, round(time.time() - started, 1), "s", flush=True)
    name = f"video_{args.source}" + (f"_limit{args.limit}" if args.limit else "")
    np.savez_compressed(ROOT / f"{name}.npz", board=board, board_every=BOARD_EVERY[args.source], decoded_frames=k, **out)
    print("saved", ROOT / f"{name}.npz", flush=True)


if __name__ == "__main__":
    main()
