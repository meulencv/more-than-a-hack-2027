"""V26 ball from the tactical video: detector, tracker, label-free validation.

Self-supervised: the only training signal is the tracking's own ball (projected to the image
with the provided homography and re-centred on the blob actually visible there). No human
label and no ground truth of the match is read anywhere in this module.

    python -m possession.video_ball extract                      # seed crops around the tracking's ball
    python -m possession.video_ball snap                         # blob-snapped labels for the seed crops
    python -m possession.v26_ball train --data seed --out net1 --epochs 8
    python -m possession.v26_ball mine --net net1 --out mined --no-columns --workers 8
    python -m possession.v26_ball train --data mined --out net2 --epochs 12 --no-columns
    python -m possession.v26_ball mine --net net2 --out mined2 --no-columns --workers 8
    python -m possession.v26_ball mine --net net2 --out mined3 --workers 8
    python -m possession.v26_ball train --data mined2,mined3 --out net3 --epochs 16
    python -m possession.v26_ball detect --net net3 --half --out detections_net3.npz
    python -m possession.v26_ball link --detections detections_net3.npz
    python -m possession.v26_ball link --detections detections_net3.npz --no-flight-filter --out video_ball_f2.npz
    python -m possession.v26_ball viz --out DIR                  # annotated frames without a tracked ball

`mine` runs the current detector on full frames of training minutes: the pseudo-label is
re-centred on the detection nearest the tracking's ball (or, for a ball in the air, straight
above it in the image), and confident detections elsewhere become hard negatives.

Held-out one-minute blocks (every eighth minute, possession.video_ball.held_out) never enter
detector training; validation hides the tracking's ball there and compares.
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from games import asset_paths
from .video_ball import ROOT, CROP, STEP, SEED, setup, to_pixel, held_out, ball_pixels, focal_loss

W, H = 1920, 1080
TOP_K = 8
PEAK_MIN = 0.03
SNAP_NET_RADIUS = 45.0          # a re-centred pseudo-label lies within this many pixels of the tracking's ball
N_MINE = 9000
COLUMN_BASE, COLUMN_SLOPE, COLUMN_HEIGHT = 20.0, 0.15, 400.0     # where an airborne ball can appear above its ground point


class Net(nn.Module):
    """Small fully-convolutional heat-map network; input 9 channels (t-2, t, t+2), output stride 2."""

    def __init__(self):
        super().__init__()

        def block(a, b, k=3, stride=1, dilation=1):
            pad = dilation * (k // 2)
            return nn.Sequential(nn.Conv2d(a, b, k, stride, pad, dilation, bias=False), nn.BatchNorm2d(b),
                                 nn.ReLU(inplace=True))
        # Everything after the stem runs at stride 4 (full-resolution layers cost 4x more and the
        # 1080p pass over the whole match is the bottleneck); PixelShuffle returns a stride-2 map.
        self.body = nn.Sequential(block(9, 32, 5, 2), block(32, 64, 3, 2), block(64, 64), block(64, 64, dilation=2),
                                  block(64, 64, dilation=4), block(64, 64), nn.Conv2d(64, 4, 1), nn.PixelShuffle(2))
        nn.init.constant_(self.body[-2].bias, -4.0)

    def forward(self, x):
        return self.body(x)


def load_net(name, device):
    net = Net()
    net.load_state_dict(torch.load(ROOT / f"{name}.pt", map_location="cpu"))
    return net.to(device).to(memory_format=torch.channels_last).eval()


# ----------------------------------------------------------------------------- training

def train(args):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device(args.device)
    if args.data == "seed":
        crops = np.load(ROOT / "train_crops.npy")
        labels = np.load(ROOT / "train_labels.npy")
        labels = np.column_stack([labels, np.full(len(labels), 24.0, dtype=np.float32), np.zeros((len(labels), 3), dtype=np.float32)])
    else:
        crops = np.concatenate([np.load(ROOT / f"{tag}_crops.npy") for tag in args.data.split(",")])
        labels = np.concatenate([np.load(ROOT / f"{tag}_labels.npy") for tag in args.data.split(",")])
    if labels.shape[1] == 4:
        raise ValueError("labels without the airborne column; run the relabel stage")
    if args.no_columns:
        labels[:, 6] = 0.0          # round 2 as it was run: airborne balls not yet left unlabelled
    n = len(crops)
    net = Net().to(device).to(memory_format=torch.channels_last)
    per_epoch = n // args.batch
    optimiser = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, 2e-3, total_steps=args.epochs * per_epoch, pct_start=0.1)
    half = CROP // 2
    gy, gx = torch.meshgrid(torch.arange(half, device=device, dtype=torch.float32),
                            torch.arange(half, device=device, dtype=torch.float32), indexing="ij")
    start = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        total = 0.0
        for k in range(per_epoch):
            pick = np.sort(order[k * args.batch:(k + 1) * args.batch])
            x = crops[pick]
            c = labels[pick].copy()
            flip = rng.random(len(pick)) < 0.5
            x[flip] = x[flip][:, :, ::-1]
            c[flip, 0] = CROP - 1 - c[flip, 0]
            c[flip, 4] = CROP - 1 - c[flip, 4]
            reverse = rng.random(len(pick)) < 0.5
            x[reverse] = x[reverse][..., [6, 7, 8, 3, 4, 5, 0, 1, 2]]
            ct = torch.from_numpy(c).to(device)
            d2 = (gx[None] - ct[:, 0, None, None] / 2) ** 2 + (gy[None] - ct[:, 1, None, None] / 2) ** 2
            kind = ct[:, 2, None, None]
            radius = ct[:, 3, None, None] / 2
            target = torch.where(kind == 1, torch.exp(-d2 / 2.0), torch.zeros_like(d2))
            target = torch.where((kind == 1) & (d2 <= 0.5), torch.ones_like(d2), target)
            weight = torch.where((kind == 2) & (d2 <= radius ** 2), torch.zeros_like(d2), torch.ones_like(d2))
            # Airborne ball: the tracking gives the ground point under it, the image shows it
            # straight above. When no ball was found the whole column above is left unlabelled.
            up = ct[:, 5, None, None] - gy[None] * 2
            across = (gx[None] * 2 - ct[:, 4, None, None]).abs()
            column = (up >= 0) & (up <= ct[:, 6, None, None]) & (across <= COLUMN_BASE + COLUMN_SLOPE * up)
            weight = torch.where(column, torch.zeros_like(d2), weight)
            xb = torch.from_numpy(np.ascontiguousarray(x)).to(device).permute(0, 3, 1, 2).float().div(255.0)
            xb = xb * (1 + 0.3 * (torch.rand(len(pick), 1, 1, 1, device=device) - 0.5))
            logits = net(xb.contiguous(memory_format=torch.channels_last))
            loss = focal_loss(logits[:, 0], target, weight)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
        print("TRAIN", args.out, "epoch", epoch, "loss", round(total / per_epoch, 4), "s", round(time.time() - start),
              flush=True)
    torch.save(net.state_dict(), ROOT / f"{args.out}.pt")


# ----------------------------------------------------------------------------- inference helpers

def peaks(net, frames_u8, device):
    """frames_u8: (B, H, W, 9) uint8 tensor on device -> per-frame (k, 3) arrays of x, y, score."""
    kind = next(net.parameters()).dtype          # float16 when the network was halved (2x faster on MPS)
    x = frames_u8.permute(0, 3, 1, 2).to(kind).div(255.0).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        heat = torch.sigmoid(net(x).float())[:, 0]
        top = (heat == nn.functional.max_pool2d(heat[:, None], 7, 1, 3)[:, 0]) & (heat > PEAK_MIN)
        values, where = (heat * top).flatten(1).topk(TOP_K, dim=1)
        hh, ww = heat.shape[-2:]
        py, px = where // ww, where % ww
        # Sub-cell refinement: score-weighted centroid of the 3x3 neighbourhood.
        padded = nn.functional.pad(heat, (1, 1, 1, 1))
        b = torch.arange(len(heat), device=heat.device)[:, None]
        sx = torch.zeros_like(values)
        sy = torch.zeros_like(values)
        sw = torch.zeros_like(values)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v = padded[b, py + 1 + dy, px + 1 + dx]
                sx += v * dx
                sy += v * dy
                sw += v
        fx = (px + sx / sw.clamp(min=1e-6)) * 2.0 + 0.5
        fy = (py + sy / sw.clamp(min=1e-6)) * 2.0 + 0.5
    out = torch.stack([fx, fy, values], dim=2).cpu().numpy()
    return out


def open_at(frame):
    capture = cv2.VideoCapture(asset_paths()["video"])
    if frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    return capture


# ----------------------------------------------------------------------------- mining

def mine_worker(job):
    worker, frames, pixels, device_name, name, tag, round_seed, columns = job
    torch.set_num_threads(1)
    cv2.setNumThreads(2)
    device = torch.device(device_name)
    net = load_net(name, device)
    rng = np.random.default_rng(SEED + 1000 * round_seed + worker)
    crops, labels, source = [], [], []
    capture = open_at(0)
    position = -10 ** 9
    found = 0
    for t in frames:
        first = int(t) - STEP
        if not 0 <= first - position < 40:
            capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            position = first
        while position < first:
            capture.grab()
            position += 1
        stack = []
        for _ in range(2 * STEP + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("decode failed at {}".format(position))
            stack.append(frame)
            position += 1
        image = np.concatenate([stack[0], stack[STEP], stack[-1]], axis=2)
        found_peaks = peaks(net, torch.from_numpy(image[None]).to(device), device)[0]
        found_peaks = found_peaks[found_peaks[:, 2] > 0]
        a = pixels[t]
        dx, dy = found_peaks[:, 0] - a[0], found_peaks[:, 1] - a[1]
        distance = np.hypot(dx, dy)
        in_disc = distance <= SNAP_NET_RADIUS
        in_column = (dy < 0) & (dy >= -COLUMN_HEIGHT) & (np.abs(dx) <= COLUMN_BASE + COLUMN_SLOPE * -dy) & ~in_disc
        near = np.flatnonzero(in_disc & (found_peaks[:, 2] >= 0.3))
        if len(near) == 0 and columns:
            near = np.flatnonzero(in_column & (found_peaks[:, 2] >= 0.4))      # airborne, straight above
        if len(near):
            best = near[np.argmax(found_peaks[near, 2])]
            ball = (found_peaks[best, 0], found_peaks[best, 1], 1.0, 0.0)
            column = 0.0
            found += 1
        else:
            ball = (a[0], a[1], 2.0, SNAP_NET_RADIUS)
            column = COLUMN_HEIGHT
        centres = [(ball[0] - rng.integers(24, CROP - 24) + CROP / 2, ball[1] - rng.integers(24, CROP - 24) + CROP / 2)]
        hard = np.flatnonzero((distance > SNAP_NET_RADIUS + 15) & (found_peaks[:, 2] >= 0.08) & ~(in_column & columns))
        if len(near):
            hard = hard[hard != best]
        for j in hard[:2]:
            centres.append((found_peaks[j, 0] + rng.integers(-50, 51), found_peaks[j, 1] + rng.integers(-50, 51)))
        centres.append((rng.integers(0, W), rng.integers(0, H)))
        for cx, cy in centres:
            x0 = int(np.clip(cx - CROP / 2, 0, W - CROP))
            y0 = int(np.clip(cy - CROP / 2, 0, H - CROP))
            bx, by = ball[0] - x0, ball[1] - y0
            margin = ball[3]
            if -margin <= bx < CROP + margin and -margin <= by < CROP + margin:
                label = (bx, by, ball[2], ball[3], a[0] - x0, a[1] - y0, column)
            else:
                label = (0.0, 0.0, 0.0, 0.0, a[0] - x0, a[1] - y0, column)
            crops.append(image[y0:y0 + CROP, x0:x0 + CROP].copy())
            labels.append(label)
            source.append((t, x0, y0))
    np.save(ROOT / f"{tag}_crops_{worker}.npy", np.stack(crops))
    return np.array(labels, dtype=np.float32), np.array(source, dtype=np.int32), found


def mine(args):
    blocks, pitch, n, ball, matrices, graded = setup()
    tag = args.out or "mined"
    round_seed = 1 if tag == "mined" else 2 + sum(map(ord, tag))
    rng = np.random.default_rng(SEED + round_seed)
    pixels = ball_pixels(ball, matrices)
    frames = np.arange(n)
    inside = (pixels[:, 0] > 8) & (pixels[:, 0] < W - 8) & (pixels[:, 1] > 8) & (pixels[:, 1] < H - 8)
    eligible = frames[graded & inside & ~held_out(frames) & (frames > STEP) & (frames < n - STEP)]
    chosen = np.sort(rng.choice(eligible, N_MINE, replace=False))
    parts = np.array_split(chosen, args.workers)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(mine_worker, [(i, part, pixels, args.device, args.net, tag, round_seed, not args.no_columns)
                                         for i, part in enumerate(parts)])
    crops = np.concatenate([np.load(ROOT / f"{tag}_crops_{i}.npy") for i in range(args.workers)])
    labels = np.concatenate([r[0] for r in results])
    source = np.concatenate([r[1] for r in results])
    np.save(ROOT / f"{tag}_crops.npy", crops)
    np.save(ROOT / f"{tag}_labels.npy", labels)
    np.save(ROOT / f"{tag}_source.npy", source)
    for i in range(args.workers):
        (ROOT / f"{tag}_crops_{i}.npy").unlink()
    print("MINE", crops.shape, "frames", len(chosen), "re-centred share", sum(r[2] for r in results) / len(chosen),
          "kinds", np.unique(labels[:, 2], return_counts=True), flush=True)


def relabel(args):
    """Add the airborne column to a set mined with --no-columns (labels with 4 fields)."""
    blocks, pitch, n, ball, matrices, graded = setup()
    pixels = ball_pixels(ball, matrices)
    for tag in args.data.split(","):
        labels = np.load(ROOT / f"{tag}_labels.npy")
        source = np.load(ROOT / f"{tag}_source.npy")
        if labels.shape[1] != 4:
            continue
        first = np.r_[True, source[1:, 0] != source[:-1, 0]]              # the first crop of a frame is the ball crop
        group = np.cumsum(first) - 1
        found = (labels[first, 2] == 1)[group]
        anchor = pixels[source[:, 0]] - source[:, 1:3]
        column = np.where(found, 0.0, COLUMN_HEIGHT)
        np.save(ROOT / f"{tag}_labels.npy", np.column_stack([labels, anchor, column]).astype(np.float32))
        print("RELABEL", tag, "frames", int(first.sum()), "with an unlabelled column", float(1 - found[first].mean()), flush=True)


# ----------------------------------------------------------------------------- detection

def detect_worker(job):
    worker, first, last, device_name, batch, name, half = job
    torch.set_num_threads(1)
    cv2.setNumThreads(3)
    device = torch.device(device_name)
    net = load_net(name, device)
    if half:
        net = net.half()
    capture = open_at(first - STEP)
    count = last - first + 1
    out = np.zeros((count, TOP_K, 3), dtype=np.float32)
    buffer = []                 # GPU uint8 frames, buffer[0] is video frame `base`
    base = first - STEP
    done = 0
    start = time.time()
    while done < count:
        size = min(batch, count - done)
        need = done + size + 2 * STEP          # frames first-STEP .. first+done+size-1+STEP
        while len(buffer) + (base - (first - STEP)) < need:
            ok, frame = capture.read()
            if not ok:
                raise ValueError("decode failed in worker {}".format(worker))
            buffer.append(torch.from_numpy(frame).to(device))
        offset = first - STEP + done - base
        stack = torch.stack([torch.cat([buffer[offset + i], buffer[offset + i + STEP], buffer[offset + i + 2 * STEP]], dim=2)
                             for i in range(size)])
        out[done:done + size] = peaks(net, stack, device)
        done += size
        drop = (first - STEP + done) - base
        del buffer[:drop]
        base += drop
        if worker == 0 and (done // batch) % 100 == 0:
            print("DETECT worker0", done, "of", count, "fps", round(done / (time.time() - start), 1), flush=True)
    return first, out


def detect(args):
    blocks, pitch, n, ball, matrices, graded = setup()
    jobs = []
    for _, a, z in blocks:
        a, z = max(a, STEP), min(z, n - 1 - STEP)
        edges = np.linspace(a, z + 1, args.workers // len(blocks) + 1).astype(int)
        for lo, hi in zip(edges[:-1], edges[1:]):
            if args.limit:
                hi = min(hi, lo + args.limit)
            jobs.append((len(jobs), int(lo), int(hi - 1), args.device, args.batch, args.net, args.half))
    start = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(detect_worker, jobs)
    peaks_all = np.zeros((n, TOP_K, 3), dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    for first, out in results:
        peaks_all[first:first + len(out)] = out
        done[first:first + len(out)] = True
    np.savez_compressed(ROOT / (args.out or "detections.npz"), peaks=peaks_all, done=done)
    print("DETECT done", int(done.sum()), "frames in", round(time.time() - start), "s; top score > 0.5 on",
          float((peaks_all[done, 0, 2] > 0.5).mean()), flush=True)


def bench(args):
    device = torch.device(args.device)
    net = Net().to(device).to(memory_format=torch.channels_last).eval()
    x = torch.randint(0, 255, (args.batch, H, W, 9), dtype=torch.uint8, device=device)
    for _ in range(2):
        peaks(net, x, device)
    start = time.time()
    for _ in range(10):
        peaks(net, x, device)
    print("BENCH fps", round(10 * args.batch / (time.time() - start), 1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--data", default="seed")
    parser.add_argument("--out", default=None)
    parser.add_argument("--net", default="net2")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--half", action="store_true", help="detect: float16 inference")
    parser.add_argument("--no-columns", action="store_true",
                        help="mine / train: early rounds, before the detector can be trusted on airborne balls")
    parser.add_argument("--threshold", type=float, default=None, help="link: fix the operating point by hand")
    parser.add_argument("--min-weight", type=float, default=3.0)
    parser.add_argument("--max-far", type=float, default=0.05,
                        help="link: tolerated share of reports more than 5 m from the hidden tracking ball")
    parser.add_argument("--detections", default="detections.npz")
    parser.add_argument("--no-flight-filter", action="store_true",
                        help="link: keep airborne stretches (ground-plane position biased away from the camera)")
    args = parser.parse_args()
    if args.batch is None:
        args.batch = 96 if args.stage == "train" else 4
    from . import v26_track
    stages = dict(train=train, mine=mine, relabel=relabel, detect=detect, bench=bench, link=v26_track.link, viz=v26_track.viz)
    stages[args.stage](args)


if __name__ == "__main__":
    main()
