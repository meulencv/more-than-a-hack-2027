"""Self-supervised ball detector on the tactical video, used to fill the tracking's ball gaps.

The automatic tracking has no ball on 41% of graded frames, always in stretches of 3 s or more.
The ball is often plainly visible in the video during those stretches. This module learns what
the ball looks like from the frames where the tracking DOES have it (its position projected to
the image through the provided homography is the pseudo-label; no human label is involved),
then looks for it where the tracking does not.

    python -m possession.video_ball extract   # one pass over the video: seed training crops
    python -m possession.video_ball snap      # pseudo-label -> the bright blob actually visible nearby

The detector that was finally used, its self-training rounds, the full-match detection pass,
the tracker and the validation live in possession.v26_ball and possession.v26_track (this
module keeps the seed extraction, the blob snap and the shared helpers; its own train /
detect / link stages are the first, unfinished design and are kept only for reference).

Held-out one-minute blocks (every eighth minute) never enter training; the detector and the
linker are checked there against the tracking's own ball. No ground truth is read anywhere.
"""
import argparse
import json
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames
from starter.load_homography import load as load_homography
from .data import load_tracking

ROOT = Path("data/work/v26")
CROP = 160
STEP = 2                        # the network sees frames t-2, t, t+2
N_TRAIN = 10000
HELD_BLOCK, HELD_EVERY, HELD_PICK = 1500, 8, 5
TOP_K = 5
PITCH_MARGIN = 3.0
SPOT_RADIUS = 1.0               # painted spots look like a resting ball
MAX_STEP_PX = 55.0              # farthest a ball moves between consecutive frames, in pixels
SWITCH_COST = 4.0               # entering or leaving a track
MIN_TRACK = 6
BRIDGE = 12                     # interpolate interior holes up to this many frames
SEED = 26


class BallNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a, b, stride=1, dilation=1):
            return nn.Sequential(nn.Conv2d(a, b, 3, stride, dilation, dilation), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.body = nn.Sequential(block(9, 24), block(24, 32, stride=2), block(32, 48), block(48, 48, dilation=2),
                                  block(48, 48, dilation=4), block(48, 32), nn.Conv2d(32, 1, 1))
        nn.init.constant_(self.body[-1].bias, -4.0)

    def forward(self, x):
        return self.body(x)


def setup():
    blocks = list(half_video_blocks(GAME))
    pitch = np.array(GAMES[GAME]["pitch"], dtype=np.float64)
    n = source_frames("tactical")
    _, ball, _ = load_tracking(cache_dir="data/work/cache")
    matrices, _, _ = load_homography(source="tactical")
    graded = np.zeros(n, dtype=bool)
    for _, a, z in blocks:
        graded[a:z + 1] = True
    return blocks, pitch, n, ball[:, :2].astype(np.float64), matrices, graded


def to_pixel(matrices, normalised):
    v = np.einsum("nij,nj->ni", matrices, np.column_stack([normalised, np.ones(len(normalised))]))
    return v[:, :2] / v[:, 2:3]


def to_pitch(matrix, pixels, pitch):
    v = (np.linalg.inv(matrix) @ np.column_stack([pixels, np.ones(len(pixels))]).T).T
    return v[:, :2] / v[:, 2:3] * pitch


def held_out(frames):
    return (frames // HELD_BLOCK) % HELD_EVERY == HELD_PICK


def ball_pixels(ball, matrices):
    pixels = np.full((len(ball), 2), np.nan)
    seen = np.isfinite(ball[:, 0])
    pixels[seen] = to_pixel(matrices[seen], ball[seen])
    return pixels


def read_triplets(wanted):
    """Yield (t, frame[t-2], frame[t], frame[t+2]) for every wanted t, in one sequential decode."""
    wanted = np.asarray(sorted(wanted))
    need = set(wanted.tolist())
    capture = cv2.VideoCapture(asset_paths()["video"])
    window = deque(maxlen=2 * STEP + 1)
    t, last = -1, int(wanted[-1]) + STEP
    while t < last:
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Video ended at frame {}".format(t))
        t += 1
        window.append(frame)
        centre = t - STEP
        if len(window) == 2 * STEP + 1 and centre in need:
            yield centre, window[0], window[STEP], window[-1]
    capture.release()


def extract(_):
    blocks, pitch, n, ball, matrices, graded = setup()
    rng = np.random.default_rng(SEED)
    pixels = ball_pixels(ball, matrices)
    frames = np.arange(n)
    inside = (pixels[:, 0] > 8) & (pixels[:, 0] < 1912) & (pixels[:, 1] > 8) & (pixels[:, 1] < 1072)
    steady = np.zeros(n, dtype=bool)
    steady[STEP:-STEP] = np.isfinite(ball[:-2 * STEP, 0]) & np.isfinite(ball[2 * STEP:, 0])
    eligible = frames[graded & inside & steady & ~held_out(frames)]
    chosen = np.sort(rng.choice(eligible, N_TRAIN, replace=False))
    crops = np.zeros((2 * N_TRAIN, CROP, CROP, 9), dtype=np.uint8)
    centre_xy = np.full((2 * N_TRAIN, 2), -1.0, dtype=np.float32)
    index = {int(t): i for i, t in enumerate(chosen)}
    for t, before, now, after in read_triplets(chosen):
        i = index[t]
        bx, by = pixels[t]
        x0 = int(np.clip(bx - rng.integers(24, CROP - 24), 0, 1920 - CROP))
        y0 = int(np.clip(by - rng.integers(24, CROP - 24), 0, 1080 - CROP))
        crops[2 * i] = np.concatenate([f[y0:y0 + CROP, x0:x0 + CROP] for f in (before, now, after)], axis=2)
        centre_xy[2 * i] = (bx - x0, by - y0)
        for _ in range(20):         # a pitch location away from the ball
            spot = to_pixel(matrices[t:t + 1], rng.random((1, 2)))[0]
            if 0 <= spot[0] < 1920 and 0 <= spot[1] < 1080 and np.hypot(*(spot - (bx, by))) > 1.5 * CROP:
                break
        x1 = int(np.clip(spot[0] - CROP // 2, 0, 1920 - CROP))
        y1 = int(np.clip(spot[1] - CROP // 2, 0, 1080 - CROP))
        if abs(bx - x1 - CROP / 2) < CROP / 2 and abs(by - y1 - CROP / 2) < CROP / 2:
            centre_xy[2 * i + 1] = (bx - x1, by - y1)   # the ball fell in the crop after clipping
        crops[2 * i + 1] = np.concatenate([f[y1:y1 + CROP, x1:x1 + CROP] for f in (before, now, after)], axis=2)
        if i % 2000 == 0:
            print("EXTRACT", i, t, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    np.save(ROOT / "train_crops.npy", crops)
    np.save(ROOT / "train_centres.npy", centre_xy)
    print("EXTRACT done", crops.shape, int((centre_xy[:, 0] >= 0).sum()), "crops with a ball", flush=True)


SNAP_RADIUS = 24               # the tracking's ball is smoothed: the real one sits within this many pixels


def blob_response(gray):
    """Determinant-of-Hessian response to compact bright blobs (a line gives almost none)."""
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.5)
    xx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
    yy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
    xy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
    return np.where((xx < 0) & (yy < 0), np.sqrt(np.clip(xx * yy - xy * xy, 0, None)), 0.0)


def snap(args):
    """Move each pseudo-label from the smoothed tracking position onto the ball actually seen.

    Round 'blob' uses a generic bright-blob filter; round 'net' uses the detector trained on the
    previous round. Output rows are (x, y, kind): 1 ball at (x, y), 0 no ball in the crop,
    2 ball expected near (x, y) but not found (that neighbourhood is ignored by the loss).
    """
    crops = np.load(ROOT / "train_crops.npy", mmap_mode="r")
    centres = np.load(ROOT / "train_centres.npy")
    labels = np.zeros((len(crops), 3), dtype=np.float32)
    yy, xx = np.mgrid[0:CROP, 0:CROP]
    has = np.flatnonzero(centres[:, 0] >= 0)
    if args.round == "net":
        device = torch.device(args.device)
        net = BallNet()
        net.load_state_dict(torch.load(ROOT / "ball_detector.pt", map_location="cpu"))
        net = net.to(device).eval()
    for start in range(0, len(has), 256):
        rows = has[start:start + 256]
        x = np.asarray(crops[rows])
        if args.round == "net":
            with torch.inference_mode():
                heat = torch.sigmoid(net(torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float().div(255.0)))[:, 0]
            response = nn.functional.interpolate(heat[:, None], scale_factor=2, mode="bilinear")[:, 0].cpu().numpy()
            accept = 0.3
        else:
            response = np.stack([blob_response(c[:, :, 3:6].min(axis=2).astype(np.float32)) for c in x])
            accept = 12.0
        for j, i in enumerate(rows):
            cx, cy = centres[i]
            near = (xx - cx) ** 2 + (yy - cy) ** 2 <= SNAP_RADIUS ** 2
            masked = np.where(near, response[j], -1.0)
            k = int(masked.argmax())
            if masked.flat[k] >= accept:
                labels[i] = (k % CROP, k // CROP, 1)
            else:
                labels[i] = (cx, cy, 2)
    np.save(ROOT / "train_labels.npy", labels)
    moved = np.linalg.norm(labels[has, :2] - centres[has], axis=1)[labels[has, 2] == 1]
    print("SNAP", args.round, "found", float((labels[has, 2] == 1).mean()), "median shift px", float(np.median(moved)),
          "p90", float(np.percentile(moved, 90)), flush=True)


def focal_loss(logits, target, weight):
    p = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
    positive = (target > 0.99).float()
    loss = -(positive * (1 - p) ** 2 * torch.log(p) + (1 - positive) * (1 - target) ** 4 * p ** 2 * torch.log(1 - p))
    return (loss * weight).sum() / positive.sum().clamp(min=1.0)


def train(args):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device(args.device)
    crops = np.load(ROOT / "train_crops.npy", mmap_mode="r")
    centres = np.load(ROOT / "train_labels.npy")
    net = BallNet().to(device).to(memory_format=torch.channels_last)
    steps = args.epochs * (len(crops) // args.batch)
    optimiser = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, 3e-3, total_steps=steps, pct_start=0.1)
    half = CROP // 2
    grid_y, grid_x = np.mgrid[0:half, 0:half].astype(np.float32)
    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(crops))
        total = 0.0
        for k in range(0, len(order) - args.batch + 1, args.batch):
            pick = np.sort(order[k:k + args.batch])
            x = np.asarray(crops[pick])
            c = centres[pick].copy()
            flip = rng.random(len(pick)) < 0.5
            x[flip] = x[flip][:, :, ::-1]
            c[flip & (c[:, 2] > 0), 0] = CROP - 1 - c[flip & (c[:, 2] > 0), 0]
            reverse = rng.random(len(pick)) < 0.5
            x[reverse] = x[reverse][..., [6, 7, 8, 3, 4, 5, 0, 1, 2]]
            target = np.zeros((len(pick), 1, half, half), dtype=np.float32)
            weight = np.ones_like(target)
            for j in np.flatnonzero(c[:, 2] > 0):
                cx, cy = c[j, :2] / 2
                distance = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
                if c[j, 2] == 1:
                    target[j, 0] = np.exp(-distance / 2.0)
                    target[j, 0, int(np.clip(round(cy), 0, half - 1)), int(np.clip(round(cx), 0, half - 1))] = 1.0
                else:
                    weight[j, 0][distance <= (SNAP_RADIUS / 2) ** 2] = 0.0
            xb = torch.from_numpy(np.ascontiguousarray(x)).to(device).permute(0, 3, 1, 2).float().div(255.0)
            xb = xb * (1 + 0.2 * (torch.rand(len(pick), 1, 1, 1, device=device) - 0.5))
            loss = focal_loss(net(xb.contiguous(memory_format=torch.channels_last)), torch.from_numpy(target).to(device),
                              torch.from_numpy(weight).to(device))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
            step += 1
        print("TRAIN epoch", epoch, "loss", round(total / (len(order) // args.batch), 4), flush=True)
    torch.save(net.state_dict(), ROOT / "ball_detector.pt")


def heldout_windows(ball, graded, n):
    """100-frame stretches with a tracked ball inside held-out blocks: pseudo-gaps for checking."""
    windows = []
    for start in range(0, n - 100, 250):
        span = np.arange(start, start + 100)
        if held_out(span).all() and graded[span].all() and np.isfinite(ball[start - STEP:start + 100 + STEP, 0]).all():
            windows.append((start, start + 100))
    return windows


def detect(args):
    blocks, pitch, n, ball, matrices, graded = setup()
    device = torch.device(args.device)
    net = BallNet()
    net.load_state_dict(torch.load(ROOT / "ball_detector.pt", map_location="cpu"))
    net = net.to(device).to(memory_format=torch.channels_last).eval()
    missing = graded & ~np.isfinite(ball[:, 0])
    wanted = set(np.flatnonzero(missing).tolist())
    for a, z in heldout_windows(ball, graded, n):
        wanted.update(range(a, z))
    wanted = sorted(t for t in wanted if STEP <= t < n - STEP)
    if args.limit:
        wanted = wanted[:args.limit]
    frames = np.array(wanted)
    xy = np.full((len(frames), TOP_K, 2), np.nan, dtype=np.float32)
    score = np.zeros((len(frames), TOP_K), dtype=np.float32)
    batch, slots = [], []

    def flush():
        x = torch.from_numpy(np.stack(batch)).to(device).permute(0, 3, 1, 2).float().div(255.0)
        with torch.inference_mode():
            heat = torch.sigmoid(net(x.contiguous(memory_format=torch.channels_last)))[:, 0]
            peak = (heat == nn.functional.max_pool2d(heat[:, None], 5, 1, 2)[:, 0]) & (heat > 0.02)
            values, where = (heat * peak).flatten(1).topk(4 * TOP_K, dim=1)
        values, where = values.cpu().numpy(), where.cpu().numpy()
        for row, i in enumerate(slots):
            keep = values[row] > 0
            px = np.column_stack([where[row][keep] % 960, where[row][keep] // 960]) * 2.0 + 1.0
            if len(px):
                metres = to_pitch(matrices[frames[i]], px, pitch)
                ok = ((metres[:, 0] > -PITCH_MARGIN) & (metres[:, 0] < pitch[0] + PITCH_MARGIN)
                      & (metres[:, 1] > -PITCH_MARGIN) & (metres[:, 1] < pitch[1] + PITCH_MARGIN))
                px, v = px[ok][:TOP_K], values[row][keep][ok][:TOP_K]
                xy[i, :len(px)] = px
                score[i, :len(px)] = v
        batch.clear()
        slots.clear()

    position = {int(t): i for i, t in enumerate(frames)}
    for count, (t, before, now, after) in enumerate(read_triplets(frames)):
        batch.append(np.concatenate([before, now, after], axis=2))
        slots.append(position[t])
        if len(batch) == args.batch:
            flush()
        if count % 5000 == 0:
            print("DETECT", count, "of", len(frames), flush=True)
    if batch:
        flush()
    np.savez_compressed(ROOT / "detections.npz", frames=frames, xy=xy, score=score)
    print("DETECT done", len(frames), "frames; top score >0.5 on", float((score[:, 0] > 0.5).mean()), flush=True)


def link_run(candidates, scores, anchors):
    """Best path through per-frame candidates (or 'no ball'), in pixels.

    candidates: list of (k, 2) arrays; scores: list of (k,) arrays in (0, 1);
    anchors: (start_xy or None, end_xy or None) known positions just outside the run.
    Returns (T, 2) pixels, NaN where the path says there is no ball.
    """
    length = len(candidates)
    start, end = anchors
    states = [np.vstack([c, [[np.nan, np.nan]]]) for c in candidates]     # last state = no ball
    unary = [np.r_[np.clip(np.log(np.clip(s, 1e-4, 1 - 1e-4)) - np.log1p(-np.clip(s, 1e-4, 1 - 1e-4)), -4, 6), 0.0]
             for s in scores]

    def transition(a, b):
        cost = np.full((len(a), len(b)), SWITCH_COST)
        cost[-1, -1] = 0.0
        d = np.linalg.norm(a[:-1, None, :] - b[None, :-1, :], axis=2)
        cost[:-1, :-1] = np.where(d <= MAX_STEP_PX, (d / MAX_STEP_PX) ** 2, np.inf)
        return cost

    first = unary[0].copy()
    if start is not None:
        first -= transition(np.vstack([[start], [[np.nan, np.nan]]]), states[0])[0]
    best, back = [first], []
    for t in range(1, length):
        total = best[-1][:, None] - transition(states[t - 1], states[t])
        back.append(total.argmax(axis=0))
        best.append(total.max(axis=0) + unary[t])
    final = best[-1].copy()
    if end is not None:
        final -= transition(states[-1], np.vstack([[end], [[np.nan, np.nan]]]))[:, 0]
    path = np.empty(length, dtype=int)
    path[-1] = int(final.argmax())
    for t in range(length - 2, -1, -1):
        path[t] = back[t][path[t + 1]]
    out = np.array([states[t][path[t]] for t in range(length)])
    seen = np.isfinite(out[:, 0])
    d = np.diff(np.r_[0, seen.astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        touches_edge = (a == 0 and start is not None) or (z == length and end is not None)
        if z - a < MIN_TRACK and not touches_edge:
            out[a:z] = np.nan
    return out


def bridge(track):
    """Linear interpolation over interior holes of at most BRIDGE frames."""
    seen = np.isfinite(track[:, 0])
    d = np.diff(np.r_[1, seen.astype(np.int8), 1])
    for a, z in zip(np.flatnonzero(d == -1), np.flatnonzero(d == 1)):
        if 0 < a and z < len(track) and z - a <= BRIDGE:
            w = (np.arange(a, z) - (a - 1)) / (z - a + 1)
            track[a:z] = track[a - 1] + w[:, None] * (track[z] - track[a - 1])
    return track


def painted_spots(pitch):
    return np.array([[11.0, pitch[1] / 2], [pitch[0] - 11.0, pitch[1] / 2], [pitch[0] / 2, pitch[1] / 2]])


def link(args):
    blocks, pitch, n, ball, matrices, graded = setup()
    saved = np.load(ROOT / "detections.npz")
    frames, xy, score = saved["frames"], saved["xy"], saved["score"].copy()
    row = np.full(n, -1)
    row[frames] = np.arange(len(frames))
    pixels = ball_pixels(ball, matrices)
    spots = painted_spots(pitch)
    # Candidates that can never be trusted: weak ones and the painted spots.
    score[score < args.min_score] = 0.0
    for r, t in enumerate(frames):
        keep = score[r] > 0
        if keep.any():
            metres = to_pitch(matrices[t], xy[r][keep], pitch)
            on_spot = (np.linalg.norm(metres[:, None, :] - spots[None], axis=2) < SPOT_RADIUS).any(axis=1)
            score[r][np.flatnonzero(keep)[on_spot]] = 0.0

    def run(a, z, use_anchors=True, hide=None):
        cands, scs = [], []
        for t in range(a, z):
            r = row[t]
            keep = score[r] > 0 if r >= 0 else np.zeros(0, dtype=bool)
            if r >= 0 and hide is not None:
                keep = keep & ~(np.linalg.norm(xy[r] - hide[t - a], axis=1) < 40)
            cands.append(xy[r][keep] if r >= 0 else np.zeros((0, 2)))
            scs.append(score[r][keep] if r >= 0 else np.zeros(0))
        anchors = (pixels[a - 1] if a > 0 and np.isfinite(pixels[a - 1, 0]) else None,
                   pixels[z] if z < n and np.isfinite(pixels[z, 0]) else None) if use_anchors else (None, None)
        track = bridge(link_run(cands, scs, anchors))
        metres = np.full((z - a, 2), np.nan)
        for i in np.flatnonzero(np.isfinite(track[:, 0])):
            metres[i] = to_pitch(matrices[a + i], track[i:i + 1], pitch)[0]
        # A track that never enters the field of play is a spare ball or a boot on the touchline.
        seen = np.isfinite(metres[:, 0])
        d = np.diff(np.r_[0, seen.astype(np.int8), 0])
        for i, j in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            inside = ((metres[i:j, 0] > 0) & (metres[i:j, 0] < pitch[0]) & (metres[i:j, 1] > 0) & (metres[i:j, 1] < pitch[1]))
            if not inside.any():
                metres[i:j] = np.nan
        return metres

    # Self-check on held-out stretches where the tracking has the ball (never used in training):
    # (a) does the linked track agree with the tracking; (b) with the true ball's detections
    # hidden, how often does the linker still invent a track (false-track rate).
    errors, covered, total, invented, top1 = [], 0, 0, 0, []
    for a, z in heldout_windows(ball, graded, n):
        if (row[a:z] < 0).any():
            continue
        truth = ball[a:z] * pitch
        metres = run(a, z, use_anchors=False)
        seen = np.isfinite(metres[:, 0])
        errors += np.linalg.norm(metres[seen] - truth[seen], axis=1).tolist()
        covered += int(seen.sum())
        total += z - a
        invented += int(np.isfinite(run(a, z, use_anchors=False, hide=pixels[a:z])[:, 0]).sum())
        for t in range(a, z):
            r = row[t]
            top1.append(np.hypot(*(xy[r, 0] - pixels[t])) if saved["score"][r, 0] > 0 else np.inf)
    errors, top1 = np.array(errors), np.array(top1)
    check = dict(heldout_frames=total, linked_share=covered / max(total, 1),
                 linked_within_3m=float((errors < 3.0).mean()) if len(errors) else None,
                 linked_within_6m=float((errors < 6.0).mean()) if len(errors) else None,
                 linked_median_error_m=float(np.median(errors)) if len(errors) else None,
                 false_track_share=invented / max(total, 1),
                 top1_within_30px=float((top1 < 30).mean()), top1_missing=float(np.isinf(top1).mean()))
    filled = np.full((n, 2), np.nan)
    missing = graded & ~np.isfinite(ball[:, 0])
    d = np.diff(np.r_[0, missing.astype(np.int8), 0])
    gaps = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    for a, z in gaps:
        a2, z2 = max(a, STEP), min(z, n - STEP)
        if z2 > a2 and (row[a2:z2] >= 0).all():
            filled[a2:z2] = run(a2, z2)
    got = np.isfinite(filled[:, 0])
    report = dict(min_score=args.min_score, check=check, gaps=len(gaps), gap_frames=int(missing.sum()),
                  filled_frames=int(got.sum()), filled_share=float(got.sum() / max(missing.sum(), 1)),
                  challenge_labels_read=False)
    if not args.dry:
        np.savez_compressed(ROOT / "video_ball.npz", ball=filled.astype(np.float32))
        (ROOT / "video_ball.json").write_text(json.dumps(report, indent=2))
    print("LINK", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["extract", "snap", "train", "detect", "link"])
    parser.add_argument("--round", choices=["blob", "net"], default="blob")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--dry", action="store_true", help="link: report only, write nothing")
    args = parser.parse_args()
    if args.batch is None:
        args.batch = 64 if args.stage == "train" else 4
    dict(extract=extract, snap=snap, train=train, detect=detect, link=link)[args.stage](args)


if __name__ == "__main__":
    main()
