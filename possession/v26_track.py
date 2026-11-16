"""V26 tracker: per-frame detector peaks -> one ball track, label-free validation, visual check.

Peaks are mapped to the pitch with the provided homography (ground plane), linked into short
tracklets, and a dynamic programme keeps the set of tracklets that is physically consistent
(no teleporting) and consistent with the tracking's ball at both ends of every gap.

Validation never reads a possession label: on held-out minutes (never seen by the detector)
the tracking's ball is hidden, the tracker runs exactly as on a real gap, and its output is
compared with the hidden ball.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from games import asset_paths
from .video_ball import ROOT, setup, held_out

MARGIN_M = 6.0                  # keep balls up to this far outside the lines
S_MIN = 0.10                    # peaks below this never enter a tracklet
LINK_GAP = 3                    # frames a tracklet may miss
V_LINK = 2.0                    # m per frame between consecutive tracklet points (ground-plane projection)
V_MAX = 1.6                     # m per frame between tracklets (40 m/s)
SLACK_M = 2.5
FAR_GAP = 70                    # beyond this many frames any jump is physically possible
CHUNK = 8
MIN_POINTS = 6
BRIDGE = 12
SMOOTH = 5
HELD_FAR_M = 5.0
CONTRADICT_M = 6.0
CONTRADICT_FRAMES = 8
COLUMN_BASE, COLUMN_SLOPE = 20.0, 0.15
FLIGHT_TOL_M = 1.0              # a smooth piece of track deviates less than this from a constant-acceleration fit
FLIGHT_ACC = 6.0                # m/s2 of apparent acceleration towards the camera that marks a flight
FLIGHT_MIN = 8                  # frames
FLIGHT_COS = 0.7                # cosine between the apparent acceleration and the direction to the camera
FLIGHT_MODE = "remove"          # "chord": straight line between take-off and landing; "remove": leave a hole
CONTACT_M = 2.5                 # a ball this close to a player (on the ground plane) may be at his feet
CONTACT_REACH = 25              # frames searched for the player who struck / received a flying ball
FPS = 25


class Geometry:
    """Ground plane <-> image for the tactical camera."""

    def __init__(self, matrices, pitch, players=None):
        self.matrices, self.pitch = matrices, np.asarray(pitch, dtype=np.float64)
        self.inverse = np.linalg.inv(matrices)
        self.players = players          # (n, slots, 2) metres, NaN when not tracked

    def nearest_player(self, frames, metres):
        if self.players is None:
            return np.full(len(frames), np.inf)
        d = np.linalg.norm(self.players[frames] - metres[:, None, :], axis=2)
        return np.where(np.isfinite(d).any(axis=1), np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1), np.inf)

    def pixel(self, frames, metres):
        v = np.einsum("nij,nj->ni", self.matrices[frames], np.column_stack([metres / self.pitch, np.ones(len(metres))]))
        return v[:, :2] / v[:, 2:3]

    def ground(self, frames, pixels):
        v = np.einsum("nij,nj->ni", self.inverse[frames], np.column_stack([pixels, np.ones(len(pixels))]))
        return v[:, :2] / v[:, 2:3] * self.pitch

    def above(self, frames, ground_m, seen_m):
        """Is `seen_m` (image point mapped to the ground) straight above `ground_m` in the image?

        That is what a ball in the air looks like: the tracking reports the point under it, the
        homography maps the ball itself to a point farther from the camera.
        """
        frames = np.atleast_1d(frames)
        a = self.pixel(frames, np.atleast_2d(ground_m))
        b = self.pixel(frames, np.atleast_2d(seen_m))
        up = a[:, 1] - b[:, 1]
        return (up > 0) & (np.abs(b[:, 0] - a[:, 0]) <= COLUMN_BASE + COLUMN_SLOPE * up)

    def away(self, frame, metres):
        """Unit vector on the pitch pointing away from the camera at this ground point."""
        f = np.array([frame])
        px = self.pixel(f, metres[None])
        d = self.ground(f, px - [[0.0, 25.0]])[0] - metres
        return d / max(np.linalg.norm(d), 1e-9)


def candidates(peaks, matrices, pitch):
    """(n, K, 2) metres and (n, K) scores; score 0 where unusable."""
    n, k, _ = peaks.shape
    inverse = np.linalg.inv(matrices)
    homogeneous = np.concatenate([peaks[:, :, :2].astype(np.float64), np.ones((n, k, 1))], axis=2)
    v = np.einsum("nij,nkj->nki", inverse, homogeneous)
    with np.errstate(divide="ignore", invalid="ignore"):
        metres = v[:, :, :2] / v[:, :, 2:3] * pitch
    score = peaks[:, :, 2].astype(np.float64).copy()
    ok = (np.isfinite(metres).all(axis=2) & (metres[:, :, 0] > -MARGIN_M) & (metres[:, :, 0] < pitch[0] + MARGIN_M)
          & (metres[:, :, 1] > -MARGIN_M) & (metres[:, :, 1] < pitch[1] + MARGIN_M))
    score[~ok] = 0.0
    return metres, score


def build_tracklets(metres, score, frames):
    """Greedy frame-to-frame linking. Returns a list of (frames, xy, scores, pixels-index) arrays."""
    active, finished = [], []
    for f in frames:
        keep = np.flatnonzero(score[f] >= S_MIN)
        still = []
        for t in active:
            (still if f - t[0][-1] <= LINK_GAP else finished).append(t)
        active = still
        if len(keep) == 0:
            continue
        xy = metres[f, keep]
        taken = np.zeros(len(keep), dtype=bool)
        if active:
            last = np.array([t[1][-1] for t in active])
            gap = f - np.array([t[0][-1] for t in active])
            distance = np.linalg.norm(xy[:, None, :] - last[None, :, :], axis=2)
            allowed = distance <= V_LINK * gap[None, :] + 0.5
            used = np.zeros(len(active), dtype=bool)
            for flat in np.argsort(distance, axis=None):
                i, j = divmod(int(flat), len(active))
                if not allowed[i, j]:
                    break
                if taken[i] or used[j]:
                    continue
                taken[i] = used[j] = True
                active[j][0].append(f)
                active[j][1].append(xy[i])
                active[j][2].append(score[f, keep[i]])
                active[j][3].append(keep[i])
        for i in np.flatnonzero(~taken):
            active.append(([f], [xy[i]], [score[f, keep[i]]], [keep[i]]))
    finished += active
    return [(np.array(a), np.array(b), np.array(c), np.array(d)) for a, b, c, d in finished if len(a) >= MIN_POINTS]


def chunk_table(tracklets, threshold, min_weight, usable, known=None, geometry=None):
    """Split tracklets at unusable frames and into chunks of CHUNK points.

    `known` is the tracking's ball in metres (NaN where hidden or missing): a tracklet that
    coexists with the tracked ball somewhere else is a second ball, a boot or a mark, and is
    dropped whole, including the part of it that extends into a gap.
    Returns arrays: start, end, start_xy, end_xy, weight, tracklet, first_point, last_point.
    """
    rows = []
    for k, (frames, xy, scores, _) in enumerate(tracklets):
        if (scores - threshold).sum() < min_weight:
            continue
        if known is not None:
            apart = np.linalg.norm(xy - known[frames], axis=1)
            overlap = np.isfinite(apart)
            elsewhere = apart[overlap] > CONTRADICT_M
            if elsewhere.any() and geometry is not None:
                elsewhere &= ~geometry.above(frames[overlap], known[frames[overlap]], xy[overlap])
            far = int(elsewhere.sum())
            if far >= CONTRADICT_FRAMES and far >= 0.5 * overlap.sum():
                continue
        ok = usable[frames]
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
        for a, z in zip(edges[::2], edges[1::2]):
            for lo in range(a, z, CHUNK):
                hi = min(lo + CHUNK, z)
                rows.append((frames[lo], frames[hi - 1], xy[lo, 0], xy[lo, 1], xy[hi - 1, 0], xy[hi - 1, 1],
                             (scores[lo:hi] - threshold).sum(), k, lo, hi))
    table = np.array(rows, dtype=np.float64).reshape(-1, 10)
    return table[np.lexsort((table[:, 1], table[:, 0]))]


def feasible(gap, distance):
    return (gap >= FAR_GAP) | (distance <= V_MAX * gap + SLACK_M)


def select(table, first, last, start_anchor, end_anchor, geometry=None):
    """Best physically consistent chain of chunks inside frames [first, last]. Returns row indices."""
    rows = np.flatnonzero((table[:, 0] >= first) & (table[:, 1] <= last))
    if len(rows) == 0:
        return rows
    t = table[rows]
    n = len(t)
    start, end, weight = t[:, 0], t[:, 1], t[:, 6]
    best = np.full(n, -np.inf)
    back = np.full(n, -1)
    by_end = np.argsort(end, kind="stable")
    end_sorted = end[by_end]
    pointer, far_best, far_arg = 0, -np.inf, -1
    for i in range(n):
        while pointer < n and end_sorted[pointer] <= start[i] - FAR_GAP:
            j = by_end[pointer]
            if best[j] > far_best:
                far_best, far_arg = best[j], j
            pointer += 1
        value, arg = -np.inf, -1
        if (start_anchor is None or feasible(start[i] - (first - 1), np.hypot(*(t[i, 2:4] - start_anchor)))
                or (geometry is not None and start[i] - first < FAR_GAP
                    and geometry.above(int(start[i]), start_anchor, t[i, 2:4])[0])):
            value = 0.0
        if far_best > value:
            value, arg = far_best, far_arg
        lo = np.searchsorted(end_sorted, start[i] - FAR_GAP, side="right")
        hi = np.searchsorted(end_sorted, start[i], side="left")
        if hi > lo:
            near = by_end[lo:hi]
            ok = feasible(start[i] - end[near], np.hypot(t[near, 4] - t[i, 2], t[near, 5] - t[i, 3])) & np.isfinite(best[near])
            if ok.any():
                j = near[ok][np.argmax(best[near][ok])]
                if best[j] > value:
                    value, arg = best[j], j
        if np.isfinite(value):
            best[i] = value + weight[i]
            back[i] = arg
    final = best.copy()
    if end_anchor is not None:
        reachable = feasible((last + 1) - end, np.hypot(t[:, 4] - end_anchor[0], t[:, 5] - end_anchor[1]))
        if geometry is not None:
            reachable |= geometry.above(end.astype(int), np.repeat(end_anchor[None], n, axis=0), t[:, 4:6])
        final[~reachable] = -np.inf
    if not np.isfinite(final).any() or final.max() <= 0:
        return rows[:0]
    chain = []
    i = int(np.argmax(final))
    while i >= 0:
        chain.append(i)
        i = int(back[i])
    return rows[np.array(chain[::-1])]


def quadratic(tt, values):
    """Least-squares constant-acceleration fit: (coefficients (3, 2), residual norms)."""
    design = np.column_stack([tt * tt, tt, np.ones(len(tt))])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return coefficients, np.linalg.norm(values - design @ coefficients, axis=1)


def smooth_pieces(t, xy, lo, hi, out):
    """Recursive split of xy[lo:hi+1] into pieces that follow a constant-acceleration model.

    Each piece is (lo, hi, acceleration in m/s2). A piece that fits is one free flight, one
    roll or one rest; a kick, a bounce or a touch is where two pieces meet.
    """
    tt = (t[lo:hi + 1] - t[lo]).astype(np.float64)
    if hi - lo < 9:
        out.append((lo, hi, np.zeros(2)))
        return
    coefficients, residual = quadratic(tt, xy[lo:hi + 1])
    chord = xy[lo] + np.outer(tt / tt[-1], xy[hi] - xy[lo])
    bow = np.linalg.norm(xy[lo:hi + 1] - chord, axis=1).max()
    if residual.max() <= FLIGHT_TOL_M + 0.05 * bow:
        out.append((lo, hi, 2.0 * coefficients[0] * FPS * FPS))
        return
    best, cut = np.inf, None
    for k in range(4, hi - lo - 3, max(1, (hi - lo) // 60)):          # the breakpoint that explains most
        cost = (quadratic(tt[:k + 1], xy[lo:lo + k + 1])[1] ** 2).sum() + (quadratic(tt[k:], xy[lo + k:hi + 1])[1] ** 2).sum()
        if cost < best:
            best, cut = cost, lo + k
    smooth_pieces(t, xy, lo, cut, out)
    smooth_pieces(t, xy, cut, hi, out)


def flatten_flights(t, xy, fixed_first, fixed_last, geometry):
    """Put airborne stretches back on the ground.

    A ball in flight maps through the homography to a point that swings away from the camera
    and back with a large apparent acceleration towards the camera (g times distance over
    camera height); its true ground track is the straight chord between take-off and landing.
    Flights whose take-off or landing was not seen cannot be corrected and are removed.
    Returns (xy, keep).
    """
    xy = xy.copy()
    keep = np.ones(len(t), dtype=bool)
    if len(t) < FLIGHT_MIN:
        return xy, keep
    pieces = []
    smooth_pieces(t, xy, 0, len(t) - 1, pieces)
    flying = np.zeros(len(t), dtype=bool)
    for lo, hi, acceleration in pieces:
        if t[hi] - t[lo] < FLIGHT_MIN - 2 or np.linalg.norm(acceleration) < FLIGHT_ACC:
            continue
        middle = (lo + hi) // 2
        towards_camera = -geometry.away(int(t[middle]), xy[middle])
        if acceleration @ towards_camera >= FLIGHT_COS * np.linalg.norm(acceleration):
            flying[lo:hi + 1] = True
    if not flying.any():
        return xy, keep
    near = geometry.nearest_player(t, xy) <= CONTACT_M
    edges = np.flatnonzero(np.diff(np.r_[0, flying.astype(np.int8), 0]))
    for lo, hi in zip(edges[::2], edges[1::2] - 1):
        # The launch and the arrival are not constant-acceleration: reach back / forward to the
        # player who struck or received the ball.
        if not near[lo]:
            back = np.flatnonzero(near[max(lo - CONTACT_REACH, 0):lo])
            if len(back):
                lo = max(lo - CONTACT_REACH, 0) + back[-1]
        if not near[hi]:
            ahead = np.flatnonzero(near[hi + 1:hi + 1 + CONTACT_REACH])
            if len(ahead):
                hi = hi + 1 + ahead[0]
        if FLIGHT_MODE == "remove" or (lo == 0 and not fixed_first) or (hi == len(t) - 1 and not fixed_last):
            keep[lo + 1:hi] = False
            if lo == 0:
                keep[0] = False
            if hi == len(t) - 1:
                keep[hi] = False
            continue
        w = (t[lo:hi + 1] - t[lo]) / (t[hi] - t[lo])
        xy[lo:hi + 1] = xy[lo] + np.outer(w, xy[hi] - xy[lo])
    return xy, keep


def track(tracklets, table, hidden, atd_m, n, geometry=None, filter_flights=True):
    """Fill every run of `hidden` frames. Returns xy (n, 2) metres, score (n,), kind (n,) 2 seen / 3 bridged."""
    xy = np.full((n, 2), np.nan)
    score = np.zeros(n)
    kind = np.zeros(n, dtype=np.int8)
    edges = np.flatnonzero(np.diff(np.r_[0, hidden.astype(np.int8), 0]))
    for a, z in zip(edges[::2], edges[1::2] - 1):
        before = atd_m[a - 1] if a > 0 and np.isfinite(atd_m[a - 1, 0]) else None
        after = atd_m[z + 1] if z + 1 < n and np.isfinite(atd_m[z + 1, 0]) else None
        chosen = select(table, a, z, before, after, geometry)
        previous = None
        for r in chosen:
            frames, points, scores, _ = tracklets[int(table[r, 7])]
            lo, hi = int(table[r, 8]), int(table[r, 9])
            f = frames[lo:hi]
            full = np.arange(f[0], f[-1] + 1)
            xy[full, 0] = np.interp(full, f, points[lo:hi, 0])
            xy[full, 1] = np.interp(full, f, points[lo:hi, 1])
            score[full] = np.interp(full, f, scores[lo:hi])
            kind[full] = 3
            kind[f] = 2
            if previous is not None and 1 < f[0] - previous <= BRIDGE + 1:
                hole = np.arange(previous + 1, f[0])
                w = (hole - previous) / (f[0] - previous)
                xy[hole] = xy[previous] + w[:, None] * (xy[f[0]] - xy[previous])
                score[hole] = min(score[previous], score[f[0]])
                kind[hole] = 3
            previous = f[-1]
    flights = dict(corrected=0, removed=0)
    if geometry is not None and filter_flights:
        seen = kind > 0
        edges = np.flatnonzero(np.diff(np.r_[0, seen.astype(np.int8), 0]))
        for a, z in zip(edges[::2], edges[1::2]):
            t = np.arange(a, z)
            points = xy[a:z]
            first = a > 0 and np.isfinite(atd_m[a - 1, 0])
            last = z < n and np.isfinite(atd_m[z, 0])
            if first:                                       # the tracking's ball is on the ground plane
                t, points = np.r_[a - 1, t], np.vstack([atd_m[a - 1], points])
            if last:
                t, points = np.r_[t, z], np.vstack([points, atd_m[z]])
            flat, keep = flatten_flights(t, points, first, last, geometry)
            flat, keep = flat[int(first):len(t) - int(last)], keep[int(first):len(t) - int(last)]
            flights["corrected"] += int((np.linalg.norm(flat - xy[a:z], axis=1) > 0.5).sum())
            flights["removed"] += int((~keep).sum())
            xy[a:z] = np.where(keep[:, None], flat, np.nan)
            kind[a:z] = np.where(keep, kind[a:z], 0)
            score[a:z] = np.where(keep, score[a:z], 0.0)
    track.flights = flights
    # Light smoothing inside continuous stretches (detector jitter would look like ball speed).
    seen = kind > 0
    edges = np.flatnonzero(np.diff(np.r_[0, seen.astype(np.int8), 0]))
    kernel = np.ones(SMOOTH) / SMOOTH
    for a, z in zip(edges[::2], edges[1::2]):
        if z - a >= SMOOTH:
            for c in range(2):
                padded = np.pad(xy[a:z, c], SMOOTH // 2, mode="edge")
                xy[a:z, c] = np.convolve(padded, kernel, mode="valid")
    return xy, score, kind


def prepare(detections):
    blocks, pitch, n, ball, matrices, graded = setup()
    saved = np.load(detections)
    peaks, done = saved["peaks"], saved["done"]
    metres, score = candidates(peaks, matrices, pitch)
    score[~done] = 0.0
    tracklets = build_tracklets(metres, score, np.flatnonzero(done))
    from .data import load_tracking
    players = load_tracking(cache_dir="data/work/cache")[0][:, :, :2].astype(np.float64) * pitch
    return dict(blocks=blocks, pitch=pitch, n=n, atd=ball * pitch, matrices=matrices, graded=graded, peaks=peaks,
                done=done, tracklets=tracklets, geometry=Geometry(matrices, pitch, players))


def heldout_check(state, threshold, min_weight):
    n, atd, done, graded = state["n"], state["atd"], state["done"], state["graded"]
    frames = np.arange(n)
    held = held_out(frames) & graded & done
    seen_atd = np.isfinite(atd[:, 0])
    hidden = (held | ~seen_atd) & done
    visible = np.where(held[:, None], np.nan, atd)
    geometry = state["geometry"] if state.get("flights", True) else None
    table = chunk_table(state["tracklets"], threshold, min_weight, hidden, visible, state["geometry"])
    xy, score, kind = track(state["tracklets"], table, hidden, visible, n, geometry,
                            filter_flights=state.get("filter_flights", True))
    compare = held & seen_atd
    found = compare & (kind > 0)
    error = np.linalg.norm(xy[found] - atd[found], axis=1)
    where = np.flatnonzero(found)
    airborne = state["geometry"].above(where, atd[where], xy[where]) if len(where) else np.zeros(0, dtype=bool)
    far = error > HELD_FAR_M
    blind = held & ~seen_atd
    return dict(threshold=threshold, min_weight=min_weight, heldout_frames_with_tracked_ball=int(compare.sum()),
                video_reports_ball=float(found.sum() / max(compare.sum(), 1)),
                within_1_5m_of_all=float((error <= 1.5).sum() / max(compare.sum(), 1)),
                within_3m_of_all=float((error <= 3.0).sum() / max(compare.sum(), 1)),
                within_1_5m_of_reported=float((error <= 1.5).mean()) if len(error) else None,
                within_3m_of_reported=float((error <= 3.0).mean()) if len(error) else None,
                far_over_5m_of_reported=float(far.mean()) if len(error) else None,
                far_airborne_of_reported=float((far & airborne).mean()) if len(error) else None,
                false_track_of_reported=float((far & ~airborne).mean()) if len(error) else None,
                flights=dict(track.flights),
                far_over_5m_of_all=float((error > HELD_FAR_M).sum() / max(compare.sum(), 1)),
                median_error_m=float(np.median(error)) if len(error) else None,
                heldout_frames_without_tracked_ball=int(blind.sum()),
                video_reports_ball_there=float((kind[blind] > 0).mean()) if blind.any() else None)


def read_codes(path, n):
    import pandas as pd
    table = pd.read_csv(path)
    column = [c for c in table.columns if table[c].astype(str).isin(list("HADX")).all()][0]
    return table[column].astype(str).to_numpy()[:n]


def link(args):
    state = prepare(ROOT / args.detections)
    n, atd, done, graded, pitch = state["n"], state["atd"], state["done"], state["graded"], state["pitch"]
    print("LINK tracklets", len(state["tracklets"]), flush=True)
    filtered = not getattr(args, "no_flight_filter", False)
    state["filter_flights"] = filtered
    thresholds = (0.2, 0.3, 0.4, 0.5, 0.6)
    sweep = []
    for threshold in thresholds:
        for min_weight in (1.0, 3.0, 6.0):
            row = heldout_check(state, threshold, min_weight)
            sweep.append(row)
            print("SWEEP", json.dumps(row), flush=True)
    # Operating point: the most permissive setting whose far reports (video ball more than
    # 5 m from the hidden tracking ball, whatever the reason) stay under --max-far of the frames
    # where it reports a ball. No possession label takes part.
    # The evidence a tracklet must accumulate (--min-weight) is fixed beforehand; only the
    # per-detection confidence threshold is chosen here.
    # Without the flight filter (--no-flight-filter) the confidence threshold is still the one
    # this rule picks WITH the filter: the filter-off sweep is dominated by airborne balls, whose
    # distance to the tracking's ground point says nothing about detector confidence.
    selection = sweep
    if not filtered:
        state["filter_flights"] = True
        selection = [heldout_check(state, t, args.min_weight) for t in thresholds]
        state["filter_flights"] = False
    allowed = [r for r in selection if r["far_over_5m_of_reported"] is not None and r["far_over_5m_of_reported"] <= args.max_far
               and r["min_weight"] == args.min_weight]
    if args.threshold is not None:
        selected = chosen = heldout_check(state, args.threshold, args.min_weight)
    elif allowed:
        selected = chosen = max(allowed, key=lambda r: r["video_reports_ball"])
    else:
        selected = chosen = min(selection, key=lambda r: r["far_over_5m_of_reported"])
    if not filtered and args.threshold is None:
        chosen = heldout_check(state, selected["threshold"], selected["min_weight"])
    threshold, min_weight = chosen["threshold"], chosen["min_weight"]
    seen_atd = np.isfinite(atd[:, 0])
    hidden = ~seen_atd & done
    table = chunk_table(state["tracklets"], threshold, min_weight, hidden, atd, state["geometry"])
    xy, score, kind = track(state["tracklets"], table, hidden, atd, n, state["geometry"], filter_flights=filtered)
    flights_in_fill = dict(track.flights)
    ball = np.where(seen_atd[:, None], atd, xy)
    source = np.where(seen_atd, 1, kind).astype(np.int8)
    confidence = np.where(seen_atd, 1.0, score).astype(np.float32)
    state["flights"] = False
    raw = heldout_check(state, threshold, min_weight)        # same point without the flight filter: what "far" is made of
    report = dict(operating_point=chosen, without_flight_filter=raw, sweep=sweep, max_far=args.max_far,
                  detections=str(args.detections), flight_mode=FLIGHT_MODE,
                  coverage_graded_tracking=float(seen_atd[graded].mean()),
                  coverage_graded_after=float((source[graded] > 0).mean()),
                  gap_frames_graded=int((graded & ~seen_atd).sum()),
                  gap_frames_filled=int((graded & ~seen_atd & (kind > 0)).sum()),
                  filled_outside_lines=int((graded & (kind > 0) & ((xy[:, 0] < 0) | (xy[:, 0] > pitch[0]) | (xy[:, 1] < 0)
                                                                 | (xy[:, 1] > pitch[1]))).sum()),
                  flights_in_fill=flights_in_fill, challenge_labels_read=False)
    if not filtered:
        report["flight_filter"] = False
        report["operating_point_selected_with_flight_filter"] = selected
    candidate = Path("evaluation/results/v24_camera_fusion.csv")
    if candidate.exists():
        codes = read_codes(candidate, n)
        gap = graded & ~seen_atd
        dead = codes == "D"
        report["gap_split_by_current_candidate"] = dict(
            dead_frames=int((gap & dead).sum()), dead_filled=float((kind[gap & dead] > 0).mean()),
            alive_frames=int((gap & ~dead).sum()), alive_filled=float((kind[gap & ~dead] > 0).mean()))
    out = ROOT / (args.out or "video_ball.npz")
    np.savez_compressed(out, ball=(ball / pitch).astype(np.float32), confidence=confidence,
                        source=source, video_metres=xy.astype(np.float32), pitch=pitch)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print("LINK", json.dumps({k: v for k, v in report.items() if k != "sweep"}, indent=2), flush=True)


def viz(args):
    """Annotated crops on frames where the tracking has no ball (debugging aid, not a label)."""
    blocks, pitch, n, ball, matrices, graded = setup()
    saved = np.load(ROOT / "video_ball.npz")
    source, track_m = saved["source"], saved["video_metres"].astype(np.float64)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.limit or 7)
    filled = np.flatnonzero(graded & (source >= 2))
    edges = np.linspace(0, len(filled), args.workers * 2 + 1).astype(int)        # spread across the match
    picks = [int(filled[rng.integers(a, z)]) for a, z in zip(edges[:-1], edges[1:])]
    capture = cv2.VideoCapture(asset_paths()["video"])
    for f in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = capture.read()
        v = matrices[f] @ np.array([track_m[f, 0] / pitch[0], track_m[f, 1] / pitch[1], 1.0])
        px, py = v[0] / v[2], v[1] / v[2]
        x0 = int(np.clip(px - 150, 0, 1920 - 300))
        y0 = int(np.clip(py - 150, 0, 1080 - 300))
        crop = cv2.resize(frame[y0:y0 + 300, x0:x0 + 300], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.circle(crop, (int((px - x0) * 3), int((py - y0) * 3)), 36, (0, 255, 255), 2)
        whole = frame.copy()
        cv2.circle(whole, (int(px), int(py)), 40, (0, 255, 255), 3)
        whole = cv2.resize(whole, (1600, 900))
        canvas = np.zeros((900, 2500, 3), dtype=np.uint8)
        canvas[:, :1600] = whole
        canvas[:, 1600:] = crop
        cv2.putText(canvas, "frame {} kind {} conf {:.2f} pitch ({:.1f},{:.1f}) m".format(
            f, int(source[f]), float(saved["confidence"][f]), track_m[f, 0], track_m[f, 1]), (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(str(out / "frame_{:06d}.jpg".format(f)), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("VIZ", len(picks), "images in", out, flush=True)
