"""V27 building blocks: fast decoder, fast spell matching, spell objects and merge actions.

Nothing here reads the challenge ground truth. Everything that is fitted or selected with this
module uses the three open Metrica sample games (event-derived references) only.

Codes: 0 = H, 1 = A, 2 = D, 3 = not graded / abstain.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import evaluation.possession_metrics as pm

FPS = 25
ROOT = Path("data/work/v27")
IOUS = pm.SPELL_IOUS
MODES = ("clean", "moderate", "stress")
REFERENCES = ("consistent", "original", "alternative")
DEFAULT = dict(c_ha=4.0, c_to_d=2.5, c_from_d=2.5, temp=0.2, gamma=0.85)


# -- emissions and decoding ------------------------------------------------------------------

def interpolate(probabilities, edges, frames, blocks):
    """25 Hz linear interpolation of the 5 Hz network output, per half (as decode_fine does)."""
    out = []
    for _, first, last in blocks:
        selected = (frames >= first) & (frames <= last)
        f = frames[selected]
        if not len(f):
            raise ValueError("No emission in half")
        grid = np.arange(first, last + 1)
        p = np.column_stack([np.interp(grid, f, probabilities[selected][:, j]) for j in range(3)])
        e = np.interp(grid, f, edges[selected])
        out.append((first, last, p, e))
    return out


def viterbi(p, edge, configs):
    """Batched 3-state Viterbi over a list of decoder configs. Returns (len(configs), T) int8.

    With DEFAULT it is exactly possession.v14_resolution.decode_fine(..., 'probability').
    """
    b = len(configs)
    transition = np.zeros((b, 3, 3))
    for i, c in enumerate(configs):
        transition[i] = [[0., c["c_ha"], c["c_to_d"]], [c["c_ha"], 0., c["c_to_d"]],
                         [c["c_from_d"], c["c_from_d"], 0.]]
    temp = np.array([c["temp"] for c in configs])[:, None, None]
    gamma = np.array([c["gamma"] for c in configs])[:, None, None]
    log_p = np.log(np.clip(p, 1e-5, 1))
    n = len(p)
    score = log_p[0][None] * temp[:, 0]
    back = np.zeros((n, b, 3), dtype=np.int8)
    index = np.arange(3)[None, :]
    rows = np.arange(b)[:, None]
    edge = edge.astype(np.float64)
    for t in range(1, n):
        previous = score[:, :, None] - transition * (1 - gamma * edge[t])
        k = previous.argmax(1)
        back[t] = k
        score = previous[rows, k, index] + log_p[t][None] * temp[:, 0]
        score -= score.max(1, keepdims=True)
    states = np.empty((b, n), dtype=np.int8)
    current = score.argmax(1)
    for t in range(n - 1, -1, -1):
        states[:, t] = current
        current = back[t, np.arange(b), current]
    return states


def decode(interpolated, n_frames, configs=(DEFAULT,)):
    codes = np.full((len(configs), n_frames), 3, dtype=np.int8)
    for first, last, p, e in interpolated:
        codes[:, first:last + 1] = viterbi(p, e, list(configs))
    return codes


# -- spells and matching ---------------------------------------------------------------------

def segments(codes, blocks):
    """[(start, stop, code)] half-open constant runs inside each half."""
    out = []
    for _, first, last in blocks:
        window = codes[first:last + 1]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)])) + first
        out += [(int(a), int(z), int(codes[a])) for a, z in zip(bounds[:-1], bounds[1:])]
    return out


def spells_of(segs):
    return [(a, z, k) for a, z, k in segs if k < 2]


def overlaps(gt, pred):
    """Same ordering as possession_metrics.spell_overlaps, without the quadratic scan."""
    out = []
    j0 = 0
    for i, (ga, gz, gk) in enumerate(gt):
        while j0 < len(pred) and pred[j0][1] <= ga:
            j0 += 1
        j = j0
        while j < len(pred) and pred[j][0] < gz:
            pa, pz, pk = pred[j]
            if pk == gk and pz > ga:
                out.append(((min(gz, pz) - max(ga, pa)) / float(max(gz, pz) - min(ga, pa)), i, j))
            j += 1
    out.sort(key=lambda item: (-item[0], item[1], item[2]))
    return out


def matching(gt, pred):
    """Per IoU threshold: (matched gt flags, matched pred flags), greedy like the official scorer."""
    pairs = overlaps(gt, pred)
    result = {}
    for threshold in IOUS:
        g = np.zeros(len(gt), dtype=bool)
        p = np.zeros(len(pred), dtype=bool)
        for iou, i, j in pairs:
            if iou < threshold:
                break
            if g[i] or p[j]:
                continue
            g[i] = True
            p[j] = True
        result[threshold] = (g, p)
    return result


def spell_f1(gt, pred):
    m = matching(gt, pred)
    total = len(gt) + len(pred)
    return float(np.mean([2.0 * m[t][0].sum() / total if total else 0.0 for t in IOUS]))


def official(reference, codes):
    """Official metric. reference: -1 where not graded."""
    gt = np.where(reference >= 0, reference, 3).astype(np.int8)
    result = pm.evaluate(gt, codes.astype(np.int8), bootstrap_n=0)
    return dict(spell_f1=float(result["spell_f1_mean"]), macro_f1=float(result["macro_f1"]),
                n_pred=int(result["n_pred_spells"]), n_gt=int(result["n_gt_spells"]))


def graded_blocks(reference):
    gt = np.where(reference >= 0, reference, 3).astype(np.int8)
    return [(i + 1, a, z - 1) for i, (a, z) in enumerate(pm.valid_runs(gt))]


def duration_stats(codes, blocks):
    spells = spells_of(segments(codes, blocks))
    d = np.array([z - a for a, z, _ in spells]) / FPS
    graded = np.concatenate([codes[a:z + 1] for _, a, z in blocks])
    live = float((graded < 2).sum()) / FPS / 60
    return dict(spells=len(d), per_live_minute=len(d) / live if live else 0.0, mean_s=float(d.mean()),
                median_s=float(np.median(d)), share_lt_1s=float((d < 1).mean()),
                share_lt_3s=float((d < 3).mean()), dead_share=float((graded == 2).mean()))


# -- data ------------------------------------------------------------------------------------

def metrica_condition(game, mode):
    """One open Metrica game under one observation regime: frozen-ensemble output + references."""
    meta = np.load(ROOT / f"metrica_{game}.npz")
    saved = np.load(f"data/work/v16/game{game}_{mode}_final.npz")
    blocks = [tuple(int(v) for v in b) for b in meta["blocks"]]
    n = len(meta["original"])
    return SimpleNamespace(game=game, mode=mode, blocks=blocks, n=n,
                           references={r: meta[r] for r in REFERENCES},
                           interpolated=interpolate(saved["probabilities"], saved["edges"], saved["frames"], blocks))


def all_conditions():
    return [metrica_condition(g, m) for g in (1, 2, 3) for m in MODES]


# -- spell objects and merge actions ---------------------------------------------------------

ACTIONS = ("absorb", "to_dead", "to_previous", "to_next")


def candidates(segs, max_frames):
    """Possible removals of one predicted possession: (segment index, action, new code).

    absorb       T-X-T  -> the surrounding team (three possessions become one)
    to_dead      X next to a stoppage (D-X-D, T-X-D, D-X-T) -> D
    to_previous  T-X-D  -> the previous team keeps the ball until the stoppage
    to_next      D-X-T  -> the next team has it from the restart
    A neighbour only counts inside the same half (segments are contiguous there).
    """
    out = []
    for i, (a, z, k) in enumerate(segs):
        if k > 1 or z - a > max_frames:
            continue
        previous = segs[i - 1][2] if i > 0 and segs[i - 1][1] == a else None
        following = segs[i + 1][2] if i + 1 < len(segs) and segs[i + 1][0] == z else None
        other = 1 - k
        if previous == other and following == other:
            out.append((i, "absorb", other))
        elif previous == 2 and following == 2:
            out.append((i, "to_dead", 2))
        elif previous == other and following == 2:
            out += [(i, "to_previous", other), (i, "to_dead", 2)]
        elif previous == 2 and following == other:
            out += [(i, "to_next", other), (i, "to_dead", 2)]
    return out


def relabel(segs, changes):
    """Apply {segment index: new code} and re-merge equal neighbours inside a half."""
    out = []
    for i, (a, z, k) in enumerate(segs):
        k = changes.get(i, k)
        if out and out[-1][2] == k and out[-1][1] == a:
            out[-1] = (out[-1][0], z, k)
        else:
            out.append((a, z, k))
    return out


def to_codes(segs, n_frames):
    codes = np.full(n_frames, 3, dtype=np.int8)
    for a, z, k in segs:
        codes[a:z] = k
    return codes


# -- features of a candidate removal (all computable on the challenge match) -----------------

FEATURES = ["duration_s", "previous_s", "next_s", "is_absorb", "is_to_dead", "is_to_team",
            "next_to_dead", "log_ratio_mean", "log_ratio_sum", "own_mean", "own_p10", "own_max",
            "new_mean", "dead_mean", "team_share_mean", "team_share_max", "edge_start", "edge_end",
            "edge_inside", "own_before", "own_after", "boundaries_10s", "decoder_gain"]
CAP_S = 30.0


def dense(interpolated, n_frames):
    """Full-length 25 Hz probability and boundary arrays (NaN outside the graded halves)."""
    p = np.full((n_frames, 3), np.nan)
    e = np.full(n_frames, np.nan)
    for first, last, q, edge in interpolated:
        p[first:last + 1] = q
        e[first:last + 1] = edge
    return p, e


def candidate_features(segs, cands, p, edge, config=DEFAULT):
    log_p = np.log(np.clip(p, 1e-5, 1))
    starts = np.array([a for a, _, _ in segs])
    rows = []
    for i, action, new in cands:
        a, z, k = segs[i]
        previous = segs[i - 1] if i > 0 and segs[i - 1][1] == a else None
        following = segs[i + 1] if i + 1 < len(segs) and segs[i + 1][0] == z else None
        inside = slice(a, z)
        ratio = log_p[inside, k] - log_p[inside, new]
        team = p[inside, k] / np.clip(p[inside, 0] + p[inside, 1], 1e-6, None)
        before = slice(max(previous[0], a - 2 * FPS), a) if previous else None
        after = slice(z, min(following[1], z + 2 * FPS)) if following else None
        # What the decoder objective loses by removing the spell: emissions plus the two
        # boundary-conditioned switch costs it no longer pays (a rough, local margin).
        cost = {(0, 1): config["c_ha"], (1, 0): config["c_ha"], (0, 2): config["c_to_d"],
                (1, 2): config["c_to_d"], (2, 0): config["c_from_d"], (2, 1): config["c_from_d"]}
        paid = 0.0
        if previous:
            paid += cost[(previous[2], k)] * (1 - config["gamma"] * edge[a])
            if previous[2] != new:
                paid -= cost[(previous[2], new)] * (1 - config["gamma"] * edge[a])
        if following:
            paid += cost[(k, following[2])] * (1 - config["gamma"] * edge[min(z, len(edge) - 1)])
            if following[2] != new:
                paid -= cost[(new, following[2])] * (1 - config["gamma"] * edge[min(z, len(edge) - 1)])
        rows.append([
            (z - a) / FPS,
            min(CAP_S, (previous[1] - previous[0]) / FPS) if previous else CAP_S,
            min(CAP_S, (following[1] - following[0]) / FPS) if following else CAP_S,
            float(action == "absorb"), float(action == "to_dead"), float(action in ("to_previous", "to_next")),
            float((previous is not None and previous[2] == 2) or (following is not None and following[2] == 2)),
            float(ratio.mean()), float(ratio.sum() * config["temp"]), float(p[inside, k].mean()),
            float(np.percentile(p[inside, k], 10)), float(p[inside, k].max()), float(p[inside, new].mean()),
            float(p[inside, 2].mean()), float(team.mean()), float(team.max()),
            float(np.nanmax(edge[max(0, a - 5):a + 6])), float(np.nanmax(edge[max(0, z - 5):z + 6])),
            float(edge[inside].mean()),
            float(p[before, k].mean()) if before is not None and before.stop > before.start else 0.0,
            float(p[after, k].mean()) if after is not None and after.stop > after.start else 0.0,
            float(((starts > a - 10 * FPS) & (starts < z + 10 * FPS)).sum()),
            float(ratio.sum() * config["temp"] - paid)])
    return np.array(rows, dtype=np.float64).reshape(len(rows), len(FEATURES))


def apply_scored(segs, score_function, threshold, max_frames, passes=3):
    """Remove the candidates whose score reaches the threshold, best first, a few passes.

    Two removals never touch neighbouring segments in the same pass (their features would be
    stale); the next pass re-reads the new segmentation. Returns the new segments and the list
    of (action, removed duration in s).
    """
    applied = []
    for _ in range(passes):
        cands = candidates(segs, max_frames)
        if not cands:
            break
        scores = score_function(segs, cands)
        order = np.argsort(-scores, kind="stable")
        changes = {}
        for j in order:
            if scores[j] < threshold:
                break
            i, _, new = cands[j]
            if any(i + d in changes for d in (-2, -1, 0, 1, 2)):
                continue
            changes[i] = new
        if not changes:
            break
        by_index = {cands[j][0]: cands[j][1] for j in order[::-1] if changes.get(cands[j][0]) == cands[j][2]}
        applied += [(by_index[i], (segs[i][1] - segs[i][0]) / FPS) for i in sorted(changes)]
        segs = relabel(segs, changes)
    return segs, applied


# -- team re-decoding inside live play (stoppages are never touched) -------------------------

def team_viterbi(p, edge, dead, configs, temp=DEFAULT["temp"]):
    """Two-state (H/A) Viterbi over one half with the D frames of `dead` held fixed.

    configs: [(switch cost, gamma)]. The cost of a change of team at frame t is
    cost * (1 - gamma * edge[t]); at a dead frame it is 0 and there is no emission, so every
    stretch of live play is decoded on its own. Returns (len(configs), T) int8 with 0/1/2.
    """
    b = len(configs)
    cost = np.array([c for c, _ in configs], dtype=np.float64)
    gamma = np.array([g for _, g in configs], dtype=np.float64)
    live = ~dead
    log_p = np.log(np.clip(p[:, :2], 1e-5, 1)) * temp * live[:, None]
    n = len(p)
    score = np.repeat(log_p[0][None], b, axis=0)
    back = np.zeros((n, b, 2), dtype=np.int8)
    edge = edge.astype(np.float64)
    for t in range(1, n):
        switch = cost * (1 - gamma * edge[t]) if live[t] and live[t - 1] else 0.0
        stay = score
        move = score[:, ::-1] - (switch[:, None] if live[t] and live[t - 1] else 0.0)
        take = move > stay
        back[t] = np.where(take, np.array([1, 0], dtype=np.int8)[None], np.array([0, 1], dtype=np.int8)[None])
        score = np.where(take, move, stay) + log_p[t][None]
        score -= score.max(1, keepdims=True)
    states = np.empty((b, n), dtype=np.int8)
    current = score.argmax(1)
    rows = np.arange(b)
    for t in range(n - 1, -1, -1):
        states[:, t] = current
        current = back[t, rows, current]
    states[:, dead] = 2
    return states


def redecode_teams(interpolated, codes, configs):
    """Baseline stoppages kept frame by frame; H/A re-decoded with other change-of-team costs."""
    out = np.repeat(codes[None], len(configs), axis=0)
    for first, last, p, e in interpolated:
        out[:, first:last + 1] = team_viterbi(p, e, codes[first:last + 1] == 2, list(configs))
    return out
