"""V28c: Metrica's automatic Smart Tagging of the challenge match as evidence (experiment).

Smart Tagging is an allowed automatic input (DATA.md). This module

1. reverse-engineers, label-free, the time structure of every code (lead / lag padding, where the
   "core" of a phase sits) from the tracking and from the frozen ensemble's confident frames;
2. measures, cross-fitted over time blocks, how informative a tag is about team and dead/alive
   as a function of the position inside the clip;
3. turns that into a bounded evidence term that is added to the (V24-fused) ensemble
   log-probabilities before the unchanged decoder and the V27 "consistent" adapter;
4. reports the label-free evidence for and against shipping it.

No challenge ground truth is read. The only "labels" are the ensemble's own confident frames.

    .venv/bin/python -m possession.v28c_tagging structure     # padding / core of each code
    .venv/bin/python -m possession.v28c_tagging evidence      # cross-fitted informativeness
    .venv/bin/python -m possession.v28c_tagging fuse          # fusion + label-free assessment
    .venv/bin/python -m possession.v28c_tagging adjudicator   # tracking check, calibrated on open games

Codes 0 = H (Barcelona), 1 = A (Athletic), 2 = D, 3 = not graded.
"""
import argparse
import json
from pathlib import Path
import numpy as np

FPS = 25
ROOT = Path("data/work/v28c")
TEAMS = {"Barcelona": 0, "Atletic Club": 1}
CONFIDENT = 0.9                 # an ensemble frame counts as a pseudo-label above this
CLIP = 2.0                      # < log(0.9 / 0.1) = 2.197: a confident frame keeps its arg-max
ALPHA_S = 1.0                   # pseudo-count, in seconds, on each side of a likelihood ratio
FOLDS_PER_HALF = 6
ADAPTER_THRESHOLD = 0.0005      # the value that reproduces evaluation/results/v27_semantic_bet.csv

# family -> (binning, cap in seconds, team evidence, dead/alive evidence)
#   "both":  1 s bins counted from the nearer edge of the clip, capped (a long phase has a core)
#   "start": 1 s bins counted from the clip start (fixed-length clips)
FAMILIES = {
    "BUILD UP": ("both", 12, True, True),
    "PROGRESSION": ("both", 12, True, True),
    "ATTACKING TRANSITION": ("both", 12, True, True),
    "LONG BALL": ("both", 12, True, True),
    "BALL IN FINAL THIRD": ("both", 14, True, True),
    "BALL IN THE BOX": ("both", 14, True, True),
    "PLAYERS IN THE BOX": ("both", 14, True, True),
    "CREATING CHANCES": ("start", 29, True, True),
    "SET PIECE": ("start", 34, False, True),          # every 35 s set-piece clip
    "SET PIECE TEAM": ("start", 34, True, False),     # corner / goal kick / free kick carry a team
    "KICK OFF": ("start", 4, False, True),
    "GOALS": ("both", 14, False, True),
}
# The DEFENDING IN ... / DEFENSIVE TRANSITION codes are the same intervals with the other team.
MIRRORS = {"DEFENDING IN ATTACKING THIRD": "BUILD UP", "DEFENDING IN MIDDLE THIRD": "PROGRESSION",
           "DEFENDING IN DEFENSIVE THIRD": "CREATING CHANCES", "DEFENSIVE TRANSITION": "ATTACKING TRANSITION"}
# Fixed before looking at any output: the families that go into the fusion.
POSSESSION_FAMILIES = ("BUILD UP", "PROGRESSION", "ATTACKING TRANSITION")
RESTART_FAMILIES = ("SET PIECE", "SET PIECE TEAM", "KICK OFF", "GOALS")
DEFAULT_FAMILIES = POSSESSION_FAMILIES + RESTART_FAMILIES


# -- tags ------------------------------------------------------------------------------------

def load_instances(events=None):
    """[{family, a, z, team, half, code, type}] half-open frame intervals; mirrors dropped."""
    if events is None:
        from starter.load_smart_tagging import load
        events = load()
    out = []
    for e in events:
        code = (e["code"] or "").strip().upper()
        tags = e["tags"]
        team = TEAMS.get(tags.get("Team"), -1)
        half = {"1st Half": 1, "2nd Half": 2}.get(tags.get("Half"), 0)
        base = dict(a=int(e["start_frame"]), z=int(e["end_frame"]), team=team, half=half, code=code,
                    type=tags.get("Type"))
        if code in MIRRORS:
            continue
        if code == "SET PIECES":
            if "kick off" in (tags.get("Type") or "").lower():
                out.append(dict(base, family="KICK OFF"))
            else:
                out.append(dict(base, family="SET PIECE"))
                if team >= 0:
                    out.append(dict(base, family="SET PIECE TEAM"))
        elif code in FAMILIES:
            out.append(dict(base, family=code))
    return out


def mirrors_are_exact(events):
    """True if every mirrored code repeats the intervals of its attacking code with the other team."""
    key = lambda e: (e["start_frame"], e["end_frame"])
    for mirror, source in MIRRORS.items():
        a = sorted((key(e), TEAMS.get(e["tags"].get("Team"), -1)) for e in events if e["code"] == mirror)
        b = sorted((key(e), TEAMS.get(e["tags"].get("Team"), -1)) for e in events if e["code"] == source)
        if len(a) != len(b):
            return False
        for (ka, ta), (kb, tb) in zip(a, b):
            if ka != kb or (ta >= 0) != (tb >= 0) or (ta >= 0 and ta != 1 - tb):
                return False
    return True


def n_bins(family):
    binning, cap = FAMILIES[family][:2]
    return 2 * (cap + 1) if binning == "both" else cap + 1


def bin_index(family, a, z, frames):
    """Position bin of each frame (a <= frame < z) inside one clip."""
    binning, cap = FAMILIES[family][:2]
    from_start = (frames - a) // FPS
    if binning == "start":
        return np.minimum(from_start, cap)
    to_end = (z - 1 - frames) // FPS
    return np.where(from_start <= to_end, np.minimum(from_start, cap), cap + 1 + np.minimum(to_end, cap))


def bin_name(family, b):
    binning, cap = FAMILIES[family][:2]
    if binning == "start" or b <= cap:
        return f"S{b}{'+' if b == cap else ''}"
    return f"E{b - cap - 1}{'+' if b - cap - 1 == cap else ''}"


def assign_folds(instances, blocks, folds_per_half=FOLDS_PER_HALF):
    """Contiguous time blocks; a clip belongs to the block of its midpoint (-1 outside the halves)."""
    for inst in instances:
        mid = (inst["a"] + inst["z"]) // 2
        inst["fold"] = -1
        for k, (_, first, last) in enumerate(blocks):
            if first <= mid <= last:
                inst["fold"] = k * folds_per_half + min(folds_per_half - 1, (mid - first) * folds_per_half // (last + 1 - first))
    return instances


# -- pseudo-labels and likelihood ratios -----------------------------------------------------

def dense_probabilities(probabilities, edges, frames, blocks, n_frames):
    from . import v27_core as c
    p, _ = c.dense(c.interpolate(np.asarray(probabilities), np.asarray(edges), np.asarray(frames), blocks), n_frames)
    return p


def pseudo_labels(p, confident=CONFIDENT):
    """Arg-max state where the ensemble is confident, -1 elsewhere (and outside the halves)."""
    q = np.nan_to_num(p, nan=0.0)
    return np.where(q.max(1) >= confident, q.argmax(1), -1).astype(np.int8)


def count_table(instances, state, family, n_frames):
    """(bins, 3) confident frames: tagged team / other team / dead (H / A / dead without a team)."""
    table = np.zeros((n_bins(family), 3))
    for inst in instances:
        if inst["family"] != family:
            continue
        a, z = max(0, inst["a"]), min(n_frames, inst["z"])
        f = np.arange(a, z)
        s = state[f]
        b = bin_index(family, inst["a"], inst["z"], f)
        team = inst["team"] if inst["team"] >= 0 else 0
        for column, code in enumerate((team, 1 - team, 2)):
            np.add.at(table[:, column], b[s == code], 1)
    return table


def log_ratios(table, base_dead_odds, team, dead, alpha=ALPHA_S):
    """Per bin: log LR tagged team vs other, log LR dead vs alive (0 where the family has none)."""
    seconds = table / FPS
    llr_team = np.log((seconds[:, 0] + alpha) / (seconds[:, 1] + alpha)) if team else np.zeros(len(table))
    live = seconds[:, 0] + seconds[:, 1]
    llr_dead = (np.log((seconds[:, 2] + alpha) / (live + alpha)) - np.log(base_dead_odds)) if dead else np.zeros(len(table))
    return llr_team, llr_dead


def base_dead_odds(state, mask):
    s = state[mask]
    return max(1.0, float((s == 2).sum())) / max(1.0, float(((s == 0) | (s == 1)).sum()))


def fit_tables(instances, state, graded, n_frames, held_fold=None, families=FAMILIES):
    """{family: (llr_team, llr_dead)} from every clip outside `held_fold` (all clips if None)."""
    train = [i for i in instances if held_fold is None or i["fold"] != held_fold]
    mask = graded.copy()
    if held_fold is not None:           # base rate from outside the held block as well
        for inst in instances:
            if inst["fold"] == held_fold:
                mask[max(0, inst["a"]):inst["z"]] = False
    odds = base_dead_odds(state, mask)
    return {family: log_ratios(count_table(train, state, family, n_frames), odds, *FAMILIES[family][2:])
            for family in families}


def evidence(instances, tables_of, n_frames, graded, families=DEFAULT_FAMILIES, clip=CLIP):
    """(n_frames, 3) bounded log-evidence for H / A / D; zero where no tag speaks.

    tables_of(instance) -> {family: (llr_team, llr_dead)}; with cross-fitting it is the table that
    never saw the instance's own time block. Overlapping clips add up (naive Bayes) and the sum is
    then bounded: the best state is 0 and no state is more than `clip` below it, so a frame whose
    ensemble probability is >= 0.9 for one state keeps that state whatever the tags say.
    """
    adj = np.zeros((n_frames, 3))
    for inst in instances:
        family = inst["family"]
        if family not in families:
            continue
        _, _, has_team, has_dead = FAMILIES[family]
        if has_team and not has_dead and inst["team"] < 0:
            continue
        llr_team, llr_dead = tables_of(inst)[family]
        a, z = max(0, inst["a"]), min(n_frames, inst["z"])
        f = np.arange(a, z)
        b = bin_index(family, inst["a"], inst["z"], f)
        if has_team and inst["team"] >= 0:
            adj[f, inst["team"]] += llr_team[b] / 2
            adj[f, 1 - inst["team"]] -= llr_team[b] / 2
        if has_dead:
            adj[f, 2] += llr_dead[b]
    adj[~graded] = 0.0
    adj -= adj.max(1, keepdims=True)
    return np.maximum(adj, -clip)


def cross_fitted_evidence(instances, state, graded, n_frames, families=DEFAULT_FAMILIES, clip=CLIP):
    folds = sorted({i["fold"] for i in instances})
    tables = {k: fit_tables(instances, state, graded, n_frames, held_fold=k) for k in folds}
    return evidence(instances, lambda inst: tables[inst["fold"]], n_frames, graded, families, clip), tables


def fuse(probabilities, frames, adj):
    """Add the evidence to the emission log-probabilities on the network grid and renormalise."""
    log_p = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0)) + adj[np.asarray(frames)]
    log_p -= log_p.max(1, keepdims=True)
    out = np.exp(log_p)
    return out / out.sum(1, keepdims=True)


# -- challenge-match plumbing ----------------------------------------------------------------

def graded_mask(blocks, n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return mask


def fused_v24(probabilities_path, alignment_path):
    """V24 camera fusion of a probabilities file (cached by content hash under data/work/v28c)."""
    from . import v27_semantic as vs
    from games import GAME, half_video_blocks, source_frames
    saved = np.load(probabilities_path)
    key = vs.digest(probabilities_path)[:16]
    cache = ROOT / f"fused_v24_{key}.npz"
    blocks, n = list(half_video_blocks(GAME)), source_frames("tactical")
    if cache.exists():
        return np.load(cache)["probabilities"], saved["edges"], saved["frames"], blocks, n
    fused, blocks, n = vs.fused_probabilities(saved["probabilities"], saved["edges"], saved["frames"], alignment_path)
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, probabilities=fused)
    return fused, saved["edges"], saved["frames"], blocks, n


def tagging_adapter(probabilities_path="data/work/final_cpu/probabilities.npz",
                    alignment_path="data/work/v24/replay_bundle/alignment.json",
                    families=DEFAULT_FAMILIES, clip=CLIP, mode="consistent", threshold=ADAPTER_THRESHOLD,
                    events=None):
    """Re-apply entry point: probabilities file -> dict(codes, baseline, decoded, report, ...).

    codes     V24 fusion + cross-fitted Smart Tagging evidence + decoder + V27 adapter
    baseline  the same pipeline without the tags (= evaluation/results/v27_semantic_bet.csv for
              the default arguments)
    """
    from . import v27_semantic as vs
    fused, edges, frames, blocks, n = fused_v24(probabilities_path, alignment_path)
    graded = graded_mask(blocks, n)
    instances = assign_folds(load_instances(events), blocks)
    state = pseudo_labels(dense_probabilities(fused, edges, frames, blocks, n))
    adj, tables = cross_fitted_evidence(instances, state, graded, n, families, clip)
    with_tags = fuse(fused, frames, adj)
    codes, decoded, report = vs.semantic_adapter(with_tags, edges, frames, blocks, n, mode, threshold)
    baseline, baseline_decoded, _ = vs.semantic_adapter(fused, edges, frames, blocks, n, mode, threshold)
    return dict(codes=codes, decoded=decoded, baseline=baseline, baseline_decoded=baseline_decoded, report=report,
                adj=adj, state=state, instances=instances, tables=tables, blocks=blocks, n=n, graded=graded,
                fused=fused, with_tags=with_tags, frames=frames, edges=edges)


# -- task 1: time structure ------------------------------------------------------------------

def tracking():
    from .data import load_tracking
    players, ball, slots = load_tracking(cache_dir="data/work/cache")
    return players, ball, slots


def attack_direction(players, slots, blocks):
    """+1 if the home team attacks towards x = 1 in that half (from the shape at its first frames)."""
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    out = {}
    for (half, _, _), kick in zip(blocks, sorted(KICKOFF_VIDEO_FRAMES.values())):
        window = players[kick - 50:kick]
        out[half] = 1 if np.nanmean(window[:, slots == 0, 0]) < np.nanmean(window[:, slots == 1, 0]) else -1
    return out


def edge_trim(table_start_end, cap, minimum=0.9, min_seconds=2.0):
    """Seconds to cut at each edge: leading / trailing 1 s bins whose agreement is below `minimum`.

    Agreement = tagged team / (tagged + other) among confident live frames. minimum=0.5 is the
    trim that maximises (agreeing - disagreeing) confident seconds; 0.9 asks for a clean core.
    A bin with fewer than `min_seconds` of confident live play (dead ball, or the far end of
    short clips) is cut as well, so a lead that is still a stoppage counts as padding.
    """
    def cut(rows):
        k = 0
        for own, other, _ in rows:
            if (own + other) / FPS >= min_seconds and own / (own + other) >= minimum:
                break
            k += 1
        return k
    return cut(table_start_end[:cap + 1]), cut(table_start_end[cap + 1:])


def structure(args=None):
    from starter.load_smart_tagging import load, KICKOFF_VIDEO_FRAMES
    from games import GAME, GAMES, half_video_blocks, source_frames
    events = load()
    blocks, n = list(half_video_blocks(GAME)), source_frames("tactical")
    graded = graded_mask(blocks, n)
    saved = np.load(ROOT / "fused_v24.npz") if (ROOT / "fused_v24.npz").exists() else None
    if saved is None:
        fused, edges, frames, _, _ = fused_v24("data/work/final_cpu/probabilities.npz", "data/work/v24/replay_bundle/alignment.json")
    else:
        fused, edges, frames = saved["probabilities"], saved["edges"], saved["frames"]
    p = dense_probabilities(fused, edges, frames, blocks, n)
    state = pseudo_labels(p)
    instances = assign_folds(load_instances(events), blocks)
    players, ball, slots = tracking()
    direction = attack_direction(players, slots, blocks)
    report = dict(instances=len(events), mirrors_exact=mirrors_are_exact(events), attack_direction=direction, codes={})
    print("STRUCTURE mirrors exact:", report["mirrors_exact"], "home attacks towards x=1:", direction, flush=True)

    # (a) clip lengths
    for family in FAMILIES:
        lengths = np.array([i["z"] - i["a"] for i in instances if i["family"] == family])
        report["codes"][family] = dict(n=int(len(lengths)), length_min=int(lengths.min()), length_median=float(np.median(lengths)),
                                       length_max=int(lengths.max()), fixed_length=bool(lengths.min() == lengths.max()))

    # (b) zone codes against the tracked ball: first / last frame of the clip with the ball in the zone
    pitch = GAMES[GAME]["pitch"]
    box_x, box_y = 1 - 16.5 / pitch[0], 20.16 / pitch[1]
    zone = {"BUILD UP": lambda x, y: x < 1 / 3, "PROGRESSION": lambda x, y: (x >= 1 / 3) & (x < 2 / 3),
            "BALL IN FINAL THIRD": lambda x, y: x >= 2 / 3, "BALL IN THE BOX": lambda x, y: (x >= box_x) & (np.abs(y - 0.5) <= box_y)}
    for family, inside_zone in zone.items():
        lead, lag = [], []
        for inst in instances:
            if inst["family"] != family or inst["team"] < 0:
                continue
            f = np.arange(inst["a"], inst["z"])
            x, y = ball[f, 0], ball[f, 1]
            if direction[inst["half"]] * (1 if inst["team"] == 0 else -1) < 0:
                x = 1 - x
            with np.errstate(invalid="ignore"):
                inside = np.flatnonzero(inside_zone(x, y))
            if len(inside):
                lead.append(int(inside[0]))
                lag.append(int(len(f) - 1 - inside[-1]))
        lead, lag = np.array(lead), np.array(lag)
        mode = lambda v: int(np.bincount(v[v > 0]).argmax()) if (v > 0).any() else 0
        row = dict(ball_zone_lead_frames=dict(median=float(np.median(lead)), mode_when_positive=mode(lead), share_within_1_of_mode=float((np.abs(lead - mode(lead)) <= 1).mean()), share_0=float((lead == 0).mean())),
                   ball_zone_lag_frames=dict(median=float(np.median(lag)), mode_when_positive=mode(lag), share_within_1_of_mode=float((np.abs(lag - mode(lag)) <= 1).mean()), share_0=float((lag == 0).mean())))
        report["codes"][family].update(row)
        print("STRUCTURE zone", family, json.dumps(row), flush=True)

    # (c) agreement with the ensemble's confident frames by position, whole match and per half
    for family, (binning, cap, has_team, _) in FAMILIES.items():
        tables = {"all": count_table(instances, state, family, n)}
        for half in (1, 2):
            tables[f"half{half}"] = count_table([i for i in instances if i["half"] == half], state, family, n)
        profile = {bin_name(family, b): {k: dict(agreement=round(float(t[b, 0] / max(1.0, t[b, 0] + t[b, 1])), 3),
                                                 live_s=round(float((t[b, 0] + t[b, 1]) / FPS), 1),
                                                 dead_share=round(float(t[b, 2] / max(1.0, t[b].sum())), 3)) for k, t in tables.items()}
                   for b in range(n_bins(family))}
        report["codes"][family].update(profile=profile)
        if has_team and binning == "both":
            trims = {f"{k}_min{m}": edge_trim(t, cap, m) for k, t in tables.items() for m in (0.5, 0.9)}
            report["codes"][family].update(trim_lead_lag_seconds=trims)
            print("STRUCTURE trim (lead s, lag s)", family, trims, flush=True)

    # (d) restarts: kick-off inside its clip, goal clip vs following kick-off, set-piece restart
    kicks = sorted(KICKOFF_VIDEO_FRAMES.values())
    starts = sorted(i["a"] for i in instances if i["family"] == "KICK OFF")
    report["kickoff_frames_after_clip_start"] = [int(k - max(s for s in starts if s <= k)) for k in kicks]
    goals = [i for i in instances if i["family"] == "GOALS"]
    report["goal_end_minus_next_kickoff_clip_start"] = [int(g["z"] - min(s for s in starts if s > g["a"])) for g in goals]
    path = Path("evaluation/results/v27_semantic_bet.csv")
    if path.exists():
        import pandas as pd
        cand = pd.read_csv(path)["label"].map(dict(H=0, A=1, D=2, X=3)).to_numpy()
        offsets = []
        for inst in instances:
            if inst["family"] != "SET PIECE":
                continue
            w = cand[inst["a"] - 100:inst["a"] + 500]
            t = np.flatnonzero((w[1:] < 2) & (w[:-1] == 2)) + 1 - 100
            if len(t):
                offsets.append(int(t[np.argmin(np.abs(t - 150))]))
        offsets = np.array(offsets)
        report["set_piece_restart_of_candidate_after_clip_start"] = dict(
            n=int(len(offsets)), median=float(np.median(offsets)), q25=float(np.percentile(offsets, 25)), q75=float(np.percentile(offsets, 75)),
            share_within_1s_of_150=float((np.abs(offsets - 150) <= 25).mean()), share_within_3s_of_150=float((np.abs(offsets - 150) <= 75).mean()))
    print("STRUCTURE restarts", json.dumps({k: v for k, v in report.items() if k not in ("codes",)}), flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "structure.json").write_text(json.dumps(report, indent=1))
    return report


# -- task 2: cross-fitted informativeness ----------------------------------------------------

def held_out_skill(instances, state, p, graded, n_frames, tables_of, strong=1.0):
    """Per family and half: does the tag, read with a table that never saw it, predict the ensemble?

    team: frames of held-out clips where |log LR team| >= strong and the ensemble is confident and
    live -> share where the tag points at the ensemble's team. dead: the same for dead vs alive.
    The same on the ensemble's *unconfident* frames (arg-max, 0.5 <= p < 0.9), next to what
    independence predicts from the ensemble's own probability there.
    """
    rows = []
    q = np.nan_to_num(p, nan=0.0)
    top, arg = q.max(1), q.argmax(1)
    for family, (_, _, has_team, has_dead) in FAMILIES.items():
        for half in (1, 2):
            acc = dict(team=[0, 0], dead=[0, 0], soft=[0, 0, 0.0])
            for inst in instances:
                if inst["family"] != family or inst["half"] != half:
                    continue
                llr_team, llr_dead = tables_of(inst)[family]
                f = np.arange(max(0, inst["a"]), min(n_frames, inst["z"]))
                f = f[graded[f]]
                b = bin_index(family, inst["a"], inst["z"], f)
                s = state[f]
                if has_team and inst["team"] >= 0:
                    predicted = np.where(llr_team[b] > 0, inst["team"], 1 - inst["team"])
                    use = (np.abs(llr_team[b]) >= strong) & (s >= 0) & (s < 2)
                    acc["team"][0] += int((predicted[use] == s[use]).sum())
                    acc["team"][1] += int(use.sum())
                    soft = (np.abs(llr_team[b]) >= strong) & (s < 0) & (arg[f] < 2)
                    acc["soft"][0] += int((predicted[soft] == arg[f][soft]).sum())
                    acc["soft"][1] += int(soft.sum())
                    share = q[f][:, :2] / np.clip(q[f][:, :2].sum(1, keepdims=True), 1e-9, None)
                    acc["soft"][2] += float(share[soft].max(1).sum())
                if has_dead:
                    use = (np.abs(llr_dead[b]) >= strong) & (s >= 0)
                    acc["dead"][0] += int(((llr_dead[b][use] > 0) == (s[use] == 2)).sum())
                    acc["dead"][1] += int(use.sum())
            row = dict(family=family, half=half)
            if has_team:
                row.update(team_frames=acc["team"][1], team_accuracy=acc["team"][0] / max(1, acc["team"][1]),
                           unconfident_frames=acc["soft"][1], unconfident_agreement=acc["soft"][0] / max(1, acc["soft"][1]),
                           unconfident_ensemble_share=acc["soft"][2] / max(1, acc["soft"][1]))
            if has_dead:
                row.update(dead_frames=acc["dead"][1], dead_accuracy=acc["dead"][0] / max(1, acc["dead"][1]))
            rows.append(row)
    return rows


def evidence_report(args=None):
    from games import GAME, half_video_blocks, source_frames
    blocks, n = list(half_video_blocks(GAME)), source_frames("tactical")
    graded = graded_mask(blocks, n)
    fused, edges, frames, _, _ = fused_v24("data/work/final_cpu/probabilities.npz", "data/work/v24/replay_bundle/alignment.json")
    p = dense_probabilities(fused, edges, frames, blocks, n)
    state = pseudo_labels(p)
    instances = assign_folds(load_instances(), blocks)
    folds = sorted({i["fold"] for i in instances})
    cross = {k: fit_tables(instances, state, graded, n, held_fold=k) for k in folds}
    report = dict(confident_share_of_graded=float((state[graded] >= 0).mean()), folds=len(folds))
    report["time_block_cross_fit"] = held_out_skill(instances, state, p, graded, n, lambda i: cross[i["fold"]])
    # other-half transfer: table from one half only, read on the other
    by_half = {}
    for half in (1, 2):
        train = [dict(i, fold=0 if i["half"] == half else 1) for i in instances]
        by_half[half] = fit_tables(train, state, graded, n, held_fold=1)
    report["other_half_transfer"] = held_out_skill(instances, state, p, graded, n, lambda i: by_half[3 - i["half"]])
    # stability of the tables themselves between halves, on bins with >= 5 s of data in both
    stability = {}
    for family in FAMILIES:
        counts = [count_table([i for i in instances if i["half"] == h], state, family, n) for h in (1, 2)]
        ok = (counts[0].sum(1) >= 5 * FPS) & (counts[1].sum(1) >= 5 * FPS)
        a, b = by_half[1][family], by_half[2][family]
        row = dict(bins=int(ok.sum()))
        for name, x, y in (("team", a[0], b[0]), ("dead", a[1], b[1])):
            if ok.sum() >= 3 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                row[name + "_llr_correlation"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
                row[name + "_sign_agreement"] = float((np.sign(np.round(x[ok], 6)) == np.sign(np.round(y[ok], 6))).mean())
        stability[family] = row
    report["half_stability"] = stability
    full = fit_tables(instances, state, graded, n)
    report["tables_full_fit"] = {family: {bin_name(family, b): [round(float(full[family][0][b]), 2), round(float(full[family][1][b]), 2)]
                                          for b in range(n_bins(family))} for family in FAMILIES}
    for key in ("time_block_cross_fit", "other_half_transfer"):
        for row in report[key]:
            print("EVIDENCE", key, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    for family, row in stability.items():
        print("EVIDENCE half stability", family, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "evidence.json").write_text(json.dumps(report, indent=1))
    return report


# -- tasks 3-4: fusion and label-free assessment ---------------------------------------------

def nearest_team(players, ball, slots, pitch, reach=2.0):
    """Team of the player nearest the tracked ball if within `reach` metres, else -1."""
    xy = ball[:, :2] * np.asarray(pitch)
    out = np.full(len(ball), -1, dtype=np.int8)
    for a in range(0, len(ball), 20000):
        z = min(len(ball), a + 20000)
        d = np.linalg.norm(players[a:z, :, :2] * np.asarray(pitch) - xy[a:z, None], axis=2)
        d = np.where(np.isnan(d), 1e9, d)
        i = d.argmin(1)
        out[a:z] = np.where(d.min(1) <= reach, slots[i], -1)
    return out


def agreement_inside(codes, instances, family, n_frames, trim=(0, 0)):
    """Frames of the tagged team / other team / dead inside (trimmed) clips of one family."""
    own = other = dead = 0
    for inst in instances:
        if inst["family"] != family or inst["team"] < 0:
            continue
        x = codes[max(0, inst["a"] + trim[0] * FPS):min(n_frames, inst["z"] - trim[1] * FPS)]
        own += int((x == inst["team"]).sum())
        other += int((x == 1 - inst["team"]).sum())
        dead += int((x == 2).sum())
    total = max(1, own + other + dead)
    return dict(own=own / total, other=other / total, dead=dead / total, frames=total)


def changed_runs(before, after, graded):
    """[(start, stop, old code, new code)] maximal runs of changed frames with one (old, new) pair."""
    diff = (before != after) & graded
    out = []
    index = np.flatnonzero(diff)
    if not len(index):
        return out
    start = index[0]
    for i, j in zip(index[:-1], index[1:]):
        if j != i + 1 or before[j] != before[i] or after[j] != after[i]:
            out.append((int(start), int(i + 1), int(before[start]), int(after[start])))
            start = j
    out.append((int(start), int(index[-1] + 1), int(before[start]), int(after[start])))
    return out


def fuse_report(args):
    from . import v27_core as c
    from games import GAME, GAMES
    result = tagging_adapter(args.probabilities, args.alignment)
    blocks, n, graded = result["blocks"], result["n"], result["graded"]
    instances, state = result["instances"], result["state"]
    base, new = result["baseline"], result["codes"]
    report = dict(probabilities=str(args.probabilities), families=list(DEFAULT_FAMILIES), clip=CLIP,
                  adapter=dict(mode="consistent", threshold=ADAPTER_THRESHOLD), challenge_labels_read=False)
    path = Path("evaluation/results/v27_semantic_bet.csv")
    if path.exists():
        import pandas as pd
        bet = pd.read_csv(path)["label"].map(dict(H=0, A=1, D=2, X=3)).to_numpy()
        report["baseline_equals_v27_semantic_bet_csv"] = bool(np.array_equal(bet, base))
    report["baseline"] = c.duration_stats(base, blocks)
    report["with_tags"] = c.duration_stats(new, blocks)
    report["per_half"] = {str(h): dict(baseline=c.duration_stats(base, [b]), with_tags=c.duration_stats(new, [b])) for b in blocks for h in [b[0]]}
    report["evidence_frames"] = dict(any=int((result["adj"].min(1) < 0).sum()), saturated=int((result["adj"].min(1) <= -CLIP + 1e-9).sum()),
                                     share_of_graded=float((result["adj"].min(1) < 0)[graded].mean()))
    confident = (state >= 0) & graded
    report["confident_frames_overturned_before_decoding"] = int((np.nan_to_num(dense_probabilities(result["with_tags"], result["edges"], result["frames"], blocks, n)).argmax(1)[confident] != state[confident]).sum())
    report["confident_frames_changed_in_output"] = dict(
        baseline_disagrees_with_confident=int((base[confident] != state[confident]).sum()),
        with_tags_disagrees_with_confident=int((new[confident] != state[confident]).sum()))

    runs = changed_runs(base, new, graded)
    names = "HAD"
    kinds = {}
    for a, z, old, fresh in runs:
        key = f"{names[old]}->{names[fresh]}"
        kinds.setdefault(key, [0, 0])
        kinds[key][0] += 1
        kinds[key][1] += z - a
    report["changes"] = dict(frames=int(((base != new) & graded).sum()), share_of_graded=float(((base != new) & graded).sum() / graded.sum()),
                             runs=len(runs), by_kind={k: dict(runs=v[0], seconds=round(v[1] / FPS, 1)) for k, v in sorted(kinds.items())},
                             per_half={str(h): int(((base != new)[a:z + 1]).sum()) for h, a, z in blocks})

    # independent of the tags: what the tracking says on the frames that changed team
    players, ball, slots = tracking()
    near = nearest_team(players, ball, slots, GAMES[GAME]["pitch"])
    changed = (base != new) & graded
    team_flip = changed & (base < 2) & (new < 2)
    seen = team_flip & (near >= 0)
    report["tracking_on_team_flips"] = dict(frames=int(team_flip.sum()), with_close_ball=int(seen.sum()),
                                            nearest_is_new_team=float((near[seen] == new[seen]).mean()) if seen.any() else None,
                                            nearest_is_old_team=float((near[seen] == base[seen]).mean()) if seen.any() else None)
    ball_seen = np.isfinite(ball[:, 0])
    revived, killed = changed & (base == 2) & (new < 2), changed & (base < 2) & (new == 2)
    report["tracking_on_dead_alive_changes"] = dict(
        dead_to_alive_frames=int(revived.sum()), dead_to_alive_ball_tracked=float(ball_seen[revived].mean()) if revived.any() else None,
        alive_to_dead_frames=int(killed.sum()), alive_to_dead_ball_tracked=float(ball_seen[killed].mean()) if killed.any() else None,
        reference_ball_tracked_when_alive=float(ball_seen[graded & (base < 2)].mean()), reference_ball_tracked_when_dead=float(ball_seen[graded & (base == 2)].mean()))

    # agreement with the tags: circular for the injected families, listed only as a description
    report["agreement_inside_clips_CIRCULAR"] = {family: dict(baseline=agreement_inside(base, instances, family, n), with_tags=agreement_inside(new, instances, family, n))
                                                 for family in ("BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "LONG BALL", "CREATING CHANCES")}

    # leave-one-family-out: inject everything except one family, read that family's own tags.
    # Not independent of the provider (its codes share one internal possession estimate), but the
    # held-out family's intervals were not injected.
    from . import v27_semantic as vs
    loo = {}
    for held in POSSESSION_FAMILIES + ("SET PIECE TEAM",):
        families = tuple(f for f in DEFAULT_FAMILIES if f != held)
        adj, _ = cross_fitted_evidence(instances, state, graded, n, families, CLIP)
        codes = vs.semantic_adapter(fuse(result["fused"], result["frames"], adj), result["edges"], result["frames"], blocks, n,
                                    "consistent", ADAPTER_THRESHOLD)[0]
        trim = {"BUILD UP": (4, 1), "PROGRESSION": (0, 4), "ATTACKING TRANSITION": (4, 2), "SET PIECE TEAM": (8, 20)}[held]
        loo[held] = dict(trim_seconds=trim, baseline=agreement_inside(base, instances, held, n, trim),
                         others_injected=agreement_inside(codes, instances, held, n, trim))
        print("FUSE leave-one-family-out", held, json.dumps(loo[held]), flush=True)
    report["leave_one_family_out"] = loo

    # ablation: which evidence moves what
    ablation = {}
    for name, families, clip in (("possession_phases_only", POSSESSION_FAMILIES, CLIP), ("restarts_only", RESTART_FAMILIES, CLIP),
                                 ("default_clip_1", DEFAULT_FAMILIES, 1.0), ("all_families", tuple(FAMILIES), CLIP)):
        adj, _ = cross_fitted_evidence(instances, state, graded, n, families, clip)
        codes = vs.semantic_adapter(fuse(result["fused"], result["frames"], adj), result["edges"], result["frames"], blocks, n,
                                    "consistent", ADAPTER_THRESHOLD)[0]
        flips = (codes != base) & graded & (base < 2) & (codes < 2) & (near >= 0)
        ablation[name] = dict(spells=c.duration_stats(codes, blocks)["spells"], frames_changed=int(((codes != base) & graded).sum()),
                              team_flip_frames_with_close_ball=int(flips.sum()),
                              nearest_is_new_team=float((near[flips] == codes[flips]).mean()) if flips.any() else None)
        print("FUSE ablation", name, json.dumps(ablation[name]), flush=True)
    report["ablation"] = ablation

    report["changed_runs"] = [dict(start=a, stop=z, seconds=round((z - a) / FPS, 2), old=names[old], new=names[fresh],
                                   nearest_new=round(float((near[a:z][near[a:z] >= 0] == fresh).mean()), 2) if (near[a:z] >= 0).any() else None,
                                   ball_tracked=round(float(ball_seen[a:z].mean()), 2),
                                   families=sorted({i["family"] for i in instances if i["family"] in DEFAULT_FAMILIES and i["a"] < z and i["z"] > a}))
                              for a, z, old, fresh in runs]
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "fuse_codes.npz", with_tags=new, baseline=base, adj=result["adj"].astype(np.float32))
    (ROOT / "fuse.json").write_text(json.dumps(report, indent=1))
    print("FUSE", json.dumps({k: v for k, v in report.items() if k != "changed_runs"}, indent=1), flush=True)
    if args.output:
        from starter.baseline import write_submission
        write_submission(str(args.output), np.array(["H", "A", "D", "X"])[new])
        print("WROTE", args.output, flush=True)
    return report


# -- calibration of the tracking adjudicator on the open Metrica games -----------------------

def spell_length_at(codes, blocks):
    """Per frame: length in frames of the constant run it belongs to (0 outside the halves)."""
    from . import v27_core as c
    out = np.zeros(len(codes), dtype=np.int64)
    for a, z, _ in c.segments(codes, blocks):
        out[a:z] = z - a
    return out


def adjudicator(args=None):
    """How often is the nearest player of the labelled team when the label is right / wrong?

    Open Metrica games, clean tracking, candidate = frozen ensemble + V27 consistent adapter (the
    pooled verifier, so mildly in-sample; only used to define which frames the candidate labels).
    For candidate-live frames with a player within 2 m of the ball:
        r_right = P(nearest = candidate team | reference = candidate team)
        r_wrong = P(nearest = candidate team | reference = other team)
    On the challenge match the tag fusion flips the team on some frames; if a share w of those
    flips is wrong (the old label was right), P(nearest = old team) = w r_right + (1 - w) r_wrong.
    """
    from . import v27_core as c, v27_semantic as vs
    from .sources import external_match
    bands = {"all": (0, 10 ** 9), "spell<=8s": (0, 8 * FPS), "spell<=4s": (0, 4 * FPS), "spell>8s": (8 * FPS + 1, 10 ** 9)}
    counts = {r: {b: np.zeros((2, 2)) for b in bands} for r in c.REFERENCES}
    per_game = {}
    for game in (1, 2, 3):
        match = external_match(game)
        saved = np.load(f"data/work/v16/game{game}_clean_final.npz")
        k = c.metrica_condition(game, "clean")
        cand = vs.semantic_adapter(saved["probabilities"], saved["edges"], saved["frames"], k.blocks, k.n, "consistent", ADAPTER_THRESHOLD)[0]
        near = nearest_team(match.players, match.ball, match.team, (1.0, 1.0))
        length = spell_length_at(cand, k.blocks)
        per_game[game] = {}
        for reference in c.REFERENCES:
            truth = k.references[reference]
            use = (cand < 2) & (near >= 0) & (truth >= 0) & (truth < 2)
            for band, (low, high) in bands.items():
                m = use & (length >= low) & (length <= high)
                right = m & (truth == cand)
                wrong = m & (truth != cand)
                table = np.array([[(near[right] == cand[right]).sum(), right.sum()], [(near[wrong] == cand[wrong]).sum(), wrong.sum()]], dtype=float)
                counts[reference][band] += table
                if band == "spell<=8s":
                    per_game[game][reference] = dict(r_right=float(table[0, 0] / max(1, table[0, 1])), r_wrong=float(table[1, 0] / max(1, table[1, 1])),
                                                     frames_right=int(table[0, 1]), frames_wrong=int(table[1, 1]))
    report = dict(per_game_spell_le_8s=per_game, pooled={})
    for reference in c.REFERENCES:
        report["pooled"][reference] = {band: dict(r_right=float(t[0, 0] / max(1, t[0, 1])), r_wrong=float(t[1, 0] / max(1, t[1, 1])),
                                                 frames_right=int(t[0, 1]), frames_wrong=int(t[1, 1])) for band, t in counts[reference].items()}
    fuse_path = ROOT / "fuse_codes.npz"
    if fuse_path.exists():
        from games import GAME, GAMES, half_video_blocks
        blocks = list(half_video_blocks(GAME))
        saved = np.load(fuse_path)
        base, new = saved["baseline"], saved["with_tags"]
        players, ball, slots = tracking()
        near = nearest_team(players, ball, slots, GAMES[GAME]["pitch"])
        length = spell_length_at(base, blocks)
        flips = (base != new) & (base < 2) & (new < 2)
        seen = flips & (near >= 0)
        observed = float((near[seen] == base[seen]).mean())
        steady = (base == new) & (base < 2) & (near >= 0)
        challenge = dict(flip_frames=int(flips.sum()), with_close_ball=int(seen.sum()), nearest_is_old_team=observed,
                         flip_frames_in_baseline_spells_le_8s=float((length[flips] <= 8 * FPS).mean()),
                         unchanged_live_frames_nearest_is_label=float((near[steady] == base[steady]).mean()),
                         unchanged_frames_in_spells_le_8s_nearest_is_label=float((near[steady & (length <= 8 * FPS)] == base[steady & (length <= 8 * FPS)]).mean()))
        challenge["per_half_nearest_is_old_team"] = {str(h): float((near[a:z + 1][seen[a:z + 1]] == base[a:z + 1][seen[a:z + 1]]).mean()) for h, a, z in blocks}
        implied = {}
        for reference in c.REFERENCES:
            for band in ("all", "spell<=8s"):
                r = report["pooled"][reference][band]
                implied[f"{reference} / {band}"] = float(np.clip((observed - r["r_wrong"]) / (r["r_right"] - r["r_wrong"]), 0, 1))
        challenge["implied_share_of_flips_that_are_wrong"] = implied
        report["challenge"] = challenge
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "adjudicator.json").write_text(json.dumps(report, indent=1))
    print("ADJUDICATOR", json.dumps(report, indent=1), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["structure", "evidence", "fuse", "adjudicator"])
    parser.add_argument("--probabilities", type=Path, default=Path("data/work/final_cpu/probabilities.npz"))
    parser.add_argument("--alignment", type=Path, default=Path("data/work/v24/replay_bundle/alignment.json"))
    parser.add_argument("--output", type=Path, help="write a submission CSV (only if the result is to be shipped)")
    arguments = parser.parse_args()
    dict(structure=structure, evidence=evidence_report, fuse=fuse_report, adjudicator=adjudicator)[arguments.step](arguments)
