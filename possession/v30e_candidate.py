"""V30e candidate: the submission's pipeline + stoppage onsets re-timed by physical events.

    data/work/v26/run_f2/probabilities.npz (frozen ensemble on the filled ball)
      -> V24 camera fusion, unchanged                 (--student camera, the default)
         or the same fusion with a camera + sound + image student (--student camera+audio+video)
      -> stoppages of the frozen ensemble that the camera student vetoed are restored when
         physical evidence backs them (whistle, ball beyond the lines, ball at rest)
      -> unchanged 25 Hz decoder -> unchanged w = 0.5 spell verifier (artifacts/verifier)
      -> re-timing of stoppage ONSETS only (no possession is created or removed):
           whistle   a referee whistle heard in both recordings in decoded play, followed by the
                     decoded stoppage within 5 s, AND agreeing physical evidence (the ball is
                     not played after it, or the players slow down by 30 %): onset = whistle;
                     up to 10 s when the ball is never played between whistle and stoppage
           crossing  the seen ball crosses a line in decoded play and the stoppage is decoded
                     within 2 s, the ball not returning inside in between: onset = crossing
           constant  a stoppage with no physical event near its onset starts CONSTANT_SHIFT
                     frames earlier (the frozen decoder's onsets are late on every open Metrica
                     game; value chosen leave-one-game-out there, possession.v30e_retime)
      -> kick-off law: the first possession of the second half belongs to the team that did not
         kick off the first half (checked against who stands on the centre spot)

    .venv/bin/python -m possession.v30e_candidate build      # evaluation/results/v30e_multimodal.csv
    .venv/bin/python -m possession.v30e_candidate build --variant retime_validated   # crossing + constant only
    .venv/bin/python -m possession.v30e_candidate control    # the same re-timer on events moved in time
    .venv/bin/python -m possession.v30e_candidate external   # crossing rule + constant shift on the open Metrica games

A whistle is evidence, not a decision: it never creates a stoppage, and it moves an onset only
when the pipeline already stops play within 5 s and the ball / players agree. No ground truth of
the challenge match is read; the official scorer is not run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from games import GAME, half_video_blocks, source_frames
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
from . import v27_core as c
from . import v30e_events as ev
from . import v30e_features as F
from . import v30e_retime as rt

ROOT = Path("data/work/v30e")
OUTPUT = Path("evaluation/results/v30e_multimodal.csv")
REFERENCE = Path("evaluation/results/v29_final.csv")
VERIFIER = Path("artifacts/verifier/frozen_v27_w0.50.joblib")
FPS = 25
WHISTLE_REACH = 5 * FPS         # the decoded stoppage must begin this soon after the whistle
WHISTLE_SURE = FPS              # ... within 1 s no further evidence is asked
WHISTLE_REACH_QUIET = 10 * FPS  # ... or within 10 s when the ball is never played in between
CROSSING_REACH = 2 * FPS
BALL_PLAYED = 6.0               # m/s: a ball moving faster than this after the whistle is being played
SLOW_DOWN = 0.7                 # players' mean speed 1-2 s after the whistle / 2 s before it
EVENT_NEAR = (5 * FPS, 3 * FPS)  # a stoppage has "an event near its onset" within -5 s .. +3 s
CONSTANT_SHIFT = 10             # frames; possession.v30e_retime external, leave-one-game-out
RESTORE_MIN = FPS               # a vetoed stoppage of the frozen ensemble lasts at least 1 s
RESTORE_OVERLAP = 0.2           # ... and less than this share of it is dead after the camera fusion
RESTORE_WHISTLE = 3 * FPS       # a whistle this soon before it (or inside it) backs it
RESTORE_REST = 0.8              # ... or the ball seen and at rest (< 1.5 m/s) on this share of it
REST_SPEED = 1.5
LABELS = np.array(["H", "A", "D", "X"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blocks():
    return list(half_video_blocks(GAME))


# -- the submission's pipeline ----------------------------------------------------------------

def fused_probabilities(triple, student="camera", folds=None):
    """(fused probabilities, report). 'camera' is the unchanged V24 fusion."""
    from . import v29_final as final
    from . import v24_camera_fusion as v24
    from . import v30e_lab as lab
    if student == "camera" and folds is None:
        fused, _, _ = final.fused(triple)
        return fused, dict(student="camera (V24, unchanged)")
    n = source_frames("tactical")
    frames = triple[2]
    teacher = lab.teacher_codes(triple, n)
    dead = (teacher[frames] == 2).astype(np.int8)
    x, names = lab.matrix(lab.feature_table(), student, frames)
    s, prior = lab.cross_fit(x, dead, lab.fold_index(frames, folds or v24.FOLDS_PER_HALF))
    fused = v24.fuse(triple[0], s, prior)
    fused[v24.before_published_kickoff(frames, blocks())] = [5e-5, 5e-5, 1 - 1e-4]
    return fused, dict(student=student, features=len(names), auc_against_teacher=lab.auc(dead, s), folds_per_half=folds or v24.FOLDS_PER_HALF)


def verified(fused, triple):
    import joblib
    from . import v29_final as final
    saved = joblib.load(VERIFIER)
    after, before, row = final.verify(fused, triple[1], triple[2], blocks(), source_frames("tactical"), saved)
    return after, before, row


# -- stoppages vetoed by the camera student ------------------------------------------------------

def physical_support(a, z, whistles, ball):
    """Evidence, independent of the camera operators, that the ball is dead in [a, z)."""
    reasons = []
    if ((whistles >= a - RESTORE_WHISTLE) & (whistles < z)).any():
        reasons.append("whistle")
    seen = ball["b_seen"][a:z] > 0
    if int((seen & (np.nan_to_num(ball["b_outside_m"][a:z], nan=-99.0) > ev.OUTSIDE_M)).sum()) >= 3:
        reasons.append("ball beyond the lines")
    rest = seen & (np.nan_to_num(ball["b_speed"][a:z], nan=99.0) < REST_SPEED)
    if seen.mean() >= RESTORE_REST and rest.mean() >= RESTORE_REST:
        reasons.append("ball at rest")
    return reasons


def vetoed_stoppages(teacher, decoded, whistles, ball):
    """Stoppages decoded from the frozen ensemble alone that are live after the camera fusion."""
    rows = []
    for a, z, k in c.segments(teacher, blocks()):
        if k != 2 or z - a < RESTORE_MIN or (decoded[a:z] == 2).mean() >= RESTORE_OVERLAP:
            continue
        rows.append(dict(start=int(a), stop=int(z), frames=int(z - a), support=physical_support(a, z, whistles, ball)))
    return rows


def restore(fused, frames, rows):
    """P(D) -> certain on the 5 Hz rows inside the vetoed stoppages that have physical support."""
    out = fused.copy()
    for row in rows:
        if row["support"]:
            out[(frames >= row["start"]) & (frames < row["stop"])] = [5e-5, 5e-5, 1 - 1e-4]
    return out


def support_rate_in_live_play(teacher, decoded, whistles, ball, lengths, draws=2000, seed=0):
    """Control: how often does the same physical test pass on windows both decodes call live?"""
    rng = np.random.default_rng(seed)
    live = (teacher < 2) & (decoded < 2)
    hits, total = {"whistle": 0, "ball beyond the lines": 0, "ball at rest": 0, "any": 0}, 0
    starts = np.flatnonzero(live)
    while total < draws:
        a = int(rng.choice(starts))
        z = a + int(rng.choice(lengths))
        if z >= len(live) or not live[a:z].all():
            continue
        total += 1
        reasons = physical_support(a, z, whistles, ball)
        for r in reasons:
            hits[r] += 1
        hits["any"] += bool(reasons)
    return {k: v / total for k, v in hits.items()}


# -- re-timing by physical events --------------------------------------------------------------

def whistle_agrees(w, onset, ball, kin):
    """Does the ball / player evidence agree that play stopped at the whistle?"""
    if onset - w <= WHISTLE_SURE:
        return True, "stoppage decoded within 1 s"
    seen = ball["b_seen"][w + 12:onset] > 0
    speed = ball["b_speed"][w + 12:onset]
    if seen.mean() >= 0.5 and np.nanmax(np.where(seen, np.nan_to_num(speed, nan=0.0), 0.0)) < BALL_PLAYED:
        return True, "ball not played after the whistle"
    if onset - w > WHISTLE_REACH:
        return False, "stoppage decoded more than 5 s later and the ball was played"
    before = np.nanmean(kin["k_speed_mean"][w - 2 * FPS:w])
    after = np.nanmean(kin["k_speed_mean"][w + FPS:w + 2 * FPS])
    if np.isfinite(before) and np.isfinite(after) and after <= SLOW_DOWN * before:
        return True, "players slow down"
    return False, "ball played and players keep moving"


def crossing_agrees(t, onset, ball):
    inside = (ball["b_seen"][t + 1:onset] > 0) & (np.nan_to_num(ball["b_outside_m"][t + 1:onset], nan=0.0) < -1.0)
    return int(inside.sum()) <= 5


def physical_onsets(codes, whistles, crossings, ball, kin, constant=CONSTANT_SHIFT, halves=None):
    """{index of D segment: new onset} and the reason of each move."""
    segs = c.segments(codes, halves or blocks())
    targets, reasons, skipped = {}, {}, []
    for i, (a, z, k) in enumerate(segs):
        if k != 2 or i == 0 or segs[i - 1][1] != a or segs[i - 1][2] > 1:
            continue
        floor = segs[i - 1][0] + rt.MIN_KEEP
        chosen = []
        for w in whistles[(whistles >= a - WHISTLE_REACH_QUIET) & (whistles < a - 2) & (whistles >= floor)]:
            ok, why = whistle_agrees(int(w), a, ball, kin)
            if ok:
                chosen.append((int(w), "whistle: " + why))
            else:
                skipped.append(dict(onset=int(a), whistle=int(w), reason=why))
        for t in crossings[(crossings >= a - CROSSING_REACH) & (crossings < a - 2) & (crossings >= floor)]:
            if crossing_agrees(int(t), a, ball):
                chosen.append((int(t), "ball crosses the line"))
        if chosen:
            targets[i], reasons[i] = min(chosen)
            continue
        near = lambda events: bool(((events >= a - EVENT_NEAR[0]) & (events <= a + EVENT_NEAR[1])).any())
        if constant and not near(whistles) and not near(crossings):
            targets[i], reasons[i] = a - constant, "constant"
    return segs, targets, reasons, skipped


def retime(codes, whistles, crossings, ball, kin, constant=CONSTANT_SHIFT, halves=None):
    segs, targets, reasons, skipped = physical_onsets(codes, whistles, crossings, ball, kin, constant, halves)
    out, applied = rt.move(codes, segs, onsets=targets)
    rows = [dict(onset_before=int(old), onset_after=int(new), frames_earlier=int(old - new), reason=reasons[i],
                 previous_possession_frames=int(segs[i - 1][1] - segs[i - 1][0])) for i, _, old, new in applied]
    return out, rows, skipped


# -- kick-off law ------------------------------------------------------------------------------

def centre_spot_team(match, frame, window=FPS):
    """Team of the player nearest to the centre spot over the second before `frame`."""
    centre = np.array(match.pitch) / 2
    d = np.linalg.norm(match.players[frame - window:frame + 1] - centre, axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    nearest = d.argmin(axis=1)
    teams = match.team[nearest]
    return int(np.round(teams.mean())), float(np.median(d.min(axis=1)))


def kickoff_law(codes, match, limit=3 * FPS):
    """First possession of the second half -> the team that did not kick off the first half."""
    out = codes.copy()
    first, second = sorted(KICKOFF_VIDEO_FRAMES.values())
    team1, d1 = centre_spot_team(match, first)
    team2, d2 = centre_spot_team(match, second)
    segs = c.segments(codes, blocks())
    j = next(i for i, (a, z, k) in enumerate(segs) if z > second and k < 2)
    a, z, k = segs[j]
    report = dict(first_half_kickoff_team="HA"[team1], second_half_centre_spot_team="HA"[team2], centre_spot_distance_m=[round(d1, 2), round(d2, 2)],
                  law_and_tracking_agree=team1 != team2, decoded_first_possession=dict(start=int(a), stop=int(z), team="HA"[k]), changed=False)
    first_decoded = next(s for s in segs if s[2] < 2)
    report["first_half_decoded_first_team"] = "HA"[first_decoded[2]]
    if team1 != team2 and first_decoded[2] == team1 and k != team2 and z - a <= limit and j + 1 < len(segs) and segs[j + 1][2] == team2:
        out[a:z] = team2
        report["changed"] = True
    return out, report


def goal_restarts(codes, match):
    """After each change of the burnt-in score: is play dead, and who restarts?"""
    events = json.loads((F.OUT / "events.json").read_text())
    segs = c.segments(codes, blocks())
    rows = []
    for frame in events.get("scoreboard", {}).get("score_change_frames", []):
        i = next((j for j, (a, z, k) in enumerate(segs) if a <= frame < z), None)
        if i is None:
            continue
        a, z, k = segs[i]
        row = dict(score_change_frame=int(frame), decoded_state="HAD"[k])
        if k == 2 and i + 1 < len(segs):
            team, d = centre_spot_team(match, z)
            row.update(stoppage=[int(a), int(z)], stoppage_seconds=round((z - a) / FPS, 1), decoded_restart_team="HA"[segs[i + 1][2]],
                       centre_spot_team_at_restart="HA"[team], centre_spot_distance_m=round(d, 2), agree=bool(team == segs[i + 1][2]))
        rows.append(row)
    return rows


# -- candidate ---------------------------------------------------------------------------------

def codes_of(path):
    import pandas as pd
    return pd.read_csv(path)["label"].map(dict(H=0, A=1, D=2, X=3)).to_numpy().astype(np.int8)


def difference(codes, reference):
    graded = ev.graded_mask(len(codes))
    changed = graded & (codes != reference)
    by_half = {str(half): int(changed[first:last + 1].sum()) for half, first, last in blocks()}
    kinds = {}
    for a, b in zip(reference[changed], codes[changed]):
        key = "HAD"[a] + "->" + "HAD"[b]
        kinds[key] = kinds.get(key, 0) + 1
    return dict(graded_frames=int(graded.sum()), frames_changed=int(changed.sum()), share_changed=float(changed.sum() / graded.sum()),
                by_half=by_half, by_kind=kinds)


def make(student="camera", constant=CONSTANT_SHIFT, whistles=None, crossings=None, folds=None, use_whistles=True, use_crossings=True,
         use_restore=True, law=True):
    from . import v29_final as final
    from . import v30e_lab as lab
    triple = lab.load_triple()
    fused, fusion_report = fused_probabilities(triple, student, folds)
    ball, kin = F.load_group("ball"), F.load_group("kinematics")
    w = ev.whistles() if whistles is None else whistles
    x = ev.crossings(ball) if crossings is None else crossings
    empty = np.array([], dtype=np.int64)
    vetoed = []
    if use_restore:
        n = source_frames("tactical")
        teacher = lab.teacher_codes(triple, n)
        decoded = lab.teacher_codes((fused, triple[1], triple[2]), n)
        vetoed = vetoed_stoppages(teacher, decoded, w if use_whistles else empty, ball)
        lengths = [row["frames"] for row in vetoed] or [75]
        fusion_report["support_rate_in_live_play_control"] = support_rate_in_live_play(teacher, decoded, w, ball, lengths)
        fused = restore(fused, triple[2], vetoed)
    after, before, verifier_report = verified(fused, triple)
    codes, moves, skipped = retime(after, w if use_whistles else empty, x if use_crossings else empty, ball, kin, constant)
    match, _ = final.filled_match(ROOT)
    report_only, law_report = kickoff_law(codes, match)
    if law:
        codes = report_only
    else:
        law_report["changed"] = False
    law = law_report
    return dict(codes=codes, verified=after, decoded=before, moves=moves, skipped=skipped, law=law, fusion=fusion_report, vetoed=vetoed,
                verifier=dict(removals=verifier_report["removals"], removals_by_kind=verifier_report["removals_by_kind"]), match=match)


VARIANTS = {
    # everything: restored stoppages + whistle / crossing / constant re-timing + kick-off law
    "multimodal": dict(output=OUTPUT, use_whistles=True, use_restore=True, law=True),
    # only what the open Metrica games could score: crossing + constant re-timing
    "retime_validated": dict(output=Path("evaluation/results/v30e_retime_validated.csv"), use_whistles=False, use_restore=False, law=False),
}


def build(args):
    from starter.baseline import write_submission
    spec = VARIANTS[args.variant]
    args.output = args.output or spec["output"]
    result = make(args.student, use_whistles=spec["use_whistles"], use_restore=spec["use_restore"], law=spec["law"])
    codes = result["codes"]
    reference = codes_of(REFERENCE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(args.output), LABELS[codes])
    reasons = {}
    for row in result["moves"]:
        key = row["reason"].split(":")[0]
        reasons.setdefault(key, []).append(row["frames_earlier"])
    report = dict(variant=args.variant, csv=str(args.output), csv_sha256=digest(args.output), rows=int(len(codes)), student=result["fusion"],
                  verifier=result["verifier"], verified_equals_submission=bool(np.array_equal(result["verified"], reference)),
                  stoppages_vetoed_by_camera_student=result["vetoed"], stoppages_restored=sum(bool(r["support"]) for r in result["vetoed"]),
                  frames_changed_by_restoring=difference(result["verified"], reference)["frames_changed"],
                  stats=c.duration_stats(codes, blocks()), stats_submission=c.duration_stats(reference, blocks()),
                  stats_by_half={str(h): c.duration_stats(codes, [(h, a, z)]) for h, a, z in blocks()},
                  counts={k: int((LABELS[codes] == k).sum()) for k in "HADX"},
                  onsets_moved=len(result["moves"]),
                  onsets_moved_by_reason={k: dict(n=len(v), median_frames=float(np.median(v)), max_frames=int(max(v)), total_frames=int(sum(v))) for k, v in reasons.items()},
                  whistles_not_used=result["skipped"], kickoff_law=result["law"], goal_restarts=goal_restarts(codes, result["match"]),
                  versus_submission=difference(codes, reference), moves=result["moves"],
                  events_after=ev.report(codes, "v30e"), challenge_labels_read=False, official_scorer_run=False)
    (ROOT / ("candidate.json" if args.variant == "multimodal" else f"candidate_{args.variant}.json")).write_text(json.dumps(report, indent=1))
    print("V30E", json.dumps({k: v for k, v in report.items() if k not in ("moves", "events_after")}, indent=1), flush=True)
    for row in result["moves"]:
        if not row["reason"].startswith("constant"):
            print("MOVE", row, flush=True)


def control(args):
    """The same re-timer with the whistles and crossings moved in time by 1-4 minutes."""
    from . import v30e_lab as lab
    triple = lab.load_triple()
    fused, _ = fused_probabilities(triple, "camera")
    after, _, _ = verified(fused, triple)
    ball, kin = F.load_group("ball"), F.load_group("kinematics")
    graded = ev.graded_mask(len(after))
    w, x = ev.whistles(), ev.crossings(ball)
    _, real, _ = retime(after, w, x, ball, kin, constant=0)
    count = lambda rows, key: sum(r["reason"].startswith(key) for r in rows)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(args.draws):
        moved = []
        for events in (w, x):
            e = events + rng.integers(60 * FPS, 240 * FPS) * rng.choice([-1, 1])
            e = e[(e >= 0) & (e < len(after))]
            moved.append(np.sort(e[graded[e]]))
        _, rows, _ = retime(after, moved[0], moved[1], ball, kin, constant=0)
        draws.append((count(rows, "whistle"), count(rows, "ball"), sum(r["frames_earlier"] for r in rows)))
    draws = np.array(draws)
    report = dict(real=dict(whistle_moves=count(real, "whistle"), crossing_moves=count(real, "ball"), frames=int(sum(r["frames_earlier"] for r in real))),
                  control=dict(draws=int(len(draws)), whistle_moves_mean=float(draws[:, 0].mean()), whistle_moves_max=int(draws[:, 0].max()),
                               crossing_moves_mean=float(draws[:, 1].mean()), crossing_moves_max=int(draws[:, 1].max()),
                               frames_mean=float(draws[:, 2].mean()), frames_max=int(draws[:, 2].max())))
    (ROOT / "candidate_control.json").write_text(json.dumps(report, indent=1))
    print("V30E CONTROL", json.dumps(report, indent=1), flush=True)


def external(args):
    """The crossing rule and the constant shift on the three open Metrica games (no sound there).

    Base = the submission recipe leave-one-game-out (frozen decoder -> w = 0.5 verifier fitted on
    the two other games). The ball is the one the networks saw in each condition (clean /
    moderate / stress). Nothing is fitted here: CONSTANT_SHIFT was chosen leave-one-game-out in
    possession.v30e_retime and the crossing rule has no fitted parameter.
    """
    from . import v28a_core as k28
    conds = rt.verified_conditions()
    names = ("consistent", "alternative")
    variants = ("verified", "constant_all", "crossing_only", "crossing+constant", "crossing+constant+verifier_again")
    from . import v29_lab as lab29
    rows = []
    empty = np.array([], dtype=np.int64)
    for k in conds:
        s = k28.cached_signals(k.game, k.mode)
        xy = np.asarray(s["ball"], dtype=np.float64)
        length, width = 105.0, 68.0
        ball = dict(b_seen=np.asarray(s["seen"], dtype=np.float32), b_speed=np.asarray(s["speed"], dtype=np.float32),
                    b_outside_m=np.maximum.reduce([-xy[:, 0], xy[:, 0] - length, -xy[:, 1], xy[:, 1] - width]).astype(np.float32))
        graded = np.zeros(k.n, dtype=bool)
        for _, first, last in k.blocks:
            graded[first:last + 1] = True
        x = ev.crossings(ball, graded)
        out = dict(verified=k.verified, constant_all=rt.constant(k.verified, k.blocks, CONSTANT_SHIFT)[0])
        out["crossing_only"], moves, _ = retime(k.verified, empty, x, ball, None, constant=0, halves=k.blocks)
        out["crossing+constant"], moves2, _ = retime(k.verified, empty, x, ball, None, constant=CONSTANT_SHIFT, halves=k.blocks)
        again, removed = lab29.apply(k.model, c.segments(out["crossing+constant"], k.blocks), k.p, k.edge, None)
        out["crossing+constant+verifier_again"] = c.to_codes(again, k.n)
        reference = np.where(k.references["consistent"] >= 0, k.references["consistent"], 3).astype(np.int8)
        onsets = np.array([a for i, (a, z, code) in enumerate(c.segments(k.verified, k.blocks)) if code == 2 and i > 0])
        truth = np.array([a for i, (a, z, code) in enumerate(c.segments(reference, k.blocks)) if code == 2 and i > 0])
        j = np.clip(np.searchsorted(truth, onsets), 1, len(truth) - 1)
        lag = np.where(np.abs(truth[j] - onsets) < np.abs(truth[j - 1] - onsets), onsets - truth[j], onsets - truth[j - 1])
        near = np.abs(lag) <= 5 * FPS
        has = np.array([((x >= a - CROSSING_REACH) & (x <= a + FPS)).any() for a in onsets])
        row = dict(game=k.game, mode=k.mode, crossings=int(len(x)), stoppages=int(sum(seg[2] == 2 for seg in c.segments(k.verified, k.blocks))),
                   removed_by_second_verifier_pass=len(removed),
                   onset_lag_median_with_crossing=float(np.median(lag[near & has])), onset_lag_median_without_crossing=float(np.median(lag[near & ~has])),
                   onset_lag_quartiles_without_crossing=[float(v) for v in np.percentile(lag[near & ~has], [25, 75])],
                   onsets_moved_to_crossing=len(moves), median_frames_earlier=float(np.median([m["frames_earlier"] for m in moves])) if moves else 0.0,
                   onsets_moved_constant=sum(m["reason"] == "constant" for m in moves2))
        for reference in names:
            for name in variants:
                row[f"{reference}_{name}"] = c.spell_f1(k.gt[reference], c.spells_of(c.segments(out[name], k.blocks)))
        rows.append(row)
        print("EXTERNAL", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
    summary = {}
    for reference in names:
        for name in variants:
            per_game = [float(np.mean([r[f"{reference}_{name}"] for r in rows if r["game"] == g])) for g in (1, 2, 3)]
            delta = [r[f"{reference}_{name}"] - r[f"{reference}_verified"] for r in rows]
            summary[f"{reference}_{name}"] = dict(per_game=[round(v, 4) for v in per_game], mean=round(float(np.mean(per_game)), 4),
                                                  worst_condition_delta=round(float(min(delta)), 4), conditions_better=int(sum(d > 0 for d in delta)),
                                                  conditions_worse=int(sum(d < 0 for d in delta)))
            print("EXTERNAL SUMMARY", f"{reference}_{name:<18}", summary[f"{reference}_{name}"], flush=True)
    (ROOT / "candidate_external.json").write_text(json.dumps(dict(rows=rows, summary=summary, constant_shift=CONSTANT_SHIFT, challenge_data_read=False), indent=1))


def sensitivity(args):
    """How the re-timing depends on the whistle threshold and on each component (label-free)."""
    from . import v30e_lab as lab
    triple = lab.load_triple()
    fused, _ = fused_probabilities(triple, "camera")
    after, _, _ = verified(fused, triple)
    ball, kin, audio = F.load_group("ball"), F.load_group("kinematics"), F.load_group("audio")
    graded = ev.graded_mask(len(after))
    x = ev.crossings(ball)
    empty = np.array([], dtype=np.int64)
    rows = []

    def row(name, w, cross, constant):
        codes, moves, skipped = retime(after, w, cross, ball, kin, constant)
        count = lambda key: sum(m["reason"].startswith(key) for m in moves)
        out = dict(variant=name, whistles=int(len(w)), whistles_in_decoded_play=int((after[w] < 2).sum()) if len(w) else 0,
                   whistle_moves=count("whistle"), crossing_moves=count("ball"), constant_moves=count("constant"),
                   whistles_rejected=len(skipped), frames_changed=int((codes != after).sum()),
                   possessions=len(c.spells_of(c.segments(codes, blocks()))))
        rows.append(out)
        print("SENSITIVITY", out, flush=True)

    for threshold in (10.0, 12.0, 15.0, 18.0):
        events = [a for a, _, _, _ in F.events_of(audio["a_whistle_both_db"], threshold, F.WHISTLE_GAP, F.WHISTLE_MIN) if graded[a]]
        row(f"whistle >= {threshold:.0f} dB in both recordings", np.array(events, dtype=np.int64), x, CONSTANT_SHIFT)
    for name, key in (("tactical sound only >= 12 dB", "a_whistle_tactical_db"), ("tactical sound only >= 18 dB", "a_whistle_tactical_db")):
        threshold = float(name.split(">= ")[1].split(" ")[0])
        events = [a for a, _, _, _ in F.events_of(audio[key], threshold, F.WHISTLE_GAP, F.WHISTLE_MIN) if graded[a]]
        row(name, np.array(events, dtype=np.int64), x, CONSTANT_SHIFT)
    w = ev.whistles()
    row("whistles only", w, empty, 0)
    row("crossings only", empty, x, 0)
    row("constant only (all stoppages)", empty, empty, CONSTANT_SHIFT)
    row("whistles + crossings, no constant", w, x, 0)
    for constant in (5, 15):
        row(f"all, constant {constant}", w, x, constant)
    (ROOT / "candidate_sensitivity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["build", "control", "external", "sensitivity"])
    parser.add_argument("--student", default="camera")
    parser.add_argument("--variant", choices=list(VARIANTS), default="multimodal")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=200)
    args = parser.parse_args()
    dict(build=build, control=control, external=external, sensitivity=sensitivity)[args.step](args)
