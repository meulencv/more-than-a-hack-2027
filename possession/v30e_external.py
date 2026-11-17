"""V30e: how much spellF1 is at stake in stoppage boundaries? Open Metrica games only.

The three open Metrica games have no video or sound, so no multimodal signal can be scored on
them. What they can say is the CEILING of any stoppage evidence, however good:

    bias      predicted minus reference frame of each stoppage onset / restart (frozen decoder)
    retime    every predicted stoppage boundary within 3 s of a reference boundary of the same
              kind is moved onto it (the possessions' teams are untouched)
    dead      the reference dead-ball mask replaces the predicted one; H/A re-decoded inside
              live play with the unchanged team costs (perfect knowledge of every stoppage)

    .venv/bin/python -m possession.v30e_external

Writes data/work/v30e/external_ceiling.json. No challenge data is read.
"""
import json
from pathlib import Path
import numpy as np
from . import v27_core as c

ROOT = Path("data/work/v30e")
REACH = 3 * c.FPS


def boundaries(codes, blocks):
    """(onsets, restarts): first frame of each D run / first live frame after it, inside halves."""
    onsets, restarts = [], []
    for a, z, k in c.segments(codes, blocks):
        if k != 2:
            continue
        inside_start = any(first < a for _, first, last in blocks if first <= a <= last)
        inside_end = any(z <= last for _, first, last in blocks if first <= z - 1 <= last)
        if inside_start:
            onsets.append(a)
        if inside_end:
            restarts.append(z)
    return np.array(onsets), np.array(restarts)


def nearest(pred, ref, reach=5 * c.FPS):
    if not len(pred) or not len(ref):
        return np.array([])
    j = np.clip(np.searchsorted(ref, pred), 1, len(ref) - 1)
    d = np.where(np.abs(ref[j] - pred) < np.abs(ref[j - 1] - pred), pred - ref[j], pred - ref[j - 1])
    return d[np.abs(d) <= reach]


def retimed(codes, reference, blocks):
    """Move predicted D boundaries onto reference D boundaries within REACH (teams kept)."""
    out = codes.copy()
    ref_on, ref_re = boundaries(reference, blocks)
    segs = c.segments(codes, blocks)
    for i, (a, z, k) in enumerate(segs):
        if k != 2:
            continue
        previous = segs[i - 1] if i > 0 and segs[i - 1][1] == a else None
        following = segs[i + 1] if i + 1 < len(segs) and segs[i + 1][0] == z else None
        if previous is not None and len(ref_on):
            target = ref_on[np.abs(ref_on - a).argmin()]
            if abs(target - a) <= REACH and previous[0] < target < z:
                out[min(a, target):max(a, target)] = 2 if target < a else previous[2]
        if following is not None and len(ref_re):
            target = ref_re[np.abs(ref_re - z).argmin()]
            if abs(target - z) <= REACH and a < target < following[1]:
                out[min(z, target):max(z, target)] = following[2] if target < z else 2
    return out


def main():
    rows = []
    for k in c.all_conditions():
        base = c.decode(k.interpolated, k.n)[0]
        row = dict(game=k.game, mode=k.mode)
        for name in ("consistent", "alternative"):
            ref = np.where(k.references[name] >= 0, k.references[name], 3).astype(np.int8)
            gt = c.spells_of(c.segments(ref, k.blocks))
            score = lambda codes: c.spell_f1(gt, c.spells_of(c.segments(codes, k.blocks)))
            p_on, p_re = boundaries(base, k.blocks)
            r_on, r_re = boundaries(ref, k.blocks)
            d_on, d_re = nearest(p_on, r_on), nearest(p_re, r_re)
            dead = c.redecode_teams(k.interpolated, ref, [(c.DEFAULT["c_ha"], c.DEFAULT["gamma"])])[0]
            dead[ref == 3] = 3
            row[name] = dict(base=score(base), retime=score(retimed(base, ref, k.blocks)), dead=score(dead),
                             onset_bias_median=float(np.median(d_on)), onset_abs_median=float(np.median(np.abs(d_on))),
                             onset_iqr=[float(v) for v in np.percentile(d_on, [25, 75])],
                             restart_bias_median=float(np.median(d_re)), restart_abs_median=float(np.median(np.abs(d_re))),
                             restart_iqr=[float(v) for v in np.percentile(d_re, [25, 75])],
                             stoppages_pred=int(len(p_on)), stoppages_ref=int(len(r_on)),
                             onsets_matched_5s=int(len(d_on)), restarts_matched_5s=int(len(d_re)))
        rows.append(row)
        print("EXTERNAL", k.game, k.mode, {n: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in row[n].items()} for n in ("consistent", "alternative")}, flush=True)
    summary = {}
    for name in ("consistent", "alternative"):
        summary[name] = {key: round(float(np.mean([r[name][key] for r in rows])), 4) for key in ("base", "retime", "dead")}
        summary[name]["per_game"] = {str(g): {key: round(float(np.mean([r[name][key] for r in rows if r["game"] == g])), 4)
                                              for key in ("base", "retime", "dead")} for g in (1, 2, 3)}
        for key in ("onset_bias_median", "restart_bias_median", "onset_abs_median", "restart_abs_median"):
            summary[name][key] = {m: round(float(np.mean([r[name][key] for r in rows if r["mode"] == m])), 1) for m in c.MODES}
    print("EXTERNAL SUMMARY", json.dumps(summary, indent=1))
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "external_ceiling.json").write_text(json.dumps(dict(rows=rows, summary=summary, challenge_data_read=False), indent=1))


if __name__ == "__main__":
    main()
