"""Audit reference construction without altering the frozen v3 implementation.

The alternative fixes a demonstrable self-match in one-touch detection. It is
still a weak reference, not a claim of manually verified possession labels.
"""
import json
from pathlib import Path
from types import FunctionType
import numpy as np
from . import reference


def corrected_one_touch(rows, i, max_frames=reference.TOUCH_FRAMES):
    current = rows[i]
    if current.type not in ("RECOVERY", "BALL LOST"):
        return False
    # The current BALL LOST cannot prove that its own possession was a short touch.
    for row in rows[i + 1:]:
        if row.start - current.start > max_frames:
            return False
        if row.team != current.team and row.type in reference.CONTROL:
            return True
        if row.team == current.team and row.type in ("PASS", "CARRY", "SHOT", "SET PIECE"):
            return False
        if row.team == current.team and row.type == "BALL LOST":
            return True
    return False


def alternative_reference(events, n):
    # Copy globals for isolation: no monkeypatch and no change to frozen code.
    namespace = dict(reference.event_reference.__globals__)
    namespace["_one_touch"] = corrected_one_touch
    function = FunctionType(reference.event_reference.__code__, namespace)
    return function(events, n)


def main():
    from .sources import external_match
    out = Path("data/work/audit_reference")
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for game in (1, 2, 3):
        match = external_match(game)
        events = match.meta["events"].sort_values(["start", "end"], kind="stable").reset_index(drop=True)
        rows = list(events.itertuples())
        lost = [i for i, r in enumerate(rows) if r.type == "BALL LOST"]
        before = sum(reference._one_touch(rows, i) for i in lost)
        after = sum(corrected_one_touch(rows, i) for i in lost)
        corrected, blocks = alternative_reference(events, len(match.ball))
        original = match.meta["labels"]
        keep = (original >= 0) & (corrected >= 0)
        changed = keep & (original != corrected)
        def spells(labels):
            starts = np.r_[True, labels[1:] != labels[:-1]]
            return int((starts & np.isin(labels, [0, 1])).sum())
        record = {"game": game, "ball_lost_events": len(lost),
                  "self_touch_original": before, "one_touch_corrected": after,
                  "changed_frames": int(changed.sum()),
                  "changed_seconds": float(changed.sum() / 25),
                  "changed_fraction": float(changed.sum() / max(keep.sum(), 1)),
                  "original_spells": spells(original), "alternative_spells": spells(corrected)}
        report.append(record)
        np.savez_compressed(out / f"references_{game}.npz", original=original, alternative=corrected)
        print(json.dumps(record), flush=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
