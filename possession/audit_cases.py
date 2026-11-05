"""Case-level errors on external native references; never reads challenge labels.

Uses the already-frozen candidate probabilities. These are development diagnostics,
not independent leaderboard estimates. Case masks overlap. No subset spellF1 is
computed because masking frames would invent possession boundaries.
"""
import json
from pathlib import Path
import numpy as np
from .sportec_native import external_native_match
from .audit_stress import blind_corruption
from .sequence_quality_gate import combine
from .v3 import load_config, decode_blocks


def boundary_window(mask, radius, blocks):
    out = np.zeros(len(mask), dtype=bool)
    for _, first, last in blocks:
        points = np.flatnonzero(mask[first:last + 1]) + first
        for i in points:
            out[max(first, i - radius):min(last + 1, i + radius + 1)] = True
    return out


def case_masks(match, labels):
    live = np.isin(labels, [0, 1])
    seen = np.isfinite(match.ball).all(axis=1)
    restart = np.r_[False, (labels[:-1] == 2) & live[1:]]
    turnover = np.r_[False, live[:-1] & live[1:] & (labels[:-1] != labels[1:])]
    distance = np.linalg.norm(match.players - match.ball[:, None, :], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    home = distance[:, match.team == 0].min(axis=1)
    away = distance[:, match.team == 1].min(axis=1)
    near_both = (home < 3) & (away < 3)
    margin = np.zeros(len(labels))
    finite = np.isfinite(home) & np.isfinite(away)
    margin[finite] = abs(home[finite] - away[finite])
    return {
        'all_native_reference_frames': labels >= 0,
        'live_ball_observed': live & seen,
        'live_ball_missing': live & ~seen,
        'dead_ball_observed': (labels == 2) & seen,
        'dead_ball_missing': (labels == 2) & ~seen,
        'within_2s_restart': boundary_window(restart, 50, match.blocks),
        'within_1s_turnover': boundary_window(turnover, 25, match.blocks),
        'contested_geometry_proxy': live & seen & near_both & (margin < 1),
        'ball_over_3m_from_nearest_player_proxy': live & seen & (np.minimum(home, away) > 3),
    }


def main():
    root = Path('data/work/sequence_quality_gated')
    source = external_native_match('J03WN1')
    cfg = load_config(Path('possession/config_external_selected.json'))
    labels = source.meta['labels']
    rows = []
    for mode in ('clean', 'blind_stress'):
        match = source if mode == 'clean' else blind_corruption(source, 907)
        saved = np.load(root / f'heldout_{mode}.npz')
        pred = decode_blocks(saved['probabilities'], saved['frames'], match, cfg['decoder'])
        for name, mask in case_masks(match, labels).items():
            mask &= labels >= 0
            n = int(mask.sum())
            confusion = np.zeros((3, 3), dtype=int)
            np.add.at(confusion, (labels[mask], pred[mask]), 1)
            row = dict(mode=mode, case=name, frames=n,
                       accuracy=float((labels[mask] == pred[mask]).mean()) if n else None,
                       seconds=n / 25, confusion=confusion.tolist())
            rows.append(row)
            print(mode, name, n, None if not n else round(row['accuracy'], 4), flush=True)
    report = dict(reference='native external Sportec J03WN1; external development, not untouched test',
                  challenge_labels_read=False, candidate_changed=False,
                  warning='Masks overlap; geometry proxies do not establish a duel, flight or physical occlusion',
                  rows=rows)
    Path('data/work/case_audit.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
