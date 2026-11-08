"""Compare V24 beta=4 against fixed-scale controls on the external V13 folds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from possession import mbr_decoder, v13_cv, v24_mbr
from possession.benchmark_external import evaluate_prediction
from possession.v14_resolution import decode_fine
from possession.v6_experiment import as_match


DURATION_BINS = ((0, 25), (25, 75), (75, 250), (250, None))
FIXED_SCALES = (0.6, 0.8)


def score_case(job):
    fold, game, mode = job
    record = v13_cv.record(game, mode)
    match = as_match(record)
    with np.load(v13_cv.ROOT / f'fold{fold}' / f'{game}_{mode}.npz') as saved:
        probabilities, edges, frames = saved['combined'], saved['edges'], saved['frames']
    if not np.array_equal(frames, record['frames']):
        raise ValueError(f'Timeline mismatch for {game} / {mode}')

    frozen = decode_fine(probabilities, edges, frames, match, 'probability')
    report = []
    beta4 = mbr_decoder.decode_mbr(probabilities, edges, frames, match,
                                   beta=4.0, refine=False, report=report)
    predictions = {'frozen': frozen, 'cand_b4': beta4}
    for scale in FIXED_SCALES:
        predictions[f'fixed_{scale}'] = np.full_like(frozen, 3)
    for _, first, last in match.blocks:
        emission, gate = mbr_decoder.prepare_half(probabilities, edges, frames, first, last)
        for scale in FIXED_SCALES:
            predictions[f'fixed_{scale}'][first:last + 1] = mbr_decoder.viterbi(emission, gate, scale)

    gt = np.where(record['full_labels'] >= 0, record['full_labels'], 3)
    runs = pm.valid_runs(gt)
    truth = pm.possession_spells(gt, runs)
    lengths = np.asarray([stop - start for start, stop, _ in truth])
    rows = []
    for name, codes in predictions.items():
        spells = pm.possession_spells(codes, runs)
        matched = pm.matched_gt_flags(pm.spell_overlaps(truth, spells), len(truth), 0.5)
        duration = []
        for start, stop in DURATION_BINS:
            use = lengths >= start
            if stop is not None:
                use &= lengths < stop
            duration.append({'min_frames': start, 'max_frames_exclusive': stop,
                             'n_gt': int(use.sum()), 'n_matched': int((use & matched).sum())})
        rows.append(dict(game=game, mode=mode, config=name, recall_by_duration_iou_0_5=duration,
                         dead_frames_changed_vs_frozen=int(np.count_nonzero((codes == 2) != (frozen == 2))),
                         report=report if name == 'cand_b4' else [], **evaluate_prediction(gt, codes)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--out', type=Path, default=Path('data/work/v24_diagnostics/results.json'))
    args = parser.parse_args()
    jobs = [(index, game, mode)
            for index, held in enumerate(v13_cv.settings()['test_folds'])
            for game in held for mode in v24_mbr.MODES]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(score_case, jobs):
            rows.extend(result)
            print(result[0]['game'], result[0]['mode'],
                  {row['config']: round(row['spell_f1_mean'], 4) for row in result}, flush=True)
    summary = v24_mbr.summarise(rows, list(dict.fromkeys(row['config'] for row in rows)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(rows=rows, summary=summary, exploratory=True,
                                        challenge_labels_read=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
