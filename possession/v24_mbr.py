"""V24 experiment: MBR decoding for spellF1 on the frozen out-of-fold predictions.

Reuses exactly the V13 out-of-fold probabilities that selected the 25 Hz decoder
(data/work/v13/fold*/<game>_<mode>.npz, method 'combined'), so the only change is
the decoder.  Adoption protocol of V17-V23: choose on J03WOH/J03WPY, then the chosen
option must improve spellF1 on *each* of J03WOY and J03WQQ, with mean macroF1 not
lower by more than 0.002.  Challenge labels are never read.

    OMP_NUM_THREADS=1 .venv/bin/python -m possession.v24_mbr --workers 6
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

CONFIGS = {
    'frozen': None,
    'cand_b3': dict(beta=3.0, refine=False),
    'cand_b5': dict(beta=5.0, refine=False),
    'cand_b8': dict(beta=8.0, refine=False),
    'refine_b4': dict(beta=4.0),
    'refine_b8': dict(beta=8.0),
}
SELECTION = ('J03WOH', 'J03WPY')
CONFIRMATION = ('J03WOY', 'J03WQQ')
MODES = ('clean', 'moderate', 'stress')
ROOT = Path('data/work/v24')


def run_case(args):
    index, game, mode, names = args
    from . import v13_cv
    from .v6_experiment import as_match
    from .v14_resolution import decode_fine
    from .benchmark_external import evaluate_prediction
    from .mbr_decoder import decode_mbr
    r = v13_cv.record(game, mode)
    match = as_match(r)
    saved = np.load(v13_cv.ROOT / f'fold{index}' / f'{game}_{mode}.npz')
    p, e, f = saved['combined'], saved['edges'], saved['frames']
    assert np.array_equal(f, r['frames'])
    rows = []
    for name in names:
        cfg = CONFIGS[name]
        report = []
        codes = decode_fine(p, e, f, match, 'probability') if cfg is None else decode_mbr(p, e, f, match, report=report, **cfg)
        metric = evaluate_prediction(r['full_labels'], codes)
        rows.append(dict(game=game, mode=mode, config=name, report=report, **metric))
    return rows


def summarise(rows, names):
    out = {}
    base = {(r['game'], r['mode']): r for r in rows if r['config'] == 'frozen'}
    for name in names:
        use = [r for r in rows if r['config'] == name]
        games = sorted({r['game'] for r in use})
        by_game = {g: float(np.mean([r['spell_f1_mean'] for r in use if r['game'] == g])) for g in games}
        delta = {g: by_game[g] - float(np.mean([base[(g, m)]['spell_f1_mean'] for m in MODES if (g, m) in base])) for g in games}
        out[name] = dict(spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in use])),
                         worst_spell_f1=float(min(r['spell_f1_mean'] for r in use)),
                         macro_f1=float(np.mean([r['macro_f1'] for r in use])),
                         n_pred_spells=float(np.mean([r['n_pred_spells'] for r in use])),
                         by_game=by_game, delta_vs_frozen=delta)
    return out


def decision(summary):
    def mean_on(name, games, key='spell_f1_mean'):
        return float(np.mean([summary[name]['by_game'][g] for g in games]))
    candidates = [n for n in summary if n != 'frozen']
    chosen = max(candidates, key=lambda n: mean_on(n, SELECTION))
    selection_gain = mean_on(chosen, SELECTION) - mean_on('frozen', SELECTION)
    each = {g: summary[chosen]['delta_vs_frozen'][g] for g in CONFIRMATION}
    macro_ok = summary[chosen]['macro_f1'] >= summary['frozen']['macro_f1'] - .002
    adopt = selection_gain > 0 and all(v > 0 for v in each.values()) and macro_ok
    return dict(chosen=chosen, selection_gain=selection_gain, confirmation_deltas=each,
                macro_ok=macro_ok, adopt=adopt)


def main():
    from . import v13_cv
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--configs', nargs='*', default=list(CONFIGS))
    args = parser.parse_args()
    names = ['frozen'] + [n for n in args.configs if n != 'frozen']
    jobs = []
    for index, held in enumerate(v13_cv.settings()['test_folds']):
        if not v13_cv.fold_complete(index):
            raise ValueError('Out-of-fold predictions missing: run possession.v13_cv first')
        jobs += [(index, game, mode, names) for game in held for mode in MODES]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for case in pool.map(run_case, jobs):
            rows += case
            print('V24', case[0]['game'], case[0]['mode'],
                  {r['config']: round(r['spell_f1_mean'], 4) for r in case}, flush=True)
    summary = summarise(rows, names)
    verdict = decision(summary)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'cross_validation.json').write_text(json.dumps(
        dict(rows=rows, summary=summary, decision=verdict, challenge_labels_read=False), indent=2))
    for name, s in summary.items():
        print(f"{name:10s} spellF1 {s['spell_f1_mean']:.4f}  peor {s['worst_spell_f1']:.4f}  "
              f"macroF1 {s['macro_f1']:.4f}  posesiones {s['n_pred_spells']:.0f}")
    print('DECISION', json.dumps(verdict, indent=2))


if __name__ == '__main__':
    main()
