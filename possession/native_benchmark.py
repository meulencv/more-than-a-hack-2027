"""Independent cross-domain check on native tracking labels from Sportec/DFL.

Models are trained only on Metrica samples. Sportec labels never enter features.
No competition labels are read. These two games are external development checks,
not a promise of equivalent annotation semantics or challenge performance.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from .sportec_native import external_native_match
from .audit_stress import blind_corruption
from .v3 import symmetric_proba, decode_blocks, load_config
from .sequence_run import load_sequence_model, match_probabilities
from .benchmark_external import evaluate_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-model', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, default=Path('data/work/native_benchmark'))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path('possession/config_external_selected.json'))
    baseline = joblib.load('data/work/temporal_v3/model.joblib')
    tcn, norm = load_sequence_model(args.sequence_model, 'mps')
    rows = []
    for game in ('J03WMX', 'J03WN1'):
        source = external_native_match(game)
        for mode in ('clean', 'blind_stress'):
            match = source if mode == 'clean' else blind_corruption(source, seed=907)
            p, frames = symmetric_proba(baseline, match)
            q, qframes = match_probabilities(tcn, norm, match, 'mps')
            assert np.array_equal(frames, qframes)
            np.savez_compressed(args.work_dir / f'{game}_{mode}.npz', frames=frames, hgb=p, tcn=q)
            for method, prob in [('hgb', p), ('tcn', q), ('ensemble', (p + q) / 2)]:
                codes = decode_blocks(prob, frames, match, cfg['decoder'])
                metric = evaluate_prediction(source.meta['labels'], codes)
                rows.append(dict(game=game, corruption=mode, method=method, **metric))
                print(game, mode, method, 'spellF1', round(metric['spell_f1_mean'], 4),
                      'macroF1', round(metric['macro_f1'], 4), flush=True)
            (args.work_dir / 'results.json').write_text(json.dumps(rows, indent=2))
    summary = {'competition_labels_read': False, 'training_domain': 'Metrica 1/2/3 only',
               'external_reference': 'Sportec native BallStatus and BallPossession',
               'source': 'https://doi.org/10.6084/m9.figshare.28196177',
               'license': 'CC BY 4.0',
               'mean_spell_f1': {method: float(np.mean([r['spell_f1_mean'] for r in rows
                                                      if r['method'] == method]))
                                 for method in ('hgb', 'tcn', 'ensemble')}}
    (args.work_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
