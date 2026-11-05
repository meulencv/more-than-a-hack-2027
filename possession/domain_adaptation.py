"""Train with corrected Metrica references and one native-reference Sportec game.

Native game J03WN1 is excluded from training. Evaluation is external development,
not an untouched benchmark. No Barcelona/Athletic reference labels are read.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .sources import external_match, challenge_match
from .sportec_native import external_native_match
from .audit_reference import alternative_reference
from .audit_stress import blind_corruption
from .degrade import challenge_profile
from .v3 import training_rows, fit, load_config, symmetric_proba, decode_blocks, file_sha256
from .benchmark_external import evaluate_prediction
from starter.baseline import write_submission


def main():
    root = Path('data/work/domain_adaptation')
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path('possession/config_external_selected.json'))
    profile = challenge_profile(challenge_match())
    xs, ys = [], []
    cache = Path('data/work/hierarchical_v4')
    for game in (1, 2, 3):
        match = external_match(game)
        labels, _ = alternative_reference(match.meta['events'], len(match.ball))
        frames = np.concatenate([np.arange(a, b + 1, 5) for _, a, b in match.blocks])
        target = labels[frames]
        values = []
        for seed in cfg['training_seeds']:
            for swap in (False, True):
                y = np.array([1, 0, 2])[np.clip(target, 0, 2)] if swap else target
                values.append(y[target >= 0])
        x, old = joblib.load(next(cache.glob(f'training_{game}_*.joblib')))
        y = np.concatenate(values)
        assert len(y) == len(x)
        xs.append(x); ys.append(y)
    x = pd.concat(xs, ignore_index=True); y = np.concatenate(ys)
    print('Fitting corrected-Metrica model', len(y), flush=True)
    corrected = fit(x, y, cfg)
    joblib.dump(corrected, root / 'corrected_metrica.joblib')
    print('Preparing Sportec training game J03WMX only', flush=True)
    native = external_native_match('J03WMX')
    nx, ny = training_rows(native, profile, cfg['training_seeds'], cfg)
    print('Fitting mixed-domain model', len(y) + len(ny), flush=True)
    mixed = fit(pd.concat([x, nx], ignore_index=True), np.concatenate([y, ny]), cfg)
    joblib.dump(mixed, root / 'mixed.joblib')
    source = external_native_match('J03WN1')
    rows = []
    baseline = joblib.load('data/work/temporal_v3/model.joblib')
    for name, model in [('legacy_hgb', baseline), ('corrected_hgb', corrected), ('mixed_hgb', mixed)]:
        for mode in ('clean', 'blind_stress'):
            match = source if mode == 'clean' else blind_corruption(source, 907)
            p, frames = symmetric_proba(model, match)
            np.savez_compressed(root / f'{name}_{mode}.npz', probabilities=p, frames=frames)
            pred = decode_blocks(p, frames, match, cfg['decoder'])
            metric = evaluate_prediction(source.meta['labels'], pred)
            rows.append(dict(model=name, corruption=mode, heldout='J03WN1', **metric))
            print(name, mode, 'spellF1', metric['spell_f1_mean'], flush=True)
            (root / 'results.json').write_text(json.dumps(rows, indent=2))
    report = dict(challenge_labels_read=False, config=cfg,
                  training='corrected weak Metrica 1/2/3 + native Sportec J03WMX',
                  validation='native Sportec J03WN1 never used in fitting')
    (root / 'provenance.json').write_text(json.dumps(report, indent=2))
    # Export both new candidates, without choosing on challenge labels.
    challenge = challenge_match()
    for name, model in [('corrected_hgb', corrected), ('mixed_hgb', mixed)]:
        p, frames = symmetric_proba(model, challenge)
        labels = np.array(['H', 'A', 'D', 'X'])[decode_blocks(p, frames, challenge, cfg['decoder'])]
        path = Path('evaluation/results') / (name + '.csv')
        write_submission(str(path), labels)
        print('Exported', path, file_sha256(path), flush=True)


if __name__ == '__main__':
    main()
