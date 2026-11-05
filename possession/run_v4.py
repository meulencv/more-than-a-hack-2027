"""Fit hierarchical/ensemble models on external games and export a full-match CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from starter.baseline import write_submission
from .sources import challenge_match, external_match
from .degrade import challenge_profile
from .features_v3 import feature_table
from .v3 import training_rows, fit, load_config, decode_blocks, file_sha256
from .hierarchical import train, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['hierarchy', 'ensemble'])
    parser.add_argument('--config', type=Path, default=Path('possession/config_v3.json'))
    parser.add_argument('--output', type=Path, default=Path('evaluation/results/temporal_v4.csv'))
    parser.add_argument('--work-dir', type=Path, default=Path('data/work/temporal_v4'))
    parser.add_argument('--model', type=Path)
    parser.add_argument('--training-cache', type=Path,
                        help='Optional feature cache produced by experiment_v4 with identical config')
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = args.work_dir
    root.mkdir(parents=True, exist_ok=True)
    match = challenge_match()
    model_path = args.model or root / 'models.joblib'
    if args.model:
        bundle = joblib.load(args.model)
        if bundle['config'] != cfg or bundle['method'] != args.method:
            raise ValueError('Stored model method/config does not match invocation')
    else:
        profile = challenge_profile(match)
        rows = []
        fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
        if args.training_cache and json.loads((args.training_cache / 'config.json').read_text()) != cfg:
            raise ValueError('Training cache configuration mismatch')
        for game in (1, 2, 3):
            print('Preparing external game', game, flush=True)
            if args.training_cache:
                rows.append(joblib.load(args.training_cache / f'training_{game}_{fingerprint}.joblib'))
            else:
                rows.append(training_rows(external_match(game), profile, cfg['training_seeds'], cfg))
        x = pd.concat([r[0] for r in rows], ignore_index=True)
        y = np.concatenate([r[1] for r in rows])
        print('Training hierarchical models', len(y), 'examples', flush=True)
        bundle = {'hierarchy': train(x, y, cfg), 'method': args.method, 'config': cfg}
        if args.method == 'ensemble':
            bundle['baseline'] = fit(x, y, cfg)
        joblib.dump(bundle, model_path)
    x, frames = feature_table(match)
    mirrored, mirrored_frames = feature_table(match, swap=True)
    assert np.array_equal(frames, mirrored_frames)
    p = probabilities(bundle['hierarchy'], x, mirrored)
    if args.method == 'ensemble':
        model = bundle['baseline']
        base = (model.predict_proba(x) + model.predict_proba(mirrored)[:, [1, 0, 2]]) / 2
        p = (p + base) / 2
    pred = decode_blocks(p, frames, match, cfg['decoder'], stage2_cfg=cfg.get('stage2'))
    labels = np.array(['H', 'A', 'D', 'X'])[pred]
    write_submission(str(args.output), labels)
    report = {'method': args.method, 'config': cfg, 'challenge_labels_read': False,
              'training': 'external Metrica games 1, 2, 3; weak event references',
              'model_sha256': file_sha256(model_path), 'csv_sha256': file_sha256(args.output),
              'counts': {k: int((labels == k).sum()) for k in 'HADX'}}
    (root / 'run.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
