"""CatBoost alternative and ensemble, evaluated exclusively on held-out external games."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from .sources import challenge_match, external_match
from .degrade import challenge_profile, degrade
from .features_v3 import feature_table
from .v3 import load_config, decode_blocks, fit
from .audit_reference import alternative_reference
from .benchmark_external import evaluate_prediction

PARAMS = dict(iterations=400, depth=6, learning_rate=0.08, l2_leaf_reg=5,
              loss_function='MultiClass', random_seed=2027, thread_count=6,
              verbose=False, allow_writing_files=False, nan_mode='Min')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, default=Path('data/work/catboost_v5'))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = Path('data/work/hierarchical_v4')
    cfg = load_config(Path('possession/config_external_selected.json'))
    cached_config = json.loads((root / 'config.json').read_text())
    fingerprint = hashlib.sha256(json.dumps(cached_config, sort_keys=True).encode()).hexdigest()[:12]
    cache = {g: joblib.load(root / f'training_{g}_{fingerprint}.joblib') for g in (1, 2, 3)}
    profile = challenge_profile(challenge_match())
    matches = {g: external_match(g) for g in (1, 2, 3)}
    alternatives = {g: alternative_reference(m.meta['events'], len(m.ball))[0]
                    for g, m in matches.items()}
    # Feature rows remain the same: the correction changes H/A, never dead/live.
    # Rebuild labels for each seed and team orientation, rather than reusing stale labels.
    for g, m in matches.items():
        frames = np.concatenate([np.arange(a, b + 1, 5) for _, a, b in m.blocks])
        y = alternatives[g][frames]
        labels = []
        for seed in cached_config['training_seeds']:
            for swap in (False, True):
                target = np.array([1, 0, 2])[np.clip(y, 0, 2)] if swap else y
                labels.append(target[y >= 0])
        new_y = np.concatenate(labels)
        if len(new_y) != len(cache[g][1]):
            raise ValueError('Corrected reference changed valid row count')
        cache[g] = cache[g][0], new_y
    results = []
    for heldout in (1, 2, 3):
        others = [g for g in cache if g != heldout]
        x = pd.concat([cache[g][0] for g in others], ignore_index=True)
        y = np.concatenate([cache[g][1] for g in others])
        print('Training CatBoost; excluded external game', heldout, 'rows', len(y), flush=True)
        model = CatBoostClassifier(**PARAMS)
        model.fit(x, y)
        model.save_model(str(args.work_dir / f'fold_{heldout}.cbm'))
        baseline = fit(x, y, cfg)
        joblib.dump(baseline, args.work_dir / f'baseline_fold_{heldout}.joblib')
        source = matches[heldout]
        for seed in (42 + heldout, 202711 + heldout):
            match = degrade(source, profile, seed=seed, sigma=1.1, p_live=.35, p_dead_sighting=.5)
            a, frames = feature_table(match)
            b, mirror_frames = feature_table(match, swap=True)
            assert np.array_equal(frames, mirror_frames)
            p = (model.predict_proba(a) + model.predict_proba(b)[:, [1, 0, 2]]) / 2
            base = (baseline.predict_proba(a) + baseline.predict_proba(b)[:, [1, 0, 2]]) / 2
            np.savez_compressed(args.work_dir / f'probabilities_{heldout}_{seed}.npz', frames=frames,
                                baseline=base, catboost=p)
            for name, prob in [('baseline', base), ('catboost', p), ('ensemble', (p + base) / 2)]:
                pred = decode_blocks(prob, frames, match, cfg['decoder'], stage2_cfg=cfg.get('stage2'))
                for reference_name, labels in [('legacy', source.meta['labels']),
                                                ('corrected', alternatives[heldout])]:
                    metric = evaluate_prediction(labels, pred)
                    results.append(dict(game=heldout, seed=seed, method=name,
                                        reference=reference_name, **metric))
                    print(heldout, seed, name, reference_name, 'spellF1',
                          round(metric['spell_f1_mean'], 4), flush=True)
            (args.work_dir / 'results.json').write_text(json.dumps(results, indent=2))
    summary = {'challenge_labels_read': False, 'parameters': PARAMS, 'decoder_config': cfg,
               'training_reference': 'corrected weak event labels',
               'mean_corrected_spell_f1': {method: float(np.mean([r['spell_f1_mean'] for r in results
                                                      if r['method'] == method and r['reference'] == 'corrected']))
                                 for method in ('baseline', 'catboost', 'ensemble')}}
    (args.work_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
