"""External-only comparison of hierarchical estimation and probability ensembling.

Stores fold models, feature caches and all results locally in ignored data/work.
Also checks a second, independently degraded version of each held-out game.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .sources import challenge_match, external_match
from .degrade import challenge_profile, degrade
from .features_v3 import feature_table
from .v3 import fit, training_rows, load_config, decode_blocks
from .hierarchical import train, probabilities
from .benchmark_external import evaluate_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, default=Path('data/work/hierarchical_v4'))
    args = parser.parse_args()
    root = args.work_dir
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    (root / 'config.json').write_text(json.dumps(cfg, indent=2))
    profile = challenge_profile(challenge_match())
    matches = {g: external_match(g) for g in (1, 2, 3)}
    training = {}
    for game, match in matches.items():
        cache = root / f'training_{game}_{fingerprint}.joblib'
        if cache.exists():
            training[game] = joblib.load(cache)
        else:
            print('Preparing external training game', game, flush=True)
            training[game] = training_rows(match, profile, cfg['training_seeds'], cfg)
            joblib.dump(training[game], cache)
    results = []
    for heldout in (1, 2, 3):
        other = [g for g in matches if g != heldout]
        x = pd.concat([training[g][0] for g in other], ignore_index=True)
        y = np.concatenate([training[g][1] for g in other])
        print('Training fold; held out', heldout, 'rows', len(y), flush=True)
        baseline = fit(x, y, cfg)
        hierarchy = train(x, y, cfg)
        joblib.dump({'baseline': baseline, 'hierarchy': hierarchy}, root / f'fold_{heldout}.joblib')
        for seed in (42 + heldout, 202711 + heldout):
            match = degrade(matches[heldout], profile, seed=seed,
                            sigma=1.1, p_live=0.35, p_dead_sighting=0.5)
            a, frames = feature_table(match)
            b, mirrored_frames = feature_table(match, swap=True)
            assert np.array_equal(frames, mirrored_frames)
            base = (baseline.predict_proba(a) + baseline.predict_proba(b)[:, [1, 0, 2]]) / 2
            hier = probabilities(hierarchy, a, b)
            np.savez_compressed(root / f'probabilities_{heldout}_{seed}.npz',
                                frames=frames, baseline=base, hierarchy=hier)
            for name, p in [('v3', base), ('hierarchy', hier), ('ensemble', (base + hier) / 2)]:
                pred = decode_blocks(p, frames, match, cfg['decoder'], stage2_cfg=cfg.get('stage2'))
                metrics = evaluate_prediction(matches[heldout].meta['labels'], pred)
                record = dict(game=heldout, seed=seed, method=name, **metrics)
                results.append(record)
                print(heldout, seed, name, 'spellF1', round(metrics['spell_f1_mean'], 4), flush=True)
            (root / 'results.json').write_text(json.dumps(results, indent=2))
    report = {'reference': 'weak event-derived external references, not challenge truth',
              'challenge_labels_read': False, 'config': cfg,
              'mean_spell_f1': {method: float(np.mean([r['spell_f1_mean'] for r in results
                                                     if r['method'] == method]))
                                for method in ('v3', 'hierarchy', 'ensemble')}}
    (root / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
