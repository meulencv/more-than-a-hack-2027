"""Different model family, evaluated only on held-out external games."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from .sources import external_match, challenge_match
from .degrade import challenge_profile, degrade
from .features_v3 import feature_table
from .v3 import decode_blocks
from .benchmark_external import evaluate_prediction


def main():
    root = Path('data/work/hierarchical_v4')
    cfg = json.loads((root / 'config.json').read_text())
    training = {g: joblib.load(next(root.glob(f'training_{g}_*.joblib'))) for g in (1, 2, 3)}
    profile = challenge_profile(challenge_match())
    results = []
    for game in (1, 2, 3):
        others = [g for g in training if g != game]
        x = pd.concat([training[g][0] for g in others], ignore_index=True)
        y = np.concatenate([training[g][1] for g in others])
        forest = ExtraTreesClassifier(n_estimators=160, max_depth=20, min_samples_leaf=30,
                                      max_features=0.7, class_weight='balanced',
                                      random_state=2027, n_jobs=8)
        print('Training randomized forest; held out', game, flush=True)
        forest.fit(x, y)
        joblib.dump(forest, root / f'forest_fold_{game}.joblib')
        baseline = joblib.load(root / f'fold_{game}.joblib')['baseline']
        source = external_match(game)
        for seed in (42 + game, 202711 + game):
            match = degrade(source, profile, seed=seed, sigma=1.1, p_live=0.35, p_dead_sighting=0.5)
            a, frames = feature_table(match)
            b, _ = feature_table(match, swap=True)
            p = (forest.predict_proba(a) + forest.predict_proba(b)[:, [1, 0, 2]]) / 2
            base = (baseline.predict_proba(a) + baseline.predict_proba(b)[:, [1, 0, 2]]) / 2
            for name, prob in [('forest', p), ('forest_ensemble', (p + base) / 2)]:
                pred = decode_blocks(prob, frames, match, cfg['decoder'], stage2_cfg=cfg.get('stage2'))
                metric = evaluate_prediction(source.meta['labels'], pred)
                results.append(dict(method=name, game=game, seed=seed, **metric))
                print(game, seed, name, 'spellF1', round(metric['spell_f1_mean'], 4), flush=True)
            (root / 'forest_results.json').write_text(json.dumps(results, indent=2))
    print({name: np.mean([r['spell_f1_mean'] for r in results if r['method'] == name])
           for name in ('forest', 'forest_ensemble')}, flush=True)


if __name__ == '__main__':
    main()
