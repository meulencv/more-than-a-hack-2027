"""Prepare only the external feature caches required by sequence experiments.

This avoids running older, expensive model comparisons merely to get feature rows.
Rebuilds when config, source code, or external source signatures change.
"""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from possession.sources import external_match, challenge_match
from possession.degrade import challenge_profile
from possession.v3 import training_rows, load_config


def main():
    root = Path('data/work/hierarchical_v4')
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path('possession/config_v3.json'))
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    files = list(Path('data/external/metrica_sample').glob('Sample_Game_*'))
    files += [Path('possession') / name for name in
              ('reference.py', 'sources.py', 'features_v3.py', 'degrade.py', 'v3.py')]
    signature = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    signature['configuration'] = fingerprint
    manifest = root / 'feature_cache_manifest.json'
    verified = manifest.exists() and json.loads(manifest.read_text()) == signature
    profile = None
    for game in (1, 2, 3):
        path = root / f'training_{game}_{fingerprint}.joblib'
        if verified and path.exists():
            print('Verified cache', game, flush=True)
            continue
        if profile is None:
            profile = challenge_profile(challenge_match())
        data = training_rows(external_match(game), profile, cfg['training_seeds'], cfg)
        joblib.dump(data, path)
        print('Prepared cache', game, len(data[1]), flush=True)
    # Scripts written before provenance manifests use next(glob(...)); reject an
    # ambiguous cache instead of silently choosing a different configuration.
    for game in (1, 2, 3):
        matches = list(root.glob(f'training_{game}_*.joblib'))
        if len(matches) != 1:
            raise RuntimeError('Multiple cache configurations: move obsolete caches out of ' + str(root))
    (root / 'config.json').write_text(json.dumps(cfg, indent=2))
    manifest.write_text(json.dumps(signature, indent=2))


if __name__ == '__main__':
    main()
