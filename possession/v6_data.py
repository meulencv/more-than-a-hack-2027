"""Expanded native-only development protocol and label-blind sensor augmentations.

The confirmation match is deliberately not loaded by preparation or development.
All outputs are under ignored data/work. The frozen candidate is never overwritten.
"""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from .sportec_native import external_native_match
from .sources import challenge_match, external_match
from .degrade import challenge_profile, _ar_noise, _alternating_mask
from .audit_stress import blind_corruption
from .audit_reference import alternative_reference
from .features_v3 import feature_table
from .sequence_experiment import split_by_halves

ROOT = Path('data/work/v6')
PROTOCOL = Path('docs/PROTOCOLO_V6.json')


def protocol():
    return json.loads(PROTOCOL.read_text())


def sensor_view(match, mode, seed, profile):
    if mode == 'clean':
        return match
    if mode == 'stress':
        return blind_corruption(match, seed)
    if mode != 'moderate':
        raise ValueError('Unknown observation regime')
    rng = np.random.default_rng(seed)
    ball = match.ball.astype(float).copy()
    players = match.players.astype(float).copy()
    for _, first, last in match.blocks:
        n = last - first + 1
        ball[first:last+1] += _ar_noise(rng, n, .9, 15)
        missing = _alternating_mask(rng, n, profile.missing_runs, profile.seen_runs, .3)
        ball[first:last+1][missing] = np.nan
        for j in range(players.shape[1]):
            players[first:last+1, j] += _ar_noise(rng, n, .12, 12)
    return match.copy_with(players=players.astype('float32'), ball=ball.astype('float32'))


def source_signature():
    files = [Path(__file__), Path('possession/features_v3.py'), Path('possession/sportec_native.py'), PROTOCOL]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def record(game, mode, role):
    path = ROOT / 'features' / f'{role}_{game}_{mode}.joblib'
    saved = joblib.load(path)
    if saved['signature'] != source_signature():
        raise ValueError('Feature cache provenance mismatch; rerun v6_data')
    return saved


def training_sequences(include_metrica=False):
    xs, ys = [], []
    for game in protocol()['native_train']:
        for mode in ('clean', 'moderate', 'stress'):
            r = record(game, mode, 'train')
            for key in ('direct', 'mirror'):
                for first, last in r['slices']:
                    xs.append(r[key][first:last])
                    target = r['labels'][first:last]
                    if key == 'mirror':
                        target = np.where(target >= 0, np.array([1,0,2])[np.clip(target,0,2)], target)
                    ys.append(target)
    if include_metrica:
        for game in (1, 2, 3):
            source = external_match(game)
            labels, _ = alternative_reference(source.meta['events'], len(source.ball))
            frames = np.concatenate([np.arange(a, b+1, 5) for _, a, b in source.blocks])
            target = labels[frames]
            y = np.concatenate([target, np.where(target >= 0, np.array([1,0,2])[np.clip(target,0,2)], target)] * 2)
            x, previous = joblib.load(next(Path('data/work/hierarchical_v4').glob(f'training_{game}_*.joblib')))
            assert len(x) == len(y)
            seq, targets = split_by_halves(x.to_numpy(), y, source, repetitions=4)
            xs.extend(seq); ys.extend(targets)
    return xs, ys


def prepare():
    root = ROOT / 'features'
    root.mkdir(parents=True, exist_ok=True)
    profile = challenge_profile(challenge_match())
    signature = source_signature()
    p = protocol()
    for role, games in [('train', p['native_train']), ('development', p['native_development'])]:
        for i, game in enumerate(games):
            print('Loading', role, game, flush=True)
            source = external_native_match(game)
            for mode in ('clean', 'moderate', 'stress'):
                path = root / f'{role}_{game}_{mode}.joblib'
                if path.exists() and joblib.load(path).get('signature') == signature:
                    print('Verified', path.name, flush=True)
                    continue
                match = sensor_view(source, mode, 310 + i if role == 'train' else 907 + i, profile)
                a, frames = feature_table(match)
                b, mirrored_frames = feature_table(match, swap=True)
                assert np.array_equal(frames, mirrored_frames)
                lengths = [(z-a)//5+1 for _,a,z in match.blocks]
                limits = np.r_[0, np.cumsum(lengths)]
                saved = dict(signature=signature, direct=a.to_numpy(), mirror=b.to_numpy(),
                             columns=list(a.columns), frames=frames,
                             labels=source.meta['labels'][frames],
                             full_labels=source.meta['labels'], blocks=source.blocks,
                             slices=list(zip(limits[:-1].tolist(), limits[1:].tolist())),
                             name=game, mode=mode, role=role,
                             ball_seen=np.isfinite(match.ball).all(axis=1))
                joblib.dump(saved, path, compress=1)
                print('Prepared', path.name, len(frames), flush=True)
    (ROOT / 'data_provenance.json').write_text(json.dumps(dict(signature=signature, protocol=p), indent=2))


if __name__ == '__main__':
    prepare()
