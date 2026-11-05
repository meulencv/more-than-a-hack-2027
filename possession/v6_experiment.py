"""Run predefined V6 families on development only; confirmation remains locked."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
import torch
from . import v6_data as data
from .v6_temporal import train, predict
from .v3 import load_config, fit, decode_blocks
from .temporal_v3 import decode
from .sequence_run import load_sequence_model
from .sequence_tcn import predict as legacy_predict
from .sequence_quality_gate import combine
from .benchmark_external import evaluate_prediction


def subsequences(array, slices):
    return [array[a:b] for a, b in slices]


def as_match(record):
    return SimpleNamespace(ball=np.empty((len(record['full_labels']), 2)), blocks=record['blocks'])


def dynamic_decode(probabilities, boundaries, frames, match):
    """Boundary-conditioned switch costs, without a hard one-second minimum."""
    codes = np.full(len(match.ball), 3, dtype=np.int8)
    transition = np.array([[0., 4., 2.5], [4., 0., 2.5], [2.5, 2.5, 0.]])
    e = np.log(np.clip(probabilities, 1e-5, 1))
    for _, first, last in match.blocks:
        use = (frames >= first) & (frames <= last)
        emission, edge, f = e[use], boundaries[use], frames[use]
        score = emission[0].copy()
        back = np.zeros((len(f), 3), dtype=np.int8)
        for t in range(1, len(f)):
            cost = transition * (1 - .85 * float(edge[t]))
            candidate = score[:, None] - cost
            back[t] = candidate.argmax(axis=0)
            score = candidate[back[t], np.arange(3)] + emission[t]
            score -= score.max()
        states = np.empty(len(f), dtype=np.int8)
        current = int(score.argmax())
        for t in range(len(f)-1, -1, -1):
            states[t] = current
            current = int(back[t, current])
        codes[first:last+1] = np.repeat(states, np.r_[f[1:], last+1] - f)
    return codes


def run(family, steps=800):
    root = data.ROOT / family
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path('possession/config_external_selected.json'))
    torch.set_num_threads(6)
    include_metrica = family.endswith('mixed')
    if family == 'baseline':
        model = joblib.load('data/work/domain_adaptation/mixed.joblib')
        neural, norm = load_sequence_model('data/work/sequence_native_mix/mixed_model.pt', 'mps')
    elif family.startswith('hgb'):
        xs, ys = data.training_sequences(include_metrica)
        columns = data.record(data.protocol()['native_train'][0], 'clean', 'train')['columns']
        x = pd.DataFrame(np.concatenate(xs), columns=columns)
        y = np.concatenate(ys)
        keep = y >= 0
        print('Fitting', family, int(keep.sum()), flush=True)
        model = fit(x[keep], y[keep], cfg)
        joblib.dump(model, root / 'model.joblib')
    else:
        kind = family.removeprefix('tcn_')
        if kind == 'mixed':
            kind = 'local'
        xs, ys = data.training_sequences(include_metrica)
        print('Fitting', family, len(xs), 'sequences', flush=True)
        neural, norm = train(xs, ys, kind=kind, steps=steps, device='mps')
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in neural.state_dict().items()},
                        mean=norm.mean, std=norm.std, kind=kind, steps=steps, seed=2027), root / 'model.pt')
    rows = []
    for game in data.protocol()['native_development']:
        for mode in ('clean', 'moderate', 'stress'):
            r = data.record(game, mode, 'development')
            if family.startswith('hgb') or family == 'baseline':
                a = pd.DataFrame(r['direct'], columns=r['columns'])
                b = pd.DataFrame(r['mirror'], columns=r['columns'])
                p = (model.predict_proba(a) + model.predict_proba(b)[:, [1,0,2]]) / 2
                edge = np.zeros(len(p))
            else:
                pp, ee = predict(neural, norm, subsequences(r['direct'], r['slices']), 'mps')
                ps, es = predict(neural, norm, subsequences(r['mirror'], r['slices']), 'mps')
                p = (np.concatenate(pp) + np.concatenate(ps)[:, [1,0,2]]) / 2
                edge = (np.concatenate(ee) + np.concatenate(es)) / 2
            if family == 'baseline':
                q = np.concatenate(legacy_predict(neural, norm, subsequences(r['direct'],r['slices']), 'mps'))
                qs = np.concatenate(legacy_predict(neural,norm,subsequences(r['mirror'],r['slices']),'mps'))
                q = (q + qs[:, [1,0,2]]) / 2
                missing = pd.Series(1 - r['ball_seen'].astype(float))
                weights = np.zeros(len(missing))
                for _, a, b in r['blocks']:
                    weights[a:b+1] = missing.iloc[a:b+1].rolling(125,center=True,min_periods=1).mean()
                p = combine(p, q, weights[r['frames']])
            np.savez_compressed(root / f'{game}_{mode}.npz', probabilities=p, edges=edge, frames=r['frames'])
            for decoder in (['fixed', 'boundary'] if family == 'tcn_boundary' else ['fixed']):
                m = as_match(r)
                pred = (dynamic_decode(p, edge, r['frames'], m) if decoder == 'boundary' else
                        decode_blocks(p, r['frames'], m, cfg['decoder']))
                metrics = evaluate_prediction(r['full_labels'], pred)
                row = dict(family=family, decoder=decoder, game=game, mode=mode, **metrics)
                rows.append(row)
                print(family, decoder, game, mode, round(metrics['spell_f1_mean'],4),
                      round(metrics['macro_f1'],4), flush=True)
            (root / 'results.json').write_text(json.dumps(rows, indent=2))
    summary = {dec: {'spell_f1_mean': float(np.mean([r['spell_f1_mean'] for r in rows if r['decoder']==dec])),
                     'worst_spell_f1': float(min(r['spell_f1_mean'] for r in rows if r['decoder']==dec)),
                     'macro_f1': float(np.mean([r['macro_f1'] for r in rows if r['decoder']==dec]))}
               for dec in sorted({r['decoder'] for r in rows})}
    (root / 'summary.json').write_text(json.dumps(dict(family=family, summary=summary,
          development_only=True, confirmation_loaded=False, challenge_labels_read=False), indent=2))
    print(family, summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('families', nargs='+')
    parser.add_argument('--steps', type=int, default=800)
    args = parser.parse_args()
    allowed = {'baseline','hgb_native','hgb_mixed','tcn_local','tcn_global','tcn_segment',
               'tcn_boundary','tcn_multistage','tcn_mixed'}
    for family in args.families:
        if family not in allowed:
            parser.error('Unknown family: '+family)
        run(family, args.steps)
