"""Counter-intuitive ablation: ignore ball observations entirely.

Uses collective player movement and shape only. Missing-ball quality remains a
possible mixture signal, not an input to this expert. No claim that the expert
should outperform models that observe a reliable ball.
"""
import json
from pathlib import Path
import numpy as np
import torch
from . import v6_data as data
from .v6_temporal import train, predict
from .v6_experiment import subsequences, as_match
from .v3 import load_config, decode_blocks
from .benchmark_external import evaluate_prediction


def main():
    root = data.ROOT / 'tcn_ballfree'
    root.mkdir(parents=True, exist_ok=True)
    columns = data.record(data.protocol()['native_train'][0], 'clean', 'train')['columns']
    prefixes = ('count_', 'sx_', 'sy_', 'cx_', 'cy_', 'line_', 'speed_med', 'speed_p90',
                'speed_h', 'speed_a', 'fwd_h', 'fwd_a', 'slow_')
    keep = np.array([c.startswith(prefixes) for c in columns])
    xs, ys = data.training_sequences(False)
    xs = [x[:, keep] for x in xs]
    print('Ball-free features:', int(keep.sum()), flush=True)
    model, norm = train(xs, ys, kind='local', steps=800, device='mps')
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    mean=norm.mean, std=norm.std, keep=keep, kind='local', steps=800), root/'model.pt')
    cfg = load_config(Path('possession/config_external_selected.json'))
    rows = []
    for game in data.protocol()['native_development']:
        for mode in ('clean','moderate','stress'):
            r = data.record(game,mode,'development')
            p,_ = predict(model,norm,subsequences(r['direct'][:,keep],r['slices']),'mps')
            q,_ = predict(model,norm,subsequences(r['mirror'][:,keep],r['slices']),'mps')
            prob = (np.concatenate(p)+np.concatenate(q)[:,[1,0,2]])/2
            np.savez_compressed(root/f'{game}_{mode}.npz',probabilities=prob,frames=r['frames'])
            pred=decode_blocks(prob,r['frames'],as_match(r),cfg['decoder'])
            metric=evaluate_prediction(r['full_labels'],pred)
            rows.append(dict(family='tcn_ballfree',game=game,mode=mode,**metric))
            print(game,mode,round(metric['spell_f1_mean'],4),flush=True)
    (root/'results.json').write_text(json.dumps(rows,indent=2))
    summary=dict(mean_spell_f1=float(np.mean([r['spell_f1_mean'] for r in rows])),
                 worst_spell_f1=float(min(r['spell_f1_mean'] for r in rows)),
                 feature_names=[c for c,k in zip(columns,keep) if k],
                 challenge_labels_read=False,confirmation_loaded=False)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
