"""External-only, leave-one-match-out TCN test; no challenge truth is loaded."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from .sources import external_match, challenge_match
from .degrade import challenge_profile, degrade
from .features_v3 import feature_table
from .v3 import load_config, decode_blocks
from .benchmark_external import evaluate_prediction
from .sequence_tcn import fit, predict


def split_by_halves(x, labels, match, repetitions=1):
    counts = [(end - start) // 5 + 1 for _, start, end in match.blocks]
    if sum(counts) * repetitions != len(x):
        raise ValueError('Sequence reconstruction failed: cache no longer matches frame blocks')
    result, targets, cursor = [], [], 0
    for _ in range(repetitions):
        for n in counts:
            result.append(x[cursor:cursor+n])
            targets.append(labels[cursor:cursor+n])
            cursor += n
    return result, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--work-dir', type=Path, default=Path('data/work/sequence_tcn'))
    args = parser.parse_args()
    root = args.work_dir
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config('possession/config_external_selected.json')
    matches = {g: external_match(g) for g in (1,2,3)}
    training = {}
    for g in matches:
        x,y = joblib.load(next(Path('data/work/hierarchical_v4').glob(f'training_{g}_*.joblib')))
        training[g] = split_by_halves(x.to_numpy(), y, matches[g], repetitions=4)
    profile = challenge_profile(challenge_match())
    results = []
    for test_game in (1,2,3):
        train_x = [x for g in matches if g != test_game for x in training[g][0]]
        train_y = [y for g in matches if g != test_game for y in training[g][1]]
        print('Fold',test_game,'training sequences',len(train_x),flush=True)
        model,norm = fit(train_x,train_y,steps=args.steps,seed=2027,device=args.device)
        torch.save({'state_dict':model.cpu().state_dict(),'mean':norm.mean,'std':norm.std,
                    'steps':args.steps,'architecture':'TCN48x6'},root/f'fold_{test_game}.pt')
        model.to(args.device)
        baseline = joblib.load(Path('data/work/hierarchical_v4')/f'fold_{test_game}.joblib')['baseline']
        test = degrade(matches[test_game],profile,seed=42+test_game,sigma=1.1,p_live=.35,p_dead_sighting=.5)
        x,frames = feature_table(test)
        xs,fs = feature_table(test,swap=True)
        seq,_ = split_by_halves(x.to_numpy(),np.zeros(len(x)),test)
        mirror,_ = split_by_halves(xs.to_numpy(),np.zeros(len(xs)),test)
        p = np.concatenate(predict(model,norm,seq,args.device))
        ps = np.concatenate(predict(model,norm,mirror,args.device))[:,[1,0,2]]
        p=(p+ps)/2
        base=(baseline.predict_proba(x)+baseline.predict_proba(xs)[:,[1,0,2]])/2
        for name,prob in [('tcn',p),('tcn_ensemble',(p+base)/2),('v3',base)]:
            pred=decode_blocks(prob,frames,test,cfg['decoder'],stage2_cfg=cfg.get('stage2'))
            result=dict(game=test_game,method=name,**evaluate_prediction(matches[test_game].meta['labels'],pred))
            results.append(result)
            print(test_game,name,result['spell_f1_mean'],result['macro_f1'],flush=True)
        np.savez_compressed(root/f'probabilities_{test_game}.npz',tcn=p,baseline=base,frames=frames)
        (root/'results.json').write_text(json.dumps({'note':'weak event references, not challenge truth; provisional comparison', 'steps':args.steps,'results':results},indent=2))
    print({m:float(np.mean([r['spell_f1_mean'] for r in results if r['method']==m])) for m in ['v3','tcn','tcn_ensemble']},flush=True)


if __name__=='__main__': main()
