"""Train temporal CNN on external games or reuse a local checkpoint for inference."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from .sources import external_match, challenge_match
from .features_v3 import feature_table
from .v3 import load_config, decode_blocks
from .sequence_tcn import TemporalCNN, Normalizer, fit, predict
from .sequence_experiment import split_by_halves
from starter.baseline import write_submission


def load_sequence_model(path, device='cpu'):
    """Load only locally generated checkpoints. Returned normalizer is training-only."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    norm=Normalizer()
    norm.mean=np.asarray(checkpoint['mean'],dtype='float32')
    norm.std=np.asarray(checkpoint['std'],dtype='float32')
    model=TemporalCNN(len(norm.mean)*2,width=48)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(),norm


def match_probabilities(model,norm,match,device='cpu'):
    x,frames=feature_table(match)
    xm,fm=feature_table(match,swap=True)
    if not np.array_equal(frames,fm): raise ValueError('Mirrored frames misaligned')
    seq,_=split_by_halves(x.to_numpy(),np.zeros(len(x)),match)
    mirrored,_=split_by_halves(xm.to_numpy(),np.zeros(len(xm)),match)
    p=np.concatenate(predict(model,norm,seq,device))
    ps=np.concatenate(predict(model,norm,mirrored,device))[:,[1,0,2]]
    return (p+ps)/2,frames


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path)
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--hgb-model',type=Path,help='Optional frozen HGB checkpoint for 50/50 probability ensemble')
    parser.add_argument('--device',choices=['cpu','mps'],default='mps')
    parser.add_argument('--output',type=Path,default=Path('evaluation/results/sequence_tcn.csv'))
    parser.add_argument('--work-dir',type=Path,default=Path('data/work/sequence_tcn_final'))
    args=parser.parse_args()
    args.work_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_path=args.model or args.work_dir/'sequence_model.pt'
    if args.model:
        model,norm=load_sequence_model(args.model,args.device)
    else:
        xs,ys=[],[]
        for game in (1,2,3):
            match=external_match(game)
            x,y=joblib.load(next(Path('data/work/hierarchical_v4').glob(f'training_{game}_*.joblib')))
            seq,target=split_by_halves(x.to_numpy(),y,match,repetitions=4)
            xs.extend(seq);ys.extend(target)
        model,norm=fit(xs,ys,steps=args.steps,device=args.device)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        torch.save(dict(state_dict=state,mean=norm.mean,std=norm.std,steps=args.steps,
                        seed=2027,architecture='TCN48x6',references='legacy Metrica event-derived'),checkpoint_path)
    match=challenge_match()
    p,frames=match_probabilities(model,norm,match,args.device)
    if args.hgb_model:
        hgb=joblib.load(args.hgb_model)
        x,_=feature_table(match);xm,_=feature_table(match,swap=True)
        base=(hgb.predict_proba(x)+hgb.predict_proba(xm)[:,[1,0,2]])/2
        p=(p+base)/2
    cfg=load_config('possession/config_external_selected.json')
    codes=decode_blocks(p,frames,match,cfg['decoder'],stage2_cfg=cfg.get('stage2'))
    labels=np.array(['H','A','D','X'])[codes]
    write_submission(str(args.output),labels)
    np.savez_compressed(args.work_dir/'probabilities.npz',probabilities=p,frames=frames)
    report={'architecture':'TCN48x6; symmetric inference; half-local sequences',
            'training':'Metrica external games1,2,3; legacy weak references',
            'challenge_ground_truth_read':False,'steps':args.steps,
            'hgb_ensemble':str(args.hgb_model) if args.hgb_model else None,
            'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'csv_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
            'label_counts':{k:int((labels==k).sum()) for k in 'HADX'}}
    (args.work_dir/'run.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
