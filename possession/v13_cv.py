"""Out-of-fold comparison and final refit, with no competition labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from . import v6_data,v10_data
from .v6_temporal import train as train_temporal,predict as predict_temporal,SegmentNet
from .sequence_tcn import Normalizer
from .v10_relational import fit as fit_relational,load_model,predict as predict_relational
from .v6_experiment import as_match,subsequences,dynamic_decode
from .benchmark_external import evaluate_prediction

ROOT=Path('data/work/v13')
PROTOCOL=Path('docs/PROTOCOLO_V13.json')


def settings():return json.loads(PROTOCOL.read_text())


def record(game,mode):
    p=v6_data.protocol();role='train' if game in p['native_train'] else 'development'
    if game not in settings()['games']:raise ValueError('Closed/unregistered match')
    return v10_data.record(role,game,mode)


def training_rows(games,width):
    xx=[];yy=[]
    for game in games:
        for mode in ('clean','moderate','stress'):
            r=record(game,mode)
            for key in ('direct','mirror'):
                for a,b in r['slices']:
                    y=r['labels'][a:b];xx.append(r[key][a:b,:width])
                    yy.append(y if key=='direct' else np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y))
    return xx,yy


def load_temporal(path,device):
    c=torch.load(path,map_location='cpu',weights_only=False);norm=Normalizer()
    norm.mean=np.asarray(c['mean'],dtype='float32');norm.std=np.asarray(c['std'],dtype='float32')
    net=SegmentNet(len(norm.mean)*2,local=True);net.load_state_dict(c['state_dict'])
    return net.to(device).eval(),norm


def fit_components(games,folder,device='mps'):
    folder.mkdir(parents=True,exist_ok=True);out=[]
    for name,width,seed in [('standard',123,2027),('physics',156,2027),('trajectory2027',None,2027),('trajectory2028',None,2028)]:
        path=folder/f'{name}.pt'
        if path.exists():
            c=torch.load(path,map_location='cpu',weights_only=False)
            if c.get('train_games')!=games:raise ValueError('Checkpoint training partition mismatch')
            net,norm=load_temporal(path,device) if width else load_model(path,device)
        elif width:
            xx,yy=training_rows(games,width)
            net,norm=train_temporal(xx,yy,kind='boundary',steps=800,seed=seed,device=device)
        else:net,norm=fit_relational('trajectory',seed=seed,steps=3000,device=device,cosine=True,games=games)
        if not path.exists():
            checkpoint=dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=norm.mean,std=norm.std,
                kind='boundary' if width else 'trajectory',steps=800 if width else 3000,seed=seed,train_games=games,
                protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest())
            temporary=path.with_suffix('.part');torch.save(checkpoint,temporary);temporary.replace(path)
        out.append((name,net,norm,width))
        print('CV COMPONENT',folder.name,name,'games',games,flush=True)
    return out


def probabilities(models,r,device):
    output=[];edge=None
    for name,net,norm,width in models:
        if width:
            a,e=predict_temporal(net,norm,subsequences(r['direct'][:,:width],r['slices']),device)
            b,f=predict_temporal(net,norm,subsequences(r['mirror'][:,:width],r['slices']),device)
            p=(np.concatenate(a)+np.concatenate(b)[:,[1,0,2]])/2
            if name=='standard':edge=(np.concatenate(e)+np.concatenate(f))/2
        else:p,_,_=predict_relational(net,norm,r,device)
        output.append(p)
    return {'baseline':(output[0]+output[1])/2,'combined':np.mean(output,axis=0)},edge


def fold(index,device='mps'):
    p=settings();held=p['test_folds'][index];games=[g for g in p['games'] if g not in held]
    folder=ROOT/f'fold{index}';models=fit_components(games,folder,device);rows=[]
    for game in held:
        for mode in ('clean','moderate','stress'):
            r=record(game,mode);probs,edge=probabilities(models,r,device)
            for name,prob in probs.items():
                codes=dynamic_decode(prob,edge,r['frames'],as_match(r));metric=evaluate_prediction(r['full_labels'],codes)
                row=dict(fold=index,game=game,mode=mode,method=name,train_games=games,**metric)
                rows.append(row);print('CV HELD',row,flush=True)
            np.savez_compressed(folder/f'{game}_{mode}.npz',**probs,edges=edge,frames=r['frames'])
            (folder/'results.json').write_text(json.dumps(rows,indent=2))
    del models
    if device=='mps':torch.mps.empty_cache()


def summarize():
    rows=[]
    for i in range(3):rows+=json.loads((ROOT/f'fold{i}'/'results.json').read_text())
    report={'rows':rows,'summary':{},'challenge_labels_read':False,'closed_confirmation_used':False}
    for method in ['baseline','combined']:
        use=[r for r in rows if r['method']==method]
        report['summary'][method]=dict(spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in use])),
            worst_spell_f1=float(min(r['spell_f1_mean'] for r in use)),macro_f1=float(np.mean([r['macro_f1'] for r in use])),
            by_game={g:float(np.mean([r['spell_f1_mean'] for r in use if r['game']==g])) for g in settings()['games']})
    a,b=report['summary']['baseline'],report['summary']['combined']
    delta=np.array([b['by_game'][g]-a['by_game'][g] for g in settings()['games']])
    rng=np.random.default_rng(2027);boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    report['game_mean_deltas']=delta.tolist();report['descriptive_paired_game_bootstrap_95']=np.quantile(boot,[.025,.975]).tolist()
    report['adopt']=bool(b['spell_f1_mean']>a['spell_f1_mean'] and (delta>0).sum()>=4 and b['macro_f1']>=a['macro_f1'])
    (ROOT/'cross_validation.json').write_text(json.dumps(report,indent=2))
    print('CV SUMMARY',report['summary'],'DELTA',delta,'ADOPT',report['adopt'],flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=[0,1,2]);p.add_argument('--refit',action='store_true')
    p.add_argument('--device',choices=['cpu','mps'],default='mps');args=p.parse_args();torch.set_num_threads(6)
    if args.fold is not None:fold(args.fold,args.device)
    else:
        for i in range(3):
            if not (ROOT/f'fold{i}'/'results.json').exists():fold(i,args.device)
        report=summarize()
        if args.refit and report['adopt']:fit_components(settings()['games'],ROOT/'final',args.device)
