"""A new representation of control: coupling between ball and player velocities.

Proximity alone confuses a reception with a ball passing overhead/nearby. These
signals measure relative speed, approach and impulses without using identities,
reference labels, native ball height or the competition answer key.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from . import v6_data as data
from .sportec_native import external_native_match
from .sources import challenge_match
from .degrade import challenge_profile
from .features_v3 import _gap_features
from .v6_temporal import train,predict
from .v6_experiment import subsequences,as_match,dynamic_decode
from .v3 import load_config,fit,decode_blocks
from .benchmark_external import evaluate_prediction

ROOT=data.ROOT/'physics_features'


def physical_table(match,swap=False):
    p=match.players.astype(float);b=match.ball.astype(float)
    team=1-match.team if swap else match.team
    n=len(b);lag=5
    pv=np.full_like(p,np.nan);bv=np.full_like(b,np.nan)
    pv[lag:]=(p[lag:]-p[:-lag])*5
    bv[lag:]=(b[lag:]-b[:-lag])*5
    ps=np.linalg.norm(pv,axis=2);pv[ps>12]=np.nan
    speed=np.linalg.norm(bv,axis=1);bv[speed>45]=np.nan
    delta=p-b[:,None]
    distance=np.linalg.norm(delta,axis=2)
    relative=np.linalg.norm(pv-bv[:,None],axis=2)
    score=np.exp(-.5*(distance/3)**2-.5*(relative/5)**2)
    seen=np.isfinite(b).all(axis=1)
    values={}
    for name,side in [('h',0),('a',1)]:
        valid=(team==side)[None,:]
        dd=np.where(valid & np.isfinite(distance),distance,np.inf)
        closest=dd.argmin(axis=1)
        nearest=dd.min(axis=1)
        rv=relative[np.arange(n),closest]
        values['relative_'+name]=np.where(np.isfinite(nearest),np.minimum(rv,50),np.nan)
        candidate=np.where(valid & np.isfinite(score),score,-np.inf).max(axis=1)
        values['control_'+name]=np.where(np.isfinite(candidate),candidate,np.nan)
        closing=np.full(n,np.nan)
        valid=np.isfinite(nearest[:-lag]) & np.isfinite(nearest[lag:])
        np.subtract(nearest[:-lag],nearest[lag:],out=closing[lag:],where=valid)
        closing[lag:]*=5
        closing[~np.isfinite(closing)]=np.nan
        values['closing_'+name]=np.clip(closing,-45,45)
    values['control_margin']=values['control_h']-values['control_a']
    acceleration=np.full(n,np.nan)
    acceleration[lag:]=np.linalg.norm(bv[lag:]-bv[:-lag],axis=1)*5
    values['ball_impulse']=np.minimum(acceleration,150)
    turn=np.full(n,np.nan)
    denom=np.linalg.norm(bv[lag:],axis=1)*np.linalg.norm(bv[:-lag],axis=1)
    dot=(bv[lag:]*bv[:-lag]).sum(axis=1)
    np.divide(dot,denom,out=turn[lag:],where=denom>.25)
    values['ball_turn']=np.clip(turn,-1,1)
    output=[]
    for _,a,z in match.blocks:
        raw=pd.DataFrame({k:v[a:z+1] for k,v in values.items()})
        cols={k:raw[k].to_numpy() for k in raw.columns}
        for key in raw.columns:
            for seconds in (1,3):
                cols[f'{key}_physical_c{seconds}']=raw[key].rolling(seconds*25,center=True,min_periods=1).mean().to_numpy()
        for key in ('control_margin','control_h','control_a'):
            last,nxt,_,_=_gap_features(raw[key].to_numpy(),seen[a:z+1])
            cols[key+'_physical_last']=last;cols[key+'_physical_next']=nxt
        frame=pd.DataFrame(cols).iloc[::5].reset_index(drop=True)
        output.append(frame)
    return pd.concat(output,ignore_index=True).astype('float32')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    profile=challenge_profile(challenge_match())
    for role,key in [('train','native_train'),('development','native_development')]:
        for i,game in enumerate(data.protocol()[key]):
            source=external_native_match(game)
            for mode in ('clean','moderate','stress'):
                r=data.record(game,mode,role)
                match=data.sensor_view(source,mode,(310 if role=='train' else 907)+i,profile)
                a=physical_table(match);b=physical_table(match,True)
                r['direct']=np.column_stack((r['direct'],a.to_numpy()))
                r['mirror']=np.column_stack((r['mirror'],b.to_numpy()))
                r['columns']+=list(a.columns)
                joblib.dump(r,ROOT/f'{role}_{game}_{mode}.joblib',compress=1)
                print('Physics prepared',game,mode,r['direct'].shape,flush=True)


def run(kind):
    root=data.ROOT/('physics_'+kind);root.mkdir(parents=True,exist_ok=True)
    xs,ys=[],[]
    for game in data.protocol()['native_train']:
        for mode in ('clean','moderate','stress'):
            r=joblib.load(ROOT/f'train_{game}_{mode}.joblib')
            for key in ('direct','mirror'):
                for a,b in r['slices']:
                    xs.append(r[key][a:b])
                    y=r['labels'][a:b]
                    ys.append(y if key=='direct' else np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y))
    cfg=load_config(Path('possession/config_external_selected.json'))
    if kind=='hgb':
        x=pd.DataFrame(np.concatenate(xs),columns=r['columns']);y=np.concatenate(ys);keep=y>=0
        model=fit(x[keep],y[keep],cfg);joblib.dump(model,root/'model.joblib')
    else:
        model,norm=train(xs,ys,kind='boundary',steps=800,device='mps')
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=norm.mean,std=norm.std,kind='boundary',steps=800,physics=True),root/'model.pt')
    rows=[]
    for game in data.protocol()['native_development']:
        for mode in ('clean','moderate','stress'):
            r=joblib.load(ROOT/f'development_{game}_{mode}.joblib')
            if kind=='hgb':
                p=(model.predict_proba(pd.DataFrame(r['direct'],columns=r['columns']))+
                   model.predict_proba(pd.DataFrame(r['mirror'],columns=r['columns']))[:,[1,0,2]])/2
                edge=np.zeros(len(p))
            else:
                q,e=predict(model,norm,subsequences(r['direct'],r['slices']),'mps')
                qm,em=predict(model,norm,subsequences(r['mirror'],r['slices']),'mps')
                p=(np.concatenate(q)+np.concatenate(qm)[:,[1,0,2]])/2
                edge=(np.concatenate(e)+np.concatenate(em))/2
            np.savez_compressed(root/f'{game}_{mode}.npz',probabilities=p,edges=edge,frames=r['frames'])
            for decoder in (['fixed'] if kind=='hgb' else ['fixed','boundary']):
                pred=(dynamic_decode(p,edge,r['frames'],as_match(r)) if decoder=='boundary' else
                      decode_blocks(p,r['frames'],as_match(r),cfg['decoder']))
                metric=evaluate_prediction(r['full_labels'],pred)
                rows.append(dict(family='physics_'+kind,game=game,mode=mode,decoder=decoder,**metric))
                print('physics_'+kind,game,mode,decoder,round(metric['spell_f1_mean'],4),flush=True)
            (root/'results.json').write_text(json.dumps(rows,indent=2))
    summary={d:float(np.mean([r['spell_f1_mean'] for r in rows if r['decoder']==d])) for d in {r['decoder'] for r in rows}}
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--kind',choices=['hgb','tcn'],default='hgb');args=parser.parse_args()
    if args.prepare:prepare()
    run(args.kind)
