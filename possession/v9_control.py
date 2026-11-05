"""Control anchors are observations with memory, never target labels.

Stable player/ball co-motion supports an anchor. A nearby ball passing a player
at high relative speed does not. Last/next anchors persist through missing ball
positions and are reset at every half boundary. The classifier may ignore them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .sources import challenge_match
from .sportec_native import external_native_match
from .degrade import challenge_profile
from .features_v3 import _gap_features
from . import v6_data
from .v6_physics import ROOT as PHYSICS
from .v6_temporal import train
from .v8_objects import evaluate
from .v3 import load_config,fit,decode_blocks
from .v6_experiment import as_match
from .benchmark_external import evaluate_prediction

ROOT=Path('data/work/v9')


def stable_runs(candidate,minimum=6):
    cuts=np.r_[0,np.flatnonzero(candidate[1:]!=candidate[:-1])+1,len(candidate)]
    valid=np.zeros(len(candidate),dtype=bool)
    for a,b in zip(cuts[:-1],cuts[1:]):
        if candidate[a]>=0 and b-a>=minimum:valid[a:b]=True
    return valid


def control_table(match,swap=False):
    team=1-match.team if swap else match.team;output=[]
    for _,a,z in match.blocks:
        p=match.players[a:z+1].astype(float);b=match.ball[a:z+1].astype(float);n=len(b)
        pv=np.full_like(p,np.nan);bv=np.full_like(b,np.nan)
        pv[5:]=(p[5:]-p[:-5])*5;bv[5:]=(b[5:]-b[:-5])*5
        pv[np.linalg.norm(pv,axis=2)>12]=np.nan
        speed=np.linalg.norm(bv,axis=1);bv[speed>45]=np.nan
        distance=np.linalg.norm(p-b[:,None],axis=2)
        relative=np.linalg.norm(pv-bv[:,None],axis=2)
        score=np.exp(-.5*(distance/2)**2-.5*(relative/4)**2)
        finite=np.isfinite(score);safe=np.where(finite,score,-np.inf)
        closest=safe.argmax(axis=1);best=safe.max(axis=1)
        home=np.where((team==0)[None]&finite,score,-np.inf).max(axis=1)
        away=np.where((team==1)[None]&finite,score,-np.inf).max(axis=1)
        margin=np.full(n,np.nan);np.subtract(home,away,out=margin,where=np.isfinite(home)&np.isfinite(away))
        candidate=np.where((best>=.55)&(np.abs(margin)>=.25),closest,-1)
        anchor=stable_runs(candidate)
        signed=np.where(team[closest]==0,1.,-1.)
        anchor_team=np.where(anchor,signed,np.nan)
        last,nxt,since,until=_gap_features(anchor_team,anchor)
        last_score,next_score,_,_=_gap_features(np.where(anchor,best,np.nan),anchor)
        index=np.arange(n);last_id=np.maximum.accumulate(np.where(anchor,index,-1))
        next_id=np.minimum.accumulate(np.where(anchor,index,n)[::-1])[::-1]
        past_player=closest[np.maximum(last_id,0)];next_player=closest[np.minimum(next_id,n-1)]
        past_distance=distance[index,past_player];next_distance=distance[index,next_player]
        past_distance[last_id<0]=np.nan;next_distance[next_id>=n]=np.nan
        both=np.isfinite(last)&np.isfinite(nxt)
        agree=np.where(both,(last==nxt).astype(float),np.nan)
        same_player=np.where(both,(past_player==next_player).astype(float),np.nan)
        blend=np.where(both,(last*until+nxt*since)/np.maximum(since+until,.04),np.nan)
        blend=np.where(both&(since+until==0),last,blend)
        flight=np.where(np.isfinite(best)&np.isfinite(speed),((speed>8)&(best<.3)).astype(float),np.nan)
        values=dict(anchor_last_team=last,anchor_next_team=nxt,anchor_since=since,anchor_until=until,
                    anchor_last_score=last_score,anchor_next_score=next_score,anchor_agreement=agree,
                    anchor_same_player=same_player,anchor_interpolated_team=blend,
                    anchor_last_distance=np.minimum(past_distance,60),anchor_next_distance=np.minimum(next_distance,60),
                    anchor_best_score=np.where(np.isfinite(best),best,np.nan),anchor_team_margin=margin,
                    anchor_flight_proxy=flight)
        signal=pd.DataFrame(dict(anchor_rate=anchor.astype(float),anchor_signed_rate=np.where(anchor,signed,0.),
                                 anchor_flight_rate=flight))
        for key in signal:
            for seconds in (1,3,8):
                values[f'{key}_c{seconds}']=signal[key].rolling(seconds*25,center=True,min_periods=1).mean().to_numpy()
        output.append(pd.DataFrame(values).iloc[::5].reset_index(drop=True))
    return pd.concat(output,ignore_index=True).astype('float32')


def signature():
    files=[Path(__file__),Path('docs/PROTOCOLO_V9.json')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def prepare():
    root=ROOT/'features';root.mkdir(parents=True,exist_ok=True)
    profile=challenge_profile(challenge_match())
    for role,key in [('train','native_train'),('development','native_development')]:
        for i,game in enumerate(v6_data.protocol()[key]):
            source=external_native_match(game)
            for mode in ('clean','moderate','stress'):
                path=root/f'{role}_{game}_{mode}.joblib'
                if path.exists() and joblib.load(path).get('control_signature')==signature():continue
                r=joblib.load(PHYSICS/path.name)
                # Check unchanged base extraction provenance before extending its cache.
                if r['signature']!=v6_data.source_signature():raise ValueError('Changed base cache')
                match=v6_data.sensor_view(source,mode,(310 if role=='train' else 907)+i,profile)
                direct=control_table(match);mirror=control_table(match,True)
                r['direct']=np.column_stack((r['direct'],direct.to_numpy()))
                r['mirror']=np.column_stack((r['mirror'],mirror.to_numpy()))
                r['columns']+=list(direct.columns);r['control_signature']=signature()
                joblib.dump(r,path,compress=1);print('ANCHORS',role,game,mode,r['direct'].shape,flush=True)


def record(role,game,mode):
    r=joblib.load(ROOT/'features'/f'{role}_{game}_{mode}.joblib')
    if r['control_signature']!=signature():raise ValueError('Changed control features')
    return r


def training():
    xx=[];yy=[]
    for game in v6_data.protocol()['native_train']:
        for mode in ('clean','moderate','stress'):
            r=record('train',game,mode)
            for key in ('direct','mirror'):
                for a,b in r['slices']:
                    y=r['labels'][a:b];xx.append(r[key][a:b])
                    yy.append(y if key=='direct' else np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y))
    return xx,yy,r['columns']


def run(kind,seed):
    root=ROOT/f'{kind}_{seed}';root.mkdir(parents=True,exist_ok=True)
    if (root/('model.pt' if kind=='tcn' else 'model.joblib')).exists():raise ValueError('Preserve existing model')
    xs,ys,columns=training()
    if kind=='tcn':
        net,norm=train(xs,ys,kind='boundary',steps=800,seed=seed,device='mps')
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=norm.mean,std=norm.std,
                        seed=seed,steps=800,kind='boundary',physics=True,anchors=True),root/'model.pt')
        evaluate(net,norm,root,feature_root=ROOT/'features')
    else:
        cfg=load_config(Path('possession/config_external_selected.json'))
        x=pd.DataFrame(np.concatenate(xs),columns=columns);y=np.concatenate(ys);known=y>=0
        model=fit(x[known],y[known],cfg);joblib.dump(model,root/'model.joblib');rows=[]
        for game in v6_data.protocol()['native_development']:
            for mode in ('clean','moderate','stress'):
                r=record('development',game,mode)
                p=(model.predict_proba(pd.DataFrame(r['direct'],columns=columns))+
                   model.predict_proba(pd.DataFrame(r['mirror'],columns=columns))[:,[1,0,2]])/2
                codes=decode_blocks(p,r['frames'],as_match(r),cfg['decoder']);metric=evaluate_prediction(r['full_labels'],codes)
                rows.append(dict(game=game,mode=mode,**metric));print('ANCHOR HGB',game,mode,metric['spell_f1_mean'],flush=True)
        (root/'results.json').write_text(json.dumps(dict(rows=rows,summary=dict(spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in rows])),
                           worst_spell_f1=float(min(r['spell_f1_mean'] for r in rows)))),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--kind',choices=['tcn','hgb']);parser.add_argument('--seed',type=int,default=2027)
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.kind:run(args.kind,args.seed)
