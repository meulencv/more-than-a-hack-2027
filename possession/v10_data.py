"""Raw per-player trajectories; no reference channels enter the observations."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from . import v6_data
from .features_v3 import attack_direction
from .v6_physics import ROOT as PHYSICS
from .sources import challenge_match
from .sportec_native import external_native_match
from .degrade import challenge_profile

ROOT=Path('data/work/v10')
MAX_PLAYERS=64


def raw_table(match):
    nodes=[];balls=[];frames=[]
    if len(match.team)>MAX_PLAYERS:raise ValueError('More player slots than supported')
    for _,a,z in match.blocks:
        f=np.arange(a,z+1,5);p=match.players[f].astype(float);b=match.ball[f].astype(float)
        direction=attack_direction(match.players,match.team,a,z)
        center=np.array(match.pitch)/2
        p=(p-center)*direction;b=(b-center)*direction
        pv=np.full_like(p,np.nan);bv=np.full_like(b,np.nan)
        pv[1:]=(p[1:]-p[:-1])*5;bv[1:]=(b[1:]-b[:-1])*5
        ps=np.linalg.norm(pv,axis=2);bs=np.linalg.norm(bv,axis=1)
        pv[ps>12]=np.nan;bv[bs>45]=np.nan
        seen=np.isfinite(p).all(2);seenb=np.isfinite(b).all(1);seenv=np.isfinite(pv).all(2)
        rel=p-b[:,None];rv=pv-bv[:,None]
        values=[p[:,:,0]/center[0],p[:,:,1]/center[1],pv[:,:,0]/12,pv[:,:,1]/12,
                np.linalg.norm(pv,axis=2)/12,rel[:,:,0]/15,rel[:,:,1]/15,np.linalg.norm(rel,axis=2)/15,
                rv[:,:,0]/12,rv[:,:,1]/12,seen.astype(float),seenv.astype(float),
                np.broadcast_to(seenb[:,None],seen.shape).astype(float),np.isfinite(rv).all(2).astype(float)]
        x=np.stack(values,axis=2)
        x=np.clip(np.nan_to_num(x,nan=0,posinf=0,neginf=0),-8,8)
        padded=np.zeros((len(f),MAX_PLAYERS,x.shape[2]),dtype='float32');padded[:,:len(match.team)]=x
        vv=np.column_stack((b[:,0]/center[0],b[:,1]/center[1],bv[:,0]/45,bv[:,1]/45,
                            np.linalg.norm(bv,axis=1)/45,seenb.astype(float),np.isfinite(bv).all(1).astype(float)))
        nodes.append(padded.astype('float16'));balls.append(np.clip(np.nan_to_num(vv,nan=0,posinf=0,neginf=0),-8,8).astype('float32'))
        frames.append(f)
    team=np.full(MAX_PLAYERS,-1,dtype='int8');team[:len(match.team)]=match.team
    return np.concatenate(nodes),np.concatenate(balls),team,np.concatenate(frames)


def signature():
    paths=[Path(__file__),Path('docs/PROTOCOLO_V10.json'),Path('possession/v6_data.py'),Path('possession/features_v3.py')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare():
    root=ROOT/'raw';root.mkdir(parents=True,exist_ok=True)
    challenge=challenge_match();profile=challenge_profile(challenge)
    for role,key in [('train','native_train'),('development','native_development')]:
        for i,game in enumerate(v6_data.protocol()[key]):
            source=external_native_match(game)
            for mode in ('clean','moderate','stress'):
                path=root/f'{role}_{game}_{mode}.joblib'
                if path.exists() and joblib.load(path).get('raw_signature')==signature():continue
                r=joblib.load(PHYSICS/path.name)
                if r['signature']!=v6_data.source_signature():raise ValueError('Changed base features')
                match=v6_data.sensor_view(source,mode,(310 if role=='train' else 907)+i,profile)
                nodes,ball,team,frames=raw_table(match)
                if not np.array_equal(frames,r['frames']):raise ValueError('Feature timeline differs')
                r.update(nodes=nodes,ball_tokens=ball,team=team,raw_signature=signature())
                joblib.dump(r,path,compress=1);print('RAW',role,game,mode,nodes.shape,flush=True)
    path=root/'unlabeled_challenge.joblib'
    if not path.exists() or joblib.load(path).get('raw_signature')!=signature():
        nodes,ball,team,frames=raw_table(challenge)
        sizes=[(z-a)//5+1 for _,a,z in challenge.blocks];cuts=np.r_[0,np.cumsum(sizes)]
        r=dict(nodes=nodes,ball_tokens=ball,team=team,frames=frames,slices=list(zip(cuts[:-1],cuts[1:])),
               labels=np.full(len(frames),-1,dtype='int8'),raw_signature=signature(),no_human_labels=True)
        joblib.dump(r,path,compress=1);print('RAW unlabeled challenge',nodes.shape,flush=True)


def record(role,game=None,mode=None):
    name='unlabeled_challenge' if role=='unlabeled_challenge' else f'{role}_{game}_{mode}'
    r=joblib.load(ROOT/'raw'/f'{name}.joblib')
    if r['raw_signature']!=signature():raise ValueError('Changed raw representation')
    return r
