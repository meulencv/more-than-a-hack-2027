"""Factorial external comparison: representation, seed averaging and second view."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from . import v6_data,v10_data
from .v6_physics import physical_table
from .features_v3 import feature_table
from .v6_bundle import Bundle
from .v6_temporal import predict as predict_temporal
from .v6_experiment import subsequences,as_match,dynamic_decode
from .v10_relational import load_model,predict as predict_relational
from .camera_sync import SyncResult,fused_ball
from .v11_camera import synthetic_view
from .sources import challenge_match
from .sportec_native import external_native_match
from .degrade import challenge_profile
from .benchmark_external import evaluate_prediction

ROOT=Path('data/work/v12')


def match_record(match):
    a,frames=feature_table(match);b,fm=feature_table(match,swap=True)
    assert np.array_equal(frames,fm)
    nodes,ball,team,raw_frames=v10_data.raw_table(match)
    assert np.array_equal(frames,raw_frames)
    sizes=[(z-a)//5+1 for _,a,z in match.blocks];cuts=np.r_[0,np.cumsum(sizes)]
    return dict(direct=np.column_stack((a.to_numpy(),physical_table(match).to_numpy())),
                mirror=np.column_stack((b.to_numpy(),physical_table(match,True).to_numpy())),
                nodes=nodes,ball_tokens=ball,team=team,frames=frames,blocks=match.blocks,
                full_labels=match.meta.get('labels',np.full(len(match.ball),-1,dtype='int8')),
                slices=list(zip(cuts[:-1],cuts[1:])))


def frozen_probabilities(bundle,r):
    probs=[];edges=[]
    for spec,net,norm in bundle.models:
        d=len(norm.mean)
        p,e=predict_temporal(net,norm,subsequences(r['direct'][:,:d],r['slices']),bundle.device)
        q,f=predict_temporal(net,norm,subsequences(r['mirror'][:,:d],r['slices']),bundle.device)
        probs.append((np.concatenate(p)+np.concatenate(q)[:,[1,0,2]])/2)
        if spec['boundary_head']:edges.append((np.concatenate(e)+np.concatenate(f))/2)
    return np.mean(probs,axis=0),np.mean(edges,axis=0)


def load_alignment(game):
    path=Path('data/work/v11')/f'{game}_alignment.npz';x=np.load(path);info=json.loads(path.with_suffix('.json').read_text())
    return SyncResult(x['target_frame'],x['reflection'],x['geometry_m'],info['runs'],info['proposals'])


def main(device='cpu'):
    torch.set_num_threads(6);ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'features').mkdir(exist_ok=True);(ROOT/'probabilities').mkdir(exist_ok=True)
    frozen=Bundle(json.loads(Path('docs/V6_SELECTION.json').read_text()),device)
    model_names=['trajectory_long','trajectory_long2028','trajectory_augmented']
    models={name:load_model(v10_data.ROOT/name/'model.pt',device) for name in model_names}
    profile=challenge_profile(challenge_match());rows=[]
    for i,game in enumerate(v6_data.protocol()['native_development']):
        source=external_native_match(game);other,truth,valid=synthetic_view(source,profile,5021+i);aligned=load_alignment(game)
        for mode in ('clean','moderate','stress'):
            base=v6_data.sensor_view(source,mode,907+i,profile)
            for view,gate in [('baseline',None),('fill_missing',None),('fill_near8',8.)]:
                stem=f'{game}_{mode}_{view}';cache=ROOT/'features'/f'{stem}.joblib';folder=ROOT/'probabilities'/stem;folder.mkdir(exist_ok=True)
                if view=='baseline':r=v10_data.record('development',game,mode);new_observations=0
                elif cache.exists():r=joblib.load(cache);new_observations=r['new_observations']
                else:
                    match,filled=fused_ball(base,other,aligned,gate);r=match_record(match)
                    new_observations=int(filled.sum());r['new_observations']=new_observations
                    temporary=cache.with_suffix('.part')
                    joblib.dump(r,temporary,compress=1);temporary.replace(cache)
                native_path=folder/'v6.npz'
                if native_path.exists():saved=np.load(native_path);p,e=saved['probabilities'],saved['edges']
                elif view=='baseline':
                    a=np.load(v6_data.ROOT/'tcn_boundary'/f'{game}_{mode}.npz');b=np.load(v6_data.ROOT/'physics_tcn'/f'{game}_{mode}.npz')
                    p=(a['probabilities']+b['probabilities'])/2;e=a['edges']
                    np.savez_compressed(native_path,probabilities=p,edges=e)
                else:p,e=frozen_probabilities(frozen,r);np.savez_compressed(native_path,probabilities=p,edges=e)
                neural=[]
                for name,(net,norm) in models.items():
                    path=folder/f'{name}.npz'
                    if path.exists():q=np.load(path)['probabilities']
                    elif view=='baseline':q=np.load(v10_data.ROOT/name/f'{game}_{mode}.npz')['probabilities'];np.savez_compressed(path,probabilities=q)
                    else:
                        q,edge,xy=predict_relational(net,norm,r,device);np.savez_compressed(path,probabilities=q,edges=edge,coordinates=xy)
                    neural.append(q)
                candidates={'v6':p,'long2027':(p+neural[0])/2,'two_seeds':(p+(neural[0]+neural[1])/2)/2,
                            'augmented_pair':(p+(neural[0]+neural[2])/2)/2}
                for name,q in candidates.items():
                    codes=dynamic_decode(q,e,r['frames'],as_match(r));metric=evaluate_prediction(r['full_labels'],codes)
                    rows.append(dict(game=game,mode=mode,view=view,model=name,new_observations=new_observations,**metric))
                print('FACTOR',stem,{x['model']:round(x['spell_f1_mean'],4) for x in rows[-4:]},flush=True)
                (ROOT/'results.json').write_text(json.dumps(rows,indent=2))
    summary=[]
    for view in ['baseline','fill_missing','fill_near8']:
        for model in ['v6','long2027','two_seeds','augmented_pair']:
            use=[r for r in rows if r['view']==view and r['model']==model]
            summary.append(dict(view=view,model=model,spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in use])),
                 worst_spell_f1=float(min(r['spell_f1_mean'] for r in use)),macro_f1=float(np.mean([r['macro_f1'] for r in use])),
                 by_game={g:float(np.mean([r['spell_f1_mean'] for r in use if r['game']==g])) for g in v6_data.protocol()['native_development']}))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print('FACTOR SUMMARY',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps'],default='cpu');main(p.parse_args().device)
