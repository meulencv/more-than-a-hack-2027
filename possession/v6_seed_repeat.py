"""Repeat the two strongest families with a second initialization, without test access."""
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from . import v6_data as data
from .v6_temporal import train,predict
from .v6_experiment import subsequences,as_match,dynamic_decode
from .benchmark_external import evaluate_prediction


def physics_training():
    xs,ys=[],[]
    for game in data.protocol()['native_train']:
        for mode in ('clean','moderate','stress'):
            r=joblib.load(data.ROOT/'physics_features'/f'train_{game}_{mode}.joblib')
            for key in ('direct','mirror'):
                for a,b in r['slices']:
                    xs.append(r[key][a:b]);y=r['labels'][a:b]
                    ys.append(y if key=='direct' else np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y))
    return xs,ys


def main():
    for physical in (False,True):
        family='physics_tcn' if physical else 'tcn_boundary'
        root=data.ROOT/('seed2028_'+family);root.mkdir(parents=True,exist_ok=True)
        xs,ys=physics_training() if physical else data.training_sequences(False)
        model,norm=train(xs,ys,kind='boundary',steps=800,seed=2028,device='mps')
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        mean=norm.mean,std=norm.std,kind='boundary',steps=800,seed=2028,physics=physical),root/'model.pt')
        for game in data.protocol()['native_development']:
            for mode in ('clean','moderate','stress'):
                r=(joblib.load(data.ROOT/'physics_features'/f'development_{game}_{mode}.joblib')
                   if physical else data.record(game,mode,'development'))
                p,e=predict(model,norm,subsequences(r['direct'],r['slices']),'mps')
                q,f=predict(model,norm,subsequences(r['mirror'],r['slices']),'mps')
                prob=(np.concatenate(p)+np.concatenate(q)[:,[1,0,2]])/2
                edge=(np.concatenate(e)+np.concatenate(f))/2
                np.savez_compressed(root/f'{game}_{mode}.npz',probabilities=prob,edges=edge,frames=r['frames'])
                print('Repeated',family,game,mode,flush=True)
    rows=[]
    for game in data.protocol()['native_development']:
        for mode in ('clean','moderate','stress'):
            r=data.record(game,mode,'development')
            old=np.load(data.ROOT/'tcn_boundary'/f'{game}_{mode}.npz')
            oldp=np.load(data.ROOT/'physics_tcn'/f'{game}_{mode}.npz')
            new=np.load(data.ROOT/'seed2028_tcn_boundary'/f'{game}_{mode}.npz')
            newp=np.load(data.ROOT/'seed2028_physics_tcn'/f'{game}_{mode}.npz')
            options={
                'pair_2027':((old['probabilities']+oldp['probabilities'])/2,old['edges']),
                'pair_2028':((new['probabilities']+newp['probabilities'])/2,new['edges']),
                'four_model_mean':((old['probabilities']+oldp['probabilities']+new['probabilities']+newp['probabilities'])/4,
                                   (old['edges']+new['edges'])/2)}
            for name,(prob,edge) in options.items():
                folder=data.ROOT/name;folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{game}_{mode}.npz',probabilities=prob,edges=edge,frames=r['frames'])
                pred=dynamic_decode(prob,edge,r['frames'],as_match(r))
                metric=evaluate_prediction(r['full_labels'],pred)
                rows.append(dict(family=name,game=game,mode=mode,**metric))
    root=data.ROOT/'seed_repeat';root.mkdir(exist_ok=True)
    (root/'results.json').write_text(json.dumps(rows,indent=2))
    summary={name:dict(spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in rows if r['family']==name])),
                       worst_spell_f1=float(min(r['spell_f1_mean'] for r in rows if r['family']==name)))
             for name in ('pair_2027','pair_2028','four_model_mean')}
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
