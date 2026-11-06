"""Continuous-time interpolation of frozen probabilities, external evaluation only."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import v6_data,v10_data
from .v6_experiment import dynamic_decode,as_match
from .benchmark_external import evaluate_prediction


def decode_fine(probabilities,boundaries,frames,match,space='probability'):
    codes=np.full(len(match.ball),3,dtype='int8')
    transition=np.array([[0.,4.,2.5],[4.,0.,2.5],[2.5,2.5,0.]])
    for _,first,last in match.blocks:
        selected=(frames>=first)&(frames<=last);f=frames[selected];grid=np.arange(first,last+1)
        if not len(f):raise ValueError('No emission in half')
        source=probabilities[selected]
        if space=='log':source=np.log(np.clip(source,1e-5,1))
        interpolated=np.column_stack([np.interp(grid,f,source[:,j]) for j in range(3)])
        if space=='probability':emission=np.log(np.clip(interpolated,1e-5,1))/5
        elif space=='log':emission=(interpolated-interpolated.max(1,keepdims=True))/5
        else:raise ValueError('Unknown interpolation space')
        edge=np.interp(grid,f,boundaries[selected]);score=emission[0].copy()
        back=np.zeros((len(grid),3),dtype='int8')
        for t in range(1,len(grid)):
            previous=score[:,None]-transition*(1-.85*float(edge[t]));back[t]=previous.argmax(0)
            score=previous[back[t],np.arange(3)]+emission[t];score-=score.max()
        current=int(score.argmax());states=np.empty(len(grid),dtype='int8')
        for t in range(len(grid)-1,-1,-1):states[t]=current;current=int(back[t,current])
        codes[first:last+1]=states
    return codes


def development():
    root=Path('data/work/v14');root.mkdir(parents=True,exist_ok=True);rows=[]
    for game in v6_data.protocol()['native_development']:
        for mode in ['clean','moderate','stress']:
            r=v10_data.record('development',game,mode)
            for view in ['baseline','fill_missing']:
                folder=Path('data/work/v12/probabilities')/f'{game}_{mode}_{view}'
                original=np.load(folder/'v6.npz');a=np.load(folder/'trajectory_long.npz')['probabilities'];b=np.load(folder/'trajectory_long2028.npz')['probabilities']
                p=(original['probabilities']+(a+b)/2)/2;e=original['edges'];match=as_match(r)
                for kind in ['original','probability','log']:
                    codes=dynamic_decode(p,e,r['frames'],match) if kind=='original' else decode_fine(p,e,r['frames'],match,kind)
                    metric=evaluate_prediction(r['full_labels'],codes);row=dict(game=game,mode=mode,view=view,decoder=kind,**metric)
                    rows.append(row);print('RESOLUTION',game,mode,view,kind,round(metric['spell_f1_mean'],4),flush=True)
    summary=[dict(view=view,decoder=kind,spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in rows if r['view']==view and r['decoder']==kind])),
                  worst_spell_f1=float(min(r['spell_f1_mean'] for r in rows if r['view']==view and r['decoder']==kind)))
             for view in ['baseline','fill_missing'] for kind in ['original','probability','log']]
    (root/'development.json').write_text(json.dumps(dict(rows=rows,summary=summary,challenge_labels_read=False),indent=2))
    print('RESOLUTION SUMMARY',summary,flush=True)


if __name__=='__main__':development()
