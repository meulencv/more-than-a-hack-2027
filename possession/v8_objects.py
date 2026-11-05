"""Train on temporal possession objects using external native references only.

Soft IoU is computed for each complete H/A training run within the local region
formed by that run and its two neighbors. Other same-team runs do not contaminate
its denominator. No target run is ever supplied to inference.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from . import v6_data
from .v6_physics import ROOT as PHYSICS
from .v6_temporal import SegmentNet,segment_weights,boundary_targets,predict
from .sequence_tcn import Normalizer
from .v6_experiment import subsequences,as_match,dynamic_decode
from .v3 import decode_blocks,load_config
from .benchmark_external import evaluate_prediction

ROOT=Path('data/work/v8')


def object_regions(batch_labels):
    regions=[]
    for batch,y in enumerate(batch_labels):
        cuts=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
        for i in range(1,len(cuts)-2):
            a,b=int(cuts[i]),int(cuts[i+1]);label=int(y[a])
            if label not in (0,1) or y[cuts[i-1]]<0 or y[cuts[i+1]]<0:continue
            regions.append([batch,label,a,b,int(cuts[i-1]),int(cuts[i+2])])
    return np.asarray(regions,dtype='int64').reshape(-1,6)


def soft_object_iou(logits,regions):
    if len(regions)==0:return logits.sum()*0
    idx=torch.as_tensor(regions,dtype=torch.long,device=logits.device)
    batch,label,a,b,lo,hi=idx.unbind(1)
    cumulative=F.pad(logits.softmax(1).cumsum(2),(1,0))
    inside=cumulative[batch,label,b]-cumulative[batch,label,a]
    local=cumulative[batch,label,hi]-cumulative[batch,label,lo]
    union=(b-a).to(logits.dtype)+local-inside
    return (1-inside/union.clamp_min(1e-6)).mean()


def inverse_segment_weights(y):
    cuts=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
    sizes=np.diff(cuts);typical=max(1.,float(np.median(sizes)))
    out=np.repeat(np.clip(typical/np.maximum(sizes,1),.1,20),sizes).astype('float32')
    out[y<0]=0
    return out


def training():
    xx=[];yy=[]
    for game in v6_data.protocol()['native_train']:
        for mode in ('clean','moderate','stress'):
            r=joblib.load(PHYSICS/f'train_{game}_{mode}.joblib')
            for key in ('direct','mirror'):
                for a,b in r['slices']:
                    y=r['labels'][a:b]
                    xx.append(r[key][a:b]);yy.append(y if key=='direct' else np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y))
    return xx,yy


def train(xs,ys,kind,seed=2027,steps=800,device='mps'):
    torch.manual_seed(seed);torch.set_num_threads(6);rng=np.random.default_rng(seed)
    norm=Normalizer().fit(xs);net=SegmentNet(len(norm.mean)*2,local=True).to(device)
    tx=[torch.from_numpy(norm.transform(x).T.copy()).to(device) for x in xs]
    ty=[torch.tensor(y,dtype=torch.long,device=device) for y in ys]
    sw=[torch.from_numpy(inverse_segment_weights(y) if kind=='inverse' else segment_weights(y)).to(device) for y in ys]
    edges=[torch.from_numpy(boundary_targets(y)).to(device) for y in ys]
    count=np.bincount(np.concatenate([y[y>=0] for y in ys]),minlength=3)
    cw=torch.tensor(np.sqrt(count.sum()/np.maximum(count,1)),dtype=torch.float32,device=device);cw/=cw.mean()
    changes=[np.flatnonzero(y[1:]!=y[:-1])+1 for y in ys]
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.001)
    net.train();window=384
    for step in range(steps):
        chunks=[]
        for _ in range(12):
            i=int(rng.integers(len(xs)))
            start=(int(np.clip(rng.choice(changes[i])-rng.integers(window),0,len(xs[i])-window))
                   if len(changes[i]) and rng.random()<.5 else int(rng.integers(len(xs[i])-window+1)))
            chunks.append((i,slice(start,start+window)))
        x=torch.stack([tx[i][:,s] for i,s in chunks]);y=torch.stack([ty[i][s] for i,s in chunks])
        weight=torch.stack([sw[i][s] for i,s in chunks]);edge=torch.stack([edges[i][s] for i,s in chunks])
        logits,be,_=net(x);valid=y>=0
        ce=F.cross_entropy(logits,y,weight=cw,ignore_index=-1,reduction='none')
        if kind=='focal_iou':
            confidence=logits.softmax(1).gather(1,y.clamp_min(0)[:,None]).squeeze(1)
            ce=ce*(1-confidence).pow(1.5)
        loss=(ce*weight*valid).sum()/(weight*valid).sum().clamp_min(1)
        lp=logits.log_softmax(1);same=valid[:,1:]&valid[:,:-1]&(y[:,1:]==y[:,:-1])
        smooth=((lp[:,:,1:]-lp[:,:,:-1].detach())**2).clamp(max=16).mean(1)
        loss=loss+.15*(smooth*same).sum()/same.sum().clamp_min(1)
        bce=F.binary_cross_entropy_with_logits(be,edge,reduction='none',pos_weight=be.new_tensor(8.))
        loss=loss+.2*(bce*valid).sum()/valid.sum().clamp_min(1)
        if kind in ('iou','focal_iou'):
            regions=object_regions([ys[i][s] for i,s in chunks])
            loss=loss+soft_object_iou(logits,regions)
        if not torch.isfinite(loss):raise ValueError('Non-finite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
        if step%200==0 or step==steps-1:print(kind,seed,step,float(loss.detach().cpu()),flush=True)
    return net.eval(),norm


def evaluate(net,norm,root,device='mps',feature_root=None):
    rows=[];cfg=load_config(Path('possession/config_external_selected.json'))
    feature_root=PHYSICS if feature_root is None else feature_root
    for game in v6_data.protocol()['native_development']:
        for mode in ('clean','moderate','stress'):
            r=joblib.load(feature_root/f'development_{game}_{mode}.joblib')
            q,e=predict(net,norm,subsequences(r['direct'],r['slices']),device)
            qm,em=predict(net,norm,subsequences(r['mirror'],r['slices']),device)
            p=(np.concatenate(q)+np.concatenate(qm)[:,[1,0,2]])/2
            edge=(np.concatenate(e)+np.concatenate(em))/2
            np.savez_compressed(root/f'{game}_{mode}.npz',probabilities=p,edges=edge,frames=r['frames'])
            a=np.load(v6_data.ROOT/'tcn_boundary'/f'{game}_{mode}.npz')
            b=np.load(v6_data.ROOT/'physics_tcn'/f'{game}_{mode}.npz')
            native=(a['probabilities']+b['probabilities'])/2;average=(native+p)/2
            cases={'own_fixed':(p,None),'own_boundary':(p,edge),'own_original_edge':(p,a['edges']),
                   'average_own_edge':(average,edge),'average_original_edge':(average,a['edges']),
                   'average_both_edges':(average,(edge+a['edges'])/2)}
            for name,(prob,boundary) in cases.items():
                codes=(decode_blocks(prob,r['frames'],as_match(r),cfg['decoder']) if boundary is None else
                       dynamic_decode(prob,boundary,r['frames'],as_match(r)))
                metric=evaluate_prediction(r['full_labels'],codes)
                rows.append(dict(game=game,mode=mode,configuration=name,**metric))
            print(root.name,game,mode,{r['configuration']:round(r['spell_f1_mean'],4) for r in rows[-len(cases):]},flush=True)
    summary={name:dict(spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in rows if r['configuration']==name])),
                  worst_spell_f1=float(min(r['spell_f1_mean'] for r in rows if r['configuration']==name)),
                  macro_f1=float(np.mean([r['macro_f1'] for r in rows if r['configuration']==name])))
             for name in cases}
    (root/'results.json').write_text(json.dumps(dict(rows=rows,summary=summary,challenge_labels_read=False),indent=2))
    print('SUMMARY',root.name,summary,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['inverse','iou','focal_iou'])
    parser.add_argument('--seed',type=int,default=2027);parser.add_argument('--steps',type=int,default=800)
    args=parser.parse_args();root=ROOT/f'{args.kind}_{args.seed}';root.mkdir(parents=True,exist_ok=True)
    if (root/'model.pt').exists():raise ValueError('Preserve previous training; choose another run')
    xs,ys=training();net,norm=train(xs,ys,args.kind,args.seed,args.steps)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=norm.mean,std=norm.std,
                    kind=args.kind,seed=args.seed,steps=args.steps,physics=True),root/'model.pt')
    evaluate(net,norm,root)


if __name__=='__main__':main()
