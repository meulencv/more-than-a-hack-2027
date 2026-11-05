"""Partial-label broadcast transfer, with a frozen native dead-ball expert.

No challenge labels are imported. Broadcast unknowns are excluded from the binary
loss. The learned H/A distribution cannot directly change P(D).
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from . import v6_data
from .features_v3 import feature_table
from .skillcorner import load_match
from .sequence_tcn import Normalizer
from .v6_temporal import SegmentNet
from .v6_experiment import subsequences,as_match,dynamic_decode
from .benchmark_external import evaluate_prediction

ROOT=Path('data/work/v7')


def protocol():return json.loads(Path('docs/PROTOCOLO_V7.json').read_text())


def signature():
    paths=[Path(__file__),Path('possession/skillcorner.py'),Path('possession/features_v3.py'),Path('docs/PROTOCOLO_V7.json')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare():
    (ROOT/'features').mkdir(parents=True,exist_ok=True)
    for role in ('train','development'):
        for game in protocol()['skillcorner_'+role]:
            for detected in (False,True):
                path=ROOT/'features'/f'{role}_{game}_{int(detected)}.joblib'
                if path.exists() and joblib.load(path).get('signature')==signature():continue
                match=load_match(game,detected)
                a,frames=feature_table(match);b,fm=feature_table(match,swap=True)
                assert np.array_equal(frames,fm)
                sizes=[(z-a)//5+1 for _,a,z in match.blocks];cuts=np.r_[0,np.cumsum(sizes)]
                r=dict(signature=signature(),direct=a.to_numpy(),mirror=b.to_numpy(),frames=frames,
                       labels=match.meta['labels'][frames],full_labels=match.meta['labels'],
                       slices=list(zip(cuts[:-1],cuts[1:])),blocks=match.blocks,meta=match.meta)
                joblib.dump(r,path,compress=1)
                known=r['labels']>=0
                print('PREPARED',role,game,'detected',detected,'known',round(float(known.mean()),3),
                      'groups',match.meta['raw_group_counts'],flush=True)


def record(role,game,detected):
    r=joblib.load(ROOT/'features'/f'{role}_{game}_{int(detected)}.joblib')
    if r['signature']!=signature():raise ValueError('Feature provenance changed')
    return r


def training(kind):
    xx=[];yy=[]
    for game in protocol()['skillcorner_train']:
        for detected in ([False] if kind=='all' else [True] if kind=='detected' else [False,True]):
            r=record('train',game,detected)
            for key in ('direct','mirror'):
                for a,b in r['slices']:
                    y=r['labels'][a:b]
                    xx.append(r[key][a:b]);yy.append(y if key=='direct' else np.where(y>=0,1-y,-1))
    if kind=='mixed':
        native_x,native_y=v6_data.training_sequences()
        xx.extend(native_x);yy.extend([np.where(y<2,y,-1) for y in native_y])
    return xx,yy


def fit_binary(xs,ys,steps=800,seed=2027,device='mps'):
    torch.manual_seed(seed);torch.set_num_threads(6);rng=np.random.default_rng(seed)
    norm=Normalizer().fit(xs)
    net=SegmentNet(len(norm.mean)*2,local=True).to(device)
    tx=[torch.from_numpy(norm.transform(x).T.copy()).to(device) for x in xs]
    ty=[torch.tensor(y,dtype=torch.long,device=device) for y in ys]
    changes=[np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1 for y in ys]
    opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.001)
    net.train();window=384
    for step in range(steps):
        chunks=[]
        for _ in range(12):
            i=int(rng.integers(len(xs)))
            start=(int(np.clip(rng.choice(changes[i])-rng.integers(window),0,len(xs[i])-window))
                   if len(changes[i]) and rng.random()<.5 else int(rng.integers(len(xs[i])-window+1)))
            chunks.append((i,slice(start,start+window)))
        inputs=torch.stack([tx[i][:,s] for i,s in chunks]);y=torch.stack([ty[i][s] for i,s in chunks])
        logits=net(inputs)[0][:,:2]
        ce=F.cross_entropy(logits,y,ignore_index=-1,reduction='none')
        valid=y>=0;loss=(ce*valid).sum()/valid.sum().clamp_min(1)
        same=valid[:,1:]&valid[:,:-1]&(y[:,1:]==y[:,:-1])
        lp=logits.log_softmax(1);smooth=((lp[:,:,1:]-lp[:,:,:-1].detach())**2).clamp(max=16).mean(1)
        loss=loss+.15*(smooth*same).sum()/same.sum().clamp_min(1)
        if not torch.isfinite(loss):raise ValueError('Non-finite loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if step%200==0 or step==steps-1:print('BINARY',step,float(loss.detach().cpu()),flush=True)
    return net.eval(),norm


def probabilities(net,norm,r,device):
    def infer(key):
        output=[]
        with torch.inference_mode():
            for x in subsequences(r[key],r['slices']):
                t=torch.from_numpy(norm.transform(x).T.copy()).to(device)[None]
                output.append(net(t)[0][:,:2].softmax(1)[0].T.cpu().numpy())
        return np.concatenate(output)
    return (infer('direct')+infer('mirror')[:,[1,0]])/2


def fuse(native,binary,weight):
    """Reallocate only H/A mass; keep frozen expert's D probability exactly."""
    mass=native[:,:2].sum(1,keepdims=True)
    conditional=native[:,:2]/np.maximum(mass,1e-8)
    output=native.copy()
    output[:,:2]=mass*((1-weight)*conditional+weight*binary)
    return output


def load_model(path,device='mps'):
    c=torch.load(path,map_location='cpu',weights_only=False);norm=Normalizer()
    norm.mean=np.asarray(c['mean'],dtype='float32');norm.std=np.asarray(c['std'],dtype='float32')
    net=SegmentNet(len(norm.mean)*2,local=True);net.load_state_dict(c['state_dict'])
    return net.to(device).eval(),norm


def evaluate(kind,net,norm,device='mps'):
    root=ROOT/kind;rows=[];teacher=[]
    for game in protocol()['native_development']:
        for mode in ('clean','moderate','stress'):
            r=v6_data.record(game,mode,'development')
            x=np.load(v6_data.ROOT/'tcn_boundary'/f'{game}_{mode}.npz')
            z=np.load(v6_data.ROOT/'physics_tcn'/f'{game}_{mode}.npz')
            native=(x['probabilities']+z['probabilities'])/2
            p=probabilities(net,norm,r,device)
            np.savez_compressed(root/f'{game}_{mode}.npz',binary=p,frames=r['frames'])
            for w in (.25,.5,1.):
                codes=dynamic_decode(fuse(native,p,w),x['edges'],r['frames'],as_match(r))
                metric=evaluate_prediction(r['full_labels'],codes)
                rows.append(dict(game=game,mode=mode,weight=w,**metric))
                print(kind,game,mode,w,round(metric['spell_f1_mean'],4),flush=True)
    for game in protocol()['skillcorner_development']:
        for detected in (False,True):
            r=record('development',game,detected);p=probabilities(net,norm,r,device)
            known=r['labels']>=0
            teacher.append(dict(game=game,detected=detected,coverage=float(known.mean()),
                  conditional_accuracy=float((p.argmax(1)[known]==r['labels'][known]).mean()),
                  teacher_only=True))
    summary={str(w):dict(spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in rows if r['weight']==w])),
               worst_spell_f1=float(min(r['spell_f1_mean'] for r in rows if r['weight']==w)),
               macro_f1=float(np.mean([r['macro_f1'] for r in rows if r['weight']==w]))) for w in (.25,.5,1.)}
    result=dict(kind=kind,summary=summary,native_rows=rows,teacher_rows=teacher,challenge_labels_read=False,
                confirmation_loaded=False)
    (root/'results.json').write_text(json.dumps(result,indent=2));print('SUMMARY',kind,summary,'TEACHER',teacher,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--kind',choices=['all','detected','mixed']);parser.add_argument('--steps',type=int,default=800)
    parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    if args.kind:
        root=ROOT/args.kind;root.mkdir(parents=True,exist_ok=True)
        if args.evaluate_only:net,norm=load_model(root/'model.pt')
        else:
            xs,ys=training(args.kind);print('FITTING',args.kind,len(xs),'sequences',flush=True)
            net,norm=fit_binary(xs,ys,args.steps)
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=norm.mean,std=norm.std,
                       steps=args.steps,seed=2027,kind=args.kind),root/'model.pt')
        evaluate(args.kind,net,norm)


if __name__=='__main__':main()
