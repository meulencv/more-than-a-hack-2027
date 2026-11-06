"""Typed-control transition expert: external training, selection and confirmation."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import find_peaks
from possession import v13_cv
from possession.v6_temporal import SegmentNet,segment_weights
from possession.sequence_tcn import Normalizer
from possession.v6_experiment import as_match
from possession.v14_resolution import decode_fine
from possession.benchmark_external import evaluate_prediction

ROOT=Path('data/work/v18');ROOT.mkdir(parents=True,exist_ok=True)
PAIRS=[(a,b) for a in range(3) for b in range(3) if a!=b]
KINDS=['baseline','typed1','typed2','typed4','typed2class10','class25']

class TypedNet(nn.Module):
    def __init__(self,features):
        super().__init__();self.base=SegmentNet(features);self.event=nn.Conv1d(64,7,1)
    def forward(self,x):
        h=self.base.blocks(self.base.input(x));return self.base.classifier(h),self.event(h)

def event_targets(y):
    t=np.zeros(len(y),dtype='int64');valid=(y[1:]>=0)&(y[:-1]>=0)
    for i,(a,b) in enumerate(PAIRS,1):t[1:][valid&(y[:-1]==a)&(y[1:]==b)]=i
    t[y<0]=-1
    return t

def fit(fold,device):
    games=[g for g in v13_cv.settings()['games'] if g not in v13_cv.settings()['test_folds'][fold]]
    path=ROOT/f'fold{fold}.pt';net=TypedNet(312).to(device);norm=Normalizer()
    if path.exists():
        s=torch.load(path,map_location='cpu',weights_only=False)
        if s['train_games']!=games:raise ValueError('Training partition mismatch')
        norm.mean=s['mean'];norm.std=s['std'];net.load_state_dict(s['state']);return net.eval(),norm
    torch.manual_seed(2030);np_rng=np.random.default_rng(2030)
    net=TypedNet(312).to(device);xs,ys=v13_cv.training_rows(games,156);norm.fit(xs)
    xx=[torch.from_numpy(norm.transform(x).T.copy()).to(device) for x in xs]
    yy=[torch.tensor(y.astype('int64'),device=device) for y in ys]
    et=[event_targets(y) for y in ys];events=[torch.tensor(y,device=device) for y in et]
    weights=[torch.from_numpy(segment_weights(y)).to(device) for y in ys]
    centers=[np.flatnonzero(y>0) for y in et]
    count=np.bincount(np.concatenate([y[y>=0] for y in ys]),minlength=3);cw=np.sqrt(count.sum()/np.maximum(count,1));cw/=cw.mean();cw=torch.tensor(cw,dtype=torch.float32,device=device)
    ec=np.bincount(np.concatenate([y[y>=0] for y in et]),minlength=7);ew=np.sqrt(ec.sum()/np.maximum(ec,1));ew/=ew.mean();ew=torch.tensor(ew,dtype=torch.float32,device=device)
    opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.001);net.train();window=384
    for step in range(1200):
        chunks=[]
        for _ in range(12):
            i=int(np_rng.integers(len(xs)))
            start=int(np.clip(np_rng.choice(centers[i])-np_rng.integers(window),0,len(xs[i])-window)) if len(centers[i]) and np_rng.random()<.75 else int(np_rng.integers(len(xs[i])-window+1))
            chunks.append((i,slice(start,start+window)))
        inp=torch.stack([xx[i][:,s] for i,s in chunks]);target=torch.stack([yy[i][s] for i,s in chunks]);event=torch.stack([events[i][s] for i,s in chunks]);sw=torch.stack([weights[i][s] for i,s in chunks])
        opt.zero_grad(set_to_none=True);c,e=net(inp)
        ce=F.cross_entropy(c,target,weight=cw,ignore_index=-1,reduction='none');cl=(ce*sw).sum()/sw.sum().clamp_min(1)
        el=F.cross_entropy(e,event,weight=ew,ignore_index=-1)
        # No cross-boundary smoothing: brief changes remain distinct training targets.
        loss=cl+.5*el
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training')
        loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if step%200==0 or step==1199:print('TRAIN',fold,step,float(loss.detach().cpu()),flush=True)
    torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=norm.mean,std=norm.std,train_games=games,steps=1200,seed=2030),path)
    return net.eval(),norm

def predict(net,norm,r,device):
    pp=[];ee=[]
    with torch.inference_mode():
        for a,b in r['slices']:
            inputs=torch.stack([torch.from_numpy(norm.transform(r[k][a:b]).T.copy()) for k in ['direct','mirror']]).to(device)
            c,e=net(inputs);p=c.softmax(1).cpu().numpy().transpose(0,2,1);ev=e.softmax(1).cpu().numpy().transpose(0,2,1)
            mapping=[0]+[1+PAIRS.index((1-x if x<2 else 2,1-y if y<2 else 2)) for x,y in PAIRS]
            pp.append((p[0]+p[1][:,[1,0,2]])/2);ee.append((ev[0]+ev[1][:,mapping])/2)
    return np.concatenate(pp),np.concatenate(ee)

def decode(p,edges,typed,f,match,strength):
    out=np.full(len(match.ball),3,dtype='int8');transition=np.array([[0.,4.,2.5],[4.,0.,2.5],[2.5,2.5,0.]])
    for _,a,b in match.blocks:
        use=(f>=a)&(f<=b);ff=f[use];grid=np.arange(a,b+1);pp=p[use];ev=typed[use]
        emission=np.log(np.maximum(np.column_stack([np.interp(grid,ff,pp[:,i]) for i in range(3)]),1e-5))/5
        edge=np.interp(grid,ff,edges[use]);bonuses=np.zeros((len(grid),3,3),dtype='float32')
        for j,(src,dst) in enumerate(PAIRS,1):
            peaks=find_peaks(ev[:,j],height=.15,distance=2)[0]
            for ix in peaks:bonuses[int(ff[ix]-a),src,dst]=strength*float(ev[ix,j])
        back=np.zeros((len(grid),3),dtype='int8');score=emission[0].copy()
        for t in range(1,len(grid)):
            value=score[:,None]-transition*(1-.85*edge[t])+bonuses[t]
            back[t]=value.argmax(0);score=value[back[t],np.arange(3)]+emission[t];score-=score.max()
        states=np.empty(len(grid),dtype='int8');current=int(score.argmax())
        for t in range(len(grid)-1,-1,-1):states[t]=current;current=int(back[t,current])
        out[a:b+1]=states
    return out

def evaluate(fold,kinds,device):
    net,norm=fit(fold,device);rows=[]
    for game in v13_cv.settings()['test_folds'][fold]:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode);old=np.load(v13_cv.ROOT/f'fold{fold}'/f'{game}_{mode}.npz');path=ROOT/f'{game}_{mode}.npz'
            if path.exists():s=np.load(path);p,e=s['p'],s['e']
            else:p,e=predict(net,norm,r,device);np.savez_compressed(path,p=p,e=e,frames=r['frames'])
            for kind in kinds:
                prob=.75*old['combined']+.25*p if kind=='class25' else .9*old['combined']+.1*p if kind=='typed2class10' else old['combined']
                if kind in ['baseline','class25']:codes=decode_fine(prob,old['edges'],r['frames'],as_match(r))
                else:codes=decode(prob,old['edges'],e,r['frames'],as_match(r),2 if kind=='typed2class10' else float(kind[5:]))
                metric=evaluate_prediction(r['full_labels'],codes);rows.append(dict(game=game,mode=mode,kind=kind,**metric));print('RESULT',fold,game,mode,kind,round(metric['spell_f1_mean'],5),flush=True)
    del net
    if device=='mps':torch.mps.empty_cache()
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='mps');args=parser.parse_args();torch.set_num_threads(6)
    protocol=dict(purpose='Detect directed control changes rather than one untyped boundary, motivated by brief merged native possessions.',
        train='156 input features, local TCN with 3 state and 7 event heads (none plus 6 directed changes); 1200 updates, seed2030; 75% boundary-centered crops; segment-weighted state CE plus .5 inverse-sqrt-count event CE.',
        candidates=KINDS,decoder='Only event peaks >.15, >=.4s apart per type, create instantaneous Viterbi transition bonuses of 1/2/4 times type probability. No frame-specific rules.',
        selection_fold=1,confirmation_fold=2,selection='Highest selection spellF1 with macro>=baseline-.002; freeze then evaluate selected only on fold2; adopt if both game means improve and macro>=baseline-.002.',
        limitation='Historically used external games; new-round held-game confirmation, not untouched final test.',challenge_labels_read=False)
    path=Path('docs/PROTOCOLO_V18.json')
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol mismatch')
    path.write_text(json.dumps(protocol,indent=2));rows=evaluate(1,KINDS,args.device)
    means={k:{m:float(np.mean([r[m] for r in rows if r['kind']==k])) for m in ['spell_f1_mean','macro_f1']} for k in KINDS}
    allowed=[k for k in means if means[k]['macro_f1']>=means['baseline']['macro_f1']-.002];chosen=max(allowed,key=lambda k:means[k]['spell_f1_mean'])
    result=dict(selection_rows=rows,selection_means=means,chosen=chosen,confirmation_rows=[],challenge_labels_read=False)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2));print('CHOSEN',chosen,means,flush=True)
    if chosen!='baseline':result['confirmation_rows']=evaluate(2,['baseline',chosen],args.device)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
