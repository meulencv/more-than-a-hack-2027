"""Residual sequence correction trained on cross-half base predictions."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from possession import v13_cv,v21_correction
from possession.v6_temporal import SegmentNet,segment_weights
from possession.sequence_tcn import Normalizer
from possession.v6_experiment import as_match
from possession.v14_resolution import decode_fine
from possession.benchmark_external import evaluate_prediction

ROOT=Path('data/work/v22');ROOT.mkdir(parents=True,exist_ok=True)
KINDS=['baseline','sequence25','sequence50','sequence100']

class Refiner(nn.Module):
    def __init__(self,width):
        super().__init__();self.core=SegmentNet(width,width=48);self.core.classifier.weight.data.zero_();self.core.classifier.bias.data.zero_()
    def forward(self,x):return self.core(x)[0]

def fit(fold,device):
    games=[g for g in v13_cv.settings()['games'] if g not in v13_cv.settings()['test_folds'][fold]]
    path=ROOT/f'fold{fold}.pt';net=Refiner(322).to(device);norm=Normalizer()
    if path.exists():
        s=torch.load(path,map_location='cpu',weights_only=False)
        if s['games']!=games:raise ValueError('Wrong training partition')
        net.load_state_dict(s['state']);norm.mean=s['mean'];norm.std=s['std'];return net.eval(),norm
    v21_correction.inner_predictions(fold,device);xx=[];yy=[];pp=[]
    for game in games:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode)
            for half,(a,b) in enumerate(r['slices']):
                s=np.load(v21_correction.ROOT/f'fold{fold}'/f'{game}_{mode}_half{half}.npz')
                if not np.array_equal(s['frames'],r['frames'][a:b]):raise ValueError('Timeline mismatch')
                for key in ['direct','mirror']:
                    p=s['p'] if key=='direct' else s['p'][:,[1,0,2]];y=r['labels'][a:b]
                    xx.append(v21_correction.observation(r[key][a:b],p,s['e']));pp.append(p)
                    yy.append(y if key=='direct' else np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y))
    norm.fit(xx);torch.manual_seed(2033);rng=np.random.default_rng(2033);net=Refiner(322).to(device)
    inputs=[torch.from_numpy(norm.transform(x).T.copy()).to(device) for x in xx]
    labels=[torch.tensor(y.astype('int64'),device=device) for y in yy]
    base=[torch.from_numpy(np.log(np.maximum(p,1e-5)).T.copy()).to(device) for p in pp]
    weight=[torch.from_numpy(segment_weights(y)).to(device) for y in yy]
    errors=[np.flatnonzero((p.argmax(1)!=y)&(y>=0)) for p,y in zip(pp,yy)]
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001);net.train();window=384
    for step in range(1200):
        batch=[]
        for _ in range(12):
            i=int(rng.integers(len(inputs)));n=len(yy[i])
            start=int(np.clip(rng.choice(errors[i])-rng.integers(window),0,n-window)) if len(errors[i]) and rng.random()<.75 else int(rng.integers(n-window+1))
            batch.append((i,slice(start,start+window)))
        x=torch.stack([inputs[i][:,s] for i,s in batch]);y=torch.stack([labels[i][s] for i,s in batch]);p=torch.stack([base[i][:,s] for i,s in batch]);w=torch.stack([weight[i][s] for i,s in batch])
        opt.zero_grad(set_to_none=True);delta=net(x);out=p+delta
        ce=F.cross_entropy(out,y,ignore_index=-1,reduction='none');ce=(ce*w).sum()/w.sum().clamp_min(1)
        valid=(y[:,:-1]>=0)&(y[:,1:]>=0)&(y[:,:-1]==y[:,1:]);lp=out.log_softmax(1)
        smooth=(lp[:,:,1:]-lp[:,:,:-1]).square().clamp(max=16).mean(1)
        loss=ce+.05*(smooth*valid).sum()/valid.sum().clamp_min(1)+.01*delta.square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training')
        loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if step%200==0 or step==1199:print('TRAIN',fold,step,float(loss.detach().cpu()),flush=True)
    torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=norm.mean,std=norm.std,games=games,seed=2033,steps=1200),path)
    return net.eval(),norm

def residual(net,norm,r,p,e,device):
    out=[]
    with torch.inference_mode():
        for a,b in r['slices']:
            values=[]
            for key in ['direct','mirror']:
                pp=p[a:b] if key=='direct' else p[a:b][:,[1,0,2]]
                obs=v21_correction.observation(r[key][a:b],pp,e[a:b]);tensor=torch.from_numpy(norm.transform(obs).T.copy()).to(device)[None]
                d=net(tensor)[0].T.cpu().numpy();values.append(d if key=='direct' else d[:,[1,0,2]])
            out.append((values[0]+values[1])/2)
    return np.concatenate(out)

def evaluate(fold,kinds,device):
    net,norm=fit(fold,device);rows=[]
    for game in v13_cv.settings()['test_folds'][fold]:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode);s=np.load(v13_cv.ROOT/f'fold{fold}'/f'{game}_{mode}.npz');p=s['combined'];e=s['edges'];d=residual(net,norm,r,p,e,device)
            np.savez_compressed(ROOT/f'{game}_{mode}.npz',delta=d,frames=r['frames'])
            for kind in kinds:
                w=0 if kind=='baseline' else float(kind[8:])/100
                logits=np.log(np.maximum(p,1e-5))+w*d;logits-=logits.max(1,keepdims=True);q=np.exp(logits);q/=q.sum(1,keepdims=True)
                pred=decode_fine(q,e,r['frames'],as_match(r));metric=evaluate_prediction(r['full_labels'],pred)
                rows.append(dict(game=game,mode=mode,kind=kind,**metric));print('RESULT',fold,game,mode,kind,round(metric['spell_f1_mean'],5),flush=True)
    del net
    if device=='mps':torch.mps.empty_cache()
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='mps');args=parser.parse_args();torch.set_num_threads(6)
    protocol=dict(purpose='Test a temporal residual corrector after the snapshot corrector improved frames but worsened possession objects.',
        training='V21 cross-half predictions on outer training matches only; input161, local width48 TCN, zero-initialized residual head; final logits=log(base p)+delta. 1200 updates seed2033, 75% error-centered crops, segment-weighted CE, .05 same-label smoothness, .01 residual L2.',
        candidates=KINDS,selection_fold=1,confirmation_fold=2,selection='Highest selection spellF1 with macro>=baseline-.002; freeze then evaluate selected only on fold2; adopt if both game means improve and macro>=baseline-.002.',
        limitation='Inner cross-half versus outer cross-match probability mismatch; historically seen development games, repeated selection not an unbiased final test.',challenge_labels_read=False)
    path=Path('docs/PROTOCOLO_V22.json')
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol mismatch')
    path.write_text(json.dumps(protocol,indent=2));rows=evaluate(1,KINDS,args.device)
    means={k:{m:float(np.mean([r[m] for r in rows if r['kind']==k])) for m in ['spell_f1_mean','macro_f1']} for k in KINDS}
    allowed=[k for k in means if means[k]['macro_f1']>=means['baseline']['macro_f1']-.002];chosen=max(allowed,key=lambda k:means[k]['spell_f1_mean'])
    result=dict(selection_rows=rows,selection_means=means,chosen=chosen,confirmation_rows=[],challenge_labels_read=False)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2));print('CHOSEN',chosen,means,flush=True)
    if chosen!='baseline':result['confirmation_rows']=evaluate(2,['baseline',chosen],args.device)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
