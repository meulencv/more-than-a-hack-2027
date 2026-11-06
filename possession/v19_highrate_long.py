"""Follow-up: give the 25Hz expert more updates, keeping the held games excluded."""
import argparse,json,math
from pathlib import Path
import joblib,numpy as np,torch
from possession import v13_cv,v17_highrate
from possession.v6_temporal import objective,segment_weights,boundary_targets

ROOT=Path('data/work/v19');ROOT.mkdir(parents=True,exist_ok=True)

def fit(fold,device):
    games=[g for g in v13_cv.settings()['games'] if g not in v13_cv.settings()['test_folds'][fold]]
    path=ROOT/f'fold{fold}.pt'
    if path.exists():
        s=torch.load(path,map_location='cpu',weights_only=False)
        if s['train_games']!=games:raise ValueError('Training partition mismatch')
        return v13_cv.load_temporal(path,device)
    net,norm=v17_highrate.fit(fold,device);torch.manual_seed(2031);rng=np.random.default_rng(2031)
    xx=[];yy=[];ss=[];bb=[];centers=[]
    for game in games:
        for mode in ['clean','moderate','stress']:
            r=joblib.load(v17_highrate.ROOT/f'{game}_{mode}.joblib')
            for key in ['direct','mirror']:
                for a,b in r['slices']:
                    y=r['labels'][a:b]
                    if key=='mirror':y=np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y)
                    xx.append(torch.from_numpy(norm.transform(r[key][a:b]).T.copy()).to(device))
                    yy.append(torch.tensor(y.astype('int64'),device=device));ss.append(torch.from_numpy(segment_weights(y)).to(device));bb.append(torch.from_numpy(boundary_targets(y)).to(device))
                    centers.append(np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1)
    all_y=np.concatenate([y.cpu().numpy() for y in yy]);counts=np.bincount(all_y[all_y>=0],minlength=3)
    cw=np.sqrt(counts.sum()/np.maximum(counts,1));cw/=cw.mean();cw=torch.tensor(cw,dtype=torch.float32,device=device)
    opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.001);net.train();steps=3200;window=768
    for step in range(steps):
        chunks=[]
        for _ in range(8):
            i=int(rng.integers(len(xx)));n=xx[i].shape[1]
            start=int(np.clip(rng.choice(centers[i])-rng.integers(window),0,n-window)) if len(centers[i]) and rng.random()<.75 else int(rng.integers(n-window+1))
            chunks.append((i,slice(start,start+window)))
        for group in opt.param_groups:group['lr']=.00005+.00065*(1+math.cos(math.pi*step/steps))/2
        inputs=torch.stack([xx[i][:,s] for i,s in chunks]);labels=torch.stack([yy[i][s] for i,s in chunks]);weights=torch.stack([ss[i][s] for i,s in chunks]);bound=torch.stack([bb[i][s] for i,s in chunks])
        opt.zero_grad(set_to_none=True);loss=objective(net(inputs),labels,cw,weights,bound,use_boundary=True)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if step%400==0 or step==steps-1:print('LONG',fold,step,float(loss.detach().cpu()),flush=True)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=norm.mean,std=norm.std,train_games=games,rate=25,steps=4000,initial_seed=2029,continuation_seed=2031),path)
    return net.eval(),norm

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='mps');args=parser.parse_args()
    protocol=dict(purpose='Test whether V17 highrate failed because identical update count exposed it to five times less match time.',
        training='Continue 800-step highrate checkpoint for 3200 updates, fixed normalizer, batch8/window768, 75% boundary crops, cosine .0007 to .00005, seed2031; no held-game fitting.',
        candidates=v17_highrate.KINDS,selection_fold=1,confirmation_fold=2,selection='Highest selection spellF1 with macro>=baseline-.002; freeze then evaluate selected only on fold2; adopt if both game means improve and macro>=baseline-.002.',
        limitation='Historically used games; repeated selection data, not an unbiased final estimate.',challenge_labels_read=False)
    path=Path('docs/PROTOCOLO_V19.json')
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol mismatch')
    path.write_text(json.dumps(protocol,indent=2))
    rows=v17_highrate.evaluate(1,v17_highrate.KINDS,args.device,model_fitter=fit,output_root=ROOT)
    means={k:{m:float(np.mean([r[m] for r in rows if r['kind']==k])) for m in ['spell_f1_mean','macro_f1']} for k in v17_highrate.KINDS}
    allowed=[k for k in means if means[k]['macro_f1']>=means['baseline']['macro_f1']-.002];chosen=max(allowed,key=lambda k:means[k]['spell_f1_mean'])
    result=dict(selection_rows=rows,selection_means=means,chosen=chosen,confirmation_rows=[],challenge_labels_read=False)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2));print('CHOSEN',chosen,means,flush=True)
    if chosen!='baseline':result['confirmation_rows']=v17_highrate.evaluate(2,['baseline',chosen],args.device,model_fitter=fit,output_root=ROOT)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
