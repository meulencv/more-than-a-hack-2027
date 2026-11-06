"""Invariant player-set models and label-free masked ball reconstruction.

Original implementation inspired by per-object TCNs, Ball Radar and PathCRF.
This is not a reproduction of their complete architectures or reported scores.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from . import v10_data as data
from . import v6_data
from .sequence_tcn import Normalizer
from .v6_temporal import ChannelNorm,Residual,SegmentNet,objective,segment_weights,boundary_targets
from .v6_experiment import as_match,dynamic_decode
from .v3 import decode_blocks,load_config
from .benchmark_external import evaluate_prediction


class PlayerSetNet(nn.Module):
    def __init__(self,kind,global_features=312):
        super().__init__();self.kind=kind
        self.embedding=nn.Sequential(nn.Linear(14,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.player_temporal=(nn.Sequential(*(Residual(32,2**i,True) for i in range(3))) if kind in ('trajectory','fused') else None)
        self.attention=nn.Linear(32,1)
        self.temporal=SegmentNet(192+7+(global_features if kind=='fused' else 0),width=64,local=True)
        self.reconstruction=nn.Conv1d(64,2,1)

    def forward(self,nodes,ball,team,global_x=None):
        batch,length,slots,_=nodes.shape
        h=self.embedding(nodes)
        if self.player_temporal is not None:
            h=self.player_temporal(h.permute(0,2,3,1).reshape(batch*slots,32,length))
            h=h.reshape(batch,slots,32,length).permute(0,3,1,2)
        scores=self.attention(h)[:,:,:,0];visible=nodes[:,:,:,10]>.5;pooled=[]
        for side in (0,1):
            mask=visible&(team[:,None,:]==side)
            count=mask.sum(2,keepdim=True).clamp_min(1)
            mean=(h*mask[:,:,:,None]).sum(2)/count
            attention=scores.masked_fill(~mask,-1e4).softmax(2)*mask
            attention=attention/attention.sum(2,keepdim=True).clamp_min(1e-6)
            attended=(h*attention[:,:,:,None]).sum(2)
            maximum=h.masked_fill(~mask[:,:,:,None],-1e4).max(2).values
            maximum=torch.where(mask.any(2,keepdim=True),maximum,0.)
            pooled.extend([mean,attended,maximum])
        values=torch.cat(pooled+[ball]+([global_x] if self.kind=='fused' else []),dim=2).transpose(1,2)
        hidden=self.temporal.blocks(self.temporal.input(values))
        logits=self.temporal.classifier(hidden);edge=self.temporal.boundary(hidden)[:,0]
        coordinates=self.reconstruction(hidden).transpose(1,2)
        return logits,edge,coordinates


def mirror(nodes,ball,team):
    nodes=nodes.copy();ball=ball.copy();team=team.copy()
    nodes[:,:,[0,1,2,3,5,6,8,9]]*=-1
    ball[:,:4]*=-1
    team=np.where(team>=0,1-team,-1)
    return nodes,ball,team


def hide_ball(nodes,ball,mask):
    """Remove every ball-derived node attribute, not only its coordinate token."""
    nodes=nodes.copy();ball=ball.copy()
    nodes[mask,:,5:10]=0;nodes[mask,:,12:14]=0;ball[mask]=0
    return nodes,ball


def sequences():
    output=[];normalizer=Normalizer();features=[]
    for game in v6_data.protocol()['native_train']:
        for mode in ('clean','moderate','stress'):
            r=data.record('train',game,mode)
            for a,b in r['slices']:
                output.append(dict(nodes=r['nodes'][a:b],ball=r['ball_tokens'][a:b],team=r['team'],
                                   y=r['labels'][a:b],direct=r['direct'][a:b],mirror=r['mirror'][a:b]))
                features.extend([r['direct'][a:b],r['mirror'][a:b]])
    normalizer.fit(features)
    for item in output:
        item['global_direct']=normalizer.transform(item.pop('direct'))
        item['global_mirror']=normalizer.transform(item.pop('mirror'))
        item['segments']=segment_weights(item['y']);item['edges']=boundary_targets(item['y'])
        item['changes']=np.flatnonzero(item['y'][1:]!=item['y'][:-1])+1
    return output,normalizer


def fit(kind,seed=2027,steps=1000,device='mps',cosine=False,augment=False):
    torch.set_num_threads(6);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    samples,norm=sequences();native_count=len(samples)
    if kind=='selfsup':
        r=data.record('unlabeled_challenge')
        for a,b in r['slices']:
            y=r['labels'][a:b]
            samples.append(dict(nodes=r['nodes'][a:b],ball=r['ball_tokens'][a:b],team=r['team'],y=y,
                                segments=np.zeros(len(y),dtype='float32'),edges=np.zeros(len(y),dtype='float32'),changes=[]))
    count=np.bincount(np.concatenate([s['y'][s['y']>=0] for s in samples[:native_count]]),minlength=3)
    cw=torch.tensor(np.sqrt(count.sum()/np.maximum(count,1)),dtype=torch.float32,device=device);cw/=cw.mean()
    net=PlayerSetNet(kind,len(norm.mean)*2).to(device);net.train()
    opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.0001) if cosine else None
    window=384;batch=8
    for step in range(steps):
        xx=[];bb=[];tt=[];yy=[];ww=[];ee=[];gg=[];targets=[];known=[]
        for j in range(batch):
            unlabeled=kind=='selfsup' and j>=6
            i=int(rng.integers(native_count,len(samples))) if unlabeled else int(rng.integers(native_count))
            s=samples[i];length=len(s['y'])
            start=(int(np.clip(rng.choice(s['changes'])-rng.integers(window),0,length-window))
                   if len(s['changes']) and rng.random()<.5 else int(rng.integers(length-window+1)))
            a=slice(start,start+window);nodes=s['nodes'][a].astype('float32');ball=s['ball'][a].copy();team=s['team'].copy();y=s['y'][a].copy()
            swap=rng.random()<.5
            if swap:
                nodes,ball,team=mirror(nodes,ball,team)
                y=np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y)
            if kind=='selfsup':
                target=ball[:,:2].copy();valid=ball[:,5]>.5
                mask=rng.random(window)<.15
                for _ in range(3):
                    first=int(rng.integers(window));mask[first:first+int(rng.integers(5,101))]=True
                nodes,ball=hide_ball(nodes,ball,mask)
                targets.append(target);known.append(valid&mask)
            if augment and not unlabeled and rng.random()<.5:
                mask=np.zeros(window,dtype=bool)
                for _ in range(2):
                    first=int(rng.integers(window));mask[first:first+int(rng.integers(1,31))]=True
                nodes,ball=hide_ball(nodes,ball,mask)
                missing=rng.random(len(team))<.08
                nodes[:,missing]=0
            xx.append(nodes);bb.append(ball);tt.append(team);yy.append(y)
            ww.append(s['segments'][a]);ee.append(s['edges'][a])
            if kind=='fused':gg.append(s['global_mirror' if swap else 'global_direct'][a])
        def tensor(values,dtype=torch.float32):return torch.as_tensor(np.stack(values),dtype=dtype,device=device)
        logits,edge,coords=net(tensor(xx),tensor(bb),tensor(tt,torch.long),tensor(gg) if gg else None)
        label=tensor(yy,torch.long);loss=objective((logits,edge,logits),label,cw,tensor(ww),tensor(ee),True)
        if kind=='selfsup':
            mask=tensor(known);reconstruction=F.smooth_l1_loss(coords,tensor(targets),beta=.1,reduction='none').mean(2)
            loss=loss+.5*(reconstruction*mask).sum()/mask.sum().clamp_min(1)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if scheduler is not None:scheduler.step()
        if step%100==0 or step==steps-1:print('SET',kind,seed,step,float(loss.detach().cpu()),flush=True)
    return net.eval(),norm


def predict(net,norm,r,device='mps',chunk=2048):
    outputs=[];boundaries=[];coordinates=[]
    with torch.inference_mode():
        for a,b in r['slices']:
            direct=[];mirrored=[];de=[];me=[];dc=[];mc=[]
            for swapped in (False,True):
                p=[];e=[];c=[]
                for start in range(a,b,chunk):
                    stop=min(start+chunk,b);lo=max(a,start-160);hi=min(b,stop+160)
                    nodes=r['nodes'][lo:hi].astype('float32');ball=r['ball_tokens'][lo:hi];team=r['team']
                    if swapped:nodes,ball,team=mirror(nodes,ball,team)
                    gx=norm.transform(r['mirror' if swapped else 'direct'][lo:hi]) if net.kind=='fused' else None
                    def t(x,dtype=torch.float32):return torch.as_tensor(x,dtype=dtype,device=device)[None]
                    logits,edge,xy=net(t(nodes),t(ball),t(team,torch.long),t(gx) if gx is not None else None)
                    use=slice(start-lo,stop-lo)
                    p.append(logits.softmax(1)[0,:,use].T.cpu().numpy());e.append(edge.sigmoid()[0,use].cpu().numpy());c.append(xy[0,use].cpu().numpy())
                if swapped:mirrored=np.concatenate(p)[:,[1,0,2]];me=np.concatenate(e);mc=-np.concatenate(c)
                else:direct=np.concatenate(p);de=np.concatenate(e);dc=np.concatenate(c)
            outputs.append((direct+mirrored)/2);boundaries.append((de+me)/2);coordinates.append((dc+mc)/2)
    return np.concatenate(outputs),np.concatenate(boundaries),np.concatenate(coordinates)


def evaluate(net,norm,root,device='mps'):
    rows=[];cfg=load_config(Path('possession/config_external_selected.json'))
    for game in v6_data.protocol()['native_development']:
        for mode in ('clean','moderate','stress'):
            r=data.record('development',game,mode);p,e,c=predict(net,norm,r,device)
            np.savez_compressed(root/f'{game}_{mode}.npz',probabilities=p,edges=e,coordinates=c,frames=r['frames'])
            a=np.load(v6_data.ROOT/'tcn_boundary'/f'{game}_{mode}.npz');b=np.load(v6_data.ROOT/'physics_tcn'/f'{game}_{mode}.npz')
            native=(a['probabilities']+b['probabilities'])/2
            configs={'own_fixed':(p,None),'own_edge':(p,e),'original_edge':(p,a['edges']),
                     'average_original_edge':((p+native)/2,a['edges'])}
            for name,(prob,edge) in configs.items():
                codes=(decode_blocks(prob,r['frames'],as_match(r),cfg['decoder']) if edge is None else
                       dynamic_decode(prob,edge,r['frames'],as_match(r)))
                metric=evaluate_prediction(r['full_labels'],codes);rows.append(dict(game=game,mode=mode,configuration=name,**metric))
            print('EVAL',root.name,game,mode,{r['configuration']:round(r['spell_f1_mean'],4) for r in rows[-4:]},flush=True)
    summary={name:dict(spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in rows if r['configuration']==name])),
                      worst_spell_f1=float(min(r['spell_f1_mean'] for r in rows if r['configuration']==name)),
                      macro_f1=float(np.mean([r['macro_f1'] for r in rows if r['configuration']==name]))) for name in configs}
    (root/'results.json').write_text(json.dumps(dict(summary=summary,rows=rows,challenge_labels_read=False),indent=2))
    print('SUMMARY',root.name,summary,flush=True)


def load_model(path,device='mps'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);norm=Normalizer()
    norm.mean=np.asarray(checkpoint['mean'],dtype='float32');norm.std=np.asarray(checkpoint['std'],dtype='float32')
    net=PlayerSetNet(checkpoint['kind'],len(norm.mean)*2);net.load_state_dict(checkpoint['state_dict'])
    return net.to(device).eval(),norm


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--kind',choices=['set','trajectory','fused','selfsup']);parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=2027)
    parser.add_argument('--run-name');parser.add_argument('--cosine',action='store_true')
    parser.add_argument('--augment',action='store_true');args=parser.parse_args()
    if args.prepare:data.prepare()
    if args.kind:
        if args.run_name is not None and (not args.run_name.replace('_','').isalnum()):raise ValueError('Invalid run name')
        root=data.ROOT/(args.run_name or f'{args.kind}_{args.seed}');root.mkdir(parents=True,exist_ok=True)
        if (root/'model.pt').exists():raise RuntimeError('Preserve existing model')
        net,norm=fit(args.kind,args.seed,args.steps,cosine=args.cosine,augment=args.augment)
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=norm.mean,std=norm.std,
                        kind=args.kind,seed=args.seed,steps=args.steps,cosine=args.cosine,augment=args.augment,raw_signature=data.signature()),root/'model.pt')
        evaluate(net,norm,root)
