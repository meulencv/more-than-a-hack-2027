"""25 Hz transient-control expert, fitted and validated on separate external games."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,torch
from possession import v6_data,v13_cv
from possession.sources import challenge_match
from possession.sportec_native import external_native_match
from possession.degrade import challenge_profile
from possession.features_v3 import feature_table
from possession.v6_temporal import train,predict
from possession.v6_experiment import as_match,subsequences
from possession.v14_resolution import decode_fine
from possession.benchmark_external import evaluate_prediction

ROOT=Path('data/work/v17/highrate');ROOT.mkdir(parents=True,exist_ok=True)
KINDS=['baseline','rate10','rate25','rate25edge']

def prepare():
    profile=challenge_profile(challenge_match());protocol=v6_data.protocol()
    for game in v13_cv.settings()['games']:
        role='train' if game in protocol['native_train'] else 'development'
        i=protocol['native_train' if role=='train' else 'native_development'].index(game)
        source=None
        for mode in ['clean','moderate','stress']:
            path=ROOT/f'{game}_{mode}.joblib'
            if path.exists():continue
            if source is None:source=external_native_match(game)
            match=v6_data.sensor_view(source,mode,(310 if role=='train' else 907)+i,profile)
            a,f=feature_table(match,step=1);b,g=feature_table(match,step=1,swap=True)
            if not np.array_equal(f,g):raise ValueError('Mirror timeline mismatch')
            coarse=v13_cv.record(game,mode)
            if not np.allclose(a.to_numpy()[np.searchsorted(f,coarse['frames'])],coarse['direct'][:,:123],equal_nan=True):raise ValueError('Sensor realization mismatch')
            lengths=[z-a+1 for _,a,z in match.blocks];cuts=np.r_[0,np.cumsum(lengths)]
            record=dict(direct=a.to_numpy(),mirror=b.to_numpy(),frames=f,labels=source.meta['labels'][f],
                slices=list(zip(cuts[:-1],cuts[1:])),blocks=match.blocks,full_labels=source.meta['labels'])
            temporary=path.with_suffix('.part');joblib.dump(record,temporary,compress=1);temporary.replace(path)
            print('PREPARED',game,mode,len(f),flush=True)

def fit(fold,device):
    games=[g for g in v13_cv.settings()['games'] if g not in v13_cv.settings()['test_folds'][fold]]
    path=ROOT/f'fold{fold}.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['train_games']!=games:raise ValueError('Training partition mismatch')
        return v13_cv.load_temporal(path,device)
    xx=[];yy=[]
    for game in games:
        for mode in ['clean','moderate','stress']:
            r=joblib.load(ROOT/f'{game}_{mode}.joblib')
            for key in ['direct','mirror']:
                for a,b in r['slices']:
                    xx.append(r[key][a:b]);y=r['labels'][a:b]
                    yy.append(y if key=='direct' else np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y))
    net,norm=train(xx,yy,kind='boundary',steps=800,seed=2029,device=device)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=norm.mean,std=norm.std,train_games=games,rate=25,seed=2029,steps=800),path)
    return net,norm

def infer(net,norm,r,device):
    # Short valid-convolution halos prevent a full-half GPU activation allocation.
    pp=[];ee=[]
    for a,b in r['slices']:
        ps=[];es=[]
        for start in range(a,b,4096):
            end=min(start+4096,b);lo=max(a,start-160);hi=min(b,end+160)
            p,e=predict(net,norm,[r['direct'][lo:hi],r['mirror'][lo:hi]],device)
            ps.append(((p[0]+p[1][:,[1,0,2]])/2)[start-lo:end-lo]);es.append(((e[0]+e[1])/2)[start-lo:end-lo])
        pp.append(np.concatenate(ps));ee.append(np.concatenate(es))
    return np.concatenate(pp),np.concatenate(ee)

def evaluate(fold,kinds,device,model_fitter=fit,output_root=None):
    output_root=ROOT if output_root is None else output_root
    net,norm=model_fitter(fold,device);rows=[]
    for game in v13_cv.settings()['test_folds'][fold]:
        for mode in ['clean','moderate','stress']:
            r=joblib.load(ROOT/f'{game}_{mode}.joblib');old=np.load(v13_cv.ROOT/f'fold{fold}'/f'{game}_{mode}.npz')
            path=output_root/f'{game}_{mode}.npz'
            if path.exists():saved=np.load(path);p,e=saved['p'],saved['e']
            else:
                p,e=infer(net,norm,r,device);np.savez_compressed(path,p=p,e=e,frames=r['frames'])
            base=[];edges=[]
            for _,a,b in r['blocks']:
                use=(old['frames']>=a)&(old['frames']<=b);grid=np.arange(a,b+1)
                base.append(np.column_stack([np.interp(grid,old['frames'][use],old['combined'][use,j]) for j in range(3)]))
                edges.append(np.interp(grid,old['frames'][use],old['edges'][use]))
            base=np.concatenate(base);edges=np.concatenate(edges)
            for kind in kinds:
                weight=0 if kind=='baseline' else .1 if kind=='rate10' else .25
                blended=(1-weight)*base+weight*p
                bound=.75*edges+.25*e if kind=='rate25edge' else edges
                codes=decode_fine(blended,bound,r['frames'],as_match(r))
                metric=evaluate_prediction(r['full_labels'],codes);rows.append(dict(game=game,mode=mode,kind=kind,**metric))
                print('RESULT',fold,game,mode,kind,round(metric['spell_f1_mean'],5),flush=True)
    del net
    if device=='mps':torch.mps.empty_cache()
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='mps');args=p.parse_args()
    protocol=dict(purpose='Resolve observed subsecond missed possessions with a 25Hz temporal expert; original features computed at every frame, not interpolated.',
        train='Same four non-held matches, 3 sensor regimes and team mirroring; 800 boundary-TCN updates, seed2029, original architecture/optimizer; 384-frame windows now 15.36s, boundary supervision radius now .08s.',
        candidates=KINDS,selection_fold=1,confirmation_fold=2,selection='Highest selection spellF1 with macro>=baseline-.002; freeze then test selected only on fold2; adopt if both confirmation game means improve and macro>=baseline-.002.',
        limitation='Previously used external matches, not untouched final test.',challenge_labels_read=False)
    path=Path('docs/PROTOCOLO_V17_HIGHRATE.json')
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol changed')
    path.write_text(json.dumps(protocol,indent=2));prepare()
    rows=evaluate(1,KINDS,args.device)
    means={k:{m:float(np.mean([r[m] for r in rows if r['kind']==k])) for m in ['spell_f1_mean','macro_f1']} for k in KINDS}
    allowed=[k for k in means if means[k]['macro_f1']>=means['baseline']['macro_f1']-.002];chosen=max(allowed,key=lambda k:means[k]['spell_f1_mean'])
    result=dict(selection_rows=rows,selection_means=means,chosen=chosen,confirmation_rows=[],challenge_labels_read=False)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2));print('CHOSEN',chosen,means,flush=True)
    if chosen!='baseline':result['confirmation_rows']=evaluate(2,['baseline',chosen],args.device)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
