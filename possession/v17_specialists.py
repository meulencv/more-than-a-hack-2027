"""Held-game specialist ensemble experiment. Never reads challenge labels."""
import json,os
os.environ.setdefault('OMP_NUM_THREADS','6')
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from possession import v13_cv
from possession.v6_temporal import segment_weights
from possession.v6_experiment import as_match
from possession.v14_resolution import decode_fine
from possession.benchmark_external import evaluate_prediction

ROOT=Path('data/work/v17/specialists');ROOT.mkdir(parents=True,exist_ok=True)
KINDS=['baseline','short10','short25','dead25','owner25','factor25','uncertain25']

def clean(x):
    x=x.astype('float32',copy=True);x[~np.isfinite(x)]=np.nan;return x

def fit(fold):
    held=v13_cv.settings()['test_folds'][fold];games=[g for g in v13_cv.settings()['games'] if g not in held]
    path=ROOT/f'fold{fold}.joblib'
    if path.exists():
        saved=joblib.load(path)
        if saved['games']!=games:raise ValueError('Training games changed')
        return saved['models']
    xx=[];yy=[];ww=[]
    for game in games:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode)
            for key in ['direct','mirror']:
                for a,b in r['slices']:
                    y=r['labels'][a:b].copy()
                    if key=='mirror':y=np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y)
                    cuts=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
                    lengths=np.repeat(np.diff(cuts),np.diff(cuts))
                    # A sampled timeline plus dense examples of brief live possessions.
                    keep=(np.arange(len(y))%4==0)|((lengths<=15)&(y<2)&(np.arange(len(y))%2==0))
                    keep&=y>=0
                    w=segment_weights(y)**2;w=np.clip(w,.1,8)
                    xx.append(clean(r[key][a:b][keep]));yy.append(y[keep]);ww.append(w[keep])
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww)
    models=[]
    for name in ['short','dead','owner']:
        use=y<2 if name=='owner' else np.ones(len(y),bool)
        target=(y==2).astype('int8') if name=='dead' else y
        # Disable random internal frame validation: a match is the validation unit.
        net=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=31,learning_rate=.06,l2_regularization=10,
            min_samples_leaf=60,early_stopping=False,random_state=2027)
        print('FIT',fold,name,x[use].shape,flush=True)
        with threadpool_limits(limits=6):net.fit(x[use],target[use],sample_weight=w[use])
        models.append(net)
    joblib.dump(dict(games=games,models=models),path,compress=1)
    return models

def probabilities(models,r):
    out=[]
    with threadpool_limits(limits=6):
        for i,net in enumerate(models):
            a=net.predict_proba(clean(r['direct']));b=net.predict_proba(clean(r['mirror']))
            if i!=1:b=b[:,[1,0,2]] if i==0 else b[:,[1,0]]
            out.append((a+b)/2)
    return out

def blend(p,qs,kind):
    if kind=='baseline':return p
    short,dead,owner=qs
    d=dead[:,1];fact=np.column_stack((owner[:,0]*(1-d),owner[:,1]*(1-d),d))
    if kind.startswith('short'):return p*(1-float(kind[5:])/100)+short*float(kind[5:])/100
    if kind=='factor25':return .75*p+.25*fact
    if kind=='dead25':
        d=.75*p[:,2]+.25*d;ratio=p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-8)
        return np.column_stack((ratio*(1-d[:,None]),d))
    if kind=='owner25':
        ratio=.75*p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-8)+.25*owner
        return np.column_stack((ratio*(1-p[:,2,None]),p[:,2]))
    if kind=='uncertain25':
        weight=.25*(1-p.max(1))/.6666667
        return p*(1-weight[:,None])+short*weight[:,None]
    raise ValueError(kind)

def evaluate(fold,kinds):
    models=fit(fold);rows=[]
    for game in v13_cv.settings()['test_folds'][fold]:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode);s=np.load(v13_cv.ROOT/f'fold{fold}'/f'{game}_{mode}.npz')
            qs=probabilities(models,r)
            np.savez_compressed(ROOT/f'{game}_{mode}.npz',short=qs[0],dead=qs[1],owner=qs[2],frames=r['frames'])
            for kind in kinds:
                pred=decode_fine(blend(s['combined'],qs,kind),s['edges'],s['frames'],as_match(r))
                row=dict(game=game,mode=mode,kind=kind,**evaluate_prediction(r['full_labels'],pred));rows.append(row)
                print('RESULT',fold,game,mode,kind,round(row['spell_f1_mean'],5),flush=True)
    return rows

def main():
    protocol=dict(purpose='Test short-possession, dead-ball and ownership specialists trained only on the four non-held games.',
        candidates=KINDS,selection_fold=1,confirmation_fold=2,training='Three HGB models, 180 iterations, 31 leaves, learning rate .06; short-spell inverse-duration weight capped at 8; sampled every .8s plus .4s in short live spells; mirrored teams.',
        selection='Highest mean spellF1 on fold1 with macroF1 >= baseline-.002. Confirm only selected candidate on fold2; adopt if both game spell means improve and aggregate macro >= baseline-.002.',
        limitation='Previously used external games; new-round confirmation, not untouched final test.',challenge_labels_read=False)
    path=Path('docs/PROTOCOLO_V17_SPECIALISTS.json')
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol mismatch')
    path.write_text(json.dumps(protocol,indent=2))
    rows=evaluate(1,KINDS)
    means={k:{m:float(np.mean([r[m] for r in rows if r['kind']==k])) for m in ['spell_f1_mean','macro_f1']} for k in KINDS}
    allowed=[k for k in means if means[k]['macro_f1']>=means['baseline']['macro_f1']-.002]
    chosen=max(allowed,key=lambda k:means[k]['spell_f1_mean'])
    result=dict(selection_rows=rows,selection_means=means,chosen=chosen,confirmation_rows=[],challenge_labels_read=False)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2));print('CHOSEN',chosen,means,flush=True)
    if chosen!='baseline':result['confirmation_rows']=evaluate(2,['baseline',chosen])
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
