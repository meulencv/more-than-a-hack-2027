"""Learn when and how to correct cross-fitted errors, not in-sample failures."""
import argparse,json,os
os.environ.setdefault('OMP_NUM_THREADS','6')
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from possession import v13_cv
from possession.v6_temporal import train,predict
from possession.v6_experiment import as_match,subsequences
from possession.v14_resolution import decode_fine
from possession.benchmark_external import evaluate_prediction

ROOT=Path('data/work/v21');ROOT.mkdir(parents=True,exist_ok=True)
KINDS=['baseline','gate25','gate50','gate75','direct10','direct25']

def observation(x,p,edge):
    entropy=-(p*np.log(np.maximum(p,1e-6))).sum(1)
    z=np.column_stack((x,np.log(np.maximum(p,1e-5)),entropy,edge)).astype('float32')
    z[~np.isfinite(z)]=np.nan
    return z

def inner_predictions(fold,device):
    folder=ROOT/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
    games=[g for g in v13_cv.settings()['games'] if g not in v13_cv.settings()['test_folds'][fold]]
    for held_half in [0,1]:
        path=folder/f'inner_half{held_half}.pt'
        if path.exists():net,norm=v13_cv.load_temporal(path,device)
        else:
            xx=[];yy=[]
            for game in games:
                for mode in ['clean','moderate','stress']:
                    r=v13_cv.record(game,mode)
                    if len(r['slices'])!=2:raise ValueError('Expected two halves')
                    a,b=r['slices'][1-held_half]
                    for key in ['direct','mirror']:
                        xx.append(r[key][a:b]);y=r['labels'][a:b]
                        yy.append(y if key=='direct' else np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y))
            net,norm=train(xx,yy,kind='boundary',steps=800,seed=2032+held_half,device=device)
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=norm.mean,std=norm.std,train_games=games,excluded_half=held_half),path)
        for game in games:
            for mode in ['clean','moderate','stress']:
                path=folder/f'{game}_{mode}_half{held_half}.npz'
                if path.exists():continue
                r=v13_cv.record(game,mode);a,b=r['slices'][held_half]
                p,e=predict(net,norm,[r['direct'][a:b],r['mirror'][a:b]],device)
                prob=(p[0]+p[1][:,[1,0,2]])/2;edge=(e[0]+e[1])/2
                np.savez_compressed(path,p=prob,e=edge,frames=r['frames'][a:b])
        del net
        if device=='mps':torch.mps.empty_cache()
    return games

def fit(fold,device):
    path=ROOT/f'fold{fold}'/'correctors.joblib'
    if path.exists():return joblib.load(path)['models']
    games=inner_predictions(fold,device);xx=[];yy=[];ww=[];ee=[]
    for game in games:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode)
            for half,(a,b) in enumerate(r['slices']):
                saved=np.load(ROOT/f'fold{fold}'/f'{game}_{mode}_half{half}.npz')
                if not np.array_equal(saved['frames'],r['frames'][a:b]):raise ValueError('Inner alignment mismatch')
                p=saved['p'];edge=saved['e'];y=r['labels'][a:b];wrong=p.argmax(1)!=y
                # Keep all misclassified frames plus a regular sample of correctly classified frames.
                keep=(y>=0)&((np.arange(len(y))%3==0)|wrong)
                # Correct selection sampling for the error gate: correct frames represent 3 frames.
                gate_weight=np.where(wrong,1.,3.)
                for key in ['direct','mirror']:
                    pp=p if key=='direct' else p[:,[1,0,2]];target=y if key=='direct' else np.where(y>=0,np.array([1,0,2])[np.clip(y,0,2)],y)
                    xx.append(observation(r[key][a:b],pp,edge)[keep]);yy.append(target[keep]);ee.append(wrong[keep].astype('int8'));ww.append(gate_weight[keep])
    x=np.concatenate(xx);y=np.concatenate(yy);error=np.concatenate(ee);weight=np.concatenate(ww);models=[]
    for kind in ['gate','correction']:
        net=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=31,learning_rate=.06,l2_regularization=10,min_samples_leaf=80,early_stopping=False,random_state=2032)
        with threadpool_limits(limits=6):net.fit(x,error if kind=='gate' else y,sample_weight=weight if kind=='gate' else np.where(error,4.,1.))
        models.append(net);print('CORRECTOR FIT',fold,kind,x.shape,float(error.mean()),flush=True)
    joblib.dump(dict(models=models,games=games,inner_crossfit='Other half only; all outer held games excluded from base models and correctors'),path,compress=1)
    return models

def predict_corrector(models,r,p,e):
    gate=[];prob=[]
    with threadpool_limits(limits=6):
        for key in ['direct','mirror']:
            obs=observation(r[key],p if key=='direct' else p[:,[1,0,2]],e)
            gate.append(models[0].predict_proba(obs)[:,1]);q=models[1].predict_proba(obs)
            prob.append(q if key=='direct' else q[:,[1,0,2]])
    return (gate[0]+gate[1])/2,(prob[0]+prob[1])/2

def evaluate(fold,kinds,device):
    models=fit(fold,device);rows=[]
    for game in v13_cv.settings()['test_folds'][fold]:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode);s=np.load(v13_cv.ROOT/f'fold{fold}'/f'{game}_{mode}.npz');p=s['combined'];e=s['edges']
            gate,q=predict_corrector(models,r,p,e);np.savez_compressed(ROOT/f'{game}_{mode}.npz',gate=gate,q=q,frames=r['frames'])
            for kind in kinds:
                if kind=='baseline':w=0
                elif kind.startswith('gate'):w=float(kind[4:])/100*gate[:,None]
                else:w=float(kind[6:])/100
                pred=decode_fine((1-w)*p+w*q,e,r['frames'],as_match(r));metric=evaluate_prediction(r['full_labels'],pred)
                rows.append(dict(game=game,mode=mode,kind=kind,**metric));print('RESULT',fold,game,mode,kind,round(metric['spell_f1_mean'],5),flush=True)
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='mps');args=parser.parse_args();torch.set_num_threads(6)
    protocol=dict(purpose='Train a corrector on observed cross-fitted base errors, with a learned intervention gate.',
        inner_crossfit='Within the four outer training matches only: train one 156-feature boundary TCN on each half across games and predict the opposite half. 800 updates per model. Outer test matches never enter base training, normalizers, correctors or gates.',
        correction='HGB error gate and state corrector, 200 iterations, 31 leaves. Input156 automatic signals plus log state probabilities, entropy and boundary. Error sampling with inverse-selection gate weights; corrector error weight4. Team-symmetric averaging.',
        limitation='Inner errors generalize across halves of seen training matches, not fully new games; inner one-TCN probabilities differ from outer four-expert probabilities. Outer held-game evaluation checks this mismatch. Previously used external games; repeated selection, not an unbiased final estimate.',
        candidates=KINDS,selection_fold=1,confirmation_fold=2,selection='Highest selection spellF1 with macro>=baseline-.002; freeze then evaluate selected only on fold2; adopt if both game means improve and macro>=baseline-.002.',challenge_labels_read=False)
    path=Path('docs/PROTOCOLO_V21.json')
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol mismatch')
    path.write_text(json.dumps(protocol,indent=2));rows=evaluate(1,KINDS,args.device)
    means={k:{m:float(np.mean([r[m] for r in rows if r['kind']==k])) for m in ['spell_f1_mean','macro_f1']} for k in KINDS}
    allowed=[k for k in means if means[k]['macro_f1']>=means['baseline']['macro_f1']-.002];chosen=max(allowed,key=lambda k:means[k]['spell_f1_mean'])
    result=dict(selection_rows=rows,selection_means=means,chosen=chosen,confirmation_rows=[],challenge_labels_read=False)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2));print('CHOSEN',chosen,means,flush=True)
    if chosen!='baseline':result['confirmation_rows']=evaluate(2,['baseline',chosen],args.device)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
