"""Explicit-duration decoding learned only from each fold's training matches."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from possession import v13_cv
from possession.v6_experiment import as_match
from possession.v14_resolution import decode_fine
from possession.benchmark_external import evaluate_prediction
from evaluation.possession_metrics import segments,valid_runs

ROOT=Path('data/work/v20');ROOT.mkdir(parents=True,exist_ok=True)
KINDS=['baseline','duration0.5','duration1','duration2']

def learned_exit_adjustment(games,cap=50):
    durations=[[],[]]
    for game in games:
        r=v13_cv.record(game,'clean');y=np.where(r['full_labels']>=0,r['full_labels'],3)
        for a,b,k in segments(y,valid_runs(y)):
            if k<3:durations[int(k==2)].append(b-a)
    curves=[]
    for lengths in durations:
        lengths=np.asarray(lengths);mean=max(2,float(lengths.mean()));generic=1/mean
        # Smooth event counts but estimate risk sets from actual training lengths.
        hist=np.bincount(np.minimum(lengths,cap+1),minlength=cap+2).astype(float)
        event=gaussian_filter1d(hist,2,mode='nearest')
        risk=np.array([(lengths>=a).sum() for a in range(1,cap+1)],float)
        hazard=(event[1:cap+1]+20*generic)/(risk+20)
        hazard[-1]=np.sum(lengths>=cap)/np.maximum(np.maximum(lengths-cap+1,0).sum(),1)
        curves.append(np.clip(np.log(np.maximum(hazard,1e-7)/generic),-2,2))
    return np.array([curves[0],curves[0],curves[1]])

def decode(p,e,f,m,adjustment,strength):
    cap=adjustment.shape[1];out=np.full(len(m.ball),3,dtype='int8')
    transition=np.array([[0.,4.,2.5],[4.,0.,2.5],[2.5,2.5,0.]])
    for _,first,last in m.blocks:
        use=(f>=first)&(f<=last);ff=f[use];grid=np.arange(first,last+1)
        emission=np.log(np.maximum(np.column_stack([np.interp(grid,ff,p[use,j]) for j in range(3)]),1e-5))/5
        edge=np.interp(grid,ff,e[use]);score=np.full((3,cap),-np.inf);score[:,0]=emission[0]
        # For age>1 predecessor is fixed, except overflow age. Store only reset and overflow decisions.
        resets=np.zeros((len(grid),3),dtype='int16');overflow=np.zeros((len(grid),3),dtype=bool)
        for t in range(1,len(grid)):
            value=score[:,:,None]-np.maximum(.05,transition[:,None,:]*(1-.85*edge[t])-strength*adjustment[:,:,None])
            for j in range(3):value[j,:,j]=-np.inf
            flat=value.reshape(3*cap,3);resets[t]=flat.argmax(0)
            new=np.empty_like(score);new[:,0]=flat.max(0)
            new[:,1:]=score[:,:-1];overflow[t]=score[:,-1]>score[:,-2]
            new[:,-1]=np.maximum(score[:,-1],score[:,-2]);new+=emission[t,:,None];new-=new.max();score=new
        state,age=np.unravel_index(score.argmax(),score.shape);states=np.empty(len(grid),dtype='int8')
        for t in range(len(grid)-1,-1,-1):
            states[t]=state
            if age==0:state,age=divmod(int(resets[t,state]),cap)
            elif age==cap-1:age=cap-1 if overflow[t,state] else cap-2
            else:age-=1
        out[first:last+1]=states
    return out

def evaluate(fold,kinds):
    games=[g for g in v13_cv.settings()['games'] if g not in v13_cv.settings()['test_folds'][fold]]
    adjustment=learned_exit_adjustment(games);np.save(ROOT/f'fold{fold}_hazard.npy',adjustment);rows=[]
    for game in v13_cv.settings()['test_folds'][fold]:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode);s=np.load(v13_cv.ROOT/f'fold{fold}'/f'{game}_{mode}.npz')
            for kind in kinds:
                pred=decode_fine(s['combined'],s['edges'],s['frames'],as_match(r)) if kind=='baseline' else decode(s['combined'],s['edges'],s['frames'],as_match(r),adjustment,float(kind[8:]))
                row=dict(game=game,mode=mode,kind=kind,**evaluate_prediction(r['full_labels'],pred));rows.append(row)
                print('RESULT',fold,game,mode,kind,round(row['spell_f1_mean'],5),flush=True)
    return rows

def main():
    protocol=dict(purpose='Replace the memoryless switching penalty by a learned age-dependent exit cost, aiming to recover brief possessions without lowering every switch penalty.',
        training='Only clean training-game duration histograms; H/A pooled for symmetry, dead separate; 2-frame Gaussian smoothing, 20-risk pseudo prior, 50-frame explicit ages then geometric overflow; log hazard ratio clipped +-2; positive cost floor .05.',
        candidates=KINDS,selection_fold=1,confirmation_fold=2,selection='Highest selection spellF1 with macro>=baseline-.002; freeze then evaluate selected only on fold2; adopt if both game means improve and macro>=baseline-.002.',
        limitation='Historically used matches, repeated development selection, not untouched final test.',challenge_labels_read=False)
    path=Path('docs/PROTOCOLO_V20.json')
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol mismatch')
    path.write_text(json.dumps(protocol,indent=2));rows=evaluate(1,KINDS)
    means={k:{m:float(np.mean([r[m] for r in rows if r['kind']==k])) for m in ['spell_f1_mean','macro_f1']} for k in KINDS}
    allowed=[k for k in means if means[k]['macro_f1']>=means['baseline']['macro_f1']-.002];chosen=max(allowed,key=lambda k:means[k]['spell_f1_mean'])
    result=dict(selection_rows=rows,selection_means=means,chosen=chosen,confirmation_rows=[],challenge_labels_read=False)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2));print('CHOSEN',chosen,means,flush=True)
    if chosen!='baseline':result['confirmation_rows']=evaluate(2,['baseline',chosen])
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
