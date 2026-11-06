"""External error diagnosis and preregistered decoder experiments; no challenge GT."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from possession import v13_cv
from possession.v6_experiment import as_match
from possession.v14_resolution import decode_fine
from possession.benchmark_external import evaluate_prediction
from evaluation import possession_metrics as pm

ROOT=Path('data/work/v17');ROOT.mkdir(parents=True,exist_ok=True)

def decode(p,e,f,m,kind):
    if kind=='baseline':return decode_fine(p,e,f,m)
    if kind.startswith('cost'):
        # Scaling emissions is exactly inverse scaling all transition costs.
        scale=float(kind[4:]);q=np.maximum(p,1e-7)**(1/scale);q/=q.sum(1,keepdims=True)
        return decode_fine(q,e,f,m)
    if kind.startswith('asrf'):
        threshold=float(kind[4:]);out=np.full(len(m.ball),3,dtype='int8')
        for _,first,last in m.blocks:
            use=(f>=first)&(f<=last);pp=p[use];ee=e[use];ff=f[use]
            peaks=find_peaks(ee,height=threshold,distance=3)[0]
            cuts=np.unique(np.r_[0,peaks,len(ff)])
            states=np.empty(len(ff),dtype='int8')
            for a,b in zip(cuts[:-1],cuts[1:]):states[a:b]=pp[a:b].mean(0).argmax()
            out[first:last+1]=np.repeat(states,np.diff(np.r_[ff,last+1]))
        return out
    raise ValueError(kind)

def diagnose(r,p,e,f):
    pred=decode(p,e,f,as_match(r),'baseline');gt=np.where(r['full_labels']>=0,r['full_labels'],3)
    runs=pm.valid_runs(gt);gs=pm.possession_spells(gt,runs);ps=pm.possession_spells(pred,runs)
    ov=pm.spell_overlaps(gs,ps);matched=pm.matched_gt_flags(ov,len(gs),.5)
    best=np.zeros(len(gs));bestj=np.full(len(gs),-1)
    for iou,i,j in ov:
        if iou>best[i]:best[i]=iou;bestj[i]=j
    rows=[]
    for i,(a,b,k) in enumerate(gs):
        near=[s for s in ps if s[1]>a and s[0]<b]
        missing=1-float(r['ball_seen'][a:b].mean())
        reason='matched' if matched[i] else ('absent' if best[i]==0 else 'split' if len([s for s in near if s[2]==k])>1 else 'merged' if bestj[i]>=0 and ps[bestj[i]][1]-ps[bestj[i]][0]>2*(b-a) else 'boundary_or_team')
        rows.append(dict(start=a,stop=b,team=int(k),duration=(b-a)/25,matched=bool(matched[i]),best_iou=float(best[i]),reason=reason,ball_missing=missing,
                         preceding=int(gt[a-1]) if a else 3,following=int(gt[b]) if b<len(gt) else 3))
    return rows,evaluate_prediction(r['full_labels'],pred)

def main():
    protocol=dict(purpose='Diagnose external fold0; select one decoder on fold1; open fold2 only after selecting. All base predictions exclude the held match.',
        candidates=['baseline','cost0.25','cost0.5','cost1.5','asrf0.2','asrf0.4','asrf0.6'],
        selection='Highest mean spellF1 on fold1 subject to macroF1 >= baseline - 0.002. Confirm selected method on fold2; adopt only if both game means improve and macro does not fall > .002.',
        limitation='All six games have been used in earlier architecture development; this is a new-round holdout, not untouched data.',challenge_labels_read=False)
    path=Path('docs/PROTOCOLO_V17.json')
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol differs')
    path.write_text(json.dumps(protocol,indent=2))
    audit=[]
    for game in v13_cv.settings()['test_folds'][0]:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode);s=np.load(v13_cv.ROOT/'fold0'/f'{game}_{mode}.npz')
            spells,metric=diagnose(r,s['combined'],s['edges'],s['frames'])
            audit.append(dict(game=game,mode=mode,spells=spells,metric=metric))
    (ROOT/'error_audit.json').write_text(json.dumps(audit,indent=2))
    spells=[s for a in audit for s in a['spells']]
    summary={name:dict(n=len(ss),matched_at_05=float(np.mean([s['matched'] for s in ss]))) for name,ss in [
        ('under1s',[s for s in spells if s['duration']<1]),('1to3s',[s for s in spells if 1<=s['duration']<3]),
        ('3to10s',[s for s in spells if 3<=s['duration']<10]),('over10s',[s for s in spells if s['duration']>=10]) ]}
    summary['miss_reasons']={k:sum(s['reason']==k for s in spells) for k in ['matched','absent','split','merged','boundary_or_team']}
    (ROOT/'error_summary.json').write_text(json.dumps(summary,indent=2));print('ERRORS',summary,flush=True)
    rows=[]
    for game in v13_cv.settings()['test_folds'][1]:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode);s=np.load(v13_cv.ROOT/'fold1'/f'{game}_{mode}.npz')
            for kind in protocol['candidates']:
                pred=decode(s['combined'],s['edges'],s['frames'],as_match(r),kind)
                row=dict(game=game,mode=mode,kind=kind,**evaluate_prediction(r['full_labels'],pred));rows.append(row)
                print('SELECT',game,mode,kind,round(row['spell_f1_mean'],5),flush=True)
    means={k:{m:float(np.mean([r[m] for r in rows if r['kind']==k])) for m in ['spell_f1_mean','macro_f1']} for k in protocol['candidates']}
    allowed=[k for k in means if means[k]['macro_f1']>=means['baseline']['macro_f1']-.002]
    chosen=max(allowed,key=lambda k:means[k]['spell_f1_mean'])
    result=dict(selection_rows=rows,selection_means=means,chosen=chosen,confirmation_rows=[],challenge_labels_read=False)
    (ROOT/'decoder_selection.json').write_text(json.dumps(result,indent=2));print('CHOSEN',chosen,means,flush=True)
    for game in v13_cv.settings()['test_folds'][2]:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode);s=np.load(v13_cv.ROOT/'fold2'/f'{game}_{mode}.npz')
            for kind in dict.fromkeys(['baseline',chosen]):
                pred=decode(s['combined'],s['edges'],s['frames'],as_match(r),kind)
                row=dict(game=game,mode=mode,kind=kind,**evaluate_prediction(r['full_labels'],pred));result['confirmation_rows'].append(row)
                print('CONFIRM',game,mode,kind,round(row['spell_f1_mean'],5),flush=True)
    (ROOT/'decoder_selection.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
