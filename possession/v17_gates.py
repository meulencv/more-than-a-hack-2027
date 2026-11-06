"""Visibility/semantic conditional fusion of existing held-game experts."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from possession import v13_cv
from possession.v6_experiment import as_match
from possession.v14_resolution import decode_fine
from possession.benchmark_external import evaluate_prediction

ROOT=Path('data/work/v17/gates');ROOT.mkdir(parents=True,exist_ok=True)
KINDS=['baseline','raw25','raw75','visible_raw','missing_raw','dead_tcn','dead_raw','certain_expert']

def blend(saved,r,kind):
    base=saved['baseline'];raw=np.clip(2*saved['combined']-base,1e-7,1);raw/=raw.sum(1,keepdims=True)
    if kind=='baseline':return saved['combined']
    if kind in ['dead_tcn','dead_raw']:
        wd=.25 if kind=='dead_tcn' else .75;wa=1-wd
        dead=(1-wd)*base[:,2]+wd*raw[:,2]
        a=base[:,:2]/np.maximum(base[:,:2].sum(1,keepdims=True),1e-8);b=raw[:,:2]/np.maximum(raw[:,:2].sum(1,keepdims=True),1e-8)
        live=(1-wa)*a+wa*b
        return np.column_stack((live*(1-dead[:,None]),dead))
    if kind in ['raw25','raw75']:w=.25 if kind=='raw25' else .75
    elif kind in ['visible_raw','missing_raw']:
        seen=r['ball_seen'][r['frames']].astype(float);smooth=np.zeros(len(seen))
        for a,b in r['slices']:smooth[a:b]=uniform_filter1d(seen[a:b],size=5,mode='nearest')
        w=.25+.5*(smooth if kind=='visible_raw' else 1-smooth);w=w[:,None]
    elif kind=='certain_expert':
        ha=-(base*np.log(base)).sum(1);hb=-(raw*np.log(raw)).sum(1)
        w=(.25+.5*(ha>hb))[:,None]
    else:raise ValueError(kind)
    return (1-w)*base+w*raw

def evaluate(fold,kinds):
    rows=[]
    for game in v13_cv.settings()['test_folds'][fold]:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode);s=np.load(v13_cv.ROOT/f'fold{fold}'/f'{game}_{mode}.npz')
            for kind in kinds:
                pred=decode_fine(blend(s,r,kind),s['edges'],s['frames'],as_match(r))
                row=dict(game=game,mode=mode,kind=kind,**evaluate_prediction(r['full_labels'],pred));rows.append(row)
                print('RESULT',fold,game,mode,kind,round(row['spell_f1_mean'],5),flush=True)
    return rows

def main():
    protocol=dict(purpose='Test conditional mixture instead of a uniform four-network mean, without training on validation labels.',
        candidates=KINDS,selection_fold=1,confirmation_fold=2,parameters='Raw weight .25/.75, visibility 1s mean with per-half isolation; separate alive/dead and ownership distributions; entropy comparison is unsupervised.',
        selection='Highest selection spellF1 with macro>=baseline-.002; freeze then test selected only on fold2; adopt if both confirmation game means improve and macro>=baseline-.002.',
        limitation='Previously used external games, new-round confirmation only.',challenge_labels_read=False)
    path=Path('docs/PROTOCOLO_V17_GATES.json')
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol mismatch')
    path.write_text(json.dumps(protocol,indent=2));rows=evaluate(1,KINDS)
    means={k:{m:float(np.mean([r[m] for r in rows if r['kind']==k])) for m in ['spell_f1_mean','macro_f1']} for k in KINDS}
    allowed=[k for k in means if means[k]['macro_f1']>=means['baseline']['macro_f1']-.002];chosen=max(allowed,key=lambda k:means[k]['spell_f1_mean'])
    result=dict(selection_rows=rows,selection_means=means,chosen=chosen,confirmation_rows=[],challenge_labels_read=False)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2));print('CHOSEN',chosen,means,flush=True)
    if chosen!='baseline':result['confirmation_rows']=evaluate(2,['baseline',chosen])
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
