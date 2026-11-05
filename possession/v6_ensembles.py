"""Predefined probability combinations on external development; no public GT."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import v6_data as data
from .v6_experiment import dynamic_decode,as_match
from .v3 import load_config,decode_blocks
from .benchmark_external import evaluate_prediction


def entropy(p):
    return -(p*np.log(np.clip(p,1e-8,1))).sum(axis=1)


def combinations(records,weight):
    b=records['tcn_boundary']['probabilities']
    s=records['tcn_segment']['probabilities']
    h=records['hgb_native']['probabilities']
    m=records['tcn_multistage']['probabilities']
    q=records['tcn_ballfree']['probabilities']
    geometric=np.sqrt(np.clip(b,1e-8,1)*np.clip(h,1e-8,1))
    geometric/=geometric.sum(axis=1,keepdims=True)
    uncertainty=entropy(h)/(entropy(h)+entropy(b)+1e-8)
    result={
        'boundary_segment':(b+s)/2,
        'boundary_multistage':(b+m)/2,
        'geometric_hgb_boundary':geometric,
        'quality_hgb_boundary':(1-weight[:,None])*h+weight[:,None]*b,
        'uncertainty_hgb_boundary':uncertainty[:,None]*b+(1-uncertainty[:,None])*h,
        'ballfree_in_occlusion':(1-weight[:,None]**2)*b+weight[:,None]**2*q,
        'median_committee':np.median(np.stack([b,s,h]),axis=0),
    }
    if 'physics_tcn' in records:
        p=records['physics_tcn']['probabilities']
        result['boundary_physics']=(b+p)/2
        result['physics_segment_boundary']=(b+p+s)/3
    for name,p in result.items():
        result[name]=p/p.sum(axis=1,keepdims=True)
    return result


def main():
    root=data.ROOT/'ensembles';root.mkdir(parents=True,exist_ok=True)
    cfg=load_config(Path('possession/config_external_selected.json'))
    rows=[]
    families=['tcn_boundary','tcn_segment','hgb_native','tcn_multistage','tcn_ballfree']
    if (data.ROOT/'physics_tcn/summary.json').exists():families+=['physics_tcn']
    for game in data.protocol()['native_development']:
        for mode in ('clean','moderate','stress'):
            r=data.record(game,mode,'development')
            records={f:dict(np.load(data.ROOT/f/f'{game}_{mode}.npz')) for f in families}
            for z in records.values():assert np.array_equal(z['frames'],r['frames'])
            missing=1-r['ball_seen'].astype(float);weights=np.zeros(len(missing))
            for _,a,b in r['blocks']:
                weights[a:b+1]=pd.Series(missing[a:b+1]).rolling(125,center=True,min_periods=1).mean()
            options=combinations(records,weights[r['frames']])
            for name,p in options.items():
                folder=root/name;folder.mkdir(exist_ok=True)
                edges=records['tcn_boundary']['edges']
                np.savez_compressed(folder/f'{game}_{mode}.npz',probabilities=p,edges=edges,frames=r['frames'])
                for decoder in ('fixed','boundary'):
                    codes=(dynamic_decode(p,edges,r['frames'],as_match(r)) if decoder=='boundary' else
                           decode_blocks(p,r['frames'],as_match(r),cfg['decoder']))
                    metric=evaluate_prediction(r['full_labels'],codes)
                    rows.append(dict(family=name,decoder=decoder,game=game,mode=mode,**metric))
            (root/'results.json').write_text(json.dumps(rows,indent=2))
    summary=[]
    for family,decoder in sorted({(r['family'],r['decoder']) for r in rows}):
        group=[r for r in rows if (r['family'],r['decoder'])==(family,decoder)]
        summary.append(dict(family=family,decoder=decoder,
                            spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in group])),
                            worst_spell_f1=float(min(r['spell_f1_mean'] for r in group)),
                            macro_f1=float(np.mean([r['macro_f1'] for r in group]))))
    summary.sort(key=lambda r:r['spell_f1_mean'],reverse=True)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    for r in summary:print(r,flush=True)


if __name__=='__main__':main()
