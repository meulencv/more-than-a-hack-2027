"""Frozen cross-provider check; references are explicitly weak and external."""
import json
from pathlib import Path
import numpy as np
import torch
from .sources import external_match,challenge_match
from .degrade import challenge_profile
from .v6_data import sensor_view
from .audit_reference import alternative_reference
from .v12_compare import match_record
from .competitive_bundle import CompetitiveBundle
from .v6_experiment import as_match,dynamic_decode
from .v14_resolution import decode_fine
from .benchmark_external import evaluate_prediction

ROOT=Path('data/work/v16')


def main():
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(6)
    final=json.loads(Path('docs/FINAL_SELECTION.json').read_text());prior=json.loads(Path('docs/V12_SELECTION.json').read_text())
    bundles={'v12':CompetitiveBundle(prior,'mps'),'final':CompetitiveBundle(final,'mps')}
    v6=dict(prior);v6['components']=[dict(s,weight=.5) for s in prior['components'][:2]]
    bundles['v6']=CompetitiveBundle(v6,'mps');profile=challenge_profile(challenge_match());rows=[]
    for game in (1,2,3):
        source=external_match(game);corrected,blocks=alternative_reference(source.meta['events'],len(source.ball))
        if blocks!=source.blocks:raise ValueError('Reference change altered period boundaries')
        for mode in ('clean','moderate','stress'):
            match=sensor_view(source,mode,907+game,profile);r=match_record(match)
            for name,bundle in bundles.items():
                p,e=bundle.probabilities(r)
                codes=(decode_fine(p,e,r['frames'],match,'probability') if name=='final' else dynamic_decode(p,e,r['frames'],match))
                np.savez_compressed(ROOT/f'game{game}_{mode}_{name}.npz',probabilities=p,edges=e,frames=r['frames'])
                for ref,y in [('corrected',corrected),('legacy',source.meta['labels'])]:
                    metric=evaluate_prediction(y,codes);rows.append(dict(game=game,mode=mode,model=name,reference=ref,weak_reference=True,**metric))
                print('PROVIDER CHECK',game,mode,name,'corrected',round(rows[-2]['spell_f1_mean'],4),'legacy',round(rows[-1]['spell_f1_mean'],4),flush=True)
            (ROOT/'results.json').write_text(json.dumps(rows,indent=2))
    summary=[dict(model=name,reference=ref,spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in rows if r['model']==name and r['reference']==ref])),
                  macro_f1=float(np.mean([r['macro_f1'] for r in rows if r['model']==name and r['reference']==ref])))
             for ref in ['corrected','legacy'] for name in bundles]
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print('PROVIDER SUMMARY',summary,flush=True)


if __name__=='__main__':main()
