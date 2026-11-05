"""Compare predefined temporal hypotheses on held-out external probabilities only."""
import json
from pathlib import Path
import numpy as np
from .sources import external_match
from .v3 import decode_blocks
from .benchmark_external import evaluate_prediction


def main():
 root=Path('data/work/hierarchical_v4')
 cfg=json.loads((root/'config.json').read_text())
 options={
 'current':(cfg['decoder'],cfg.get('stage2')),
 'original_v3':(dict(min_steps=[5,5,5],cost_turnover=4.,cost_to_dead=2.5,cost_to_live=2.5,emission_weight=1.),None),
 'short_control':(dict(min_steps=[3,3,3],cost_turnover=4.,cost_to_dead=2.5,cost_to_live=2.5,emission_weight=1.),None),
 'light_temporal':(dict(min_steps=[3,3,3],cost_turnover=2.,cost_to_dead=1.5,cost_to_live=1.5,emission_weight=1.),None)}
 results=[]
 for game in (1,2,3):
  match=external_match(game)
  for seed in (42+game,202711+game):
   p=np.load(root/f'probabilities_{game}_{seed}.npz')
   for name,(dec,stage2) in options.items():
    pred=decode_blocks(p['baseline'],p['frames'],match,dec,stage2_cfg=stage2)
    metric=evaluate_prediction(match.meta['labels'],pred)
    results.append(dict(game=game,seed=seed,method=name,**metric))
 summary={k:float(np.mean([r['spell_f1_mean'] for r in results if r['method']==k])) for k in options}
 (root/'decoder_results.json').write_text(json.dumps(dict(options=options,results=results,means=summary),indent=2))
 print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
