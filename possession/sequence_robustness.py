"""Second-occlusion-seed test of already-frozen temporal fold checkpoints."""
import json
from pathlib import Path
import numpy as np
from .sources import external_match,challenge_match
from .degrade import challenge_profile,degrade
from .sequence_run import load_sequence_model,match_probabilities
from .audit_reference import alternative_reference
from .v3 import load_config,decode_blocks
from .benchmark_external import evaluate_prediction
import joblib
from .features_v3 import feature_table


def main():
 root=Path('data/work/sequence_tcn')
 profile=challenge_profile(challenge_match())
 cfg=load_config('possession/config_external_selected.json')
 results=[]
 for game in (1,2,3):
  source=external_match(game)
  match=degrade(source,profile,seed=202711+game,sigma=1.1,p_live=.35,p_dead_sighting=.5)
  model,norm=load_sequence_model(root/f'fold_{game}.pt','mps')
  p,frames=match_probabilities(model,norm,match,'mps')
  baseline=joblib.load(Path('data/work/hierarchical_v4')/f'fold_{game}.joblib')['baseline']
  x,_=feature_table(match);xm,_=feature_table(match,swap=True)
  base=(baseline.predict_proba(x)+baseline.predict_proba(xm)[:,[1,0,2]])/2
  corrected,_=alternative_reference(source.meta['events'],len(source.ball))
  for method,prob in [('tcn',p),('tcn_ensemble',(p+base)/2),('v3',base)]:
   pred=decode_blocks(prob,frames,match,cfg['decoder'],stage2_cfg=cfg.get('stage2'))
   for ref,y in [('legacy',source.meta['labels']),('corrected',corrected)]:
    record=dict(game=game,method=method,reference=ref,**evaluate_prediction(y,pred))
    results.append(record)
    print(game,method,ref,round(record['spell_f1_mean'],4),flush=True)
  np.savez_compressed(root/f'probabilities_robustness_{game}.npz',tcn=p,baseline=base,frames=frames)
  (root/'robustness.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
