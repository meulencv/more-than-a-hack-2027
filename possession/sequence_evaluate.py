"""Re-evaluate frozen temporal probabilities against both external references."""
import argparse
import json
from pathlib import Path
import numpy as np
from .sources import external_match
from .v3 import load_config, decode_blocks
from .audit_reference import alternative_reference
from .benchmark_external import evaluate_prediction


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--work-dir',type=Path,default=Path('data/work/sequence_tcn'))
 args=parser.parse_args()
 cfg=load_config('possession/config_external_selected.json')
 results=[]
 for game in (1,2,3):
  path=args.work_dir/f'probabilities_{game}.npz'
  if not path.exists(): continue
  p=np.load(path)
  match=external_match(game)
  corrected,_=alternative_reference(match.meta['events'],len(match.ball))
  for method,prob in [('tcn',p['tcn']),('tcn_ensemble',(p['tcn']+p['baseline'])/2),('v3',p['baseline'])]:
   pred=decode_blocks(prob,p['frames'],match,cfg['decoder'],stage2_cfg=cfg.get('stage2'))
   for reference,y in [('legacy',match.meta['labels']),('corrected',corrected)]:
    result=dict(game=game,method=method,reference=reference,**evaluate_prediction(y,pred))
    results.append(result)
    print(game,method,reference,round(result['spell_f1_mean'],4),flush=True)
 (args.work_dir/'reference_audit_results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
