"""Mixed-domain development: Metrica1/2/3 + native SportecWMX; hold out SportecWN1."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from .sources import external_match,challenge_match
from .sportec_native import external_native_match
from .v3 import load_config,degraded_copy,decode_blocks
from .features_v3 import feature_table
from .sequence_experiment import split_by_halves
from .sequence_tcn import fit
from .sequence_run import match_probabilities,load_sequence_model
from .audit_stress import blind_corruption
from .benchmark_external import evaluate_prediction


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--work-dir',type=Path,default=Path('data/work/sequence_native_mix'))
 parser.add_argument('--train-only',action='store_true')
 args=parser.parse_args()
 root=args.work_dir
 root.mkdir(parents=True,exist_ok=True)
 cfg=load_config('possession/config_external_selected.json')
 xs,ys=[],[]
 for g in (1,2,3):
  x,y=joblib.load(next(Path('data/work/hierarchical_v4').glob(f'training_{g}_*.joblib')))
  seq,labels=split_by_halves(x.to_numpy(),y,external_match(g),repetitions=4)
  xs.extend(seq);ys.extend(labels)
 native=external_native_match('J03WMX')
 for seed in (101,102):
  # Blind degradation does not consult the native reference labels.
  d=blind_corruption(native,seed=seed)
  for swap in (False,True):
   x,frames=feature_table(d,swap=swap)
   labels=native.meta['labels'][frames].astype(int)
   if swap:labels=np.where(labels>=0,np.array([1,0,2])[np.clip(labels,0,2)],labels)
   seq,target=split_by_halves(x.to_numpy(),labels,native)
   xs.extend(seq);ys.extend(target)
 print('Training',len(xs),'sequences: Metrica3 + nativeWMX, 500steps',flush=True)
 model,norm=fit(xs,ys,steps=500,device='mps')
 state={k:v.detach().cpu() for k,v in model.state_dict().items()}
 torch.save(dict(state_dict=state,mean=norm.mean,std=norm.std,steps=500,seed=2027,
                 architecture='TCN48x6',training='Metrica1/2/3 + SportecJ03WMX only'),root/'mixed_model.pt')
 if args.train_only:
  print('Saved model',root/'mixed_model.pt',flush=True)
  return
 baseline=joblib.load('data/work/temporal_v3/model.joblib')
 source=external_native_match('J03WN1')
 results=[]
 for mode in ('clean','blind_stress'):
  match=source if mode=='clean' else blind_corruption(source,seed=907)
  p,frames=match_probabilities(model,norm,match,'mps')
  x,_=feature_table(match);xm,_=feature_table(match,swap=True)
  base=(baseline.predict_proba(x)+baseline.predict_proba(xm)[:,[1,0,2]])/2
  for name,prob in [('mixed_tcn',p),('mixed_ensemble',(p+base)/2)]:
   pred=decode_blocks(prob,frames,match,cfg['decoder'])
   result=dict(game='J03WN1',mode=mode,method=name,**evaluate_prediction(source.meta['labels'],pred))
   results.append(result)
   print(mode,name,result['spell_f1_mean'],result['macro_f1'],flush=True)
  np.savez_compressed(root/f'heldout_{mode}.npz',mixed_tcn=p,baseline=base,frames=frames)
  (root/'results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
