"""Compare observation distributions without loading competition ground truth."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
from possession.data import load_tracking
from possession.external_training import read_game, feature_matrix, FEATURE_NAMES, occlude
from games import half_video_blocks, GAME

root=Path('data/external/metrica_sample')
p,b,_=read_game(root,2)
external=feature_matrix(p,occlude(b,2029))[::25]
p,b,_=load_tracking(cache_dir='data/work/cache')
challenge=feature_matrix(p,b)
idx=np.concatenate([np.arange(a,z+1,25) for _,a,z in half_video_blocks(GAME)])
challenge=challenge[idx]
report=[]
for j,name in enumerate(FEATURE_NAMES):
 x,y=external[:,j],challenge[:,j]
 xf,yf=x[np.isfinite(x)],y[np.isfinite(y)]
 report.append({'feature':name,'external_missing':float((~np.isfinite(x)).mean()),
 'challenge_missing':float((~np.isfinite(y)).mean()),
 'external_q10_q50_q90':np.percentile(xf,[10,50,90]).tolist() if len(xf) else [],
 'challenge_q10_q50_q90':np.percentile(yf,[10,50,90]).tolist() if len(yf) else [],
 'ks_distance':float(ks_2samp(xf,yf).statistic) if len(xf) and len(yf) else None})
report.sort(key=lambda r:r['ks_distance'] or 0,reverse=True)
out=Path('data/work/transfer_audit.json')
out.write_text(json.dumps({'competition_labels_read':False,'accuracy_estimate':False,
 'note':'Distribution differences are diagnostic, not an error rate. External data includes stoppages and live play.',
 'features':report},indent=2))
for r in report: print(r['feature'],'KS',round(r['ks_distance'],3),'external',r['external_q10_q50_q90'],'challenge',r['challenge_q10_q50_q90'])
