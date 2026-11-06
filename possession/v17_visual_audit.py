"""Plot external failures for inspection, with native references and automatic inputs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from possession import v13_cv
from possession.v14_resolution import decode_fine
from possession.v6_experiment import as_match
from possession.sportec_native import external_native_match

def main():
    root=Path('data/work/v17/visual_audit');root.mkdir(parents=True,exist_ok=True)
    audits=json.loads(Path('data/work/v17/error_audit.json').read_text());out=[]
    for audit in audits:
        if audit['mode']!='clean':continue
        game=audit['game'];r=v13_cv.record(game,'clean');s=np.load(v13_cv.ROOT/'fold0'/f'{game}_clean.npz')
        pred=decode_fine(s['combined'],s['edges'],s['frames'],as_match(r));native=external_native_match(game)
        selected=[z for z in audit['spells'] if not z['matched'] and .5<z['duration']<2 and z['reason']=='merged'][:3]
        for z in selected:
            a,b=z['start'],z['stop'];lo=max(0,a-100);hi=min(len(pred),b+100)
            frames=np.arange(lo,hi);time=(frames-a)/25;use=(s['frames']>=lo)&(s['frames']<hi);t=(s['frames'][use]-a)/25
            fig,axes=plt.subplots(4,1,figsize=(11,8),sharex=True,layout='constrained')
            axes[0].step(time,r['full_labels'][frames],where='post',label='Native reference');axes[0].step(time,pred[frames]+.04,where='post',label='Prediction',alpha=.8)
            axes[0].set_yticks([0,1,2],['Home','Away','Dead']);axes[0].legend(loc='upper right');axes[0].set_title(f'{game}: missed {z["duration"]:.2f}s possession; best IoU {z["best_iou"]:.3f}')
            for j,label in enumerate(['Home','Away','Dead']):axes[1].plot(t,s['combined'][use,j],label=label)
            axes[1].plot(t,s['edges'][use],':',label='Boundary');axes[1].legend(loc='upper right',ncols=4);axes[1].set_ylabel('Probability')
            for i,label in [(6,'Nearest home'),(10,'Nearest away')]:axes[2].plot(t,r['direct'][use,i],label=label)
            axes[2].set_ylabel('Ball distance (m)');axes[2].set_ylim(0,12);axes[2].legend(loc='upper right')
            for i,label in [(4,'Ball speed'),(16,'Median player speed')]:axes[3].plot(t,r['direct'][use,i],label=label)
            axes[3].set_ylabel('m/s');axes[3].legend(loc='upper right');axes[3].set_xlabel('Seconds relative to native possession start')
            for ax in axes:ax.axvspan(0,(b-a)/25,color='gold',alpha=.18);ax.grid(alpha=.2)
            path=root/f'{game}_{a}.png';fig.savefig(path,dpi=130);plt.close(fig)
            # Input-level evidence: nearest-team assignment is not assumed to be control.
            pp=native.players[a:b];bb=native.ball[a:b];dist=np.linalg.norm(pp-bb[:,None],axis=2)
            h=np.nanmin(dist[:,native.team==0],axis=1);aw=np.nanmin(dist[:,native.team==1],axis=1)
            out.append(dict(game=game,**z,nearest_team_agrees_with_reference=float(np.mean((h<aw)==(z['team']==0))),plot=str(path)))
    (root/'cases.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
