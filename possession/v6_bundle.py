"""Reproducible inference and one locked confirmation of the selected V6 bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from .sequence_tcn import Normalizer
from .v6_temporal import SegmentNet,predict
from .v6_experiment import subsequences,dynamic_decode
from .features_v3 import feature_table
from .v6_physics import physical_table
from .sources import challenge_match
from .sportec_native import external_native_match
from .v6_data import protocol,sensor_view
from .degrade import challenge_profile
from .v3 import symmetric_proba,decode_blocks,load_config
from .sequence_run import load_sequence_model,match_probabilities
from .sequence_quality_gate import combine,missing_fraction
from .benchmark_external import evaluate_prediction
from starter.baseline import write_submission


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Bundle:
    def __init__(self,selection,device):
        self.selection=selection;self.device=device;self.models=[]
        for spec in selection['components']:
            if digest(spec['path'])!=spec['sha256']:raise ValueError('Changed component weights')
            checkpoint=torch.load(spec['path'],map_location='cpu',weights_only=False)
            norm=Normalizer();norm.mean=np.asarray(checkpoint['mean'],dtype='float32');norm.std=np.asarray(checkpoint['std'],dtype='float32')
            model=SegmentNet(len(norm.mean)*2,local=True,multistage=False)
            model.load_state_dict(checkpoint['state_dict'])
            self.models.append((spec,model.to(device).eval(),norm))

    def probabilities(self,match):
        a,frames=feature_table(match);b,fm=feature_table(match,swap=True)
        assert np.array_equal(frames,fm)
        direct=a.to_numpy();mirror=b.to_numpy()
        lengths=[(z-a)//5+1 for _,a,z in match.blocks];cuts=np.r_[0,np.cumsum(lengths)]
        slices=list(zip(cuts[:-1],cuts[1:]));probs=[];edges=[]
        for spec,model,norm in self.models:
            x,y=direct,mirror
            if spec['physics']:
                x=np.column_stack((x,physical_table(match).to_numpy()))
                y=np.column_stack((y,physical_table(match,True).to_numpy()))
            p,e=predict(model,norm,subsequences(x,slices),self.device)
            q,f=predict(model,norm,subsequences(y,slices),self.device)
            probs.append((np.concatenate(p)+np.concatenate(q)[:,[1,0,2]])/2)
            if spec['boundary_head']:
                edges.append((np.concatenate(e)+np.concatenate(f))/2)
        p=probs[0].copy()
        for q in probs[1:]:p+=q
        p/=len(probs)
        edge=edges[0].copy()
        for e in edges[1:]:edge+=e
        edge/=len(edges)
        if not np.isfinite(p).all() or not np.isfinite(edge).all():raise ValueError('Non-finite predictions')
        return p,edge,frames

    def labels(self,match):
        p,e,f=self.probabilities(match)
        return dynamic_decode(p,e,f,match),dict(probabilities=p,edges=e,frames=f)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selection',type=Path,default=Path('docs/V6_SELECTION.json'))
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--confirm',action='store_true')
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('evaluation/results/v6_selected.csv'))
    parser.add_argument('--work-dir',type=Path,default=Path('data/work/v6_selected'))
    args=parser.parse_args();args.work_dir.mkdir(parents=True,exist_ok=True)
    device=('mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(6)
    selected=json.loads(args.selection.read_text());bundle=Bundle(selected,device)
    if args.confirm:
        report_path=args.work_dir/'confirmation.json'
        if report_path.exists():raise RuntimeError('Confirmation already evaluated; preserve result and do not tune on it')
        old=joblib.load('data/work/domain_adaptation/mixed.joblib')
        oldnet,oldnorm=load_sequence_model('data/work/sequence_native_mix/mixed_model.pt',device)
        cfg=load_config(Path('possession/config_external_selected.json'))
        profile=challenge_profile(challenge_match());rows=[]
        for game in protocol()['native_confirmation_locked']:
            source=external_native_match(game)
            for mode in ('clean','moderate','stress'):
                match=sensor_view(source,mode,1201,profile)
                codes,saved=bundle.labels(match)
                a,frames=symmetric_proba(old,match);b,fb=match_probabilities(oldnet,oldnorm,match,device)
                assert np.array_equal(frames,fb)
                base=combine(a,b,missing_fraction(match,frames))
                previous=decode_blocks(base,frames,match,cfg['decoder'])
                np.savez_compressed(args.work_dir/f'confirmation_{game}_{mode}.npz',baseline=base,**saved)
                for name,pred in [('frozen_previous',previous),('v6_selected',codes)]:
                    metric=evaluate_prediction(source.meta['labels'],pred)
                    rows.append(dict(game=game,mode=mode,method=name,**metric))
                    print('CONFIRMATION',game,mode,name,metric['spell_f1_mean'],metric['macro_f1'],flush=True)
        report_path.write_text(json.dumps(dict(selection=selected,device=device,rows=rows,
            no_training_on_confirmation=True,challenge_labels_read=False),indent=2))
    if args.export:
        match=challenge_match();codes,saved=bundle.labels(match)
        labels=np.array(['H','A','D','X'])[codes]
        write_submission(str(args.output),labels)
        np.savez_compressed(args.work_dir/'challenge_probabilities.npz',**saved)
        report=dict(selection=selected,device=device,rows=len(labels),csv_sha256=digest(args.output),
                    counts={k:int((labels==k).sum()) for k in 'HADX'},challenge_labels_read=False)
        (args.work_dir/'run.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
