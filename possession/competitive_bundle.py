"""Frozen four-network possession inference with label-free camera alignment.

No competition ground truth is imported. Existing CSVs are replaced only when the
new file is byte-identical; different results must use a different output path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .sources import Match,challenge_match
from .data import load_tracking
from starter.load_homography import load as load_homography
from starter.baseline import write_submission
from games import asset_paths
from .camera_sync import SyncResult,synchronize,fused_ball
from .v10_relational import load_model,predict as relational_predict
from .v6_temporal import SegmentNet,predict as temporal_predict
from .sequence_tcn import Normalizer
from .v6_experiment import subsequences,dynamic_decode
from .v12_compare import match_record
from .v14_resolution import decode_fine


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def input_signature():
    output={}
    for source in ['tactical','broadcast']:
        assets=asset_paths(source=source)
        for key in ['atd_raw','atd_meta','homography']:
            p=Path(assets[key]);output[f'{source}/{p.name}']=digest(p)
    return output


def mapping_digest(result):
    h=hashlib.sha256();h.update(result.target_frame.astype('<i4').tobytes());h.update(result.reflection.astype('int8').tobytes())
    return h.hexdigest()


def read_alignment(path):
    x=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
    return SyncResult(x['target_frame'],x['reflection'],x['geometry_m'],meta['runs'],meta['proposals'])


def camera_match(reference,work):
    p,b,team=load_tracking('broadcast',work/'tracking_cache');scale=np.array(reference.pitch,dtype='float32')
    other=Match('broadcast',p[:,:,:2]*scale,team,b[:,:2]*scale,reference.pitch,[(1,0,len(b)-1)])
    _,_,valid=load_homography(source='broadcast')
    return other,valid


def align(reference,other,valid,work,path=None):
    if path is not None:return read_alignment(path)
    eligible=np.zeros(len(reference.ball),dtype=bool)
    for _,a,b in reference.blocks:eligible[a:b+1]=True
    result=synchronize(reference,other,eligible,valid)
    np.savez_compressed(work/'alignment.npz',target_frame=result.target_frame,reflection=result.reflection,geometry_m=result.geometry_m)
    (work/'alignment.json').write_text(json.dumps(dict(runs=result.runs,proposals=result.proposals),indent=2))
    return result


class CompetitiveBundle:
    def __init__(self,selection,device='cpu'):
        self.selection=selection;self.device=device;self.models=[]
        weights=[c['weight'] for c in selection['components']]
        if not np.isclose(sum(weights),1.) or min(weights)<0:raise ValueError('Invalid probability weights')
        for spec in selection['components']:
            if digest(spec['path'])!=spec['sha256']:raise ValueError('Changed model checkpoint: '+spec['path'])
            if spec['kind']=='trajectory':net,norm=load_model(spec['path'],device)
            else:
                c=torch.load(spec['path'],map_location='cpu',weights_only=False);norm=Normalizer()
                norm.mean=np.asarray(c['mean'],dtype='float32');norm.std=np.asarray(c['std'],dtype='float32')
                net=SegmentNet(len(norm.mean)*2,local=True);net.load_state_dict(c['state_dict']);net=net.to(device).eval()
            self.models.append((spec,net,norm))

    def probabilities(self,r):
        probabilities=[];boundaries=[];weights=[]
        for spec,net,norm in self.models:
            if spec['kind']=='trajectory':p,e,_=relational_predict(net,norm,r,self.device)
            else:
                width=len(norm.mean)
                a,e=temporal_predict(net,norm,subsequences(r['direct'][:,:width],r['slices']),self.device)
                b,f=temporal_predict(net,norm,subsequences(r['mirror'][:,:width],r['slices']),self.device)
                p=(np.concatenate(a)+np.concatenate(b)[:,[1,0,2]])/2;e=(np.concatenate(e)+np.concatenate(f))/2
            probabilities.append(p);weights.append(spec['weight'])
            if spec['boundary_head']:boundaries.append(e)
        if not boundaries:raise ValueError('No registered boundary expert')
        if np.allclose(weights,np.full(len(weights),1/len(weights))):p=np.mean(probabilities,axis=0)
        else:p=np.sum([w*q for w,q in zip(weights,probabilities)],axis=0)
        e=np.mean(boundaries,axis=0)
        if not np.isfinite(p).all() or not np.isfinite(e).all():raise ValueError('Nonfinite prediction')
        return p,e

    def labels(self,match):
        r=match_record(match);p,e=self.probabilities(r)
        kind=self.selection.get('decoder_kind','coarse')
        if kind=='coarse':codes=dynamic_decode(p,e,r['frames'],match)
        elif kind in ('fine_probability','fine_log'):codes=decode_fine(p,e,r['frames'],match,kind.removeprefix('fine_'))
        else:raise ValueError('Unregistered decoder')
        return codes,dict(probabilities=p,edges=e,frames=r['frames'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selection',type=Path,default=Path('docs/FINAL_SELECTION.json'))
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu');parser.add_argument('--alignment',type=Path)
    parser.add_argument('--output',type=Path,default=Path('evaluation/results/final_candidate.csv'))
    parser.add_argument('--work-dir',type=Path,default=Path('data/work/final_run'))
    args=parser.parse_args();args.work_dir.mkdir(parents=True,exist_ok=True);torch.set_num_threads(6)
    selection=json.loads(args.selection.read_text());signature=input_signature()
    if selection.get('input_files_sha256')!=signature:raise ValueError('Benchmark input fingerprint changed')
    if selection.get('alignment_algorithm_sha256')!=digest('possession/camera_sync.py'):raise ValueError('Changed alignment algorithm')
    reference=challenge_match();other,valid=camera_match(reference,args.work_dir)
    alignment=align(reference,other,valid,args.work_dir,args.alignment)
    if mapping_digest(alignment)!=selection['alignment_mapping_sha256']:raise ValueError('Different automatic camera mapping')
    match,filled=fused_ball(reference,other,alignment)
    bundle=CompetitiveBundle(selection,args.device);codes,saved=bundle.labels(match)
    labels=np.array(['H','A','D','X'])[codes];args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.part.csv');write_submission(str(temporary),labels)
    new_hash=digest(temporary)
    if args.output.exists():
        if digest(args.output)!=new_hash:raise RuntimeError('Output differs from existing candidate; use a new path')
        temporary.unlink()
    else:temporary.replace(args.output)
    np.savez_compressed(args.work_dir/'probabilities.npz',**saved)
    report=dict(selection=selection,device=args.device,csv_sha256=new_hash,rows=len(labels),new_ball_observations=int(filled.sum()),
        counts={k:int((labels==k).sum()) for k in 'HADX'},challenge_labels_read=False,public_score_computed=False)
    (args.work_dir/'run.json').write_text(json.dumps(report,indent=2));print('FROZEN CANDIDATE',json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
