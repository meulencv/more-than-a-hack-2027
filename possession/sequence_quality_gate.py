"""Observation-quality mixture: native-trained temporal expert for missing-ball windows.

Fixed physical hypothesis: TCN weight is local ball-missing fraction in a5s
centered half-local window. No reference labels enter this gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .sportec_native import external_native_match
from .audit_stress import blind_corruption
from .sources import challenge_match
from .v3 import load_config,decode_blocks,symmetric_proba
from .sequence_run import load_sequence_model,match_probabilities
from .benchmark_external import evaluate_prediction
from starter.baseline import write_submission


def missing_fraction(match,frames,window_frames=125):
    weight=np.zeros(len(match.ball),dtype=float)
    missing=~np.isfinite(match.ball).all(axis=1)
    for _,start,end in match.blocks:
        weight[start:end+1]=pd.Series(missing[start:end+1].astype(float)).rolling(
            window_frames,center=True,min_periods=1).mean().to_numpy()
    return weight[np.asarray(frames,dtype=int)]


def combine(hgb,tcn,weight):
    hgb,tcn,weight=np.asarray(hgb),np.asarray(tcn),np.asarray(weight)
    if hgb.shape!=tcn.shape or hgb.ndim!=2 or hgb.shape[1]!=3 or weight.shape!=(len(hgb),):
        raise ValueError('Misaligned expert probabilities or weights')
    if not np.isfinite(hgb).all() or not np.isfinite(tcn).all() or not np.isfinite(weight).all():
        raise ValueError('Non-finite expert inputs')
    if (weight<0).any() or (weight>1).any() or (hgb<0).any() or (tcn<0).any():
        raise ValueError('Invalid expert probabilities or weights')
    p=(1-weight[:,None])*hgb+weight[:,None]*tcn
    if (p.sum(axis=1)<=0).any(): raise ValueError('Zero expert mass')
    return p/p.sum(axis=1,keepdims=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('evaluation/results/sequence_quality_gated.csv'))
    parser.add_argument('--work-dir',type=Path,default=Path('data/work/sequence_quality_gated'))
    args=parser.parse_args()
    device=('mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(8)
    args.work_dir.mkdir(parents=True,exist_ok=True)
    cfg=load_config('possession/config_external_selected.json')
    rows=[]
    if not args.export_only:
        source=external_native_match('J03WN1')
        for mode in ('clean','blind_stress'):
            match=source if mode=='clean' else blind_corruption(source,seed=907)
            a=np.load(f'data/work/domain_adaptation/mixed_hgb_{mode}.npz')
            b=np.load(f'data/work/sequence_native_mix/heldout_{mode}.npz')
            if not np.array_equal(a['frames'],b['frames']):raise ValueError('Misaligned saved frames')
            frames=a['frames']
            weight=missing_fraction(match,frames)
            p=combine(a['probabilities'],b['mixed_tcn'],weight)
            for method,prob in [('mixed_hgb',a['probabilities']),('mixed_tcn',b['mixed_tcn']),('quality_gate',p)]:
                pred=decode_blocks(prob,frames,match,cfg['decoder'])
                record=dict(game='J03WN1',mode=mode,method=method,mean_tcn_weight=float(weight.mean()),
                            **evaluate_prediction(source.meta['labels'],pred))
                rows.append(record)
                print(mode,method,record['spell_f1_mean'],record['macro_f1'],flush=True)
            np.savez_compressed(args.work_dir/f'heldout_{mode}.npz',probabilities=p,frames=frames,weight=weight)
        (args.work_dir/'native_results.json').write_text(json.dumps(rows,indent=2))
    if args.export or args.export_only:
        match=challenge_match()
        hgb_path=Path('data/work/domain_adaptation/mixed.joblib')
        tcn_path=Path('data/work/sequence_native_mix/mixed_model.pt')
        hgb=joblib.load(hgb_path)
        model,norm=load_sequence_model(tcn_path,device)
        a,frames=symmetric_proba(hgb,match)
        b,fb=match_probabilities(model,norm,match,device)
        if not np.array_equal(frames,fb):raise ValueError('Expert frames misaligned')
        weight=missing_fraction(match,frames)
        p=combine(a,b,weight)
        pred=decode_blocks(p,frames,match,cfg['decoder'])
        labels=np.array(['H','A','D','X'])[pred]
        write_submission(str(args.output),labels)
        np.savez_compressed(args.work_dir/'probabilities.npz',probabilities=p,frames=frames,weight=weight)
        report={'method':'quality-gated native/Metrica-trained HGB+TCN',
                'gate':'TCNweight=ball-missingfraction125frames; centered and half-local',
                'model_hgb_sha256':hashlib.sha256(hgb_path.read_bytes()).hexdigest(),
                'model_tcn_sha256':hashlib.sha256(tcn_path.read_bytes()).hexdigest(),
                'competition_labels_read':False,'mean_tcn_weight':float(weight.mean()),'inference_device':device,
                'decoder':cfg['decoder'],'stage2':cfg.get('stage2'),
                'gate_window_frames':125,'fps':25,
                'gate_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'training_ids':['Metrica1','Metrica2','Metrica3','SportecJ03WMX'],
                'validation_id':'SportecJ03WN1: external development; never fitted',
                'hgb_targets':'corrected weak Metrica + native Sportec',
                'tcn_targets':'legacy weak Metrica + native Sportec',
                'provenance_files':['docs/external_data_manifest.json','data/work/domain_adaptation/provenance.json'],
                'csv_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
                'counts':{k:int((labels==k).sum()) for k in 'HADX'}}
        (args.work_dir/'run.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
