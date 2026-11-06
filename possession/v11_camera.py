"""Second-camera diagnostics and external-only synthetic-view evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
from .camera_sync import synchronize,fused_ball
from .sources import Match,challenge_match
from .data import load_tracking
from starter.load_homography import load
from .sportec_native import external_native_match
from .degrade import challenge_profile
from .v6_data import sensor_view,protocol
from .v6_bundle import Bundle
from .benchmark_external import evaluate_prediction

ROOT=Path('data/work/v11')


def save(result,path):
    np.savez_compressed(path,target_frame=result.target_frame,reflection=result.reflection,geometry_m=result.geometry_m)
    path.with_suffix('.json').write_text(json.dumps(dict(runs=result.runs,proposals=result.proposals,challenge_labels_read=False),indent=2))


def challenge():
    ROOT.mkdir(parents=True,exist_ok=True)
    reference=challenge_match();p,b,team=load_tracking('broadcast',ROOT/'cache')
    scale=np.array(reference.pitch,dtype='float32')
    other=Match('broadcast',p[:,:,:2]*scale,team,b[:,:2]*scale,reference.pitch,[(1,0,len(b)-1)])
    _,error,valid=load(source='broadcast')
    reference_valid=np.zeros(len(reference.ball),dtype=bool)
    for _,a,z in reference.blocks:reference_valid[a:z+1]=True
    result=synchronize(reference,other,reference_valid,valid)
    save(result,ROOT/'challenge_alignment.npz')
    known=np.flatnonzero((result.target_frame>=0)&np.isfinite(other.ball).all(1))
    target=result.target_frame[known];both=np.isfinite(reference.ball[target]).all(1)
    xy=(other.ball[known]-scale/2)*result.reflection[known]+scale/2
    disagreement=np.linalg.norm(xy[both]-reference.ball[target[both]],axis=1)
    fused,used=fused_ball(reference,other,result)
    report=dict(accepted_source_frames=int((result.target_frame>=0).sum()),runs=len(result.runs),
        independently_observed_ball_pairs=int(both.sum()),
        ball_disagreement_quantiles_m=np.quantile(disagreement,[.25,.5,.75,.9]).tolist() if len(disagreement) else [],
        new_ball_observations=int(used.sum()),missing_before=float((~np.isfinite(reference.ball[reference_valid]).all(1)).mean()),
        missing_after=float((~np.isfinite(fused.ball[reference_valid]).all(1)).mean()),challenge_labels_read=False,
        possession_score_not_computed=True,alignment_does_not_use_ball=True)
    (ROOT/'challenge_diagnostics.json').write_text(json.dumps(report,indent=2));print('CAMERA DIAGNOSTICS',report,flush=True)


def synthetic_view(source,profile,seed):
    """Known mapping for other games, with blind cuts, replay replacements and masks."""
    view=sensor_view(source,'stress',seed,profile);indices=[]
    for _,a,z in source.blocks:
        length=z-a+1;cut=a+length//2
        indices.extend([np.arange(a+125,cut),np.arange(cut+300,z+1)])
    indices=np.concatenate(indices);truth=indices.copy()
    for start in range(6250,len(indices)-250,6250):
        indices[start:start+250]=indices[start-500:start-250]
        truth[start:start+250]=-1
    valid=np.ones(len(indices),dtype=bool)
    for a in range(2000,len(indices),5000):valid[a:a+125]=False
    players=view.players[indices].copy();ball=view.ball[indices].copy()
    # One reflected camera tests automatic coordinate-convention resolution.
    players=np.array(source.pitch)-players;ball=np.array(source.pitch)-ball
    return Match('synthetic_broadcast',players,source.team,ball,source.pitch,[(1,0,len(ball)-1)]),truth,valid


def external(games):
    import torch
    torch.set_num_threads(6);ROOT.mkdir(parents=True,exist_ok=True)
    bundle=Bundle(json.loads(Path('docs/V6_SELECTION.json').read_text()),'cpu')
    profile=challenge_profile(challenge_match());rows=[];alignment=[]
    for i,game in enumerate(games):
        if game not in protocol()['native_development']:raise ValueError('Development games only')
        i=protocol()['native_development'].index(game)
        source=external_native_match(game);base=sensor_view(source,'stress',907+i,profile)
        other,truth,valid=synthetic_view(source,profile,5021+i)
        eligible=np.zeros(len(source.ball),dtype=bool)
        for _,a,z in source.blocks:eligible[a:z+1]=True
        result=synchronize(base,other,eligible,valid);save(result,ROOT/f'{game}_alignment.npz')
        accepted=result.target_frame>=0;live=truth>=0
        errors=np.abs(result.target_frame[accepted&live]-truth[accepted&live])
        stats=dict(game=game,coverage=float(accepted[live&valid].mean()),
                   accepted_replay_fraction=float(accepted[~live].mean()) if (~live).any() else 0,
                   error_frames_quantiles=np.quantile(errors,[.5,.9,.99]).tolist() if len(errors) else [],
                   within_5_frames=float((errors<=5).mean()) if len(errors) else None)
        alignment.append(stats);print('ALIGNMENT TEST',stats,flush=True)
        for name,gate in [('baseline',None),('fill_missing',None),('fill_near8',8.)]:
            match,used=(base,np.zeros(len(base.ball),dtype=bool)) if name=='baseline' else fused_ball(base,other,result,gate)
            codes,_=bundle.labels(match);metrics=evaluate_prediction(source.meta['labels'],codes)
            row=dict(game=game,mode='stress',method=name,new_observations=int(used.sum()),**metrics)
            rows.append(row);print('SECOND CAMERA',row,flush=True)
        (ROOT/'external_results.json').write_text(json.dumps(dict(rows=rows,alignment=alignment,challenge_labels_read=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--challenge',action='store_true');p.add_argument('--external',nargs='+');args=p.parse_args()
    if args.challenge:challenge()
    if args.external:external(args.external)
