"""Label-free camera synchronization from unordered team player constellations.

Ball trajectories are deliberately absent from synchronization. They are useful
independent diagnostics afterwards. Runs going backwards in match time are rejected.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import find_peaks


@dataclass
class SyncResult:
    target_frame: np.ndarray
    reflection: np.ndarray
    geometry_m: np.ndarray
    runs: list
    proposals: list


def reflect(points,pitch,signs):
    center=np.array(pitch)/2
    return (points-center)*np.array(signs)+center


def descriptors(players,team):
    values=[];valid=np.ones(len(players),dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for side in (0,1):
            p=players[:,team==side]
            valid&=np.isfinite(p).all(2).sum(1)>=4
            for coordinate in (0,1):
                values.append(np.nanquantile(p[:,:,coordinate],[.1,.3,.5,.7,.9],axis=1).T)
    return np.nan_to_num(np.concatenate(values,axis=1),nan=0).astype('float32'),valid


def geometry(source,source_team,target,target_team):
    """Trimmed symmetric Chamfer, separately for each team; metres."""
    result=[]
    for side in (0,1):
        a=source[source_team==side];b=target[target_team==side]
        a=a[np.isfinite(a).all(1)];b=b[np.isfinite(b).all(1)]
        if min(len(a),len(b))<4:return 12.
        distance=np.sqrt(((a[:,None]-b[None])**2).sum(2))
        first=np.minimum(distance.min(1),12);second=np.minimum(distance.min(0),12)
        ka=max(1,int(np.ceil(len(first)*.8)));kb=max(1,int(np.ceil(len(second)*.8)))
        result.append((np.partition(first,ka-1)[:ka].mean()+np.partition(second,kb-1)[:kb].mean())/2)
    return float(np.mean(result))


def proposals(reference,other,reference_valid,other_valid,step=25,k=16):
    rf=np.arange(0,len(reference.ball),step);sf=np.arange(0,len(other.ball),step)
    rd,rv=descriptors(reference.players[rf],reference.team);rv&=reference_valid[rf]
    if rv.sum()<k:raise ValueError('Too few reference constellations')
    tree=cKDTree(rd[rv]);indices=rf[rv];output=[];diagnostics=[]
    for signs in [(1,1),(-1,1),(1,-1),(-1,-1)]:
        positions=reflect(other.players[sf],other.pitch,signs)
        sd,sv=descriptors(positions,other.team);sv&=other_valid[sf]
        chosen=np.flatnonzero(sv)
        if not len(chosen):continue
        neighbors=tree.query(sd[chosen],k=k,workers=2)[1]
        best=[]
        for i,row in zip(chosen,neighbors):
            costs=[geometry(positions[i],other.team,reference.players[indices[j]],reference.team) for j in row]
            j=int(np.argmin(costs))
            if costs[j]<4:best.append((int(sf[i]-indices[row[j]]),float(costs[j])))
        if not best:continue
        seconds=np.rint(np.array([v[0] for v in best])/25).astype(int)
        low,high=int(seconds.min()),int(seconds.max());hist=np.bincount(seconds-low,minlength=high-low+1)
        votes=np.convolve(np.pad(hist,(1,1)),np.ones(3,dtype=int),mode='valid')
        candidates=sorted(find_peaks(np.r_[0,votes,0])[0]-1,key=lambda q:votes[q],reverse=True)
        kept=[]
        for index in candidates:
            if votes[index]<max(5,int(len(chosen)*.005)):continue
            delta=int((index+low)*25)
            if any(abs(delta-old)<100 for old in kept):continue
            kept.append(delta);output.append((int(signs[0]),int(signs[1]),delta))
            if len(kept)>=8:break
        diagnostics.append(dict(reflection=signs,eligible=int(len(chosen)),retrieved_good=len(best),offset_frames=kept))
        print('SYNC proposals',diagnostics[-1],flush=True)
    return output,diagnostics,sf


def synchronize(reference,other,reference_valid=None,other_valid=None):
    nr,ns=len(reference.ball),len(other.ball)
    reference_valid=np.ones(nr,dtype=bool) if reference_valid is None else np.asarray(reference_valid,dtype=bool)
    other_valid=np.ones(ns,dtype=bool) if other_valid is None else np.asarray(other_valid,dtype=bool)
    states,diagnostics,frames=proposals(reference,other,reference_valid,other_valid)
    mapping=np.full(ns,-1,dtype='int32');orientation=np.ones((ns,2),dtype='int8');quality=np.full(ns,np.inf,dtype='float32')
    if not states:return SyncResult(mapping,orientation,quality,[],diagnostics)
    nstate=len(states);cost=np.full((len(frames),nstate+1),12.,dtype='float32')
    cost[:,-1]=np.where(other_valid[frames],4.,0.)
    for j,(sx,sy,delta) in enumerate(states):
        position=reflect(other.players[frames],other.pitch,(sx,sy))
        target=frames-delta
        allowed=other_valid[frames]&(target>=0)&(target<nr)
        allowed&=reference_valid[np.clip(target,0,nr-1)]
        for i in np.flatnonzero(allowed):
            cost[i,j]=geometry(position[i],other.team,reference.players[target[i]],reference.team)
    transition=np.full((nstate+1,nstate+1),8.,dtype='float32');np.fill_diagonal(transition,0)
    transition[-1,:]=1.;transition[:,-1]=1.;transition[-1,-1]=0
    score=cost[0].copy();back=np.zeros_like(cost,dtype='int16')
    for t in range(1,len(frames)):
        previous=score[:,None]+transition;back[t]=previous.argmin(0)
        score=previous[back[t],np.arange(nstate+1)]+cost[t];score-=score.min()
    path=np.empty(len(frames),dtype='int16');current=int(score.argmin())
    for t in range(len(frames)-1,-1,-1):path[t]=current;current=int(back[t,current])
    cuts=np.r_[0,np.flatnonzero(path[1:]!=path[:-1])+1,len(path)];runs=[];last_target=-1
    for a,b in zip(cuts[:-1],cuts[1:]):
        state=int(path[a])
        if state==nstate or b-a<15:continue
        sx,sy,delta=states[state];good=np.arange(a,b)[cost[a:b,state]<4]
        if len(good)<.7*(b-a):continue
        use=good[np.linspace(0,len(good)-1,min(64,len(good))).astype(int)]
        source=reflect(other.players[frames[use]],other.pitch,(sx,sy))
        candidates=[]
        for offset in range(delta-25,delta+26):
            target=frames[use]-offset
            if (target<0).any() or (target>=nr).any():continue
            scores=np.array([geometry(p,other.team,reference.players[t],reference.team) for p,t in zip(source,target)])
            candidates.append((float(np.mean(np.sort(scores)[:max(1,int(len(scores)*.8))])),offset))
        if not candidates:continue
        minimum,offset=min(candidates)
        begin=int(frames[a]);end=int(min(ns,frames[b-1]+25));target_start=begin-offset
        if minimum>=4 or target_start<last_target-25:continue
        source_index=np.arange(begin,end);target=source_index-offset
        valid=other_valid[source_index]&(target>=0)&(target<nr)
        valid&=reference_valid[np.clip(target,0,nr-1)]
        interpolated=np.interp(source_index,frames[a:b],cost[a:b,state])
        valid&=interpolated<4
        chosen=source_index[valid];mapping[chosen]=target[valid];orientation[chosen]=[sx,sy];quality[chosen]=interpolated[valid]
        last_target=end-1-offset
        run=dict(source_first=begin,source_last=end-1,offset_frames=offset,reflection=[sx,sy],geometry_m=minimum,accepted_frames=int(valid.sum()))
        runs.append(run);print('SYNC run',run,flush=True)
    return SyncResult(mapping,orientation,quality,runs,diagnostics)


def fused_ball(reference,other,result,near_gate=None):
    ball=reference.ball.copy();possible=np.flatnonzero((result.target_frame>=0)&np.isfinite(other.ball).all(1))
    possible=possible[np.argsort(result.geometry_m[possible],kind='stable')]
    used=np.zeros(len(reference.ball),dtype=bool);center=np.array(reference.pitch)/2
    for b in possible:
        t=int(result.target_frame[b])
        if used[t] or np.isfinite(ball[t]).all():continue
        xy=(other.ball[b]-np.array(other.pitch)/2)*result.reflection[b]+center
        if near_gate is not None:
            distance=np.linalg.norm(reference.players[t]-xy,axis=1)
            if not np.isfinite(distance).any() or np.nanmin(distance)>near_gate:continue
        ball[t]=xy;used[t]=True
    return reference.copy_with(ball=ball),used
