"""Pinned SkillCorner broadcast observations and partial algorithmic H/A teachers.

Unknown possession is NOT dead ball. Neither possession nor phases/events are used
as input features. Match metadata identifies slots and halves, never target frames.
"""
import json
from pathlib import Path
import numpy as np
from .sources import Match

ROOT = Path('data/external/skillcorner_v7')


def partial_label(group):
    return {'home': 0, 'away': 1, 'home team': 0, 'away team': 1}.get(group, -1)


def resample_positions(array):
    """10 -> 25 Hz without interpolating across absent coordinate observations."""
    ticks = np.arange(int(np.floor((len(array)-1)*2.5))+1, dtype=float)*.4
    left = np.floor(ticks+1e-9).astype(int)
    right = np.minimum(left+1, len(array)-1)
    weight = (ticks-left).reshape((-1,)+(1,)*(array.ndim-1))
    interpolated = array[left]*(1-weight)+array[right]*weight
    return np.where(weight < 1e-8, array[left], interpolated).astype('float32'), left


def load_match(game, detected_only=False, allow_confirmation=False):
    protocol = json.loads(Path('docs/PROTOCOLO_V7.json').read_text())
    if game in protocol['skillcorner_confirmation_locked'] and not allow_confirmation:
        raise ValueError('Locked confirmation game')
    folder=ROOT/str(game)
    metadata=json.loads((folder/f'{game}_match.json').read_text())
    if metadata['competition_edition']['competition']['id'] != 61:
        raise ValueError('Unexpected competition')
    people=metadata['players'];slot={p['id']:i for i,p in enumerate(people)}
    home=metadata['home_team']['id'];away=metadata['away_team']['id']
    team=np.array([0 if p['team_id']==home else 1 if p['team_id']==away else -1 for p in people],dtype='int8')
    if (team<0).any():raise ValueError('Unknown player team')
    n=max(p['end_frame'] for p in metadata['match_periods'])
    players=np.full((n,len(slot),2),np.nan,dtype='float32')
    ball=np.full((n,2),np.nan,dtype='float32');labels=np.full(n,-1,dtype='int8')
    periods=np.zeros(n,dtype='int8');previous=-1
    counts={};detected=0;finite=0
    with (folder/f'{game}_tracking_extrapolated.jsonl').open() as handle:
        for line in handle:
            row=json.loads(line);f=row['frame']
            if f != previous+1:raise ValueError('Nonconsecutive tracking frame')
            previous=f
            if f>=n:continue
            periods[f]=row['period'] or 0
            group=row['possession']['group'];counts[str(group)]=counts.get(str(group),0)+1
            if group not in (None,'home','away','home team','away team'):
                raise ValueError('Undocumented possession group: '+str(group))
            labels[f]=partial_label(group)
            b=row['ball_data']
            if b['x'] is not None and b['y'] is not None:
                finite+=1;detected+=bool(b['is_detected'])
                if not detected_only or b['is_detected']:ball[f]=[b['x'],b['y']]
            for player in row['player_data']:
                if player['player_id'] not in slot:raise ValueError('Unknown player ID')
                if not detected_only or player['is_detected']:
                    players[f,slot[player['player_id']]]=[player['x'],player['y']]
    pitch=(float(metadata['pitch_length']),float(metadata['pitch_width']))
    shift=np.array(pitch,dtype='float32')/2
    pp=[];bb=[];yy=[];blocks=[];offset=0
    for period in metadata['match_periods']:
        a,z=period['start_frame'],period['end_frame']
        if not np.all(periods[a:z]==period['period']):
            raise ValueError('Metadata/tracking period misalignment')
        p,index=resample_positions(players[a:z]);b,bi=resample_positions(ball[a:z])
        assert np.array_equal(index,bi)
        pp.append(p+shift);bb.append(b+shift);yy.append(labels[a:z][index])
        blocks.append((period['period'],offset,offset+len(p)-1));offset+=len(p)
        pp.append(np.full((1,len(slot),2),np.nan,dtype='float32'))
        bb.append(np.full((1,2),np.nan,dtype='float32'));yy.append(np.array([-1],dtype='int8'));offset+=1
    return Match(f'skillcorner_{game}',np.concatenate(pp),team,np.concatenate(bb),pitch,blocks,
                 dict(labels=np.concatenate(yy),teacher='SkillCorner algorithmic possession.group',
                      detected_only=detected_only,raw_group_counts=counts,raw_ball_finite=finite,
                      raw_ball_detected=detected,metadata_periods=metadata['match_periods']))
