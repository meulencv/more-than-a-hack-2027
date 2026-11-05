"""Read native Sportec/DLF possession and alive/dead annotations from OTHER games.

Source: https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0).
Native mapping agrees with PySport/kloppy SportecTrackingDataDeserializer:
BallPossession 1=home, 2=away; BallStatus 1=alive, 0=dead. Unknowns stay -1.
These reference channels are stored only in Match.meta and never features.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .sources import Match

ROOT = Path('data/external/pathcrf')
SECTION = {'firstHalf': 1, 'secondHalf': 2, 'firstHalfExtra': 3, 'secondHalfExtra': 4}
VERSION = 2


def native_label(status, possession):
    if status == '0':
        return 2
    if status == '1' and possession in ('1', '2'):
        return int(possession) - 1
    return -1


def _frames(path, ball_only):
    """Two sequential streaming passes, never building a whole-match XML tree."""
    current = None
    for event, elem in ET.iterparse(path, events=('start', 'end')):
        if event == 'start' and elem.tag == 'FrameSet':
            current = (SECTION.get(elem.get('GameSection')), elem.get('TeamId'), elem.get('PersonId'))
        elif event == 'end' and elem.tag == 'Frame':
            if current and current[0] is not None and ((current[1] == 'BALL') == ball_only):
                yield current, dict(elem.attrib)
            elem.clear()
        elif event == 'end' and elem.tag == 'FrameSet':
            current = None
            elem.clear()


def external_native_match(game_id, root=ROOT):
    root = Path(root)
    if not isinstance(game_id, str) or not game_id.isalnum():
        raise ValueError('Use an alphanumeric Sportec match ID, e.g. J03WMX')
    metas = list((root / 'sportec/metadata').glob('*' + game_id + '.xml'))
    tracks = list((root / 'sportec/tracking').glob('*' + game_id + '.xml'))
    if len(metas) != 1 or len(tracks) != 1:
        raise FileNotFoundError('Expected one completed metadata/tracking XML for ' + game_id)
    metadata, tracking = metas[0], tracks[0]
    signature = [VERSION, metadata.stat().st_size, metadata.stat().st_mtime_ns,
                 tracking.stat().st_size, tracking.stat().st_mtime_ns]
    cache_dir = root / 'native_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (game_id + '.npz')
    info_path = cache.with_suffix('.json')
    if cache.exists() and info_path.exists():
        info = json.loads(info_path.read_text())
        if info.get('signature') == signature:
            with np.load(cache) as a:
                return Match(game_id, a['players'], a['team'], a['ball'], tuple(info['pitch']),
                             [tuple(b) for b in info['blocks']],
                             dict(labels=a['labels'], native_frame_ids=a['native_frame_ids'],
                                  provenance=info))
    meta = ET.parse(metadata).getroot()
    general = next(meta.iter('General'))
    title = general.get('MatchTitle', '')
    if any(s in title.lower() for s in ('barcelona', 'athletic')):
        raise ValueError('Competition match must never enter external reference loader')
    env = next(meta.iter('Environment'))
    pitch = (float(env.get('PitchX')), float(env.get('PitchY')))
    teams, players, player_ids = [], {}, []
    for side in ('home', 'guest'):
        selected = [t for t in meta.iter('Team') if t.get('Role') == side]
        if len(selected) != 1:
            raise ValueError('Metadata must identify exactly one home and one guest team')
        team = selected[0]
        for player in team.iter('Player'):
            player_ids.append(player.get('PersonId'))
            players[player.get('PersonId')] = len(teams)
            teams.append(0 if side == 'home' else 1)
    rows = []
    per_period = {}
    for (period, _, _), f in _frames(tracking, True):
        frame = int(f['N'])
        per_period.setdefault(period, []).append(frame)
        rows.append((period, frame, f))
    if not rows:
        raise ValueError('No native ball records found')
    blocks, offsets, count = [], {}, 0
    for period in sorted(per_period):
        # Official metrics infer independent blocks from X gaps. A single unknown
        # separator preserves half separation without inventing halftime observations.
        if blocks:
            count += 1
        ids = per_period[period]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate native ball frame in period ' + str(period))
        first, last = min(ids), max(ids)
        length = last - first + 1
        if not (25 * 60 <= length <= 25 * 90 * 60):
            raise ValueError('Implausible period length: ' + str(length))
        offsets[period] = count - first
        blocks.append((period, count, count + length - 1))
        count += length
    observed = np.full((count, len(teams), 2), np.nan, dtype=np.float32)
    ball = np.full((count, 2), np.nan, dtype=np.float32)
    labels = np.full(count, -1, dtype=np.int8)
    native_ids = np.full(count, -1, dtype=np.int64)
    for period, first, last in blocks:
        native_ids[first:last + 1] = np.arange(first, last + 1) - offsets[period]
    shift = np.array(pitch, dtype=np.float32) / 2
    for period, frame, f in rows:
        index = frame + offsets[period]
        ball[index] = [float(f['X']), float(f['Y'])]
        labels[index] = native_label(f.get('BallStatus'), f.get('BallPossession'))
    ball += shift
    for (period, team_id, player), f in _frames(tracking, False):
        if player not in players or period not in offsets:
            continue
        index = int(f['N']) + offsets[period]
        first, last = next((a, b) for p, a, b in blocks if p == period)
        if first <= index <= last:
            observed[index, players[player]] = [float(f['X']) + shift[0], float(f['Y']) + shift[1]]
    info = dict(signature=signature, source='https://doi.org/10.6084/m9.figshare.28196177',
                license='CC BY 4.0', title=title, date=general.get('KickoffTime'),
                competition=general.get('CompetitionName'), native_reference=True,
                label_mapping='BallStatus0:D; BallStatus1 and BallPossession1:H / 2:A; unknown:-1',
                mapping_reference='https://github.com/PySport/kloppy/blob/master/kloppy/infra/serializers/tracking/sportec/deserializer.py',
                blocks=blocks, pitch=pitch, player_ids=player_ids,
                label_counts={str(k): int((labels == k).sum()) for k in (-1, 0, 1, 2)},
                inter_period_separator_frames=len(blocks) - 1,
                native_missing_ball_records=int(count - len(rows) - len(blocks) + 1),
                files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (metadata, tracking)})
    np.savez_compressed(cache, players=observed, team=np.asarray(teams, dtype=np.int8),
                        ball=ball, labels=labels, native_frame_ids=native_ids)
    info_path.write_text(json.dumps(info, indent=2))
    return Match(game_id, observed, np.asarray(teams, dtype=np.int8), ball, pitch, blocks,
                 dict(labels=labels, native_frame_ids=native_ids, provenance=info))
