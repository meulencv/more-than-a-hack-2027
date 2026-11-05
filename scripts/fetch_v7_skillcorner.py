"""Fetch pinned public broadcast tracks, verifying their Git LFS object hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path('data/external/skillcorner_v7')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, target, expected=None):
    if target.exists() and (expected is None or sha256(target) == expected):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix+'.part')
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
        while chunk := response.read(1024*1024):
            out.write(chunk)
    if expected is not None and sha256(temporary) != expected:
        raise ValueError('Source checksum mismatch: '+str(target))
    temporary.replace(target)


def main(include_confirmation=False):
    p = json.loads(Path('docs/PROTOCOLO_V7.json').read_text())
    commit = p['source_commit']
    raw = f'https://raw.githubusercontent.com/SkillCorner/opendata/{commit}'
    media = f'https://media.githubusercontent.com/media/SkillCorner/opendata/{commit}'
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ['LICENSE', 'README.md', 'data/matches.json']:
        fetch(raw+'/'+name, ROOT/name.replace('/', '_'))
    matches = json.loads((ROOT/'data_matches.json').read_text())
    allowed = {m['id'] for m in matches if m['competition_id'] == 61 and m['season_id'] == 95}
    games = p['skillcorner_train']+p['skillcorner_development']
    if include_confirmation:
        games += p['skillcorner_confirmation_locked']
    manifest = []
    for game in games:
        if game not in allowed:
            raise ValueError('Unexpected match/competition')
        folder = ROOT/str(game)
        for suffix in ['match.json', 'tracking_extrapolated.jsonl']:
            relative = f'data/matches/{game}/{game}_{suffix}'
            target = folder/f'{game}_{suffix}'
            if suffix.endswith('.jsonl'):
                with urllib.request.urlopen(raw+'/'+relative, timeout=60) as response:
                    pointer = response.read().decode()
                if not pointer.startswith('version https://git-lfs.github.com/spec/v1'):
                    raise ValueError('Expected Git LFS source')
                expected = pointer.split('oid sha256:')[1].splitlines()[0]
                size = int(pointer.split('size ')[1].splitlines()[0])
                fetch(media+'/'+relative, target, expected)
                if target.stat().st_size != size:
                    raise ValueError('Source size mismatch')
            else:
                fetch(raw+'/'+relative, target)
            manifest.append(dict(path=str(target), url=(media if suffix.endswith('.jsonl') else raw)+'/'+relative,
                                 bytes=target.stat().st_size, sha256=sha256(target)))
        print('Verified SkillCorner', game, flush=True)
    (ROOT/'manifest.json').write_text(json.dumps(dict(commit=commit, files=manifest),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-confirmation',action='store_true')
    main(parser.parse_args().include_confirmation)
