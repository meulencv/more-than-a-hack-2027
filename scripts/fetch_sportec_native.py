"""Download and SHA-256 verify the four pinned public CC BY Sportec XML files.

Uses curl's normal TLS verification; no challenge assets are downloaded or published.
Already verified files are left untouched. Partial downloads never replace a target.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, record):
    path = Path(path)
    return path.is_file() and path.stat().st_size == record['size'] and checksum(path) == record['sha256']


def fetch(record, output_dir):
    name = record['name']
    if Path(name).name != name or record['kind'] not in ('metadata', 'tracking'):
        raise ValueError('Unexpected path or asset type in Sportec manifest')
    target = Path(output_dir) / 'sportec' / record['kind'] / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if verified(target, record):
        print('Verified', name, flush=True)
        return target
    temporary = target.with_suffix(target.suffix + '.part')
    try:
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--retry', '3', '--connect-timeout', '30', record['url'],
                        '--output', str(temporary)], check=True)
        if not verified(temporary, record):
            raise ValueError('Sportec size/SHA-256 mismatch: ' + name)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print('Downloaded and verified', name, flush=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('data/external/pathcrf'))
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'docs/sportec_native_manifest.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if len(manifest['files']) != 4:
        raise ValueError('Expected exactly four pinned Sportec XML assets')
    for record in manifest['files']:
        fetch(record, args.output_dir)


if __name__ == '__main__':
    main()
