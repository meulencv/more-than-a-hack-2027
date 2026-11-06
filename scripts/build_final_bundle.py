"""Build a local code/weights bundle and optionally replay it without any GT/cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);selection=json.loads(Path('docs/FINAL_SELECTION.json').read_text())
    files={Path('games.py'),Path('LICENSE'),Path('requirements-local.lock.txt'),Path('docs/FINAL_SELECTION.json')}
    for folder in ['possession','starter','evaluation','tests','scripts']:
        files.update(Path(folder).rglob('*.py'))
    files.update(Path('possession').glob('*.json'))
    files.update(Path('docs').glob('PROTOCOLO*.json'));files.update(Path('docs').glob('*manifest*.json'))
    files.update([Path('docs/MEMORIA_TECNICA.md'),Path('docs/RESULTADOS_MODELO_FINAL.md'),Path('docs/INVESTIGACION_V10.md')])
    files.update(Path(spec['path']) for spec in selection['components'])
    permitted_weights={spec['path'] for spec in selection['components']}
    if any(str(p).startswith('data/') and str(p) not in permitted_weights for p in files):raise ValueError('Unexpected data in archive')
    if any(p.suffix.lower() in ['.csv','.mp4','.npz','.joblib'] for p in files):raise ValueError('Restricted data/cache export')
    target=Path('data/work/final_model_bundle.zip');target.parent.mkdir(parents=True,exist_ok=True)
    readme='''Local reproduction bundle: code and four frozen native-data-trained checkpoints.

No match videos, tracking, frame labels, ground truth or precomputed alignment is included.
Extract into a fresh directory and provide the organiser's files under data/tactical
and data/broadcast. Install requirements-local.lock.txt into a Python3.12 environment.
Then run:
  OMP_NUM_THREADS=6 python -m possession.competitive_bundle
  python evaluation/validate_submission.py evaluation/results/final_candidate.csv

The default CPU command recomputes synchronization and regenerates the CSV.
Its expected SHA256 is 75e884aea21d68e0cc46c3e5a67798e799e4ceb59431e5872638080b38dc192e.
See docs/MEMORIA_TECNICA.md for methods, licenses, limitations and references.
'''
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):archive.write(path,str(path))
        archive.writestr('README_REPRODUCTION.md',readme)
    report=dict(bundle_path=str(target),bundle_sha256=digest(target),bundle_bytes=target.stat().st_size,
                csv_path='evaluation/results/final_candidate.csv',csv_sha256=digest('evaluation/results/final_candidate.csv'),
                packaged_weights=[dict(path=s['path'],sha256=s['sha256']) for s in selection['components']],
                no_competition_data_packaged=True,no_ground_truth_packaged=True,no_alignment_cache_packaged=True)
    if args.verify:
        qa=Path(tempfile.mkdtemp(prefix='mtah-final-qa-',dir='/private/tmp'))
        with zipfile.ZipFile(target) as archive:archive.extractall(qa)
        (qa/'data').mkdir(exist_ok=True)
        for source in ['tactical','broadcast']:(qa/'data'/source).symlink_to(ROOT/'data'/source,target_is_directory=True)
        environment=os.environ.copy();environment['PYTHONPATH']=str(qa);environment['OMP_NUM_THREADS']='6'
        with (qa/'replay.log').open('w') as log:
            subprocess.run([sys.executable,'-c','import possession; print(possession.__file__)'],cwd=qa,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],cwd=qa,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,'-m','possession.competitive_bundle'],cwd=qa,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        repeated=qa/'evaluation/results/final_candidate.csv'
        report.update(clean_replay_directory=str(qa),clean_replay_sha256=digest(repeated),
                      clean_replay_identical=digest(repeated)==report['csv_sha256'],
                      replay_used_no_precomputed_alignment=True,replay_had_no_external_training_data=True,
                      packaged_unit_tests_passed=True)
        if not report['clean_replay_identical']:raise RuntimeError('Clean archive replay differs')
    Path('docs/FINAL_ARTIFACT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
