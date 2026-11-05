"""Download pinned Sportec V6 external assets; never downloads competition data."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.fetch_sportec_native import fetch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path('data/external/pathcrf'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'docs/sportec_v6_manifest.json').read_text())
    if len(manifest['files'])!=14:raise ValueError('Expected fourteen pinned V6 assets')
    for item in manifest['files']:fetch(item,args.output_dir)


if __name__=='__main__':main()
