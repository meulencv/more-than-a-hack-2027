"""Prepare final model inputs without fitting or replacing any frozen weights."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


if __name__=='__main__':
    os.chdir(ROOT)
    from possession import v6_data,v6_physics,v10_data
    v6_data.prepare();v6_physics.prepare();v10_data.prepare()
