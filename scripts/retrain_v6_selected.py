"""Repeat selected training in a fresh directory, preserving frozen checkpoints."""
import json
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from possession.v6_data import training_sequences,ROOT
from possession.v6_seed_repeat import physics_training
from possession.v6_temporal import train


def main():
    root=ROOT/'retrained_pair';root.mkdir(parents=True,exist_ok=True)
    report=[]
    for physical,family in [(False,'tcn_boundary'),(True,'physics_tcn')]:
        xs,ys=physics_training() if physical else training_sequences(False)
        model,norm=train(xs,ys,kind='boundary',steps=800,seed=2027,device='mps')
        checkpoint=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        mean=norm.mean,std=norm.std,kind='boundary',steps=800)
        if physical:checkpoint['physics']=True
        else:checkpoint['seed']=2027
        out=root/family/'model.pt';out.parent.mkdir(exist_ok=True)
        torch.save(checkpoint,out)
        expected=ROOT/family/'model.pt'
        same=out.read_bytes()==expected.read_bytes()
        report.append(dict(family=family,exact_checkpoint=same,
                           sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
        print(report[-1],flush=True)
        if not same:raise RuntimeError('Training not byte-identical: inspect before claiming exact reproduction')
    (root/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
