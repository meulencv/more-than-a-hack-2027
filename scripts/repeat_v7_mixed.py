"""Repeat V7 mixed-source binary training without replacing any existing weights."""
import argparse
import torch
from possession.v7_transfer import training,fit_binary,evaluate,ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=2028)
    args=parser.parse_args();kind=f'mixed_seed{args.seed}';root=ROOT/kind
    root.mkdir(parents=True,exist_ok=True)
    if (root/'model.pt').exists():raise RuntimeError('Preserve previous model; choose a fresh seed')
    xs,ys=training('mixed');net,norm=fit_binary(xs,ys,seed=args.seed)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},
                    mean=norm.mean,std=norm.std,kind='mixed',steps=800,seed=args.seed),root/'model.pt')
    evaluate(kind,net,norm)


if __name__=='__main__':main()
