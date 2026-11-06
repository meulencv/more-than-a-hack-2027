import unittest
import numpy as np
import torch
from torch import nn
from possession.v10_relational import PlayerSetNet,predict


class RecurrentTests(unittest.TestCase):
    def test_recurrent_forward_backward_finite(self):
        torch.manual_seed(5);net=PlayerSetNet('lstm');x=torch.randn(2,30,6,14);x[:,:,:,10]=1
        b=torch.randn(2,30,7);team=torch.tensor([[0,0,0,1,1,1]]*2)
        output=net(x,b,team);loss=sum(a.square().mean() for a in output)
        self.assertTrue(torch.isfinite(loss));loss.backward()

    def test_recurrent_inference_does_not_reset_within_half(self):
        class Fake(nn.Module):
            kind='lstm'
            def __init__(self):super().__init__();self.lengths=[]
            def forward(self,nodes,ball,team,global_x=None):
                batch,length,_,_=nodes.shape;self.lengths.append(length)
                return torch.zeros(batch,3,length),torch.zeros(batch,length),torch.zeros(batch,length,2)
        net=Fake();r=dict(nodes=np.zeros((100,4,14),dtype='float16'),ball_tokens=np.zeros((100,7),dtype='float32'),
                          team=np.array([0,0,1,1]),slices=[(0,100)])
        p,e,c=predict(net,None,r,'cpu',chunk=20)
        self.assertEqual(net.lengths,[100,100]);self.assertEqual(p.shape,(100,3))


if __name__=='__main__':unittest.main()
