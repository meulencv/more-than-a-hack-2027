import unittest
from types import SimpleNamespace
import numpy as np
import torch
from possession.v6_temporal import ChannelNorm, segment_weights, boundary_targets, objective, SegmentNet
from possession.v6_experiment import dynamic_decode


class V6Tests(unittest.TestCase):
    def test_local_normalization_does_not_depend_on_other_times(self):
        norm = ChannelNorm(4).eval()
        x = torch.randn(1,4,12)
        y = torch.cat([x,torch.randn(1,4,12)*100],dim=2)
        torch.testing.assert_close(norm(x),norm(y)[:,:,:12])

    def test_segment_and_boundary_targets_ignore_unknown(self):
        y = np.array([0]*20+[1]*2+[-1]+[2]*10)
        w = segment_weights(y)
        self.assertGreater(w[20],w[0])
        self.assertEqual(w[22],0)
        b = boundary_targets(y)
        self.assertEqual(b[20],1)
        self.assertEqual(b[22],0)

    def test_boundary_loss_finite_with_unknowns(self):
        net = SegmentNet(8,width=8)
        outputs = net(torch.randn(2,8,30))
        labels = torch.zeros((2,30),dtype=torch.long); labels[:,10]=-1
        loss = objective(outputs,labels,torch.ones(3),torch.ones((2,30)),torch.zeros((2,30)),True)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_dynamic_boundary_can_represent_a_short_real_possession(self):
        p = np.tile([.99,.005,.005],(40,1));p[20:23]=[.005,.99,.005]
        edge = np.zeros(40);edge[[20,23]]=1
        match = SimpleNamespace(ball=np.empty((200,2)),blocks=[(1,0,199)])
        states = dynamic_decode(p,edge,np.arange(0,200,5),match)
        self.assertTrue((states[100:115]==1).all())
        self.assertTrue((states[115:]==0).all())


if __name__=='__main__':unittest.main()
