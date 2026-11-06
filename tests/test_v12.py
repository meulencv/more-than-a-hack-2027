import unittest
import numpy as np
import torch
from possession.sources import Match
from possession.v12_compare import match_record
from possession.v10_relational import PlayerSetNet


class CombinedTests(unittest.TestCase):
    def test_rebuilt_features_align_all_representations_and_mirror(self):
        p=np.zeros((240,4,2),dtype='float32');p[:,0]=[10,10];p[:,1]=[20,20];p[:,2]=[80,10];p[:,3]=[90,20]
        b=np.full((240,2),[20,20],dtype='float32');m=Match('synthetic',p,np.array([0,0,1,1]),b,(105.,68.),[(1,0,239)])
        r=match_record(m)
        np.testing.assert_array_equal(r['frames'],np.arange(0,240,5))
        self.assertEqual(r['direct'].shape,(48,156));self.assertEqual(r['mirror'].shape,(48,156))
        self.assertEqual(r['nodes'].shape,(48,64,14));self.assertEqual(r['ball_tokens'].shape,(48,7))
        self.assertFalse(np.allclose(r['direct'],r['mirror'],equal_nan=True))

    def test_graph_is_equivariant_to_player_order_and_handles_empty(self):
        torch.manual_seed(19);net=PlayerSetNet('graph').eval();x=torch.randn(2,25,6,14)
        x[:,:,:,10]=1;b=torch.randn(2,25,7);team=torch.tensor([[0,0,0,1,1,1]]*2);order=torch.tensor([4,2,5,1,0,3])
        with torch.no_grad():
            a=net(x,b,team);c=net(x[:,:,order],b,team[:,order])
            for y,z in zip(a,c):torch.testing.assert_close(y,z,atol=3e-6,rtol=1e-5)
            x.zero_()
            for value in net(x,b,team):self.assertTrue(torch.isfinite(value).all())


if __name__=='__main__':unittest.main()
