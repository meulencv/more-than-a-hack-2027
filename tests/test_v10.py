import unittest
import numpy as np
import torch
from possession.sources import Match
from possession.v10_data import raw_table
from possession.v10_relational import PlayerSetNet,mirror,hide_ball


class RelationalTests(unittest.TestCase):
    def test_mask_removes_all_ball_information(self):
        x=np.ones((20,4,14),dtype='float32');b=np.ones((20,7),dtype='float32');m=np.arange(20)%2==0
        xx,bb=hide_ball(x,b,m)
        self.assertTrue((xx[m,:,5:10]==0).all());self.assertTrue((xx[m,:,12:]==0).all())
        self.assertTrue((bb[m]==0).all());np.testing.assert_array_equal(xx[~m],x[~m])
        np.testing.assert_array_equal(xx[:,:,:5],x[:,:,:5])

    def test_mirror_is_involution(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(20,4,14));b=rng.normal(size=(20,7));team=np.array([0,1,-1,0])
        y,c,t=mirror(*mirror(x,b,team));np.testing.assert_array_equal(y,x);np.testing.assert_array_equal(c,b);np.testing.assert_array_equal(t,team)

    def test_set_is_permutation_invariant_and_empty_finite(self):
        torch.manual_seed(4);net=PlayerSetNet('trajectory').eval()
        x=torch.randn(2,30,6,14);x[:,:,:,10]=1;b=torch.randn(2,30,7);team=torch.tensor([[0,0,0,1,1,1]]*2)
        order=torch.tensor([4,1,5,0,2,3])
        with torch.no_grad():
            a=net(x,b,team);c=net(x[:,:,order],b,team[:,order])
            for lhs,rhs in zip(a,c):torch.testing.assert_close(lhs,rhs,atol=2e-6,rtol=1e-5)
            x.zero_()
            for value in net(x,b,team):self.assertTrue(torch.isfinite(value).all())

    def test_raw_observations_do_not_read_reference_labels(self):
        p=np.zeros((50,4,2),dtype='float32');p[:,2:,0]=80;b=np.zeros((50,2),dtype='float32')
        m=Match('synthetic',p,np.array([0,0,1,1]),b,(105.,68.),[(1,0,49)],{'labels':np.zeros(50)})
        a=raw_table(m);m.meta['labels'][:]=2;c=raw_table(m)
        for lhs,rhs in zip(a,c):np.testing.assert_array_equal(lhs,rhs)
        self.assertTrue(np.isfinite(a[0]).all());self.assertTrue(np.isfinite(a[1]).all())


if __name__=='__main__':unittest.main()
