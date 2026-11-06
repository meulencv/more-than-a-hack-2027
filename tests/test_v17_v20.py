import unittest
from types import SimpleNamespace
import numpy as np
import torch
from possession.v14_resolution import decode_fine
from possession.v17_errors import decode as candidate_decode
from possession.v17_gates import blend,KINDS
from possession.v17_highrate import infer
from possession.v18_typed import event_targets,decode as typed_decode,PAIRS
from possession.v20_duration import decode as duration_decode
from possession.v6_temporal import SegmentNet,predict
from possession.sequence_tcn import Normalizer


class ErrorExperimentTests(unittest.TestCase):
    def test_typed_targets_preserve_rapid_changes_and_ignore_unknown(self):
        y=np.array([0,0,1,0,2,-1,1,2])
        targets=event_targets(y)
        self.assertEqual(targets[2],1+PAIRS.index((0,1)))
        self.assertEqual(targets[3],1+PAIRS.index((1,0)))
        self.assertEqual(targets[5],-1)
        self.assertEqual(targets[6],0)

    def test_duration_zero_strength_matches_original_across_halves(self):
        rng=np.random.default_rng(13);f=np.r_[np.arange(0,400,5),np.arange(411,811,5)]
        p=rng.dirichlet([.8,.8,.8],len(f));e=rng.random(len(f))
        m=SimpleNamespace(ball=np.empty((811,2)),blocks=[(1,0,399),(2,411,810)])
        expected=decode_fine(p,e,f,m)
        actual=duration_decode(p,e,f,m,np.zeros((3,12)),0)
        np.testing.assert_array_equal(expected,actual)
        self.assertTrue((actual[400:411]==3).all())

    def test_duration_team_swap(self):
        rng=np.random.default_rng(19);f=np.arange(0,500,5);p=rng.dirichlet([1,1,1],len(f));e=rng.random(len(f))
        m=SimpleNamespace(ball=np.empty((500,2)),blocks=[(1,0,499)])
        adjustment=np.tile(np.linspace(2,-1,15),(3,1));adjustment[2]*=.5
        a=duration_decode(p,e,f,m,adjustment,1);b=duration_decode(p[:,[1,0,2]],e,f,m,adjustment,1)
        np.testing.assert_array_equal(b,np.array([1,0,2,3])[a])

    def test_directed_event_can_restore_a_brief_control(self):
        f=np.arange(0,300,5);p=np.tile([.6,.399,.001],(len(f),1));e=np.zeros(len(f))
        typed=np.zeros((len(f),7));typed[:,0]=1
        for index,pair in [(20,(0,1)),(26,(1,0))]:typed[index]=0;typed[index,0]=.05;typed[index,1+PAIRS.index(pair)]=.95
        m=SimpleNamespace(ball=np.empty((300,2)),blocks=[(1,0,299)])
        a=typed_decode(p,e,typed,f,m,6)
        self.assertTrue((a[100:130]==1).all());self.assertTrue((a[:100]==0).all());self.assertTrue((a[130:]==0).all())

    def test_no_typed_bonus_exactly_preserves_original(self):
        rng=np.random.default_rng(20);f=np.arange(0,300,5);p=rng.dirichlet([1,1,1],len(f));e=rng.random(len(f))
        typed=np.zeros((len(f),7));typed[:,0]=1;m=SimpleNamespace(ball=np.empty((300,2)),blocks=[(1,0,299)])
        np.testing.assert_array_equal(typed_decode(p,e,typed,f,m,4),decode_fine(p,e,f,m))

    def test_conditional_fusion_is_symmetric_and_normalized(self):
        rng=np.random.default_rng(21);base=rng.dirichlet([1,1,1],40);raw=rng.dirichlet([1,1,1],40)
        s=dict(baseline=base,combined=(base+raw)/2);swapped={k:v[:,[1,0,2]] for k,v in s.items()}
        r=dict(ball_seen=np.r_[np.ones(100),np.zeros(100)],frames=np.arange(0,200,5),slices=[(0,20),(20,40)])
        for kind in KINDS:
            a=blend(s,r,kind);b=blend(swapped,r,kind)
            np.testing.assert_allclose(a.sum(1),1,atol=1e-6);np.testing.assert_allclose(b,a[:,[1,0,2]],atol=1e-6)

    def test_residual_refiner_starts_as_exact_identity(self):
        from possession.v22_refiner import Refiner
        torch.manual_seed(31)
        net=Refiner(10).eval()
        with torch.inference_mode():delta=net(torch.randn(2,10,120))
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))

    def test_highrate_chunking_has_no_internal_seams(self):
        torch.set_num_threads(2);torch.manual_seed(22);rng=np.random.default_rng(22)
        x=rng.normal(size=(4600,2)).astype('float32');norm=Normalizer().fit([x]);net=SegmentNet(4).eval()
        r=dict(direct=x,mirror=x,slices=[(0,len(x))])
        p,e=infer(net,norm,r,'cpu');whole,edge=predict(net,norm,[x],'cpu')
        np.testing.assert_allclose(p,(whole[0]+whole[0][:,[1,0,2]])/2,atol=2e-6)
        np.testing.assert_allclose(e,edge[0],atol=2e-6)


if __name__=='__main__':unittest.main()
