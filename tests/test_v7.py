import json
from pathlib import Path
import unittest
import numpy as np
from possession.skillcorner import resample_positions,partial_label,load_match
from possession.v7_transfer import fuse


class TransferTests(unittest.TestCase):
    def test_unknown_teacher_is_not_dead(self):
        self.assertEqual([partial_label(x) for x in ['home','away',None,'unknown','ball_out']], [0,1,-1,-1,-1])
        self.assertEqual([partial_label(x) for x in ['home team','away team']],[0,1])

    def test_resampling_time_and_missingness(self):
        x=np.array([[0.,0.],[1.,1.],[2.,2.],[np.nan,np.nan],[4.,4.]])
        y,t=resample_positions(x)
        np.testing.assert_allclose(y[:6,0],np.arange(6)*.4)
        self.assertTrue(np.isnan(y[6:10]).all())
        np.testing.assert_array_equal(y[10],x[4])
        np.testing.assert_array_equal(t[[0,5,10]],[0,2,4])

    def test_fusion_preserves_dead_and_team_symmetry(self):
        p=np.array([[.2,.3,.5],[0.,0.,1.]],dtype='float32');q=np.array([[.9,.1],[.7,.3]])
        for w in (0.,.25,.5,1.):
            r=fuse(p,q,w);np.testing.assert_array_equal(r[:,2],p[:,2])
            np.testing.assert_allclose(r.sum(1),1.,atol=1e-7)
            np.testing.assert_allclose(fuse(p[:,[1,0,2]],q[:,[1,0]],w),r[:,[1,0,2]])
        np.testing.assert_allclose(fuse(p,q,0),p)

    def test_locked_partition(self):
        p=json.loads(Path('docs/PROTOCOLO_V7.json').read_text())
        train=set(p['skillcorner_train']);dev=set(p['skillcorner_development']);test=set(p['skillcorner_confirmation_locked'])
        self.assertTrue(train.isdisjoint(dev));self.assertTrue(train.isdisjoint(test));self.assertTrue(dev.isdisjoint(test))
        with self.assertRaisesRegex(ValueError,'Locked'):load_match(next(iter(test)))


if __name__=='__main__':unittest.main()
