import unittest
import numpy as np
import torch
from possession.sequence_tcn import Normalizer, TemporalCNN, loss
from possession.sequence_experiment import split_by_halves
from possession.sources import Match


class SequenceTests(unittest.TestCase):
    def test_normalizer_is_train_only_and_preserves_missingness(self):
        n=Normalizer().fit([np.array([[1.,np.nan],[3.,np.nan]],dtype='float32')])
        x=n.transform(np.array([[np.nan,9.]],dtype='float32'))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[0,2:],[1,0])
        self.assertEqual(x[0,0],0)

    def test_temporal_shape_and_finite_loss(self):
        model=TemporalCNN(8)
        x=torch.randn(2,8,64)
        y=torch.randint(0,3,(2,64))
        prediction=model(x)
        self.assertEqual(tuple(prediction.shape),(2,3,64))
        objective=loss(prediction,y)
        self.assertTrue(torch.isfinite(objective))
        objective.backward()

    def test_cache_reconstruction_never_crosses_halves(self):
        match=Match('synthetic',np.zeros((40,2,2)),np.array([0,1]),np.zeros((40,2)),(105.,68.),[(1,0,9),(2,20,34)])
        x=np.arange(10)[:,None]
        y=np.arange(10)
        seq,target=split_by_halves(x,y,match,repetitions=2)
        self.assertEqual([len(a) for a in seq],[2,3,2,3])
        np.testing.assert_array_equal(target[2],[5,6])
        with self.assertRaises(ValueError): split_by_halves(x[:-1],y[:-1],match,repetitions=2)


if __name__=='__main__': unittest.main()
