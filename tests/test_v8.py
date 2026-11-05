import unittest
import numpy as np
import torch
from possession.v8_objects import object_regions,soft_object_iou,inverse_segment_weights


class ObjectLossTests(unittest.TestCase):
    def test_complete_runs_with_neighbors_only(self):
        y=np.array([2]*4+[0]*3+[1]*2+[0]*4+[2]*5)
        regions=object_regions([y])
        np.testing.assert_array_equal(regions,[[0,0,4,7,0,9],[0,1,7,9,4,13],[0,0,9,13,7,18]])

    def test_perfect_local_objects_and_wrong_team(self):
        y=np.array([2]*4+[0]*3+[1]*2+[0]*4+[2]*5)
        logits=torch.full((1,3,len(y)),-15.,requires_grad=True)
        with torch.no_grad():logits[0,torch.tensor(y),torch.arange(len(y))]=15.
        regions=object_regions([y]);loss=soft_object_iou(logits,regions)
        self.assertLess(float(loss.detach()),1e-5)
        wrong=soft_object_iou(logits[:,[1,0,2]],regions)
        self.assertGreater(float(wrong.detach()),.99)
        wrong.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_unknown_or_truncated_runs_are_excluded(self):
        self.assertEqual(len(object_regions([np.array([0,0,-1,1,1,2,2])])),0)
        self.assertEqual(len(object_regions([np.array([0,0,1,1])])),0)
        empty=soft_object_iou(torch.randn(1,3,3,requires_grad=True),np.empty((0,6)))
        self.assertEqual(float(empty.detach()),0);empty.backward()

    def test_inverse_weights_equalize_unclipped_runs(self):
        y=np.array([0]*20+[1]*4+[-1]*2+[2]*10)
        w=inverse_segment_weights(y)
        self.assertAlmostEqual(float(w[:20].sum()),float(w[20:24].sum()),places=5)
        self.assertTrue((w[24:26]==0).all())


if __name__=='__main__':unittest.main()
