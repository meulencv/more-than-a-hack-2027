import unittest
import numpy as np
from possession.sources import Match
from possession.sequence_quality_gate import missing_fraction,combine


class QualityGateTests(unittest.TestCase):
 def test_weights_are_half_local_and_finite(self):
  ball=np.zeros((400,2));ball[200:]=np.nan
  m=Match('x',np.zeros((400,2,2)),np.array([0,1]),ball,(105,68),[(1,0,199),(2,200,399)])
  w=missing_fraction(m,np.arange(400))
  self.assertTrue(np.isfinite(w).all())
  np.testing.assert_array_equal(w[:200],0)
  np.testing.assert_array_equal(w[200:],1)
 def test_mass_and_team_symmetry(self):
  a=np.array([[.7,.2,.1],[.2,.5,.3]])
  b=np.array([[.1,.3,.6],[.5,.4,.1]])
  w=np.array([0.,.8])
  p=combine(a,b,w)
  np.testing.assert_allclose(p.sum(axis=1),1)
  np.testing.assert_allclose(p[:,[1,0,2]],combine(a[:,[1,0,2]],b[:,[1,0,2]],w))
  np.testing.assert_allclose(p[0],a[0])
 def test_invalid_input_rejected(self):
  with self.assertRaises(ValueError):combine(np.zeros((2,3)),np.ones((2,3)),np.array([np.nan,0]))


if __name__=='__main__':unittest.main()
