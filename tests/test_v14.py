import unittest
from types import SimpleNamespace
import numpy as np
from possession.v14_resolution import decode_fine


class ResolutionTests(unittest.TestCase):
    def test_constant_emission_and_half_separation(self):
        m=SimpleNamespace(ball=np.empty((101,2)),blocks=[(1,0,49),(2,51,100)])
        f=np.r_[np.arange(0,50,5),np.arange(51,101,5)]
        p=np.array([[.99,.005,.005]]*10+[[.005,.99,.005]]*10)
        for space in ['probability','log']:
            y=decode_fine(p,np.zeros(20),f,m,space)
            self.assertTrue((y[:50]==0).all());self.assertEqual(y[50],3);self.assertTrue((y[51:]==1).all())

    def test_team_swap_symmetry(self):
        m=SimpleNamespace(ball=np.empty((200,2)),blocks=[(1,0,199)]);f=np.arange(0,200,5)
        p=np.array([[.99,.005,.005]]*20+[[.005,.99,.005]]*20);e=np.zeros(40);e[20]=1
        y=decode_fine(p,e,f,m);z=decode_fine(p[:,[1,0,2]],e,f,m)
        np.testing.assert_array_equal(z,np.array([1,0,2,3])[y])


if __name__=='__main__':unittest.main()
