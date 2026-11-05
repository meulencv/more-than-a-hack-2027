import unittest
import numpy as np
from possession.sources import Match
from possession.v6_physics import physical_table


class PhysicalControlTests(unittest.TestCase):
    def fixture(self):
        n=100
        ball=np.column_stack((20+np.arange(n)*.16,np.full(n,30.)))
        players=np.zeros((n,2,2))
        players[:,0]=ball-[.8,0]
        players[:,1]=[23.2,30.]
        return Match('synthetic',players,np.array([0,1]),ball,(105,68),[(1,0,n-1)])

    def test_co_motion_can_outweigh_a_nearer_passing_opponent(self):
        frame=physical_table(self.fixture())
        self.assertGreater(frame.loc[4,'control_h'],frame.loc[4,'control_a'])

    def test_team_swap_symmetry(self):
        a=physical_table(self.fixture());b=physical_table(self.fixture(),True)
        np.testing.assert_allclose(a['control_h'],b['control_a'],equal_nan=True)
        np.testing.assert_allclose(a['control_margin'],-b['control_margin'],equal_nan=True)
        np.testing.assert_allclose(a['ball_impulse'],b['ball_impulse'],equal_nan=True)


if __name__=='__main__':unittest.main()
