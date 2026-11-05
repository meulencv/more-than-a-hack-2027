import unittest
import numpy as np
from possession.sources import Match
from possession.v9_control import stable_runs,control_table


class ControlTests(unittest.TestCase):
    def test_brief_encounters_do_not_create_stable_control(self):
        np.testing.assert_array_equal(stable_runs(np.array([-1,0,0,1,1,1,1,1,1,-1])),[False]*3+[True]*6+[False])

    def test_control_memory_across_uncontrolled_ball(self):
        p=np.zeros((120,2,2),dtype='float32');p[:,1,0]=10
        b=np.zeros((120,2),dtype='float32');b[50:80,0]=5;b[80:,0]=10
        m=Match('synthetic',p,np.array([0,1]),b,(105.,68.),[(1,0,119)])
        x=control_table(m);y=control_table(m,True)
        self.assertEqual(x.loc[12,'anchor_last_team'],1)
        self.assertEqual(x.loc[12,'anchor_next_team'],-1)
        self.assertEqual(x.loc[12,'anchor_agreement'],0)
        for key in x:
            sign=-1 if key in ['anchor_last_team','anchor_next_team','anchor_interpolated_team','anchor_team_margin'] or key.startswith('anchor_signed_rate') else 1
            np.testing.assert_allclose(y[key],sign*x[key],equal_nan=True)

    def test_half_boundary_resets_control(self):
        p=np.zeros((100,2,2),dtype='float32');p[:,1,0]=10
        b=np.zeros((100,2),dtype='float32');b[50:]=np.nan
        m=Match('synthetic',p,np.array([0,1]),b,(105.,68.),[(1,0,49),(2,50,99)])
        x=control_table(m)
        self.assertTrue(x.loc[10:,'anchor_last_team'].isna().all())
        self.assertTrue((x.loc[10:,'anchor_since']==60).all())


if __name__=='__main__':unittest.main()
