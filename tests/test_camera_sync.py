import unittest
import numpy as np
from possession.sources import Match
from possession.camera_sync import synchronize,fused_ball,geometry


def fixture():
    rng=np.random.default_rng(9);n=2000;t=np.arange(n)/25
    team=np.array([0]*11+[1]*11,dtype='int8');base=rng.uniform([5,5],[95,63],(22,2))
    phase=rng.uniform(-3,3,(22,2));rates=rng.uniform(.015,.12,(22,2))
    players=base[None]+8*np.sin(t[:,None,None]*rates+phase)+np.column_stack((6*np.sin(t*.07),3*np.cos(t*.11)))[:,None]
    ball=players[:,0].copy();reference=Match('synthetic',players,team,ball,(105.,68.),[(1,0,n-1)])
    return reference


class CameraSyncTests(unittest.TestCase):
    def test_geometry_is_identity_and_slot_invariant(self):
        m=fixture();p=m.players[100];order=np.arange(22)[::-1]
        self.assertAlmostEqual(geometry(p,m.team,p[order],m.team[order]),0.)

    def test_recovers_offset_reflection_and_fills_only_missing(self):
        m=fixture();rng=np.random.default_rng(3);idx=np.arange(300,1700)
        p=np.array(m.pitch)-m.players[idx]+rng.normal(0,.02,m.players[idx].shape)
        b=np.array(m.pitch)-m.ball[idx]
        other=Match('other',p,m.team,b,m.pitch,[(1,0,len(idx)-1)])
        result=synchronize(m,other)
        use=result.target_frame>=0
        self.assertGreater(use.mean(),.7)
        self.assertLessEqual(np.median(np.abs(result.target_frame[use]-idx[use])),2)
        missing=m.ball.copy();missing[700:900]=np.nan;source=m.copy_with(ball=missing)
        fused,filled=fused_ball(source,other,result)
        self.assertGreater(filled[700:900].mean(),.5)
        np.testing.assert_array_equal(fused.ball[~filled],missing[~filled])


if __name__=='__main__':unittest.main()
