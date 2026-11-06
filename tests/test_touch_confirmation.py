import unittest
import numpy as np
from possession.touch_confirmation import apply_policy, dense_evidence
from possession.v23_touch_margins import check_invariants, policies


def evidence(p, control=None, observed=None, edge=None):
    n=len(p)
    return dict(p=np.asarray(p,float),control=np.zeros((n,2)) if control is None else control,
        observed=np.ones(n,bool) if observed is None else observed,
        edge=np.zeros(n) if edge is None else edge)


class TouchConfirmationTests(unittest.TestCase):
    def test_confirmed_change_starts_at_evidence_not_confirmation(self):
        base=np.zeros(60,dtype='int8');p=np.tile([.99,.009,.001],(60,1));p[20:28]=[.199,.8,.001]
        out=apply_policy(base,evidence(p),[(1,0,59)],dict(kind='confirm',seconds=.24))
        expected=base.copy();expected[20:28]=1
        np.testing.assert_array_equal(out,expected)

    def test_weak_unconfirmed_pulse_does_not_change_control(self):
        base=np.zeros(60,dtype='int8');p=np.tile([.99,.009,.001],(60,1));p[20:22]=[.299,.7,.001]
        out=apply_policy(base,evidence(p),[(1,0,59)],dict(kind='confirm',seconds=.24))
        np.testing.assert_array_equal(out,base)

    def test_overwhelming_evidence_preserves_subsecond_control(self):
        base=np.zeros(60,dtype='int8');p=np.tile([.99,.009,.001],(60,1));p[20:22]=[.009,.99,.001]
        out=apply_policy(base,evidence(p),[(1,0,59)],dict(kind='confirm',seconds=.8))
        self.assertTrue((out[20:22]==1).all())

    def test_dead_ball_and_half_hole_reset_pending_state(self):
        base=np.r_[np.zeros(10),np.full(3,2),np.ones(10),np.full(5,3),np.ones(10)].astype('int8')
        p=np.tile([.8,.199,.001],(len(base),1));p[10:]=[.199,.8,.001]
        out=apply_policy(base,evidence(p),[(1,0,22),(2,28,37)],dict(kind='confirm',seconds=.8))
        np.testing.assert_array_equal(out,base)
        check_invariants(base,out,[(1,0,22),(2,28,37)])

    def test_missing_ball_never_forces_dead_or_a_switch(self):
        base=np.zeros(60,dtype='int8');p=np.tile([.51,.48,.01],(60,1))
        out=apply_policy(base,evidence(p,observed=np.zeros(60,bool)),[(1,0,59)],dict(kind='confirm',seconds=.24,adaptive=True))
        np.testing.assert_array_equal(out,base)

    def test_recovery_requires_control_and_local_boundaries(self):
        base=np.zeros(60,dtype='int8');p=np.tile([.99,.009,.001],(60,1));p[20:25]=[.199,.8,.001]
        c=np.zeros((60,2));c[20:25,1]=.9;e=np.zeros(60);e[[20,25]]=.8
        spec=dict(kind='recover',seconds=.4)
        out=apply_policy(base,evidence(p,c,edge=e),[(1,0,59)],spec)
        expected=base.copy();expected[20:25]=1;np.testing.assert_array_equal(out,expected)
        np.testing.assert_array_equal(apply_policy(base,evidence(p),[(1,0,59)],spec),base)

    def test_weak_merge_preserves_physical_control_and_stoppages(self):
        base=np.r_[np.zeros(20),np.ones(5),np.zeros(20),np.full(3,2),np.zeros(10)].astype('int8')
        p=np.tile([.55,.44,.01],(len(base),1));p[20:25]=[.44,.55,.01];c=np.zeros((len(base),2));c[20:25,1]=.9
        spec=dict(kind='merge',seconds=.4,weak_only=True)
        np.testing.assert_array_equal(apply_policy(base,evidence(p,c),[(1,0,len(base)-1)],spec),base)
        unknown=np.ones(len(base),bool);unknown[20:25]=False
        np.testing.assert_array_equal(apply_policy(base,evidence(p,observed=unknown),[(1,0,len(base)-1)],spec),base)
        out=apply_policy(base,evidence(p),[(1,0,len(base)-1)],spec)
        self.assertTrue((out[20:25]==0).all());self.assertTrue((out[45:48]==2).all())

    def test_all_policies_are_team_symmetric_and_preserve_dead_frames(self):
        rng=np.random.default_rng(23);base=np.r_[np.repeat([0,1,0,2,1],40),np.full(5,3),np.ones(80)].astype('int8')
        p=rng.dirichlet([1,1,1],len(base));c=rng.random((len(base),2));seen=rng.random(len(base))>.3;e=rng.random(len(base));blocks=[(1,0,199),(2,205,284)]
        ev=evidence(p,c,seen,e);swapped=evidence(p[:,[1,0,2]],c[:,[1,0]],seen,e)
        swap=np.array([1,0,2,3])
        for spec in policies().values():
            a=apply_policy(base,ev,blocks,spec);b=apply_policy(swap[base],swapped,blocks,spec)
            np.testing.assert_array_equal(b,swap[a]);check_invariants(base,a,blocks)

    def test_overlapping_weak_merges_do_not_invert_the_anchor(self):
        base=np.repeat([0,1,0,1,0],3).astype('int8');p=np.tile([.495,.495,.01],(len(base),1))
        out=apply_policy(base,evidence(p),[(1,0,len(base)-1)],dict(kind='merge',seconds=.4,weak_only=True))
        np.testing.assert_array_equal(out,base)

    def test_dead_uncertainty_does_not_make_team_confidence_weak(self):
        base=np.r_[np.zeros(40),np.ones(5),np.zeros(40)].astype('int8');p=np.tile([.99,.009,.001],(len(base),1));p[40:45]=[.01,.64,.35]
        out=apply_policy(base,evidence(p),[(1,0,len(base)-1)],dict(kind='merge',seconds=.4,weak_only=True))
        np.testing.assert_array_equal(out,base)

    def test_joint_evidence_can_confirm_two_frame_control(self):
        base=np.zeros(60,dtype='int8');p=np.tile([.99,.009,.001],(60,1));p[20:22]=[.099,.9,.001];c=np.zeros((60,2));c[20:22,1]=.9
        out=apply_policy(base,evidence(p,c),[(1,0,59)],dict(kind='confirm',seconds=.4,adaptive=True,contact_immediate=True))
        expected=base.copy();expected[20:22]=1
        np.testing.assert_array_equal(out,expected)

    def test_dense_evidence_has_no_reference_label_dependency(self):
        f=np.array([0,5,10,20,25,30]);p=np.tile([.7,.2,.1],(6,1));seen=np.ones(31,bool);seen[3:8]=False
        r=dict(ball_seen=seen,columns=['control_h','control_a'],direct=np.tile([.8,.1],(6,1)),blocks=[(1,0,10),(2,20,30)])
        ev=dense_evidence(p,np.zeros(6),f,r)
        self.assertFalse(ev['observed'][3:8].any());self.assertFalse(ev['observed'][11:20].any());self.assertTrue((ev['p'][11:20]==0).all())


if __name__=='__main__':unittest.main()
