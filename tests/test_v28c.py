import unittest
import numpy as np
from possession import v28c_tagging as t


def event(code, start, end, team=None, half="1st Half", kind=None):
    tags = {"Half": half}
    if team is not None:
        tags["Team"] = team
    if kind is not None:
        tags["Type"] = kind
    return dict(code=code, start_frame=start, end_frame=end, start_s=start / 25, end_s=end / 25, tags=tags)


class Tags(unittest.TestCase):
    def test_mirrors_are_dropped_and_set_pieces_split(self):
        events = [event("PROGRESSION", 100, 600, "Barcelona"), event("DEFENDING IN MIDDLE THIRD", 100, 600, "Atletic Club"),
                  event("SET PIECES", 700, 1575, "Atletic Club", kind="Corner Kick"), event("SET PIECES", 2000, 2875, "N/A", kind="Throw-in"),
                  event("SET PIECES", 3000, 3125, "N/A", kind="Kick Off (Start)"), event("GOALS", 4000, 6000, "Barcelona")]
        self.assertTrue(t.mirrors_are_exact(events))
        families = sorted(i["family"] for i in t.load_instances(events))
        self.assertEqual(families, ["GOALS", "KICK OFF", "PROGRESSION", "SET PIECE", "SET PIECE", "SET PIECE TEAM"])
        events[1] = event("DEFENDING IN MIDDLE THIRD", 100, 601, "Atletic Club")
        self.assertFalse(t.mirrors_are_exact(events))

    def test_bins_count_from_the_nearer_edge(self):
        f = np.arange(100, 100 + 40 * 25)
        b = t.bin_index("PROGRESSION", 100, 100 + 40 * 25, f)
        cap = t.FAMILIES["PROGRESSION"][1]
        self.assertEqual(b[0], 0)
        self.assertEqual(b[24], 0)
        self.assertEqual(b[25], 1)
        self.assertEqual(b[-1], cap + 1)                     # last second = E0
        self.assertEqual(b[-26], cap + 2)                    # E1
        self.assertEqual(b[15 * 25], cap)                    # deeper than the cap from the start
        self.assertEqual(t.bin_name("PROGRESSION", cap + 1), "E0")
        self.assertLess(b.max(), t.n_bins("PROGRESSION"))
        s = t.bin_index("SET PIECE", 0, 875, np.arange(875))
        self.assertEqual(s[150], 6)
        self.assertEqual(s.max(), 34)

    def test_folds_are_time_blocks(self):
        blocks = [(1, 0, 5999), (2, 8000, 13999)]
        inst = t.assign_folds([dict(a=0, z=100), dict(a=5900, z=6000), dict(a=8000, z=8100), dict(a=6500, z=6600)], blocks, 6)
        self.assertEqual([i["fold"] for i in inst], [0, 5, 6, -1])


class Evidence(unittest.TestCase):
    def setUp(self):
        self.n = 6000
        self.graded = np.ones(self.n, dtype=bool)
        events = []
        state = np.full(self.n, 2, dtype=np.int8)
        for k in range(10):                                  # ten clips: the tagged team has the ball
            a = 200 + k * 550
            team = k % 2
            events.append(event("PROGRESSION", a, a + 400, "Barcelona" if team == 0 else "Atletic Club"))
            state[a:a + 400] = team
        self.instances = t.assign_folds(t.load_instances(events), [(1, 0, self.n - 1)], 5)
        self.state = state

    def test_table_learns_team_and_alive(self):
        tables = t.fit_tables(self.instances, self.state, self.graded, self.n)
        llr_team, llr_dead = tables["PROGRESSION"]
        self.assertTrue((llr_team[:5] > 1).all())
        self.assertTrue((llr_dead[:5] < -1).all())

    def test_cross_fitting_never_uses_the_held_block(self):
        poisoned = self.state.copy()
        held = self.instances[0]["fold"]
        for inst in self.instances:
            if inst["fold"] == held:
                poisoned[inst["a"]:inst["z"]] = 1 - inst["team"]     # the held clips say the opposite
        clean = t.fit_tables(self.instances, self.state, self.graded, self.n, held_fold=held)
        other = t.fit_tables(self.instances, poisoned, self.graded, self.n, held_fold=held)
        for family in clean:
            self.assertTrue(np.allclose(clean[family][0], other[family][0]))
            self.assertTrue(np.allclose(clean[family][1], other[family][1]))

    def test_confident_frame_is_never_overturned(self):
        adj, _ = t.cross_fitted_evidence(self.instances, self.state, self.graded, self.n, ("PROGRESSION",))
        self.assertTrue((adj.max(1) == 0).all())
        self.assertGreaterEqual(adj.min(), -t.CLIP)
        self.assertLess(t.CLIP, np.log(0.9 / 0.1))
        frames = np.arange(0, self.n, 5)
        rng = np.random.default_rng(0)
        for top in range(3):                                 # ensemble says `top` with 0.9 everywhere
            p = np.full((len(frames), 3), 0.0)
            rest = rng.uniform(0.01, 0.09, len(frames))
            others = [s for s in range(3) if s != top]
            p[:, top] = 0.9
            p[:, others[0]] = rest
            p[:, others[1]] = 0.1 - rest
            fused = t.fuse(p, frames, adj)
            self.assertTrue((fused.argmax(1) == top).all())
            self.assertTrue(np.allclose(fused.sum(1), 1))

    def test_evidence_is_zero_without_tags_and_outside_graded(self):
        graded = self.graded.copy()
        graded[:1000] = False
        adj, _ = t.cross_fitted_evidence(self.instances, self.state, graded, self.n, ("PROGRESSION",))
        self.assertTrue((adj[:1000] == 0).all())
        self.assertTrue((adj[5800:] == 0).all())             # after the last clip
        inside = self.instances[3]
        self.assertLess(adj[inside["a"] + 100, 1 - inside["team"]], -1.0)
        self.assertEqual(adj[inside["a"] + 100, inside["team"]], 0.0)

    def test_changed_runs(self):
        before = np.array([0, 0, 1, 1, 2, 2, 0, 0])
        after = np.array([0, 1, 1, 0, 0, 2, 0, 2])
        runs = t.changed_runs(before, after, np.ones(8, dtype=bool))
        self.assertEqual(runs, [(1, 2, 0, 1), (3, 4, 1, 0), (4, 5, 2, 0), (7, 8, 0, 2)])


if __name__ == "__main__":
    unittest.main()
