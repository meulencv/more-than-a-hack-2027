import unittest
import numpy as np
from possession.v26_track import build_tracklets, chunk_table, track, select, smooth_pieces, flatten_flights, Geometry


def straight(first, count, origin, velocity, score=0.9):
    frames = np.arange(first, first + count)
    xy = np.asarray(origin, dtype=float) + np.outer(np.arange(count), velocity)
    return frames, xy, np.full(count, score), np.zeros(count, dtype=int)


class TrackTests(unittest.TestCase):
    def test_linking_follows_one_moving_ball_and_ignores_noise(self):
        n = 60
        metres = np.full((n, 2, 2), np.nan)
        score = np.zeros((n, 2))
        metres[:, 0] = np.column_stack([10 + 0.5 * np.arange(n), np.full(n, 30.0)])
        score[:, 0] = 0.8
        metres[::7, 1] = [80.0, 5.0]            # isolated flickers far away
        score[::7, 1] = 0.9
        tracklets = build_tracklets(metres, score, np.arange(n))
        self.assertEqual(len(tracklets), 1)
        self.assertEqual(len(tracklets[0][0]), n)

    def test_track_does_not_teleport_between_tracklets(self):
        real = straight(10, 40, (20.0, 30.0), (0.4, 0.0), 0.7)
        far = straight(52, 30, (90.0, 60.0), (0.0, 0.0), 0.95)       # 2 frames later, 60 m away
        usable = np.ones(120, dtype=bool)
        table = chunk_table([real, far], 0.3, 1.0, usable)
        chosen = select(table, 0, 119, None, None)
        used = set(table[chosen, 7].astype(int).tolist())
        self.assertEqual(len(used), 1)                               # never both

    def test_anchor_rejects_a_track_far_from_the_tracked_ball(self):
        far = straight(5, 30, (90.0, 60.0), (0.0, 0.0), 0.95)
        near = straight(5, 30, (21.0, 30.0), (0.2, 0.0), 0.6)
        usable = np.ones(60, dtype=bool)
        table = chunk_table([far, near], 0.3, 1.0, usable)
        atd = np.full((60, 2), np.nan)
        atd[:3] = (20.0, 30.0)
        atd[40:] = (28.0, 30.0)
        hidden = ~np.isfinite(atd[:, 0])
        xy, score, kind = track([far, near], table, hidden, atd, 60)
        self.assertTrue((kind[5:35] > 0).all())
        self.assertLess(np.nanmax(np.abs(xy[5:35, 1] - 30.0)), 1.0)
        self.assertTrue((kind[~hidden] == 0).all())

    def test_weak_tracklet_is_not_reported(self):
        weak = straight(5, 30, (50.0, 30.0), (0.0, 0.0), 0.25)
        table = chunk_table([weak], 0.3, 1.0, np.ones(60, dtype=bool))
        self.assertEqual(len(table), 0)

    def test_pieces_split_at_a_kick(self):
        t = np.arange(60)
        xy = np.column_stack([np.where(t < 30, 0.3 * t, 9.0), np.where(t < 30, 20.0, 20.0 + 0.8 * (t - 30))])
        pieces = []
        smooth_pieces(t, xy, 0, 59, pieces)
        cuts = sorted(lo for lo, _, _ in pieces)[1:]
        self.assertTrue(any(abs(c - 30) <= 2 for c in cuts))

    def test_flight_seen_through_the_ground_plane_is_removed_and_rolling_ball_kept(self):
        # Camera looking along +y from behind y = -40: image x ~ pitch x, image y falls with pitch y.
        matrix = np.array([[900.0, 0.0, 960.0], [0.0, -600.0, 900.0], [0.0, 0.5, 1.0]])
        geometry = Geometry(np.repeat(matrix[None], 80, axis=0), (100.0, 60.0))
        t = np.arange(80)
        rolling = np.column_stack([20.0 + 0.3 * t, np.full(80, 30.0)])
        flat, keep = flatten_flights(t, rolling, True, True, geometry)
        self.assertTrue(keep.all())
        away = geometry.away(40, np.array([32.0, 30.0]))
        height = np.where((t >= 20) & (t <= 60), 4.0 * (t - 20) * (60 - t) / 1600.0, 0.0)     # homography bias, apex 15 m
        flight = rolling + 15.0 * np.outer(height, away)
        flat, keep = flatten_flights(t, flight, True, True, geometry)
        self.assertLess(keep[25:56].mean(), 0.2)
        self.assertTrue(keep[:15].all() and keep[66:].all())


if __name__ == "__main__":
    unittest.main()
