"""The radar: every chest at its true bearing and flat distance, north up. Offline only (renders into a numpy canvas)."""
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import hud as H_
from chestlab import live

W, H = 2560, 1440
R = max(40, int(round(H_.RADAR_R * H)))
CX, CY = H_.RADAR_AT[0] * W, H_.RADAR_AT[1] * H
K = 0.80 * R / 60.0                                  # pixels per metre


def marker(brg, hd, **kw):
    return dict(id=1, x=-999.0, y=-999.0, top=(-999.0, -999.0), ring=None, dist=hd, dz=0.0, rel=0.0, front=False, dug=False, sure=True,
                kind="chest_sure", show=True, step=0, next=False, brg=brg, hd=hd, **kw)


def draw(markers, yaw=0.0, with_yaw=True, **extra):
    hud, cv = H_.Hud(W, H), np.zeros((H, W, 4), np.uint8)
    st = dict(markers=markers, **{**dict(radar_m=60.0, radar_up=False), **extra})       # fixed 60 m, north up unless a test says otherwise
    if with_yaw:
        st.update(yaw=yaw, fov=78.0)
    hud.render(cv, st, "", H_.hex_bgr("#3dff8a"), "", t=0.1)
    return cv


def dial_window(cv):
    x0, y0 = int(CX - R), int(CY - R)
    return cv[y0:y0 + 2 * R, x0:x0 + 2 * R].astype(int)


class Radar(unittest.TestCase):
    def centroid_of_change(self, markers, **kw):
        a, b = dial_window(draw([], **kw)), dial_window(draw(markers, **kw))
        d = np.abs(b - a).sum(axis=2)
        ys, xs = np.nonzero(d > 40)
        self.assertGreater(len(xs), 5)
        w = d[ys, xs].astype(float)
        return (xs * w).sum() / w.sum() - R, (ys * w).sum() / w.sum() - R           # relative to the dial centre

    def test_no_radar_without_a_heading(self):
        self.assertEqual(int(dial_window(draw([marker(90, 30)], with_yaw=False))[..., 3].sum()), 0)

    def test_chests_sit_at_their_true_bearing_and_distance_whatever_way_you_face(self):
        for yaw in (0.0, 137.0):                              # north up: facing never moves a chest
            for brg, hd in ((90, 30), (180, 20), (270, 40), (0, 25), (45, 50)):
                dx, dy = self.centroid_of_change([marker(brg, hd)], yaw=yaw)
                ex, ey = hd * K * math.sin(math.radians(brg)), -hd * K * math.cos(math.radians(brg))
                self.assertLess(math.hypot(dx - ex, dy - ey), 4.0, (yaw, brg, hd, dx, dy, ex, ey))

    def test_a_chest_past_the_rim_is_pinned_to_it_and_one_far_beyond_is_left_out(self):
        a, b = dial_window(draw([marker(0, 3)])), dial_window(draw([marker(0, 3), marker(90, 80)]))       # a nearer chest keeps the highlight
        d = np.abs(b - a).sum(axis=2); ys, xs = np.nonzero(d > 40); w = d[ys, xs].astype(float)
        dx, dy = (xs * w).sum() / w.sum() - R, (ys * w).sum() / w.sum() - R
        self.assertLess(abs(math.hypot(dx, dy) - 0.97 * 0.80 * R), 6.0)
        base = int(dial_window(draw([]))[..., 3].sum())
        self.assertEqual(int(dial_window(draw([marker(90, 95)]))[..., 3].sum()), base)

    def test_a_dug_chest_is_not_on_the_radar(self):
        base = draw([])
        self.assertTrue(np.array_equal(draw([dict(marker(90, 30), dug=True)]), base))

    def test_the_facing_arrow_and_view_wedge_turn_with_you(self):
        a, b = draw([], yaw=0.0), draw([], yaw=90.0)
        self.assertFalse(np.array_equal(a, b))
        up, down = dial_window(draw([], yaw=0.0))[..., 3], dial_window(draw([], yaw=180.0))[..., 3]
        # the fan opens toward where you face: facing north lights the upper half more than facing south does
        self.assertGreater(up[:R].sum(), down[:R].sum())
        self.assertGreater(down[R:].sum(), up[R:].sum())

    def test_radar_only_draws_nothing_over_the_world_and_3d_mode_still_can(self):
        far = dict(marker(90, 30), front=True, x=1280.0, y=700.0, top=(1280.0, 600.0))     # a chest straight ahead on screen
        def outside(cv):
            cv = cv.copy(); x0, y0 = int(CX - R * 1.4), int(CY - R * 1.4); cv[y0:y0 + int(2.8 * R) + 60, x0:x0 + int(2.8 * R)] = 0      # the dial and the distance under it
            return int(cv[..., 3].sum())
        self.assertEqual(outside(draw([far], markers_3d=False)), 0)
        self.assertGreater(outside(draw([far], markers_3d=True)), 0)
        self.assertGreater(outside(draw([far])), 0)                                         # key absent: old behaviour

    def test_vault_and_sprite_chest_get_their_own_icons_and_the_icon_count_is_capped(self):
        ok = lambda k, sure=True: int(dial_window(draw([dict(marker(90, 30), kind=k, sure=sure)])).sum())      # colour counts: a sprite chest is the same shape, another colour
        self.assertEqual(len({ok("chest_sure"), ok("sprite"), ok("vault")}), 3)
        self.assertNotEqual(ok("vault"), ok("vault", False))
        many = [marker(i * 5 % 360, 5 + i) for i in range(60)]
        same = [marker(i * 5 % 360, 5 + i) for i in range(H_.RADAR_ICONS)]
        self.assertTrue(np.array_equal(draw(many), draw(same)) or int(dial_window(draw(many))[..., 3].sum()) <= int(dial_window(draw(same))[..., 3].sum()) * 1.2)

    def test_heading_up_puts_what_is_ahead_at_the_top_whatever_the_compass_says(self):
        for yaw in (0.0, 90.0, 213.0):
            for off, ex, ey in ((0, 0.0, -1.0), (90, 1.0, 0.0), (180, 0.0, 1.0), (270, -1.0, 0.0)):
                dx, dy = self.centroid_of_change([marker((yaw + off) % 360, 30)], yaw=yaw, radar_up=True)
                self.assertLess(math.hypot(dx - ex * 30 * K, dy - ey * 30 * K), 4.0, (yaw, off, dx, dy))

    def test_auto_range_zooms_in_on_the_target_and_back_out(self):
        hud = H_.Hud(W, H)
        def rng(hd, t):
            cv = np.zeros((H, W, 4), np.uint8)
            hud.render(cv, dict(markers=[marker(0, hd)], yaw=0.0, fov=78.0, radar_m=0.0), "", H_.hex_bgr("#3dff8a"), "", t=t)
            return hud.lvl
        self.assertEqual(rng(50, 0.0), 80.0)
        self.assertEqual(rng(30, 0.1), 40.0)
        self.assertEqual(rng(15, 0.2), 20.0)
        self.assertEqual(rng(6, 0.3), 10.0)
        self.assertEqual(rng(9.5, 0.4), 10.0)                      # hysteresis: no flicker at the boundary
        self.assertEqual(rng(21, 0.5), 40.0)

    def test_every_chest_on_the_map_reaches_the_radar_far_from_the_island(self):
        p = live.Pipeline(dict(live.DEFAULTS), reader=object())
        far = list(range(p.n_island, len(p.world.chests)))                  # every kind: sure, rare, sprite, vault, maybe, buried
        self.assertGreater(len(far), 900)
        for i in far[::37]:                                               # a sample of chests all over the map
            pos = tuple(np.asarray(p.world.chests[i]) + [0.0, 300.0])       # 26 m south of it
            ms = {m["id"]: m for m in p.project(W, H, pos, 90.0)}
            self.assertIn(i + 1, ms, i)
            self.assertAlmostEqual(ms[i + 1]["hd"], 300.0 * p.world.mpp, delta=0.05)
            self.assertAlmostEqual(ms[i + 1]["brg"], 0.0, delta=0.5)

    def test_fix_accuracy_is_finest_on_the_island_then_world_then_storm(self):
        p = live.Pipeline(dict(live.DEFAULTS), reader=object())
        isl = p.accuracy_m((p.world.w / 2, p.world.h / 2))
        world = p.accuracy_m((p.world.w + 5000.0, p.world.h / 2))
        self.assertLess(isl, world)
        self.assertGreater(p.accuracy_m((0.0, 0.0), storm=True), world)
        v = p.view(10.0, (p.world.w / 2, p.world.h / 2), False)
        self.assertEqual(v["sigma_m"], isl)

    def test_the_accuracy_disc_grows_with_the_error_and_the_zoom(self):
        small, big = draw([], sigma_m=0.2), draw([], sigma_m=3.0)
        self.assertGreater(int(dial_window(big)[..., 3].sum()), int(dial_window(small)[..., 3].sum()))

    def test_the_gold_outline_means_100_percent_spawn_whatever_the_route_says(self):
        gold, plain = H_.radar_icon("chest_sure", True, 24), H_.radar_icon("chest", True, 24)     # a maybe-chest on a route is flagged sure
        self.assertFalse(np.array_equal(gold, plain))
        self.assertTrue(np.array_equal(plain, H_.radar_icon("chest", False, 24)))
        self.assertTrue(np.array_equal(gold, H_.radar_icon("chest_sure", False, 24)))

    def test_the_pipeline_hands_the_radar_a_bearing_and_flat_distance(self):
        p = live.Pipeline(dict(live.DEFAULTS), reader=object())
        i = 3
        pos = np.asarray(p.world.chests[i]) + [0.0, 200.0]            # 200 px south of the chest (about 17 m at 0.0875 m/px)
        ms = {m["id"]: m for m in p.project(W, H, tuple(pos), 90.0)}
        m = ms[i + 1]
        self.assertAlmostEqual(m["brg"], 0.0, delta=0.5)               # the chest is due north of you
        self.assertAlmostEqual(m["hd"], 200.0 * p.world.mpp, delta=0.05)


if __name__ == "__main__":
    unittest.main()


class RadarSpot(unittest.TestCase):
    """Config radar_at: where the radar sits (default / each corner / a custom [x, y])."""

    def test_names_and_fallbacks(self):
        for name, spot in H_.RADAR_SPOTS.items():
            self.assertEqual(H_.radar_spot(name), spot)
        self.assertEqual(H_.radar_spot(None), H_.RADAR_AT)
        self.assertEqual(H_.radar_spot("nonsense"), H_.RADAR_AT)
        self.assertEqual(H_.radar_spot([0.5, 0.5]), (0.5, 0.5))
        self.assertEqual(H_.radar_spot([5, -3]), (0.96, 0.08))                 # kept on the screen
        self.assertEqual(H_.radar_spot(["a", 1]), H_.RADAR_AT)

    def test_every_spot_draws_there(self):
        for name, (fx, fy) in H_.RADAR_SPOTS.items():
            cv = draw([marker(90, 30)], radar_at=name)
            cx, cy, r = int(fx * W), int(fy * H), int(1.25 * R)
            win = cv[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r, 3]
            self.assertGreater(int((win > 0).sum()), 0.6 * np.pi * R * R, name)               # the dial fills its window
            ys, xs = np.nonzero(cv[..., 3] > 0)
            self.assertGreaterEqual(int(xs.min()), 0)
            self.assertLessEqual(int(xs.max()), W - 1)
