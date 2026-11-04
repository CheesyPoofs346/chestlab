"""Chest heights, the floor you are on, and only marking the close chests."""
import json, os, sys, tempfile, unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import live
from chestlab.hud import Hud, hex_bgr, LEVEL_M

W, H = 1600, 900


class Heights(unittest.TestCase):
    def setUp(self):
        self.saved = (live.USER, live.CHESTS, live.LEARNED, live.HEIGHTS)
        d = tempfile.mkdtemp()
        live.USER, live.CHESTS, live.LEARNED, live.HEIGHTS = d, os.path.join(d, "c.json"), os.path.join(d, "l.json"), os.path.join(d, "h.json")
        self.p = live.Pipeline(dict(live.DEFAULTS), reader=object())
        c = self.p.world.chests[0]
        self.pos = tuple(c + [0.0, 25.0])             # about 2 m south of chest #1 (map y grows southwards): it is in front facing north
        self.heading = 0.0

    def tearDown(self):
        live.USER, live.CHESTS, live.LEARNED, live.HEIGHTS = self.saved

    def marker(self, i=0):
        return self.p.project(W, H, self.pos, self.heading)[i]

    def test_flat_island_is_unchanged(self):
        m = self.marker()
        self.assertEqual(m["dz"], 0.0)
        self.assertAlmostEqual(m["dist"], np.hypot(*(self.p.world.chests[0] - np.asarray(self.pos))) * self.p.world.mpp, places=6)

    def test_chest_on_a_higher_floor_is_marked_above_and_further(self):
        flat = self.marker()
        self.p.chest_h[0] = 6.0
        up = self.marker()
        self.assertAlmostEqual(up["dz"], 6.0)
        self.assertGreater(up["dist"], flat["dist"])
        self.assertLess(up["y"], flat["y"] - 20)       # drawn higher on screen

    def test_changing_floor_flips_the_marker_below_you(self):
        flat = self.marker()
        self.assertIn("floor +1", self.p.change_floor(+1))
        below = self.marker()
        self.assertAlmostEqual(below["dz"], -self.p.cfg["floor_m"])
        self.assertGreater(below["y"], flat["y"] + 20)
        self.p.change_floor(-1)
        self.assertEqual(self.marker()["dz"], 0.0)
        self.p.change_floor(+1)
        self.p.reset_dug()
        self.assertEqual(self.p.floor, 0)              # new match: back on the ground

    def test_only_close_chests_are_shown(self):
        self.p.cfg["show_m"] = 40.0
        ms = self.p.project(W, H, self.pos, self.heading)
        self.assertTrue(any(not m["show"] for m in ms) and any(m["show"] for m in ms))
        self.assertTrue(all(m["show"] == (m["dist"] <= self.p.cfg["show_m" if m["sure"] else "maybe_m"]) for m in ms))

    def test_dig_on_another_floor_saves_the_height(self):
        self.p.change_floor(+1)
        self.p.live_pos = tuple(self.p.world.chests[0])                       # standing on chest #1
        self.p.recent.extend([self.p.world.chests[0]] * 6)
        msg = self.p.dig()
        self.assertIn("height +3 m", msg)
        self.assertIn("dug", msg)
        self.assertAlmostEqual(self.p.chest_h[0], 3.0)
        with open(live.HEIGHTS) as f:
            self.assertEqual(json.load(f)[0], 3.0)
        q = live.Pipeline(dict(live.DEFAULTS), reader=object())           # survives a restart
        self.assertAlmostEqual(q.chest_h[0], 3.0)

    def test_new_spawn_learned_upstairs_keeps_its_height(self):
        self.p.change_floor(+2)
        self.p.recent.extend([np.array([5.5, 5.5])] * 6)                  # nowhere near a known chest
        msg = self.p.dig()
        self.assertIn("NEW", msg)
        self.assertIn("height +6 m", msg)
        self.assertEqual(len(self.p.chest_h), len(self.p.world.chests))
        self.assertAlmostEqual(self.p.chest_h[-1], 2 * self.p.cfg["floor_m"])
        self.p.project(W, H, self.pos, self.heading)                      # and projecting with the longer list works

    def test_short_or_missing_heights_file_is_padded(self):
        with open(live.HEIGHTS, "w") as f:
            json.dump([2.0], f)
        q = live.Pipeline(dict(live.DEFAULTS), reader=object())
        self.assertEqual(len(q.chest_h), len(q.world.chests))
        self.assertEqual(q.chest_h[0], 2.0)
        self.assertTrue((q.chest_h[1:] == 0).all())


class HeightHud(unittest.TestCase):
    def render(self, markers):
        hud, cv = Hud(W, H), np.zeros((H, W, 4), np.uint8)
        hud.render(cv, dict(markers=markers), "", hex_bgr("#3dff8a"), "", t=0.1)
        return cv

    def marker(self, **kw):
        m = dict(id=1, x=800.0, y=500.0, top=(800.0, 380.0), ring=[(800 + 40 * np.cos(a), 500 + 12 * np.sin(a)) for a in np.linspace(0, 6.28, 12)],
                 dist=12.0, dz=0.0, rel=0.0, front=True, dug=False, show=True)
        return dict(m, **kw)

    def test_up_and_down_labels_differ_from_the_flat_one(self):
        flat, up, down = (self.render([self.marker(dz=dz)]) for dz in (0.0, 4.0, -4.0))
        self.assertGreater(int(np.abs(up.astype(int) - flat).sum()), 0)
        self.assertGreater(int(np.abs(up.astype(int) - down).sum()), 0)
        self.assertTrue(abs(0.9) < LEVEL_M)                                # a small step (stairs, a slope) does not count as another floor

    def test_small_height_differences_are_not_labelled(self):
        self.assertTrue(np.array_equal(self.render([self.marker(dz=0.0)]), self.render([self.marker(dz=1.0)])))

    def test_hidden_far_chest_draws_nothing(self):
        self.assertTrue(np.array_equal(self.render([self.marker(show=False)]), self.render([])))      # same as no marker at all

    def test_chest_above_or_below_the_screen_gets_a_pill_at_that_edge(self):
        base = self.render([])
        up = self.render([self.marker(y=-120.0, top=(800.0, -300.0), dz=6.0)])
        down = self.render([self.marker(y=H + 400.0, top=(800.0, H + 250.0), dz=-6.0)])
        ys_up = np.nonzero((up[..., 3] != base[..., 3]).any(axis=1))[0]
        ys_down = np.nonzero((down[..., 3] != base[..., 3]).any(axis=1))[0]
        self.assertTrue(len(ys_up) and ys_up.max() < H * 0.35)             # top of the screen
        self.assertTrue(len(ys_down) and ys_down.min() > H * 0.65)         # bottom of the screen
        self.assertEqual(int(self.render([self.marker(y=-120.0, top=(800.0, -300.0), show=False)])[..., 3].sum()), int(base[..., 3].sum()))

    def test_pill_for_a_marker_that_is_only_partly_off_screen_is_not_used(self):
        """A beam whose base is below the bottom edge but whose top is on screen is drawn as a normal marker."""
        base = self.render([])
        m = self.render([self.marker(y=H + 200.0, top=(800.0, H - 150.0))])
        ys = np.nonzero((m[..., 3] != base[..., 3]).any(axis=1))[0]
        self.assertTrue(len(ys) and ys.min() < H * 0.9)

    def test_edge_pill_shows_height_too(self):
        off = self.marker(front=False, rel=40.0)
        flat, up = self.render([dict(off, dz=0.0)]), self.render([dict(off, dz=5.0)])
        self.assertGreater(int(np.abs(up.astype(int) - flat).sum()), 0)


if __name__ == "__main__":
    unittest.main()
