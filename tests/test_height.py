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
        self.p.new_match()
        self.assertEqual(self.p.floor, 0)              # new match: back on the ground

    def test_only_close_chests_are_shown(self):
        self.p.cfg["show_m"] = 40.0
        ms = self.p.project(W, H, self.pos, self.heading)
        self.assertTrue(any(not m["show"] for m in ms) and any(m["show"] for m in ms))
        self.assertTrue(all(m["show"] == (m["dist"] <= self.p.cfg["show_m" if m["sure"] else "maybe_m"]) for m in ms))



if __name__ == "__main__":
    unittest.main()
