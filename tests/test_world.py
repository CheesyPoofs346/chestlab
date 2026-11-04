"""The whole map: chest data, localising the minimap anywhere on it, and marking every kind of chest."""
import glob, os, sys, unittest
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import live, worldmap as W
from chestlab.hud import Hud, hex_bgr
from chestlab.world import World

WM = W.world_map()
S4 = W.Z_REAL * 16


def minimap_at(px, py):
    """The minimap as the map itself would show it with the player at z4 pixel (px, py): crop centre = player."""
    M = np.float32([[1 / S4, 0, 115.0 - px / S4], [0, 1 / S4, 115.0 - py / S4]])
    return cv2.warpAffine(cv2.imread(os.path.join(W.GG, "full_z4.png")), M, (231, 231), flags=cv2.INTER_AREA)


class Data(unittest.TestCase):
    def test_every_chest_of_the_map_is_listed(self):
        self.assertEqual(len(WM.chests), 1001)
        self.assertEqual({k: int((WM.kind == k).sum()) for k in W.KINDS}, dict(buried=69, chest_sure=265, rare=1, sprite=45, vault=36, chest=585))

    def test_coordinate_roundtrip_and_the_islands_19_spawns_are_in_the_list(self):
        uv = np.array([[123.4, -567.8], [3000.0, 90.0]])
        self.assertTrue(np.allclose(W.px_to_uv(W.uv_to_px(uv, 3), 3), uv))
        spawns = live.Pipeline(dict(live.DEFAULTS, world=False), reader=object()).world.chests
        near = np.hypot(*(WM.chests[None] - spawns[:, None]).transpose(2, 0, 1)).min(axis=1)
        self.assertLess(float(near.max()), 2.0)                       # the island spawns are fortnite.gg's own buried chests

    def test_pipeline_lists_all_and_marks_only_the_near_ones(self):
        p = live.Pipeline(dict(live.DEFAULTS, markers_3d=True), reader=object())
        self.assertEqual((len(p.world.chests), p.n_island), (1001, 19))
        far = tuple(WM.chests[(WM.kind == "rare")][0])                # a rare chest somewhere on the mainland
        ms = p.project(1600, 900, far, 0.0)
        self.assertTrue(0 < len(ms) < 60)
        self.assertEqual(min(ms, key=lambda m: m["dist"])["kind"], "rare")
        self.assertLess(min(m["dist"] for m in ms), 0.5)
        self.assertEqual(p.project(1600, 900, (-3000.0, -3000.0), 0.0), [])    # open ocean: nothing in range


class Localise(unittest.TestCase):
    def test_real_minimaps_found_from_scratch_agree_with_the_silhouette_fit(self):
        from chestlab.estimate import SilhouetteFitter
        from chestlab.live import crop_minimap
        wf, isl = W.WorldFitter(WM), SilhouetteFitter(World())
        shots = sorted(glob.glob(os.path.join(ROOT, "real", "shot*.png")))
        self.assertGreaterEqual(len(shots), 10)
        errs = []
        for f in shots:
            mm = crop_minimap(cv2.imread(f))
            a, b = wf.acquire(mm), isl.fit(mm)
            self.assertIsNotNone(a, f)
            if b is not None:
                errs.append(np.hypot(a.u - b.u, a.v - b.v) * 0.0875)
        self.assertGreater(len(errs), 10)
        self.assertLess(float(np.median(errs)), 1.5)
        self.assertLess(float(max(errs)), 6.0)

    def test_anywhere_on_the_mainland_is_found_to_better_than_a_metre(self):
        wf = W.WorldFitter(WM)
        wf.BIAS_PX = (0.0, 0.0)                                       # the rendered minimap has no shadows to compensate
        found = 0
        for px, py in [(1500, 2100), (2300, 1700), (1700, 1300), (2600, 2300), (1200, 1750)]:
            f = wf.acquire(minimap_at(px, py))
            if f is None:
                continue
            found += 1
            self.assertLess(np.hypot(*(np.array([f.u, f.v]) - W.px_to_uv([px, py], 4))) * 0.0875, 1.0, (px, py))
        self.assertGreaterEqual(found, 4)

    def test_open_sea_is_refused_not_guessed(self):
        wf = W.WorldFitter(WM)
        self.assertIsNone(wf.acquire(np.full((231, 231, 3), (140, 93, 17), np.uint8)))

    def test_tracker_locks_on_the_map_and_follows_a_walk(self):
        tk = W.WorldTracker(World(), WM)
        tk.fitter.BIAS_PX = (0.0, 0.0)                                # (the rendered minimap has no shadows)
        px = np.array([1700.0, 1300.0])
        for i in range(24):
            r = tk.step(minimap_at(*(px + [i * 1.5, i * 0.5])), dt=1 / 15, heading=0.0)
        self.assertTrue(r["locked"] and r["on_island"])
        want = W.px_to_uv(px + [23 * 1.5, 23 * 0.5], 4)
        self.assertLess(float(np.hypot(*(np.array(r["pos"]) - want))) * 0.0875, 3.0)


class Draw(unittest.TestCase):
    def test_every_kind_of_chest_has_its_own_colour(self):
        def render(kind):
            hud, cv = Hud(1600, 900), np.zeros((900, 1600, 4), np.uint8)
            m = dict(id=1, x=800.0, y=500.0, top=(800.0, 380.0), ring=None, dist=12.0, dz=0.0, rel=0.0, front=True, dug=False, show=True, sure=True, kind=kind)
            hud.render(cv, dict(markers=[m]), "", hex_bgr("#3dff8a"), "", t=0.1)
            return cv
        imgs = {k: render(k) for k in ("buried", "chest", "rare", "sprite", "vault")}
        for a in imgs:
            for b in imgs:
                if a < b:
                    self.assertGreater(int(np.abs(imgs[a].astype(int) - imgs[b]).sum()), 0, (a, b))


if __name__ == "__main__":
    unittest.main()
