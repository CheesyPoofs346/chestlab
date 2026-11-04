import os, sys, time, unittest
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from chestlab.compass import BarComb, CompassReader, region_px
from chestlab.engine import Engine, FrameSource
from chestlab.heading import HeadingFilter
from chestlab.hud import Hud, hex_bgr
from chestlab.live import DEFAULTS, Pipeline

REAL = os.path.join(ROOT, "real")
TRUTH = {15: 170, 16: 138, 20: 150, 22: 235, 25: 271, 26: 260}   # compass numbers read off the screenshots by eye


def shot(k):
    return cv2.imread(os.path.join(REAL, "shot%d.png" % k))


class HeadingFilterTest(unittest.TestCase):
    def test_follows_a_flick_with_dropouts(self):
        import heading_sim
        e, r = heading_sim.run(seed=3, ocr_hz=80, drop=0.10, latency=0.008, comp=0.008)
        self.assertLess(np.percentile(e, 95), 2.5)
        self.assertLess(np.percentile(e[r < 5], 95), 1.0)
        old, _ = heading_sim.run(seed=3, old=True)
        self.assertLess(np.percentile(e, 95), np.percentile(old, 95) / 5)

    def test_single_misread_is_ignored_but_a_real_jump_is_taken(self):
        f = HeadingFilter()
        for i in range(20):
            f.update(i * 0.01, 100)
        f.update(0.20, 217)                                  # lone misread
        self.assertAlmostEqual(f.at(0.21)[0], 100, delta=1.0)
        f.update(0.21, 40); f.update(0.22, 40)               # two agreeing readings: a real jump
        self.assertAlmostEqual(f.at(0.23)[0], 40, delta=2.0)

    def test_wraps_through_north(self):
        f = HeadingFilter()
        for i in range(60):
            f.update(i * 0.01, round(350 + i * 0.5) % 360)   # 50 deg/s through 0
        h, om, _ = f.at(0.6)
        self.assertLess(abs((h - 20 + 180) % 360 - 180), 2.0)
        self.assertAlmostEqual(om, 50, delta=20)   # quantised 1 deg steps: rate is noisy, heading is what matters


@unittest.skipUnless(os.path.isdir(REAL), "needs the real screenshots")
class RealShots(unittest.TestCase):
    def test_compass_number_and_region_capture_agree(self):
        r = CompassReader()
        for k, h in TRUTH.items():
            img = shot(k)
            x0, y0, w, hh = region_px(img.shape[1])
            reg = img[y0:y0 + hh, x0:x0 + w]
            self.assertEqual(r.read(img), h, k)
            self.assertEqual(r.read(reg, s=img.shape[1] / 2000.0, ox=x0, oy=y0), h, k)

    def test_label_comb_matches_number(self):
        b = BarComb()
        for k, h in TRUTH.items():
            img = shot(k)
            x0, y0, w, hh = region_px(img.shape[1])
            est, q = b.estimate(img[y0:y0 + hh, x0:x0 + w], h + 3.0, s=img.shape[1] / 2000.0, ox=x0, oy=y0)
            self.assertLess(abs(est - h), 2.0, (k, est))
            self.assertGreater(q, 1.25, k)

    def test_engine_locks_on_a_screenshot_without_any_window(self):
        img = shot(15)
        H, W = img.shape[:2]
        eng = Engine(Pipeline(dict(DEFAULTS)), dict(left=0, top=0, width=W, height=H), grabber=FrameSource(img))
        eng.start()
        t0 = time.perf_counter()
        st = None
        while time.perf_counter() - t0 < 6:
            time.sleep(0.05)
            st = eng.snapshot(time.perf_counter(), W, H)
            if st["locked"] and st["heading"] is not None and st["markers"]:
                break
        eng.stop = True
        self.assertTrue(st["locked"] and st["on_island"])
        self.assertAlmostEqual(st["heading"], 170, delta=1.0)
        self.assertAlmostEqual(st["yaw"], st["heading"] + eng.pipe.cfg["yaw_off"])      # the radar's arrow
        self.assertTrue(all("brg" in m and "hd" in m for m in st["markers"]))
        self.assertGreaterEqual(len(st["markers"]), eng.pipe.n_island)     # the island's spawns (the rest of the map is out of range)
        self.assertGreater(eng.rate_c.v, 30)     # heading sensor loop rate, Hz


class HudTest(unittest.TestCase):
    def test_dirty_rects_clear_the_previous_frame(self):
        pipe = Pipeline(dict(DEFAULTS), reader=object())
        W, H = 1600, 900
        m = pipe.project(W, H, tuple(pipe.world.chests[0] + [5, 5]), 90.0)
        hud, cv = Hud(W, H), np.zeros((H, W, 4), np.uint8)
        hud.render(cv, dict(markers=m), "status", hex_bgr("#3dff8a"), "toast", t=0.1)
        self.assertGreater(int(cv[..., 3].sum()), 0)
        hud.render(cv, dict(markers=[]), "status", hex_bgr("#3dff8a"), "", t=0.2)
        # only the status line remains; markers and the toast must be gone
        ys, xs = np.nonzero(cv[..., 3])
        self.assertTrue(len(ys) and ys.min() > H * 0.85 and xs.max() < W * 0.6)

    def test_minimap_and_compass_areas_stay_transparent(self):
        pipe = Pipeline(dict(DEFAULTS), reader=object())
        W, H = 2000, 1125
        m = pipe.project(W, H, tuple(pipe.world.chests[0] + [3, 3]), 0.0)
        for mk in m:
            mk["x"], mk["y"], mk["top"] = 1900.0, 100.0, (1900.0, 20.0)        # force markers onto the minimap
        hud, cv = Hud(W, H), np.zeros((H, W, 4), np.uint8)
        hud.render(cv, dict(markers=m), "", hex_bgr("#3dff8a"), "", t=0.1)
        self.assertEqual(int(cv[6:230, 1770:1996, 3].max()), 0)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.path.isdir(REAL), "needs the real screenshots")
class WholeApp(unittest.TestCase):
    def test_main_loop_headless_on_a_screenshot(self):
        import live_app
        from chestlab.overlay import NullOverlay
        img = shot(15)
        H, W = img.shape[:2]
        out = live_app.main(overlay=NullOverlay, grabber=FrameSource(img), hotkeys=False, rect=dict(left=0, top=0, width=W, height=H), seconds=5.0)
        st = out["state"]
        self.assertGreater(out["frames"], 50)                                    # the render loop really ran (smoke test, not a speed test)
        self.assertTrue(st["locked"] and len(st["markers"]) > 0)
        self.assertIsNone(out["eng"].err)
        self.assertGreater(int(out["ov"].canvas[..., 3].sum()), 0)               # something was drawn
        self.assertEqual(int(out["ov"].canvas[6:230, 1770:1996, 3].max()), 0)    # minimap area left clear
