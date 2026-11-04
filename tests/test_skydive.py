"""A real skydive frame (pink / purple trees on the minimap, bus + ping icons, large compass digits): heading is read and the minimap locks."""
import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import compass, live

FRAME = cv2.imread(os.path.join(ROOT, "tests", "data", "gameplay.webp"))


class Skydive(unittest.TestCase):
    def test_compass_reads_the_heading(self):
        self.assertEqual(compass.CompassReader().read(FRAME), 51)

    def test_two_digit_headings_are_read(self):
        """Windows OCR returns nothing for a lone 2-digit number; the reader pads it with a rendered 0. And the map screen has no compass: None."""
        r = compass.CompassReader()
        ground = cv2.imread(os.path.join(ROOT, "tests", "data", "ground53.webp"))
        self.assertEqual(r.read(ground), 53)
        x0, y0, w, h = compass.region_px(ground.shape[1])
        self.assertEqual(r.read(ground[y0:y0 + h, x0:x0 + w], s=ground.shape[1] / 2000.0, ox=x0, oy=y0), 53)
        self.assertIsNone(r.read(cv2.imread(os.path.join(ROOT, "tests", "data", "map_ui.webp"))))

    def test_a_lock_on_the_wrong_spot_is_corrected_by_looking_at_the_whole_map(self):
        pipe = live.Pipeline(dict(live.DEFAULTS))
        mm = live.crop_minimap(cv2.imread(os.path.join(ROOT, "tests", "data", "ground53.webp")))
        wf = pipe.tracker.fitter
        self.assertIsNotNone(wf.acquire(mm))                         # a lock has been held: its zoom / storm mode is remembered
        alt = wf.verify(mm, (736.0, 3829.0), 0.5)                    # "we think we are at 736, 3829 (match 0.5)"
        self.assertIsNotNone(alt)
        self.assertLess(abs(alt.u + 5639) + abs(alt.v - 8273), 120)
        near = wf.verify(mm, (alt.u + 5, alt.v + 5), 0.7)            # already right: nothing to say
        self.assertIsNone(near)

    def test_tracking_survives_a_wrong_remembered_size_and_storm_mode(self):
        pipe = live.Pipeline(dict(live.DEFAULTS))
        mm = live.crop_minimap(cv2.imread(os.path.join(ROOT, "tests", "data", "ground53.webp")))
        wf = pipe.tracker.fitter
        fit = wf.acquire(mm)
        self.assertIsNotNone(fit)
        good = (wf.zs, wf.sx)
        wf.zs, wf.sx = 1.05, not wf.sx                      # what the last fix remembered is now wrong
        again = wf.fit(mm, guess_uv=(fit.u, fit.v))
        self.assertIsNotNone(again)                          # a neighbouring size / mode is found without a full-map search
        wf.verify(mm, (again.u, again.v), again.score)       # ... and the periodic look re-tunes the size
        self.assertAlmostEqual(wf.zs, good[0], delta=0.03)
        tuned = wf.fit(mm, guess_uv=(again.u, again.v))
        self.assertLess(abs(tuned.u - fit.u) + abs(tuned.v - fit.v), 25)

    def test_minimap_locks_on_the_ground_frame(self):
        pipe = live.Pipeline(dict(live.DEFAULTS))
        mm = live.crop_minimap(cv2.imread(os.path.join(ROOT, "tests", "data", "ground53.webp")))
        for _ in range(20):
            r = pipe.step_minimap(mm, 1 / 30, heading=53)
        self.assertTrue(r["locked"] and r["on_island"])

    def test_minimap_locks_despite_pink_trees_that_look_like_storm(self):
        pipe = live.Pipeline(dict(live.DEFAULTS))
        mm = live.crop_minimap(FRAME)
        for _ in range(30):
            r = pipe.step_minimap(mm, 1 / 30, heading=51)
        self.assertTrue(r["locked"])
        self.assertTrue(r["on_island"])
        u, v = r["pos"]
        self.assertLess(abs(u + 8053) + abs(v - 8532), 150)       # ~ the spot the lake / beach in the minimap match on the map
        self.assertFalse(pipe.tracker.fitter.sx)                 # storm masking off: the violet is vegetation here


class Hold(unittest.TestCase):
    """The open full map hides the minimap and the compass: what the sensors would 'read' there (the map screen's flat blue panels match the
    sea at 0.93, the compass comb finds garbage) must not become a position or a heading, and nothing stale may survive the close."""

    def test_no_lock_and_no_heading_while_held_then_a_fresh_lock(self):
        import time
        from chestlab.engine import Engine, FrameSource
        img = cv2.imread(os.path.join(ROOT, "real", "shot15.png"))
        H, W = img.shape[:2]
        eng = Engine(live.Pipeline(dict(live.DEFAULTS)), dict(left=0, top=0, width=W, height=H), grabber=FrameSource(img))
        eng.hold = True
        eng.start()
        try:
            time.sleep(1.5)
            st = eng.snapshot(time.perf_counter(), W, H)
            self.assertFalse(st["locked"])
            self.assertIsNone(st["heading"])
            eng.hold = False
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 6:
                time.sleep(0.05)
                st = eng.snapshot(time.perf_counter(), W, H)
                if st["locked"] and st["heading"] is not None:
                    break
            self.assertTrue(st["locked"] and st["on_island"])
            self.assertAlmostEqual(st["heading"], 170, delta=1.0)
        finally:
            eng.stop = True



class HoldInChildProcessCode(unittest.TestCase):
    """The app runs the sensors through procs.pose_main / compass_main (child processes); here the same functions run on threads."""

    def test_pose_and_compass_main_survive_hold_and_lock_after(self):
        import multiprocessing as mp
        import tempfile
        import threading
        import time
        from types import SimpleNamespace
        from chestlab import procs
        from chestlab.engine import FrameSource
        img = cv2.imread(os.path.join(ROOT, "real", "shot15.png"))
        H, W = img.shape[:2]
        P, Hh, M = mp.RawArray("d", procs.P_N), mp.RawArray("d", procs.H_N), mp.RawArray("i", 4)
        for i, v in enumerate((0, 0, W, H)):
            M[i] = v
        stop, hold = threading.Event(), SimpleNamespace(value=1)
        logf = os.path.join(tempfile.mkdtemp(), "sensors.log")
        ths = [threading.Thread(target=procs.pose_main, args=(P, M, stop, FrameSource(img), logf, hold), daemon=True),
               threading.Thread(target=procs.compass_main, args=(Hh, M, stop, FrameSource(img), logf, hold), daemon=True)]
        for t in ths:
            t.start()
        try:
            time.sleep(1.5)
            self.assertEqual(procs._read(P, procs.P_N)[3] if P[1] else 0.0, 0.0)           # held: no lock
            self.assertEqual(Hh[1], 0.0)                                                   # held: no heading
            hold.value = 0
            t0 = time.time()
            while time.time() - t0 < 8 and not (P[3] and Hh[1]):
                time.sleep(0.05)
            self.assertTrue(P[3] and P[4])                                                 # locked, on the island
            self.assertEqual(P[procs.P_BASE - 1], 0.0)                                     # no sensor error flag
        finally:
            stop.set()
            for t in ths:
                t.join(timeout=3)
        self.assertFalse(os.path.exists(logf) and "ERROR" in open(logf).read(), open(logf).read() if os.path.exists(logf) else "")



class RadarChain(unittest.TestCase):
    """Whole sensor chain (engine threads, screenshot as the screen) on the user's own frames: lock + heading + markers = the radar shows."""

    def run_frame(self, name, heading):
        import time
        from chestlab.engine import Engine, FrameSource
        img = cv2.imread(os.path.join(ROOT, "tests", "data", name))
        H, W = img.shape[:2]
        eng = Engine(live.Pipeline(dict(live.DEFAULTS)), dict(left=0, top=0, width=W, height=H), grabber=FrameSource(img))
        eng.start()
        try:
            t0 = time.perf_counter()
            st = None
            while time.perf_counter() - t0 < 10:
                time.sleep(0.05)
                st = eng.snapshot(time.perf_counter(), W, H)
                if st["locked"] and st["on_island"] and st["heading"] is not None and st["markers"]:
                    break
        finally:
            eng.stop = True
        self.assertTrue(st["locked"] and st["on_island"], name)
        self.assertAlmostEqual(st["heading"], heading, delta=1.5)
        self.assertTrue(st["markers"], name)                                  # the radar has something to draw

    def test_ground_frame(self):
        self.run_frame("ground53.webp", 53)

    def test_skydive_frame(self):
        self.run_frame("gameplay.webp", 51)


class Holding(unittest.TestCase):
    def test_hold_ends_the_moment_the_map_is_lost(self):
        from chestlab import fullmap

        class W(fullmap.MapWatch):
            active = True

        w = W(lambda: None, (100, 100))
        self.assertTrue(w.holding)                               # M just pressed, not found yet
        w.cur, w.seen = dict(M=None), True
        self.assertTrue(w.holding)                               # following it
        w.cur = None
        self.assertFalse(w.holding)                              # found, then lost: the game is back, do not wait for the watch to give up


if __name__ == "__main__":
    unittest.main()


class StormCoast(unittest.TestCase):
    """With a storm over the minimap and no fix, the tracker keeps predicting longer than in clear weather."""

    def run_misses(self, minimap, seconds):
        tr = live.Pipeline(dict(live.DEFAULTS)).tracker
        tr.fitter.fit = lambda mm, guess_uv=None: None
        tr.arrow.estimate = lambda mm: (0.0, 0.0)
        tr.filt.update(500.0, 500.0, 0.0, 1 / 30)
        r = None
        for _ in range(int(seconds * 30)):
            r = tr.step(minimap, 1 / 30, heading=0.0)
        return r

    def test_storm_coasts_longer_than_clear(self):
        clear = np.full((231, 231, 3), (140, 93, 17), np.uint8)               # open sea
        storm = np.full((231, 231, 3), (182, 51, 141), np.uint8)              # violet: under the storm
        self.assertFalse(self.run_misses(clear, 3.0)["locked"])
        self.assertTrue(self.run_misses(storm, 3.0)["locked"])
        self.assertFalse(self.run_misses(storm, 5.0)["locked"])


class StopReleasesAtOnce(unittest.TestCase):
    def test_holding_ends_the_moment_stop_is_called_even_while_the_thread_finishes(self):
        from chestlab import fullmap

        class W(fullmap.MapWatch):
            active = True                              # the thread is still in its last look

        w = W(lambda: None, (100, 100))
        self.assertTrue(w.holding)                     # M just pressed, map not found yet
        w.stop()                                       # Shift: back to the radar
        self.assertFalse(w.holding)
