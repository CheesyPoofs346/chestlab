"""A place name across the minimap (entering a named location) and a compass number on a bright lantern must not make the radar vanish."""
import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import compass, live
from chestlab.engine import compass_loop
from chestlab.heading import HeadingFilter

DATA = os.path.join(ROOT, "tests", "data")
if not os.path.isfile(os.path.join(DATA, "mm_label.png")):
    raise unittest.SkipTest("tests/data (real screenshots) is not part of the repository")
MM = cv2.imread(os.path.join(DATA, "mm_label.png"))                 # the minimap while the "LANTERN LANDING" label and a pin cover its middle
STRIP = cv2.imread(os.path.join(DATA, "compass_lantern.png"))       # the compass strip: white "77" on a glowing orange lantern


class PlaceName(unittest.TestCase):
    def test_the_minimap_locks_under_a_big_place_name(self):
        p = live.Pipeline(dict(live.DEFAULTS))
        r = p.step_minimap(MM, 1 / 15, heading=78.0)
        self.assertTrue(r["locked"] and r["on_island"])
        pos = r["pos"]
        for _ in range(5):                                           # and keeps the same spot while tracking
            r = p.step_minimap(MM, 1 / 15, heading=78.0)
            self.assertTrue(r["locked"])
        self.assertLess(float(np.hypot(*(np.asarray(r["pos"]) - pos))), 5.0)

    def test_a_far_wrong_spot_is_not_accepted_just_because_a_label_covers_the_map(self):
        f = live.Pipeline(dict(live.DEFAULTS)).tracker.fitter
        self.assertIsNone(f.fit(np.full_like(MM, 120), guess_uv=(-8098.0, 6073.0)))      # nothing to match at all


class Compass(unittest.TestCase):
    def test_a_misread_number_does_not_turn_the_heading_to_zero(self):
        class Reader:                                                # what Windows OCR does with this strip: a confident "0"
            def read(self, *a, **k):
                return 0
        W = 2000
        frame = np.zeros((1125, W, 3), np.uint8)
        x0, y0, w, h = compass.region_px(W)
        frame[y0:y0 + h, x0:x0 + w] = STRIP
        hf = HeadingFilter()
        for i in range(8):
            hf.update(0.1 * i, 78.0)
        calls = []
        stop = {"n": 0}

        def stopped():
            stop["n"] += 1
            return stop["n"] > 3
        grab = lambda box: cv2.cvtColor(frame[box["top"]:box["top"] + box["height"], box["left"]:box["left"] + box["width"]], cv2.COLOR_BGR2BGRA)
        compass_loop(grab, lambda: dict(left=0, top=0, width=W, height=1125), stopped, hf, Reader(), lambda *a: calls.append(a))
        import time
        h = hf.at(time.perf_counter())
        self.assertIsNotNone(h)
        d = abs((h[0] - 78.0 + 180.0) % 360.0 - 180.0)
        self.assertLess(d, 15.0, h)


if __name__ == "__main__":
    unittest.main()
